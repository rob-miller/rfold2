"""potential energy function based on environment grid around residue"""

from .base_mfn import BaseMfn
from data.gridExplore import gridProt
from data.dbLoad import dbLoad
from utils.rdb import openDb, pqry, pqry1  # , pqry1p
from utils.rpdb import (
    resList,
    crMapH,
    crMapNoH,
    NoAtomCR,
    resMap,
    outputArr2values,
    MaxHedron,
    MaxDihedron,
)
from Bio.PDB.Chain import Chain
from Bio.PDB.internal_coords import IC_Chain  # AtomKey

from datasets import get_dataset
from utils import parse_configuration
from models import get_model, checkpoint_load
from torchinfo import summary

# from psycopg2.extras import DictCursor
import torch

import numpy as xp
import math
import copy

# from utils.rdb import openDb, pqry, pqry1


class NNOnlyMfn(BaseMfn):
    """Motion class to compute nest step using only neural net prediction."""

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class."""
        self.configuration = configuration
        self.gp = gridProt([])  # no targList but initialize other data
        self.step = float(configuration["grid_resolution"])
        self.grids, self.points = self.gp.getGrids(self.step)
        self.dbl = dbLoad()
        self.maxd = math.sqrt(
            3 * (self.step * self.step)
        )  # diagonal between 2 grid points
        self.smArr = [n for n in range(self.gp.search_multi)]  # used by argpartition
        self.nlp = self.gp.getGPnlp(self.points, self.step)

        gridRes = self.configuration["grid_resolution"]
        hydrogens = self.configuration["hydrogens"]
        resChar = self.configuration["res_char"]

        self.conn = openDb()
        self.cur = self.conn.cursor()

        # worth it to save one input for every voxel
        if hydrogens:
            self.crMap = crMapH
        else:
            self.crMap = crMapNoH
        NoAtom = self.crMap[NoAtomCR]

        # get grid points for gridRes and resChar
        self.gref = pqry1(
            self.cur,
            "select grid_ref from grid where"
            f" res_char='{resChar}' and step={gridRes}",
        )
        pqry(
            self.cur,
            f"select index, x, y, z from grid_point where grid_ref = {self.gref} order by index",
        )

        # grid needed as dict because grid_point data is only used voxels, not all voxels
        self.gridDict = {
            gp[0]: [gp[1], gp[2], gp[3]] + NoAtom
            for gp in self.cur.fetchall()
            # gp[0]: [gp[1], gp[2], gp[3]] for gp in self.cur.fetchall()
        }

        # prepare grid normalisation parameters
        self.gridMinArr = (
            torch.tensor(
                [gp[0:3] for gp in self.gridDict.values()], dtype=torch.float32
            )
            # xp.array([gp[0:3] for gp in self.gridDict.values()], dtype=xp.float32)
            - self.step
        )

        pqry(self.cur, "select cr_class, acr_key from atom_covalent_radii;")
        self.acr = {k: v for (k, v) in self.cur.fetchall()}

        self.model = self.getNN(configuration)
        self.getResAngleMap()
        self.getNormFactors()
        # print("hello")

    def getNN(self, configDict):
        args = configDict["args"]
        print(f"Reading config file {configDict['netconfig']}...")
        config = parse_configuration(configDict["netconfig"])

        if "cudalist" in configDict:
            config["model"]["devlist"] = f"cuda:{configDict['cudalist']}"
        if args.cudalist:
            config["model"]["devlist"] = f"cuda:{args.cudalist}"
        if ("cpu" in configDict and configDict["cpu"]) or args.cpu:
            for i in range(len(config["model"]["devlist"])):
                config["model"]["devlist"][i] = "cpu"

        print(f"devlist: {config['model']['devlist']}")
        full_dataset = get_dataset(config["dataset"])  # need for in, out dims
        if hasattr(full_dataset, "inputDim"):
            config["model"]["input_dim"] = full_dataset.inputDim
            config["model"]["output_dim"] = full_dataset.outputDim
        if args.fake:
            full_dataset.__init__(config["dataset"])
            self.dataset = full_dataset
        print(f"input dimension {config['model']['input_dim']}")
        print(f"output dimension {config['model']['output_dim']}")

        if args.fake:
            model = None
            print("fake from database, no nn load")
        else:
            print("Initializing model...")
            model = get_model(config["model"])
            for d in range(len(config["model"]["devlist"])):
                model.netlist[d].to(config["model"]["devlist"][d])
                summary(model.netlist[0])
            print("------")

            summary(model)
            epochs = config["process"]["epochs"]
            load_epoch = 0
            if config["checkpoint"]["load"] != -1:  # can use "last" instead of number
                if checkpoint_load(model, config):
                    load_epoch = config["checkpoint"]["load"]
                    print(f"re-loaded model from epoch {load_epoch}...")
                    epochs += load_epoch
        self.nnconfig = config
        return model

    def getResAngleMap(self):
        self.dihedraMap = {}
        self.hedraMap = {}
        for resChar in resList:
            rk = pqry1(
                self.cur,
                f"select res_key from residue where res_char = '{resChar}' limit 1",
            )
            pqry(
                self.cur,
                "select dc.d_class from dihedron d, dihedron_class dc"
                f" where d.res_key = {rk} and d.class_key = dc.dc_key"
                " order by dc.d_class",
            )
            self.dihedraMap[resChar] = [x[0] for x in self.cur.fetchall()]
            pqry(
                self.cur,
                "select hc.h_class from hedron h, hedron_class hc"
                f" where h.res_key = {rk} and h.class_key = hc.hc_key"
                " order by hc.h_class",
            )
            self.hedraMap[resChar] = [x[0] for x in self.cur.fetchall()]

    def getNormFactors(self):
        normFactors = {}
        for leng in ("len12", "len23", "len13", "len14"):
            pqry(
                self.cur,
                f"select min, range from len_normalization where name = '{leng}'",
            )
            normFactors[leng] = self.cur.fetchall()[0]

        self.hMinLenArr = xp.array(
            [normFactors[lx][0] for lx in ("len12", "len23", "len13")]
        )
        self.hRangeArr = xp.array(
            [normFactors[lx][1] for lx in ("len12", "len23", "len13")]
        )
        (self.dhMinLen, self.dhRange) = normFactors["len14"]

    def move(self, chain):
        """Return next chain conformation"""
        if isinstance(chain, IC_Chain):
            cic = chain
        else:
            if not isinstance(chain, Chain):
                for c in chain.get_chains():
                    break
                chain = c
            if not chain.internal_coord:
                chain.atom_to_internal_coordinates()
            cic = chain.internal_coord
        distPlot = cic.distance_plot()
        if not hasattr(cic, "dihedra_signs"):
            # create arrays to hold h13, d14, dsign
            # compute hedra L13 and dihedra L14
            dhSigns = (cic.dihedral_signs()).astype(int)
            cic.distplot_to_dh_arrays(distPlot, dhSigns)

        coordDict = self.dbl.loadResidueEnvirons(cic, distPlot)
        if self.configuration["args"].fake:
            chn = (cic.chain.full_id[0] + cic.chain.full_id[2]).upper()
        resArr = xp.zeros([len(cic.ordered_aa_ic_list)], dtype=float)
        rndx = 0
        for ric in cic.ordered_aa_ic_list:
            crArr, locArr0 = coordDict[ric]
            locArr = xp.array([la[0:3] for la in locArr0])

            distArrX = xp.linalg.norm(
                locArr[:, None, :] - self.grids["X"][None, :, :], axis=-1
            )
            assignedX = self.gp.assignGridPoints(distArrX, None)

            gridDictCopy = copy.deepcopy(self.gridDict)

            for i in range(len(crArr)):
                gridDictCopy[assignedX[i]][0:3] = locArr[i]
                gridDictCopy[assignedX[i]][3:] = self.crMap[self.acr[crArr[i]]]

            gridArr = torch.tensor(
                [gp for gp in gridDictCopy.values()], dtype=torch.float32
            )

            # normalise coords to -1/+1 - so all unpopulated grid points go to 0,0,0 atom type 0
            gridArr[:, 0:3] = (
                2 * ((gridArr[:, 0:3] - self.gridMinArr) / (2 * self.step))
            ) - 1

            # get input arr for NN
            if not self.nnconfig["dataset"]["learnXres"]:
                inputArr = torch.cat(
                    (
                        gridArr.flatten(),
                        torch.tensor(resMap[ric.lc], dtype=torch.float32),
                    )
                )
            else:
                inputArr = gridArr.flatten()

            # get output arr from NN or fake from db
            if self.configuration["args"].fake:
                rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")
                rk = pqry1(
                    self.cur,
                    "select res_key from residue r, chain c where"
                    f" c.chain_name = '{chn}' and r.res_seqpos = '{rsp}'"
                    " and c.chain_key = r.chain_key",
                )

                # try:
                inp, pred = self.dataset.getRkInOut(rk, ric.lc)
                inp = inp.numpy()
                # except(TypeError):  # no data for terminal residues
                #    pred = self.model(inputArr).cpu().detach().numpy()
                inputArr = inputArr.numpy()
                # for i in range(len(inp)):
                #    print(f"{i} {inp[i]:.3f} {inputArr[i]:.3f}")
            else:
                pred = self.model(inputArr).cpu().detach().numpy()
            if torch.is_tensor(pred):
                pred = pred.numpy()

            # extract, denormalise, clean results
            dhChrArr, dhLenArr, hArr = outputArr2values(ric.lc, pred)

            # working here on angle order
            dhndx = 0
            dhs = sorted(ric.dihedra)  # AtomKey sort on tuples
            ndxlst = []
            if len(ric.rprev) != 0:
                ndxlst = [ric.pick_angle("omg").ndx, ric.pick_angle("phi").ndx]
            else:
                dhndx = 2
            ndxlst += [ric.dihedra[x].ndx for x in dhs if x[2] in ric]

            """
            # print(ndxlst)
            print(cic.dihedraL14[ndxlst])
            print(dhLenArr[dhndx:])
            print(cic.dihedra_signs[ndxlst])
            print(dhChrArr[dhndx:])
            """
            cic.dihedraL14[ndxlst] = dhLenArr[dhndx:]
            cic.dihedra_signs[ndxlst] = dhChrArr[dhndx:]

            hndx = 0
            hs = sorted(ric.hedra)
            if len(ric.rprev) != 0:
                rp = ric.rprev[0]
                ndxlst = [
                    rp.pick_angle("CA:C:1N").ndx,
                    rp.pick_angle("C:1N:1CA").ndx,
                    rp.pick_angle("1N:1CA:1C").ndx,
                ]
            else:
                ndxlst = []  # [ric.pick_angle("N:CA:C").ndx]
                hndx = 2
            pass
            ndxlst += [ric.hedra[x].ndx for x in hs if x[1] in ric]

            """
            # print(ndxlst)
            print(cic.hedraL12[ndxlst])
            print(hArr[hndx:, 0])
            print(cic.hedraL23[ndxlst])
            print(hArr[hndx:, 1])
            print(cic.hedraL13[ndxlst])
            print(hArr[hndx:, 2])
            """

            cic.hedraL12[ndxlst] = hArr[hndx:, 0]
            cic.hedraL23[ndxlst] = hArr[hndx:, 1]
            cic.hedraL13[ndxlst] = hArr[hndx:, 2]

            # compute E for last conformation because convenient
            distArrR = xp.linalg.norm(
                locArr[:, None, :] - self.grids[ric.lc][None, :, :], axis=-1
            )
            sortedR = xp.argpartition(distArrR, self.smArr)

            distribArrRC = self.gp.distributeGridPoints(
                crArr, distArrR, self.maxd, sortedR
            )

            # avg energy per contact
            # local env * neg-log-prob array, divide by local env contacts to reward contacts/compactness over fewer contacts
            resArr[rndx] = xp.sum(distribArrRC * self.nlp[ric.lc]) / xp.count_nonzero(
                distribArrRC
            )
            rndx += 1

        globAvg = xp.sum(resArr) / len(resArr)
        cic.distance_to_internal_coordinates()
        cic.internal_to_atom_coordinates()
        return globAvg, resArr, cic
