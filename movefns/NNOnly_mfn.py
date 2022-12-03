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
    MaxHedron,
    MaxDihedron,
)
from Bio.PDB.Chain import Chain
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
        rndx, dndx, hndx = 0, 0, 0
        for ric in cic.ordered_aa_ic_list:

            """
            # testing code
            chn = (cic.chain.full_id[0] + cic.chain.full_id[2]).upper()
            rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")
            rk = pqry1(
                self.cur,
                "select res_key from residue r, chain c where"
                f" c.chain_name = '{chn}' and r.res_seqpos = '{rsp}' and c.chain_key = r.chain_key",
            )
            self.curj = self.conn.cursor(cursor_factory=DictCursor)
            self.curj.execute(
                f"select jdict from eagn where res_key = {rk} and grid_ref = {self.gref}"
            )
            ea_grid = self.curj.fetchone()[0]
            gdc2 = copy.deepcopy(self.gridDict)
            for gp in gdc2.keys():
                gdc2[gp][0:3] = [0.0, 0.0, 0.0]  # pre-normalize
            for gNdx in ea_grid:
                # replace grid voxels with any voxels populated around this residue
                # this is only populated voxels, so index is only relevant to dict
                gdc2[int(gNdx)] = ea_grid[gNdx]

            # make numpy array (used voxels over all db, with changes for this residue)
            gridArr2 = torch.tensor([gp for gp in gdc2.values()], dtype=torch.float32)
            gridArr2np = xp.array([gp for gp in gdc2.values()], dtype=xp.float32)
            """

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

            """
            # testing code
            assert (torch.all(torch.eq(gridArr, gridArr2)))
            """

            if not self.nnconfig["dataset"]["learnXres"]:
                inputArr = torch.cat(
                    (
                        gridArr.flatten(),
                        torch.tensor(resMap[ric.lc], dtype=torch.float32),
                    )
                )
            else:
                inputArr = gridArr.flatten()

            if self.configuration["args"].fake:
                rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")
                rk = pqry1(
                    self.cur,
                    "select res_key from residue r, chain c where"
                    f" c.chain_name = '{chn}' and r.res_seqpos = '{rsp}' and c.chain_key = r.chain_key",
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

            dhChrArr, dhLenArr, hArr = (
                pred[0:MaxDihedron],
                pred[MaxDihedron : 2 * MaxDihedron],
                pred[2 * MaxDihedron : 2 * MaxDihedron + 3 * MaxHedron],
            )

            # truncate NN output for specific residue
            dhLenArr = dhLenArr[0 : len(self.dihedraMap[ric.lc])]
            dhChrArr = dhLenArr[0 : len(self.dihedraMap[ric.lc])]
            hArr = xp.reshape(hArr[0 : 3 * len(self.hedraMap[ric.lc])], (-1, 3))

            # denormalize from dbmng.py:gnoLengths() code:
            # "normalise dihedra len to -1/+1"
            dhLenArr = (((dhLenArr + 1) / 2) * self.dhRange) + self.dhMinLen
            # "normalise hedra len to -1/+1"
            hArr = (((hArr + 1) / 2) * self.hRangeArr) + self.hMinLenArr

            dhChrArr[dhChrArr < 0] = -1
            dhChrArr[dhChrArr >= 0] = 1

            # working here on angle order
            for dh in dict(
                sorted(ric.dihedra.items(), key=lambda item: item[1].e_class)
            ):
                dho = ric.dihedra[dh]
                print(
                    dh, dho.e_class, cic.dihedraL14[dho.ndx], cic.dihedra_signs[dho.ndx]
                )
            print(dhLenArr)
            print(dhChrArr)
            for h in dict(sorted(ric.hedra.items(), key=lambda item: item[1].e_class)):
                ho = ric.hedra[h]
                print(
                    h,
                    ho.e_class,
                    cic.hedraL12[ho.ndx],
                    cic.hedraL23[ho.ndx],
                    cic.hedraL13[ho.ndx],
                )
            print(hArr)
            dhndx = 0
            if len(ric.rprev) != 0:
                rp = ric.rprev[0]
                ndx = ric.pick_angle("omg").ndx
                print(f"p-omg d14 {cic.dihedraL14[ndx]:.3f} <- {dhLenArr[dhndx]:.3f}")
                print(
                    f"p-omg chr {cic.dihedra_signs[ndx]:.3f} <- {dhChrArr[dhndx]:.3f}"
                )
                # cic.dihedraL14[ndx] = dhLenArr[dhndx]
                # cic.dihedra_signs[ndx] = dhChrArr[dhndx]
                dhndx += 1
                ndx = ric.pick_angle("phi").ndx
                print(f"p-phi d14 {cic.dihedraL14[ndx]:.3f} <- {dhLenArr[dhndx]:.3f}")
                print(
                    f"p-phi chr {cic.dihedra_signs[ndx]:.3f} <- {dhChrArr[dhndx]:.3f}"
                )
                # cic.dihedraL14[ndx] = dhLenArr[dhndx]
                # cic.dihedra_signs[ndx] = dhChrArr[dhndx]
                dhndx += 1
            else:
                dhndx += 2
            """
            for dl, dc in zip(dhLenArr, dhChrArr):
                print("d14", cic.dihedraL14[dndx], dl)
                cic.dihedraL14[dndx] = dl
                print("dc", cic.dihedra_signs[dndx], dc)
                cic.dihedra_signs[dndx] = dc
                dndx += 1
            for hl in hArr:
                cic.hedraL12[hndx] = hl[0]
                cic.hedraL23[hndx] = hl[1]
                cic.hedraL13[hndx] = hl[2]
                hndx += 1
            """
            # compute E for last conformation because convenient

            distArrR = xp.linalg.norm(
                locArr[:, None, :] - self.grids[ric.lc][None, :, :], axis=-1
            )
            # sortedX = xp.argpartition(distArrX, self.smArr)
            sortedR = xp.argpartition(distArrR, self.smArr)

            # distribArrX += self.distributeGridPoints(
            #    crArr, distArrX, self.maxd, sortedX
            # )
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
        return globAvg, resArr
