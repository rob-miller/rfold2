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
    dhlNorm,
    getNormValues,
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

    def getNN(self, configDict):
        args = configDict["args"]
        print(f"Reading config file {configDict['netconfig']}...")
        config = parse_configuration(configDict["netconfig"])

        if "cudalist" in configDict:
            config["model"]["devlist"] = f"cuda:{configDict['cudalist']}"
            # config["model"]["devlist"] = []
            # for dev in configDict['cudalist']:
            #    config["model"]["devlist"].append(f"cuda:{dev}")
        if args.cudalist:
            config["model"]["devlist"] = []
            for dev in args.cudalist:
                config["model"]["devlist"].append(f"cuda:{dev}")
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

    def stepwise(self, cic, dndxlst, hndxlst, hndx, tdhLenArr, tdhChrArr, thArr):
        # move toward predicted angles by stepfrac
        dstepfrac = self.configuration[
            "dstepfrac"
        ]  # move dihedra this fraction of difference to predicted
        hstepfrac = self.configuration[
            "hstepfrac"
        ]  # move hedra this fraction of difference to predicted
        flipMin = self.configuration[
            "flipMin"
        ]  # abs val of normalized +/-1 dihedral must be greater to flip

        normDict = getNormValues()
        # get current dihedrals this residue
        rdhl14 = cic.dihedraL14[dndxlst]
        # get target dihedrals this residue
        tdhl14 = tdhLenArr
        # get normalized current dihedrals this residue
        nrdhl14 = dhlNorm(rdhl14)
        # get current chiralities this residue
        newChr = cic.dihedra_signs[dndxlst]
        # get chiralities that need to flip
        chrFlip = xp.not_equal(newChr, tdhChrArr)
        # get chiralities allowed to flip
        flipOk = xp.greater_equal(xp.abs(nrdhl14), flipMin)
        flipOk = xp.logical_and(flipOk, chrFlip)
        # flip those chiralities this residue
        if dstepfrac > 0:  # allow for no change
            newChr[flipOk] *= -1
        # chiralities done

        # get chiralities to flip but not allowed
        flipWait = xp.logical_and(xp.logical_not(flipOk), chrFlip)
        # blanket change all residue dhLen by fraction
        deltal14 = (tdhl14 - rdhl14) * dstepfrac
        # marker for debugging
        # deltal14[chrFlip] = 1
        # flips already moving so no dhLen change there
        deltal14[flipOk] = 0

        # compute delta for flipWait cells
        # distance between going through max
        fwMaxTour = xp.full_like(deltal14, 0.0)
        fwMaxTour[flipWait] = (normDict["maxLen14"] - tdhl14[flipWait]) + (
            normDict["maxLen14"] - rdhl14[flipWait]
        )
        # distance between going through min
        fwMinTour = xp.full_like(deltal14, 0.0)
        fwMinTour[flipWait] = (tdhl14[flipWait] - normDict["len14"][0]) + (
            rdhl14[flipWait] - normDict["len14"][0]
        )
        # choose best max tour or min tour
        fwMax = xp.greater_equal(fwMaxTour, fwMinTour)
        fwMax[~flipWait] = False
        fwMin = xp.logical_not(fwMax)
        fwMin[~flipWait] = False
        deltal14[fwMax] = fwMaxTour[fwMax] * dstepfrac
        deltal14[fwMin] = -fwMinTour[fwMin] * dstepfrac

        # now hedra
        rhl12 = cic.hedraL12[hndxlst]
        rhl23 = cic.hedraL23[hndxlst]
        rhl13 = cic.hedraL13[hndxlst]
        deltahl12 = (thArr[hndx:, 0] - rhl12) * hstepfrac
        deltahl23 = (thArr[hndx:, 1] - rhl23) * hstepfrac
        deltahl13 = (thArr[hndx:, 2] - rhl13) * hstepfrac

        return (deltal14, newChr, deltahl12, deltahl23, deltahl13)

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

        # get dict of filtered, transformed env atoms
        coordDict = self.dbl.loadResidueEnvirons(cic, distPlot)
        if self.configuration["args"].fake:
            # do here for db query so not repeated in loop
            chn = (cic.chain.full_id[0] + cic.chain.full_id[2]).upper()

        # resArr is energy at each residue position
        resArr = xp.zeros([len(cic.ordered_aa_ic_list)], dtype=float)
        rndx = 0
        for ric in cic.ordered_aa_ic_list:
            # get env atom coords in AA space
            crArr, locArr0 = coordDict[ric]
            locArr = xp.array([la[0:3] for la in locArr0])

            # assign env atoms to grid points and make local dict copy
            distArrX = xp.linalg.norm(
                locArr[:, None, :] - self.grids["X"][None, :, :], axis=-1
            )
            assignedX = self.gp.assignGridPoints(distArrX, None)

            gridDictCopy = copy.deepcopy(self.gridDict)

            for i in range(len(crArr)):
                try:
                    gridDictCopy[assignedX[i]][0:3] = locArr[i]
                    gridDictCopy[assignedX[i]][3:] = self.crMap[self.acr[crArr[i]]]
                except KeyError:
                    pass
            gridArr = torch.tensor(
                [gp for gp in gridDictCopy.values()], dtype=torch.float32
            )

            # normalise env atom coords on grid
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

                inp, pred = self.dataset.getRkInOut(rk, ric.lc)
                inp = inp.numpy()
                inputArr = inputArr.numpy()
                # for i in range(len(inp)):
                #    print(f"{i} {inp[i]:.3f} {inputArr[i]:.3f}")
            else:
                pred = self.model(inputArr).cpu().detach().numpy()
            if torch.is_tensor(pred):
                pred = pred.numpy()

            # extract, denormalise, clean results
            dhChrArr, dhLenArr, hArr, dhLenArrNorm, hArrNorm = outputArr2values(
                ric.lc, pred
            )

            # create index list mapping NN output array to chain interatom distance
            # arrays e.g. dihedraL14, hedraL12.  Problems are AtomKey sort and
            # prev residue if present holds some angles
            # copy computed values from NN to chain interatom distance arrays

            # dihedra dndxlst
            dhndx = 0
            dhs = sorted(ric.dihedra)  # AtomKey sort on tuples
            dndxlst = []
            if len(ric.rprev) != 0:
                dndxlst = [ric.pick_angle("omg").ndx, ric.pick_angle("phi").ndx]
            else:
                dhndx = 2
            dndxlst += [ric.dihedra[x].ndx for x in dhs if x[2] in ric]

            """
            # print(ndxlst)
            print(cic.dihedraL14[dndxlst])
            print(dhLenArr[dhndx:])
            print(cic.dihedra_signs[dndxlst])
            print(dhChrArr[dhndx:])
            """

            # hedra hndxlst
            hndx = 0
            hs = sorted(ric.hedra)
            if len(ric.rprev) != 0:
                rp = ric.rprev[0]
                hndxlst = [
                    rp.pick_angle("CA:C:1N").ndx,
                    rp.pick_angle("C:1N:1CA").ndx,
                    rp.pick_angle("1N:1CA:1C").ndx,
                ]
            else:
                hndxlst = []  # [ric.pick_angle("N:CA:C").ndx]
                hndx = 2
            pass
            hndxlst += [ric.hedra[x].ndx for x in hs if x[1] in ric]

            """
            # print(ndxlst)
            print(cic.hedraL12[hndxlst])
            print(hArr[hndx:, 0])
            print(cic.hedraL23[hndxlst])
            print(hArr[hndx:, 1])
            print(cic.hedraL13[hndxlst])
            print(hArr[hndx:, 2])
            """

            if self.configuration["stepwise"]:
                (deltal14, newChr, deltahl12, deltahl23, deltahl13) = self.stepwise(
                    cic,
                    dndxlst,
                    hndxlst,
                    hndx,
                    dhLenArr[dhndx:],
                    dhChrArr[dhndx:],
                    hArr,
                )

                cic.dihedraL14[dndxlst] += deltal14
                cic.dihedra_signs[dndxlst] = newChr
                cic.hedraL12[hndxlst] += deltahl12
                cic.hedraL23[hndxlst] += deltahl23
                cic.hedraL13[hndxlst] += deltahl13
            else:

                cic.dihedraL14[dndxlst] = dhLenArr[dhndx:]
                cic.dihedra_signs[dndxlst] = dhChrArr[dhndx:]
                cic.hedraL12[hndxlst] = hArr[hndx:, 0]
                cic.hedraL23[hndxlst] = hArr[hndx:, 1]
                cic.hedraL13[hndxlst] = hArr[hndx:, 2]

            # done updating chain interatom distance arrays for this residue

            # compute E for last conformation (resArr) because convenient here
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

        # done computing per residue env energy for input conformation, updated
        # interatom distance arrays for chain

        # compute global energy for input conformation by averaging per residue
        # env energies
        globAvg = xp.sum(resArr) / len(resArr)

        # compute coordinates for new conformation

        cic.distance_to_internal_coordinates()
        cic.internal_to_atom_coordinates()
        cic.atom_to_internal_coordinates()
        return globAvg, resArr, cic
