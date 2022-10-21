"""potential energy function based on environment grid around residue"""

from .base_mfn import BaseMfn
from data.gridExplore import gridProt
from data.dbLoad import dbLoad
from utils.rdb import openDb, pqry, pqry1, pqry1p
from utils.rpdb import crMapH, crMapNoH, NoAtomCR, MaxHedron, MaxDihedron
from Bio.PDB.Chain import Chain

from psycopg2.extras import DictCursor
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
        # self.gp.smArr = [n for n in range(self.gp.search_multi)]  # used by argpartition
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
            torch.tensor([gp[0:3] for gp in self.gridDict.values()], dtype=torch.float32)
            # xp.array([gp[0:3] for gp in self.gridDict.values()], dtype=xp.float32)
            - self.step
        )

        pqry(self.cur, "select cr_class, acr_key from atom_covalent_radii;")
        self.acr = {k: v for (k, v) in self.cur.fetchall()}

        # print("hello")

    def move(self, chain):
        """Return next chain conformation"""
        if not isinstance(chain, Chain):
            for c in chain.get_chains():
                break
            chain = c
        if not chain.internal_coord:
            chain.atom_to_internal_coordinates()
        cic = chain.internal_coord
        # compute hedra L13 and dihedra L14
        # dhSigns = (cic.dihedral_signs()).astype(int)
        distPlot = cic.distance_plot()
        # cic.distplot_to_dh_arrays(distPlot, dhSigns)

        coordDict = self.dbl.loadResidueEnvirons(cic, distPlot)

        resArr = xp.zeros([len(cic.ordered_aa_ic_list)], dtype=float)
        ndx = 0
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
            # gridArr2 = xp.array([gp for gp in gdc2.values()], dtype=xp.float32)
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
            print("hello")

            # for gNdx in ea_grid:
            # replace grid voxels with any voxels populated around this residue
            # this is only populated voxels, so index is only relevant to dict
            #    gridDictCopy[int(gNdx)] = ea_grid[gNdx]

            # distArrX = xp.linalg.norm(
            #    locArr[:, None, :] - self.grids["X"][None, :, :], axis=-1
            # )
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
            resArr[ndx] = xp.sum(distribArrRC * self.nlp[ric.lc]) / xp.count_nonzero(
                distribArrRC
            )
            ndx += 1

        globAvg = xp.sum(resArr) / len(resArr)

        return globAvg, resArr
