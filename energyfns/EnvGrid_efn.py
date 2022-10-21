"""potential energy function based on environment grid around residue"""

from .base_efn import BaseEfn
from data.gridExplore import gridProt
from data.dbLoad import dbLoad
from Bio.PDB.Chain import Chain

import numpy as xp
import math

# from utils.rdb import openDb, pqry, pqry1


class EnvGridEfn(BaseEfn):
    """Energy function based on membership in 3d grid of environment atoms."""

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

        # print("hello")

    def evaluate(self, chain):
        """Return global (avg) energy and array of energy per residue"""
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
            crArr, locArr0 = coordDict[ric]
            locArr = xp.array([la[0:3] for la in locArr0])
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
