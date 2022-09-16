"""potential energy function based on environment grid around residue"""

from .base_efn import BaseEfn
from data.gridExplore import gridProt
from data.dbLoad import dbLoad
from Bio.PDB.Chain import Chain

# from utils.rdb import openDb, pqry, pqry1


class EnvGridEfn(BaseEfn):
    """This class is an abstract base class (ABC) for efns."""

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class."""
        self.configuration = configuration
        self.gp = gridProt([])  # no targList but initialize other data
        self.step = float(configuration["grid_resolution"])
        self.grids, self.points = self.gp.getGrids(self.step)
        self.dbl = dbLoad()
        print("hello")

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

        resArr = []

        return 0, []
