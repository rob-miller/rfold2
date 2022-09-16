#!/usr/local/bin/python3

from utils.rdb import openDb, pqry, pqry1, pqry1p
from utils.rpdb import crMapH, crMapNoH, NoAtomCR, MaxHedron, MaxDihedron
from utils.rpdb import resMap, get_dh_counts

import torch
import torch.nn.functional as F

# import numpy as np

# from torch.utils.data import Dataset
# from Bio.PDB import ic_data
from psycopg2.extras import DictCursor
import copy

# import json

from .base_dataset import BaseDataset


class aa0Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

        learnXres = self.configuration["learnXres"]
        gridRes = self.configuration["grid_resolution"]
        hydrogens = self.configuration["hydrogens"]
        resChar = self.configuration["res_char"]

        self.conn = openDb()
        self.cur = self.conn.cursor()
        self.curj = self.conn.cursor(cursor_factory=DictCursor)

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
            gp[0]: [gp[1], gp[2], gp[3]] + NoAtom for gp in self.cur.fetchall()
        }

        # prepare grid normalisation parameters
        self.gridMinArr = (
            torch.tensor(
                [gp[0:3] for gp in self.gridDict.values()], dtype=torch.float32
            )
            - gridRes
        )

        # prepare output data
        # create dict of rc : len(di/hedra) to filter missing atoms in data
        # start with psi, omg, phi (= 3) not in ic_data_backbone
        # similar for hedra init with 3

        # need for computing output dimension
        hedra_counts, dihedra_counts = get_dh_counts(hydrogens)

        # select res_key, res_char list as index for train, test
        lim = f" limit {config['limit']}" if config["limit"] > -1 else ""
        if resChar == "X":
            self.resChar = "X"
            pqry(
                self.cur,
                f"select res_key, res_char from residue where std_all_angles{lim}",
            )
        else:
            self.resChar = None
            pqry(
                self.cur,
                "select res_key, res_char"
                f" from residue where res_char = '{resChar}' and std_all_angles"
                f" order by res_key{lim}",
            )
        self.rkclist = self.cur.fetchall()

        # set input and output dimension
        self.inputDim = len(self.gridDict) * (3 + len(self.crMap[0]))
        if resChar == "X":
            self.outputDim = MaxHedron * 3 + MaxDihedron * 2
            if learnXres:
                self.outputDim += 20
            else:
                self.inputDim += 20
        else:
            self.outputDim = hedra_counts[resChar] * 3 + dihedra_counts[resChar] * 2

        # print(
        #     f"dataset initiaised; residues: {len(self.rkclist)} inputDim: {self.inputDim} outputDim: {self.outputDim}"
        # )

    def __len__(self):
        return len(self.rkclist)

    def __getitem__(self, idx):
        rkc = self.rkclist[idx]
        rk = rkc[0]
        rc = rkc[1]

        # get environments for input
        self.curj.execute(
            f"select jdict from eagn where res_key = {rk} and grid_ref = {self.gref}"
        )
        ea_grid = self.curj.fetchone()[0]

        gridDictCopy = copy.deepcopy(self.gridDict)
        for gNdx in ea_grid:
            # replace grid voxels with any voxels populated around this residue
            # this is only populated voxels, so index is only relevant to dict
            gridDictCopy[int(gNdx)] = ea_grid[gNdx]

        # make numpy array (used voxels over all db, with changes for this residue)
        gridArr = torch.tensor(
            [gp for gp in gridDictCopy.values()], dtype=torch.float32
        )

        # get dihedra for desired output

        dhLenArr = torch.from_numpy(
            pqry1p(self.cur, f"select bytes from dhlen where res_key = {rk}")
        )
        dhChrArr = torch.from_numpy(
            pqry1p(self.cur, f"select bytes from dhchirality where res_key = {rk}")
        )
        hArr = torch.from_numpy(
            pqry1p(self.cur, f"select bytes from hlen where res_key = {rk}")
        )

        # pad each array to max if doing all residues
        if self.resChar == "X":
            Ld = MaxDihedron - len(dhLenArr)
            # dhLenArr = np.pad(dhLenArr, (0, MaxDihedron - len(dhLenArr)))
            # dhChrArr = np.pad(dhChrArr, (0, MaxDihedron - len(dhChrArr)))
            dhLenArr = F.pad(dhLenArr, (0, Ld))
            dhChrArr = F.pad(dhChrArr, (0, Ld))
            # hArr = np.pad(hArr, [(0, MaxHedron - len(hArr)), (0, 0)])
            hArr = F.pad(hArr, (0, 0, 0, MaxHedron - hArr.shape[0]))

        # output ready
        outputArr = torch.cat((dhChrArr, dhLenArr, hArr.flatten()))

        # add residue identities as needed
        if self.resChar == "X":
            if self.configuration["learnXres"]:
                outputArr = torch.cat(
                    (outputArr, torch.tensor(resMap[rc], dtype=torch.float32))
                )
                inputArr = gridArr.flatten()
            else:
                inputArr = torch.cat(
                    (gridArr.flatten(), torch.tensor(resMap[rc], dtype=torch.float32))
                )
        else:
            inputArr = gridArr.flatten()

        # input = torch.tensor(inputArr, dtype=torch.float32)
        # output = torch.tensor(outputArr, dtype=torch.float32)

        # print(f"input {inputArr.shape}")
        # print(f"output {outputArr.shape}")
        return (inputArr.float(), outputArr.float())
