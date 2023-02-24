#!/usr/local/bin/python3


from multiprocessing import Pool
import sys
import argparse
import signal

# numpy 10x faster for gridStep
try:
    import xcupy as xp  # type: ignore
except ImportError:
    print("grid-explore using numpy")
    import numpy as xp
import math

# import os

from utils.rdb import openDb, pqry, pqry1, pgetset

# , closeDb
from utils.rpdb import PDB_repository_base, pdbidre
from Bio.PDB.ic_data import covalent_radii


# multiprocessing support
PROCESSES = 8

# for multiprocessing, step sizes to try
Step_Start = 2.0
Step_Stop = 3.2
Step_Step = 0.2

# default distance between cell centers for -p 0 single run
argstep = 3.0
search_multi = 27


def sigint_handler(signal, frame):
    print("KeyboardInterrupt is caught")
    # global conn
    # conn.close()
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)


# argparse support
args = None


def parseArgs():
    global args
    global PROCESSES
    global argstep
    global search_multi

    arg_parser = argparse.ArgumentParser(
        description="evaluate different residue environment grids"
    )
    arg_parser.add_argument(
        "file",
        nargs="*",
        help="a .cif path/filename to read, or a PDB idCode with "
        "optional chain ID to read from {0} as .cif.gz".format(
            (
                PDB_repository_base
                or "[PDB resource not defined - please configure before use]"
            )
        ),
    )
    arg_parser.add_argument(
        "-f",
        dest="filelist",
        help="text file each line beginning PDB id, optional chain id (7rsaA) to read from {0} as .ent.gz".format(
            (
                PDB_repository_base
                or "[PDB resource not defined - please configure before use]"
            )
        ),
    )
    arg_parser.add_argument(
        "-r", help="process all of db 'residue' table", action="store_true"
    )
    arg_parser.add_argument("-u", help="update database", action="store_true")
    arg_parser.add_argument(
        "-nc", help="don't commit database changes", action="store_true"
    )
    arg_parser.add_argument(
        "-skip", dest="skip_count", help="count of pdb ID list entries to skip"
    )
    arg_parser.add_argument(
        "-limit",
        dest="limit_count",
        help="stop after processing this many pdb ID list entries or residues",
    )
    arg_parser.add_argument(
        "-p", dest="PROCESSES", help=f"number of subprocesses, default {PROCESSES}"
    )
    arg_parser.add_argument(
        "-step", dest="argstep", help=f"stepsize to try for -p 0, default {argstep}"
    )
    arg_parser.add_argument(
        "-pm",
        help="print residue info for multi-hit environment",
        action="store_true",
    )
    arg_parser.add_argument(
        "-sm",
        dest="search_multi",
        help=f"number of cells to try before multi-hit, default {search_multi}",
    )
    arg_parser.add_argument(
        "-cr", help="differentiate by partner cov rad class", action="store_true"
    )
    # arg_parser.add_argument(
    #    "-fg", help="fix grid table with counts", action="store_true"
    # )

    args = arg_parser.parse_args()
    if args.skip_count:
        args.skip_count = int(args.skip_count)
    if args.limit_count:
        args.limit_count = int(args.limit_count)
    if args.PROCESSES:
        PROCESSES = int(args.PROCESSES)
    if args.argstep:
        argstep = float(args.argstep)
    if args.search_multi:
        args.search_multi = int(args.search_multi)


class gridProt:
    # global start and fin range for each axis
    ranges = {}
    # global dict of residues and environs for processing
    resDict = {}
    # print residue info when multi-hit cell counted
    print_multi = False
    # how many cells to try to get each env atom in 1 cell (3x3x3=27)
    search_multi = 27
    # differentiate cell hits by partner covalent radii class
    covRad = False
    # write to db: grid def, (used) grid point, ea_grid (env atom - grid point link)
    updateDb = True
    # write to db: total multi_hit, used cells in grid table, count in grid_point
    # needs to be over entire dataset
    updateDbGlobal = False
    # don't commit database changes
    noCommit = False
    # don't print table of cell hit counts
    quiet = False

    def __init__(self, targList, residue=False, limit=0):
        # limit only applies to lines from residue table here, if residue=True
        self.getRanges()
        self.getResDict(targList, residue, limit)
        self.crNdx = {k: v for v, k in enumerate(covalent_radii.keys())}

    def getRanges(self):
        conn = openDb()
        cur = conn.cursor()

        self.ranges["X"] = {}
        pqry(
            cur, "select min(x), max(x), min(y), max(y), min(z), max(z) from env_atoms"
        )
        rslt = cur.fetchall()[0]
        self.ranges["X"]["x"] = (rslt[0], rslt[1])
        self.ranges["X"]["y"] = (rslt[2], rslt[3])
        self.ranges["X"]["z"] = (rslt[4], rslt[5])

        # distNeg = xp.linalg.norm(xp.array([rslt[n] for n in (0, 2, 4)]))
        # distPos = xp.linalg.norm(xp.array([rslt[n] for n in (1, 3, 5)]))
        # self.maxDist = max(distNeg, distPos).item()

        for rc in "GAVLIMFPSTCNQYWDEHKR":
            self.ranges[rc] = {}
            pqry(
                cur,
                "select min(ea.x), max(ea.x), min(ea.y), max(ea.y), min(ea.z), max(ea.z)"
                f" from env_atoms ea, residue r where r.res_char = '{rc}'"
                " and ea.res_key = r.res_key",
            )
            rslt = cur.fetchall()[0]
            self.ranges[rc]["x"] = (rslt[0], rslt[1])
            self.ranges[rc]["y"] = (rslt[2], rslt[3])
            self.ranges[rc]["z"] = (rslt[4], rslt[5])

        cur.close()
        conn.close()

    @staticmethod
    def frange(a, b, s):
        return (
            []
            if s > 0 and a > b or s < 0 and a < b or s == 0
            else [a] + gridProt.frange(a + s, b, s)
        )

    def getResDict(self, targList, residue=False, limit=0):
        """Generate dict of res_key: res_char, [env atom x,y,z, cr, ea_key]
        for input (targlist or all in database residue table)."""
        conn = openDb()
        cur = conn.cursor()

        if residue:
            if targList:
                print("ignoring input target list due to residue table option set")
            pqry(
                cur,
                f"select res_key, res_char from residue limit {limit}"
                if limit
                else "select res_key, res_char from residue",
            )
            rSet = cur.fetchall()
            for rk, res in rSet:
                if len(res) > 1:
                    continue  # skip UNK and other non standard residues
                pqry(
                    cur,
                    "select ea.x, ea.y, ea.z, acr.cr_class, ea.env_atom, ea.ea_key"
                    " from env_atoms ea, atom a, atom_covalent_radii acr"
                    f" where ea.res_key = {rk}"
                    " and ea.env_atom = a.atom_key and acr.acr_key = a.crc_key",
                )
                eaList = cur.fetchall()
                if len(eaList) > 0:
                    self.resDict[rk] = (res, eaList)
        else:
            # can only load from residue table so targlist redundant if residue
            # limit and skip already used for targList
            for chn in targList:
                pqry(
                    cur,
                    "select r.res_key, r.res_char from residue r, chain c where"
                    f" r.std_all_angles and r.chain_key = c.chain_key and c.chain_name = '{chn}'",
                )
                resList = cur.fetchall()
                if resList == []:
                    chn2 = chn[0:4].upper() + chn[4]
                    pqry(
                        cur,
                        "select r.res_key, r.res_char from residue r, chain c where"
                        f" r.std_all_angles and r.chain_key = c.chain_key and c.chain_name = '{chn2}'",
                    )
                    resList = cur.fetchall()

                for rk in resList:
                    if len(rk[1]) > 1:
                        continue  # skip UNK and other non standard residues
                    pqry(
                        cur,
                        "select ea.x, ea.y, ea.z, acr.cr_class, ea.env_atom, ea.ea_key from env_atoms ea,"
                        f" atom a, atom_covalent_radii acr where ea.res_key = {rk[0]}"
                        " and ea.env_atom = a.atom_key and acr.acr_key = a.crc_key",
                    )
                    eaList = cur.fetchall()
                    if len(eaList) == 0:
                        continue
                    self.resDict[rk[0]] = (rk[1], eaList)

        # print(f"gridProt loaded {len(self.resDict)} residues")

        cur.close()
        conn.close()

    def assignGridPoints(self, distArr, sorted):
        """given list of distances b/n each env atom and each grid point, return
        index of assigned gp for each env atom.  if search_multi is 0, assigned
        will be closest, otherwise will be nearest without conflicting assignment
        for up to search_multi attempts"""
        closest = xp.argmin(distArr, axis=1)  # get minimum dist over each row
        sorted = None  # might not need to sort so do later # wiped input????
        count = self.search_multi  # number of tries to find no conflicts
        if not hasattr(self, "smArr"):
            self.smArr = [n for n in range(self.search_multi)]  # used by argpartition
        while count > 0 or sorted is None:
            assignments = {}
            noConflict = True
            loc = 0
            for gp in closest:
                gp = int(gp)
                if gp in assignments:
                    assignments[gp].append(loc)
                    noConflict = False
                else:
                    assignments[gp] = [loc]
                loc += 1
            if noConflict or count == 0:
                return closest
            if sorted is None:
                # sorted = xp.argsort(distArr, axis=1)
                # https://stackoverflow.com/a/68612359/2783487
                sorted = xp.argpartition(distArr, self.smArr)
            for gp in assignments:
                if len(assignments[gp]) == 1:
                    continue
                if distArr[assignments[gp][0]][gp] < distArr[assignments[gp][1]][gp]:
                    # closest[loc] = gp
                    loc2 = assignments[gp][1]
                    ndx = xp.where(sorted[loc2] == gp)
                    ndx = ndx[0].item()
                    closest[loc2] = sorted[loc2][ndx + 1]
                else:
                    loc1 = assignments[gp][0]
                    ndx = xp.where(sorted[loc1] == gp)
                    ndx = ndx[0].item()
                    closest[loc1] = sorted[loc1][ndx + 1]
            count -= 1
        return closest

    # @profile
    def distributeGridPoints(self, crArr, distArr, maxd, sorted):
        # count fractional membership of env atoms to grid points by cov rad
        distribArr = xp.zeros([distArr.shape[1], len(self.crNdx)])
        # sorted = xp.argsort(distArr, axis=1)
        # https://stackoverflow.com/a/68612359/2783487
        # sorted2 = xp.argpartition(distArr, self.smArr)
        rowNdx = 0
        for row in sorted:
            crn = self.crNdx[crArr[rowNdx]]
            for ndx in row:
                dist = distArr[rowNdx][ndx]
                if dist > maxd:
                    break
                val = (maxd - dist) / maxd
                distribArr[ndx][crn] += val
            rowNdx += 1
        return distribArr

    def getGPnlp(self, points, step):
        conn = openDb()
        cur = conn.cursor()
        nlp = {
            rc: xp.full([points[rc], len(self.crNdx)], 100.0, dtype=float)
            for rc in points.keys()
        }
        crStr = ""
        started = False
        for cr in self.crNdx.keys():
            if started:
                crStr += ", "
            crStr += f"gpn.{cr.lower()}"
            started = True
        for rc in "XGAVLIMFPSTCNQYWDEHKR":
            gref = pqry1(
                cur,
                f"select grid_ref from grid where res_char = '{rc}' and step = {step}",
            )
            pqry(
                cur,
                f"select gp.index, {crStr} from grid_point gp, grid_point_nlp gpn where grid_ref = {gref} and gp.gp_key = gpn.gp_key",
            )
            rslt = cur.fetchall()
            for r in rslt:
                nlp[rc][r[0]] = r[1:]

        return nlp

    def getGridDef(self, step):
        conn = openDb()
        cur = conn.cursor()
        gridDef = {}
        points = {}
        for rc in "XGAVLIMFPSTCNQYWDEHKR":
            gridDef[rc] = {}
            pqry(
                cur,
                "select total_cells, x_start, x_fin, x_count,"
                " y_start, y_fin, y_count,"
                " z_start, z_fin, z_count from grid where"
                f" res_char = '{rc}' and step = {step}",
            )
            rslt = cur.fetchall()[0]
            points[rc] = rslt[0]
            ndx = 1
            for ax in ("x", "y", "z"):
                gridDef[rc][ax] = {}
                for flip in range(3):
                    gridDef[rc][ax][flip] = rslt[ndx]
                    ndx += 1
        conn.close()
        return gridDef, points

    def setGridDef(self, step):
        gridDef = {}
        points = {}
        if self.updateDb:
            conn = openDb()
            cur = conn.cursor()

        for rc in "XGAVLIMFPSTCNQYWDEHKR":
            gridDef[rc] = {}
            for ax in ("x", "y", "z"):
                minc = math.ceil(abs(self.ranges[rc][ax][0]) / step)
                maxc = math.ceil(self.ranges[rc][ax][1] / step)
                gridDef[rc][ax] = (
                    round(-(minc * step), 2),
                    round(maxc * step, 2),
                    minc
                    + maxc,  # total cells = most negative cell index + most positive (0 doesn't matter as is centre/self)
                )

            # print(gridDef)

            # for i in range(gridDef["x"][2]):
            #    print(i, round(gridDef["x"][0] + (i * step), 2))

            points[rc] = gridDef[rc]["x"][2] * gridDef[rc]["y"][2] * gridDef[rc]["z"][2]
            if self.updateDb:
                # ensure specs saved to grid table
                pqry(
                    cur,
                    "insert into grid(res_char, step, total_cells,"
                    " x_start, x_fin, x_count, y_start, y_fin, y_count,"
                    " z_start, z_fin, z_count) values"
                    f" ('{rc}', {step}, {points[rc]}, {gridDef[rc]['x'][0]},"
                    f" {gridDef[rc]['x'][1]}, {gridDef[rc]['x'][2]}, {gridDef[rc]['y'][0]}, {gridDef[rc]['y'][1]}, {gridDef[rc]['y'][2]},"
                    f" {gridDef[rc]['z'][0]}, {gridDef[rc]['z'][1]}, {gridDef[rc]['z'][2]})"
                    " on conflict (res_char, step) do nothing",
                    # " on conflict (res_char, step) do update set (x_count, y_count, z_count) ="
                    # f" ({gridDef[rc]['x'][2]}, {gridDef[rc]['y'][2]}, {gridDef[rc]['z'][2]})",
                )
        if self.updateDb:
            conn.commit()
            conn.close()

        return gridDef, points

    def buildGrids(self, step, gridDef, points):
        grids = {}
        for rc in "XGAVLIMFPSTCNQYWDEHKR":
            grids[rc] = xp.zeros([points[rc], 3])
            i = 0
            for x in range(gridDef[rc]["x"][2]):
                for y in range(gridDef[rc]["y"][2]):
                    for z in range(gridDef[rc]["z"][2]):
                        vx = round(gridDef[rc]["x"][0] + (x * step), 2)
                        vy = round(gridDef[rc]["y"][0] + (y * step), 2)
                        vz = round(gridDef[rc]["z"][0] + (z * step), 2)
                        grids[rc][i][:] = xp.asarray([vx, vy, vz])
                        i += 1
        return grids

    def setGrids(self, step):
        # generate 3d grid of step separated points that covers max self.ranges of
        # x, y, z in env_atoms (determined by self.getRanges())

        gridDef, points = self.setGridDef(step)
        grids = self.buildGrids(step, gridDef, points)
        return grids, points

    def getGrids(self, step):
        """get grid definitions from database for step (grid spec set/created in gridStep())"""
        gridDef, points = self.getGridDef(step)
        grids = self.buildGrids(step, gridDef, points)
        return grids, points

    # @profile
    def gridStep(self, step):
        """Generate 3d grid and assign env atoms"""
        step = round(step, 2)
        if not self.quiet:
            print(f"starting step {step} search_multi = {self.search_multi}")

        # gridDef = {}
        # points = {}
        # grids = {}
        grids, points = self.setGrids(step)

        gridCountsGlobal = {}
        multi_hit_cells = {}
        multi_hit_cells_cr = {}
        used_cells = {}
        gpks = {}
        distribArrGlobal = {}
        countArrGlobal = {}

        for rc in "XGAVLIMFPSTCNQYWDEHKR":
            gridCountsGlobal[rc] = xp.zeros(points[rc], dtype=int)
            multi_hit_cells[rc] = 0
            multi_hit_cells_cr[rc] = 0
            used_cells[rc] = 0
            distribArrGlobal[rc] = xp.zeros([points[rc], len(self.crNdx)], dtype=float)
            countArrGlobal[rc] = xp.zeros([len(self.crNdx)], dtype=int)
            if self.updateDb:
                gpks[rc] = {}

        # if self.updateDbGlobal:
        maxd = math.sqrt(3 * (step * step))  # diagonal between 2 grid points
        self.smArr = [n for n in range(self.search_multi)]  # used by argpartition

        conn = openDb()
        cur = conn.cursor()

        if self.updateDb:
            pqry(cur, f"select res_char, grid_ref from grid where step = {step}")
            gridRefs = {k: v for (k, v) in cur.fetchall()}

        # for each input residue (self.resDict), compute distance for every env atom
        # to every grid point (for X and all residue classes).  Assign closest
        # according to search_multi, count and optionally store in database.

        # print(f"starting residue load step {step}")
        count = 0
        total = len(self.resDict)
        for rk, (rc, eaList) in self.resDict.items():
            # print(step, rk, rc)
            gridCountsLocalX = xp.zeros(points["X"], dtype=int)
            gridCountsLocalR = xp.zeros(points[rc], dtype=int)
            if self.covRad:
                gcLocalXcr = {}
                gcLocalRcr = {}
                for cr in covalent_radii:
                    gcLocalXcr[cr] = xp.zeros(points["X"], dtype=int)
                    gcLocalRcr[cr] = xp.zeros(points[rc], dtype=int)

            # locArr = xp.array(eaList[:, [0, 1, 2]])
            locArr = xp.array([ea[0:3] for ea in eaList])
            crArr = [ea[3] for ea in eaList]
            # eaArr = [ea[4] for ea in eaList]
            eakArr = [ea[5] for ea in eaList]
            distArrX = xp.linalg.norm(
                locArr[:, None, :] - grids["X"][None, :, :], axis=-1
            )
            distArrR = xp.linalg.norm(
                locArr[:, None, :] - grids[rc][None, :, :], axis=-1
            )
            if self.updateDbGlobal:
                # argpartition sorts only points within smArr
                sortedX = xp.argpartition(distArrX, self.smArr)
                sortedR = xp.argpartition(distArrR, self.smArr)
            else:
                sortedX = sortedR = None  # might not need so delay

            # closestX = xp.argmin(distArrX, axis=1)
            assignedX = self.assignGridPoints(distArrX, sortedX)  # sorted arg ignored???
            # closestR = xp.argmin(distArrR, axis=1)
            assignedR = self.assignGridPoints(distArrR, sortedR)
            if self.updateDbGlobal:
                distribArrGlobal["X"] += self.distributeGridPoints(
                    crArr, distArrX, maxd, sortedX
                )
                distribArrGlobal[rc] += self.distributeGridPoints(
                    crArr, distArrR, maxd, sortedR
                )
                for cr in crArr:
                    crn = self.crNdx[cr]
                    countArrGlobal["X"][crn] += 1
                    countArrGlobal[rc][crn] += 1

            # loc = 0
            # print()
            eaNdx = 0
            if self.updateDb:
                grefX = gridRefs["X"]
                grefR = gridRefs[rc]
            for g in assignedX:
                g = int(g)  # cupy issue
                gridCountsLocalX[g] += 1
                gridCountsGlobal["X"][g] += 1
                # loc += 1
                if self.covRad:
                    gcLocalXcr[crArr[eaNdx]][g] += 1
                if self.updateDb:
                    gpt = grids["X"][g]
                    if g in gpks["X"]:
                        gpk = gpks["X"][g]
                    else:
                        # X:
                        # this grid point is used, first time seen, save in grid_point
                        gpk = pgetset(
                            cur,
                            f"select gp_key from grid_point where grid_ref={grefX}"
                            f" and index = {g}",
                            "insert into grid_point (grid_ref, index, x, y, z)"
                            f" values({grefX}, {g}, {gpt[0]}, {gpt[1]}, {gpt[2]})"
                            " returning gp_key",
                        )
                        gpk = gpks["X"][g] = gpk
                    # log assignment of eak to gp in ea_grid
                    pqry(
                        cur,
                        "insert into ea_grid (ea_key, gp_key, grid_ref) values"
                        f" ({eakArr[eaNdx]}, {gpk}, {grefX}) on conflict do nothing",
                    )
                eaNdx += 1

            # same log used gp as above for per-residue
            eaNdx = 0
            for g in assignedR:
                g = int(g)  # cupy issue
                gridCountsLocalR[g] += 1
                gridCountsGlobal[rc][g] += 1
                # loc += 1
                if self.covRad:
                    gcLocalRcr[crArr[eaNdx]][g] += 1
                if self.updateDb:
                    gpt = grids[rc][g]
                    if g in gpks[rc]:
                        gpk = gpks[rc][g]
                    else:
                        gpk = pgetset(
                            cur,
                            f"select gp_key from grid_point where grid_ref={grefR}"
                            f" and index = {g}",
                            "insert into grid_point (grid_ref, index, x, y, z)"
                            f" values ({grefR}, {g}, {gpt[0]}, {gpt[1]}, {gpt[2]})"
                            " returning gp_key",
                        )
                        gpks[rc][g] = gpk
                    pqry(
                        cur,
                        "insert into ea_grid (ea_key, gp_key, grid_ref) values"
                        f" ({eakArr[eaNdx]}, {gpk}, {grefR})"
                        " on conflict do nothing",  # on conflict crash
                    )
                eaNdx += 1

            # count up multi-hit-cells for X and per-residue
            eaNdx = 0
            for g in assignedX:
                g = int(g)
                if gridCountsLocalX[g] > 1:
                    multi_hit_cells["X"] += 1
                    if self.covRad and gcLocalXcr[crArr[eaNdx]][g] > 1:
                        multi_hit_cells_cr["X"] += 1
                        if self.print_multi:
                            print(f"X {rk} {rc} {crArr[eaNdx]}   {step}")

                    elif self.print_multi:
                        print(f"X {rk} {rc} {step}")

            for g in assignedR:
                g = int(g)
                if gridCountsLocalR[g] > 1:
                    multi_hit_cells[rc] += 1
                    if self.covRad and gcLocalRcr[crArr[eaNdx]][g] > 1:
                        multi_hit_cells_cr[rc] += 1
                        if self.print_multi:
                            print(f"R {rk} {rc} {crArr[eaNdx]}   {step}")
                    elif self.print_multi:
                        print(f"R {rk} {rc} {step}")

            count += 1
            if count % 100000 == 0:
                print("step=", step, ":", count, "of", total)
            # done assigning one residue env atoms to grid points

        # report results and update global counts

        if self.updateDbGlobal:
            pqry(cur, "delete from grid_atom_counts")
            pqry(cur, "delete from grid_point_counts")

        if not self.quiet:
            print("step  res  mh_cells  used_cells  fraction")
        for rc in "XGAVLIMFPSTCNQYWDEHKR":
            used_cells[rc] = xp.count_nonzero(gridCountsGlobal[rc])
            mhc = multi_hit_cells_cr[rc] if self.covRad else multi_hit_cells[rc]
            if not self.quiet:
                print(
                    f"{step:2.1f}    {rc}    {mhc:3d}         {used_cells[rc]:4d}      {xp.round(used_cells[rc]/points[rc], 2):2.2f}"
                )

            if self.updateDbGlobal:
                pqry(
                    cur,
                    "update grid set (multi_hit_cells, used_cells) ="
                    f" ({multi_hit_cells[rc]}, {used_cells[rc]})"
                    f" where res_char = '{rc}' and step = {step}",
                )

                gref = pqry1(
                    cur,
                    f"select grid_ref from grid where res_char='{rc}' and step={step}",
                )

                for ndx in xp.nonzero(gridCountsGlobal[rc])[0]:
                    gpt = grids[rc][ndx]
                    gpk = pqry1(
                        cur,
                        f"select gp_key from grid_point where grid_ref={gref} and index={ndx}",
                    )
                    if gpk is not None:
                        pqry(
                            cur,
                            f"update grid_point set count = {gridCountsGlobal[rc][ndx]} where gp_key={gpk}",
                        )
                    else:
                        print(
                            f"error: no grid point for {gpt} hits {gridCountsGlobal[rc][ndx]} ndx {ndx} ?"
                        )
                ndxDone = {}
                for ndx in xp.nonzero(distribArrGlobal[rc])[0]:
                    if ndx in ndxDone:  # nonzero returns ndx for each cr class
                        continue
                    ndxDone[ndx] = True

                    gpt = grids[rc][ndx]
                    if ndx in gpks[rc]:
                        gpk = gpks[rc][ndx]
                    else:
                        gpk = pgetset(
                            cur,
                            f"select gp_key from grid_point where grid_ref={gref}"
                            f" and index = {ndx}",
                            "insert into grid_point (grid_ref, index, x, y, z)"
                            f" values ({gref}, {ndx}, {gpt[0]}, {gpt[1]}, {gpt[2]})"
                            " returning gp_key",
                        )
                        gpks[rc][ndx] = gpk

                    cols = ""
                    vals = ""
                    started = False
                    for cr, cndx in self.crNdx.items():
                        if started:
                            cols += ", "
                            vals += ", "
                        started = True
                        cols += cr.lower()
                        vals += str(distribArrGlobal[rc][ndx][cndx])

                    pqry(
                        cur,
                        f"insert into grid_point_counts (gp_key, res_char, {cols}) values ({gpk}, '{rc}', {vals})",
                    )

                cols = ""
                vals = ""
                started = False
                for cr, cndx in self.crNdx.items():
                    if started:
                        cols += ", "
                        vals += ", "
                    started = True
                    cols += cr.lower()
                    vals += str(countArrGlobal[rc][cndx])
                pqry(
                    cur,
                    f"insert into grid_atom_counts (grid_ref, res_char, {cols}) values ({gref}, '{rc}', {vals})",
                )

        if not self.quiet:
            print("")

        cur.close()
        if self.updateDb and not self.noCommit:
            conn.commit()
        conn.close()

        return 1


if __name__ == "__main__":
    parseArgs()
    targList = []
    toProcess = args.file
    if args.filelist:
        flist = open(args.filelist, "r")
        for aline in flist:
            fields = aline.split()
            pdbidMatch = pdbidre.match(fields[0])
            if pdbidMatch:
                toProcess.append(pdbidMatch.group(0))

    if len(toProcess) == 0 and not args.r:
        print("no files to process. use '-h' for help")
        sys.exit(0)

    fileNo = 1
    limCount = args.limit_count
    for target in toProcess:
        if args.skip_count and fileNo <= args.skip_count:
            fileNo += 1
            continue
        if limCount:
            if limCount <= 0:
                # sys.exit(0)
                break
            limCount -= 1
        targList.append(target)
        fileNo += 1

    gProt = gridProt(targList, residue=args.r, limit=args.limit_count)

    if args.search_multi:
        gProt.search_multi = args.search_multi
    if args.pm:
        gProt.print_multi = True
    if args.cr:
        gProt.covRad = True
    if args.u:
        gProt.updateDb = True
    if args.nc:
        gProt.noCommit = True

    # if args.fg:
    #    gProt.setGrids(argstep)
    #    sys.exit()

    # gProt.getRanges()
    # gProt.getResDict(targList, args.r)

    if PROCESSES > 0:
        with Pool(PROCESSES) as p:
            rslts = [
                p.apply_async(gProt.gridStep, (i,))
                for i in gridProt.frange(Step_Start, Step_Stop, Step_Step)
            ]
            for i in rslts:
                i.get()
                # print(i.get())
        print("Now the pool is closed and no longer available")
    else:
        gProt.gridStep(argstep)
