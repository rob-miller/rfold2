#!/usr/local/bin/python3

from multiprocessing import Pool
import sys
import signal
import argparse
import pickle

import psycopg2
import copy
import itertools

from psycopg2.extras import Json, DictCursor
from time import time

from utils.rdb import closeDb, openDb, pqry, pqry1p, pqry1
from utils.rpdb import MaxHedron, MaxDihedron, outputArr2values, len2ang

from utils.rpdb import PDB_repository_base, getPDB, pdbidre, get_dcDict
from utils.rpdb import resList, get_phi_omg_classes, lenAng2coords
from utils.rpdb import crMapH, crMapNoH, NoAtomCR, get_dh_counts
from utils.rpdb import getNormValues, dhlNorm, harrNorm


from dbLoad import dbLoad
from gridExplore import gridProt

from Bio.PDB.Chain import Chain
from Bio.PDB.ic_data import covalent_radii

from Bio.PDB.internal_coords import Edron

import numpy as np


# multiprocessing support
PROCESSES = 7


conn0 = openDb()
cur0 = conn0.cursor()


def sigint_handler(signal, frame):
    print("KeyboardInterrupt is caught")
    # global conn
    # conn.close()
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)

# argparse support
args = None

# global defaults:
# not processing all residue table
procResidues = False
# default grid resolution for environment atoms
gridStep = 3.0
# default grid to search for unique assignment (3x3x3)
gridSearchMulti = 27
# process specific residue
resChar = "X"


def parseArgs():
    global args
    global PROCESSES
    global procResidues
    global gridStep
    global gridSearchMulti
    global resChar

    # global sdPass
    arg_parser = argparse.ArgumentParser(
        description="Load PDB data files into hal rfold database "
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
        help="text file each line beginning PDB id, optional chain id (7RSAA) to read from {0} as .ent.gz".format(
            (
                PDB_repository_base
                or "[PDB resource not defined - please configure before use]"
            )
        ),
    )
    arg_parser.add_argument(
        "-skip",
        dest="skip_count",
        help="count of pdb ID list entries to skip",
        type=int,
    )
    arg_parser.add_argument(
        "-limit",
        dest="limit_count",
        help="stop after processing this many pdb ID list entries",
        type=int,
    )

    arg_parser.add_argument(
        "-p",
        dest="PROCESSES",
        help=f"number of subprocesses, default {PROCESSES}",
        type=int,
    )

    arg_parser.add_argument(
        "-pr", help="process db residue table, not input files", action="store_true"
    )

    arg_parser.add_argument(
        "-delete", help="delete PDB code from database", action="store_true"
    )

    # protein chains local
    arg_parser.add_argument("-dbl", help="load mmCIF data to db", action="store_true")
    arg_parser.add_argument(
        "-sln",
        dest="source_list",
        help="source list name for -dbl (overrides -f)",
    )
    arg_parser.add_argument(
        "-norm",
        help="generate normalisation factors for all di/hedra lengths (need current for -gno)",
        action="store_true",
    )
    arg_parser.add_argument(
        "-gno",
        help="generate NN output arrays in dhlen, dhchirality, hlen, and dhcoords",
        action="store_true",
    )

    # env atoms
    arg_parser.add_argument(
        "-sd",
        help="run non-bonded distance std dev pass on all db chains; only valid after -dbl all chains",
        action="store_true",
    )
    arg_parser.add_argument(
        "-grid", help="assign env atoms to grid points", action="store_true"
    )

    arg_parser.add_argument(
        "-gridStep", help=f"grid resolution, default {gridStep} angstrom", type=float
    )

    arg_parser.add_argument(
        "-gridSM",
        help=f"grid search-multi: neighbour grid points to search for unique assignment, default {gridSearchMulti}",
        type=int,
    )

    arg_parser.add_argument(
        "-gridNU",
        help="no update - grid only print counts, do not update env-atom links in database",
        action="store_true",
    )
    arg_parser.add_argument(
        "-gridUG",
        help="update global (whole dataset) grid table counts",
        action="store_true",
    )

    arg_parser.add_argument(
        "-gnlp",
        help="generate negative-log-probability values for grid table counts",
        action="store_true",
    )

    arg_parser.add_argument(
        "-gni",
        help="generate NN input arrays in eagn, residue agnostic ('X'), uses -gridStep",
        action="store_true",
    )

    arg_parser.add_argument(
        "-rc",
        # dest="gniRC",
        help=f"specific single residue character for -gni, -gno, -tst default is {resChar}",
    )

    arg_parser.add_argument(
        "-gnir",
        help="generate NN input arrays in eagn per residue, uses -gridStep",
        action="store_true",
    )

    arg_parser.add_argument(
        "-tst",
        help="test each residue NN db output against angle and coordinate db values",
        action="store_true",
    )

    arg_parser.add_argument(
        "-chnchk",
        help="test input chain list against all in database chain table",
        action="store_true",
    )

    arg_parser.add_argument(
        "-load",
        help="-dbl -gno -grid -gni -gnir -tst",
        action="store_true",
    )

    arg_parser.add_argument(
        "-loadall",
        help="-dbl -norm -sd -gno -grid -gnlp -gni -gnir -tst",
        action="store_true",
    )

    args = arg_parser.parse_args()

    if args.load or args.loadall:
        args.dbl = args.gni = args.grid = args.gno = args.gnir = args.tst = True
        if args.loadall:
            args.norm = args.sd = args.gnlp = True

    if args.dbl and not (args.filelist or args.source_list):
        print("need -f or -sln for db load")
        sys.exit()

    if args.PROCESSES is not None:
        PROCESSES = args.PROCESSES
    if args.gridStep:
        gridStep = args.gridStep
    if args.gridSM:
        gridSearchMulti = args.gridSM
    if args.rc:
        resChar = args.rc

    procResidues = args.pr or args.sd


def delPDB(pdbid):
    pdbk = pqry1(cur0, f"select pdb_key from pdb where pdb_id = '{pdbid[:4]}'")
    print(f"deleting PDB id {pdbid[:4]} key {pdbk}")
    start = time()

    pqry(cur0, f"select chain_key from chain where pdb_key = {pdbk}")
    delList = cur0.fetchall()
    delList = [d[0] for d in delList]

    def doDel(tag, qry=None):
        lasttime = time()

        for chnk in delList:
            pqry(
                cur0,
                eval(f"f'{qry}'")
                if qry
                else f"delete from {tag} where res_key in (select res_key from residue where chain_key = {chnk})",
            )
        curr = time()

        print(
            f"{tag} {cur0.rowcount} rows deleted in {round((curr - lasttime), 2)} seconds {round((curr - start),2)} total seconds"
        )
        conn0.commit()
        lasttime = curr

    [doDel(table) for table in ["eagn", "dhcoords", "dhlen", "dhchirality", "hlen"]]
    doDel(
        "ea_grid",
        "delete from ea_grid where ea_key in (select ea_key from env_atoms where res_key in (select res_key from residue where chain_key = {chnk}))",
    )
    [doDel(table) for table in ["dihedron", "env_atoms", "hedron", "atom"]]
    doDel("residue", "delete from residue where chain_key = {chnk}")
    doDel("pdb", f"delete from pdb where pdb_key = {pdbk}")


def dblp(targ, sd=False):
    global args
    sln = None
    if not sd:
        if args.source_list:
            sln = args.source_list
        elif args.filelist:
            sln = args.filelist
    dbLoad(sd).loadPDB(targ, sln)


def dbLoadProtein(targList):
    global PROCESSES
    print(f"starting dbLoadProtein for {len(targList)} target(s)")
    if PROCESSES > 0:
        with Pool(PROCESSES) as p:
            rslts = [p.apply_async(dblp, (j,)) for j in targList]
            for i in rslts:
                i.get()
                # print(i.get())
        print("dbLoadProtein() pool closed.")
    else:
        for i in targList:
            dblp(i)


def sdLoadProtein():
    dbl = dbLoad(sdp=True)
    dbl.nbdMinMaxAverages()
    dbl.dlq("select chain_name from chain where rebuild")
    targList = [p[0] for p in dbl.cur.fetchall()]
    print(f"starting sdLoadProtein for {len(targList)} target(s)")
    if PROCESSES > 0:
        with Pool(PROCESSES) as p:
            # nbd data in db per chain so can be parallel
            rslts = [p.apply_async(dblp, (j, True)) for j in targList]
            for i in rslts:
                i.get()
                # print(i.get())
        print("sdLoadProtein() pool closed.")
    else:
        for i in targList:
            dblp(i, sd=True)
    dbl.nbdStdDevs()
    dbl.nbdAll()


def genNormFactors():
    # generate normalisation factors for output di/hedra lengths.  Normalize
    # on +/- 3 std devs instead of min/max to avoid outliers unduly influencing
    # di/hedra normalisation parameters

    print("generating di/hedra length normalization factors")

    pqry(cur0, "delete from len_normalization")

    r = pqry1(
        cur0,
        "select avg(len12), stddev(len12), avg(len23), stddev(len23), avg(len13), stddev(len13) from hedron",
    )
    # min len defined as avg minus 3 sd
    minLenList = [r[2 * i] - (3 * r[(2 * i) + 1]) for i in range(3)]
    # range defined as +/- 3 sd
    rangeList = [6 * r[(2 * i) + 1] for i in range(3)]

    # dihedra normalisation parameter
    r = pqry1(cur0, "select avg(len14), stddev(len14) from dihedron")
    minLen14 = r[0] - (3 * r[1])
    rangeLen14 = 6 * r[1]

    pqry(
        cur0,
        f"insert into len_normalization (name, min, range) values ('len12', {minLenList[0]}, {rangeList[0]})",
    )
    pqry(
        cur0,
        f"insert into len_normalization (name, min, range) values ('len23', {minLenList[1]}, {rangeList[1]})",
    )
    pqry(
        cur0,
        f"insert into len_normalization (name, min, range) values ('len13', {minLenList[2]}, {rangeList[2]})",
    )
    pqry(
        cur0,
        f"insert into len_normalization (name, min, range) values ('len14', {minLen14}, {rangeLen14})",
    )
    conn0.commit()


def splitList(a, n):
    # split list a into n chunks
    # https://stackoverflow.com/a/2135920/2783487
    if n == 0:
        return [a]
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def genRkList(targList, resChar=None):
    global procResidues
    global args
    if resChar == "X":
        resChar = None
    if procResidues:
        qry = (
            "select res_key from residue where std_all_angles order by res_key"
            if not resChar
            else f"select res_key from residue where res_char = '{resChar}' and std_all_angles order by res_key"
        )
        if args.limit_count is not None:
            qry += f" limit {args.limit_count}"
        pqry(
            cur0,
            qry,
            # "select res_key, res_char from residue where std_all_angles order by res_key",
        )
        rkList = [r[0] for r in cur0.fetchall()]
    else:
        rkList = []
        for chn in targList:
            qryFrag = f" and r.res_char = '{resChar}'" if resChar else ""
            qry = (
                "select r.res_key from residue r, chain c"
                f" where c.chain_name = '{chn}'"
                f" and r.chain_key = c.chain_key"
                + qryFrag
                + " and r.std_all_angles order by res_key"
            )
            pqry(cur0, qry)
            # rkList.append([r[0] for r in cur0.fetchall()])
            rkList += [r[0] for r in cur0.fetchall()]
    return rkList


"""
minLenList = None
rangeList = None
minLen14 = None
rangeLen14 = None
hMinLenArr = None
hRangeArr = None
hAvgArr = None
avgLen14 = None


def getNorms():
    global minLenList, rangeList, minLen14, rangeLen14, hMinLenArr, hRangeArr
    global hAvgArr, avgLen14
    minLenList = []
    minLenList.append(
        pqry1(cur0, "select min from len_normalization where name = 'len12'")
    )
    minLenList.append(
        pqry1(cur0, "select min from len_normalization where name = 'len23'")
    )
    minLenList.append(
        pqry1(cur0, "select min from len_normalization where name = 'len13'")
    )
    rangeList = []
    rangeList.append(
        pqry1(cur0, "select range from len_normalization where name = 'len12'")
    )
    rangeList.append(
        pqry1(cur0, "select range from len_normalization where name = 'len23'")
    )
    rangeList.append(
        pqry1(cur0, "select range from len_normalization where name = 'len13'")
    )

    minLen14 = pqry1(cur0, "select min from len_normalization where name = 'len14'")
    rangeLen14 = pqry1(cur0, "select range from len_normalization where name = 'len14'")
    avgLen14 = minLen14 / (rangeLen14 / 2)  <-- bug, compare below

    hMinLenArr = np.array(minLenList, dtype=np.float64)
    hRangeArr = np.array(rangeList, dtype=np.float64)
    hAvgArr = hMinLenArr + (hRangeArr / 2)
"""


def gnoLengths(rklist):
    """populates dhlen table with normalized di/hedron interatom distances and chiralities"""
    # global minLenList, rangeList, minLen14, rangeLen14, hMinLenArr, hRangeArr
    # global avgLen14, hAvgArr

    normDict = getNormValues()
    hAvgArr = np.array(
        [normDict[x][2] for x in ["len12", "len23", "len13"]], dtype=np.float64
    )
    avgLen14 = normDict["len14"][2]

    conn = openDb()
    cur = conn.cursor()

    (
        phiClassKey,
        omgClassKey,
        acbClassKey,
        psiH1key,
        phiH1key,
        omgH1key,
    ) = get_phi_omg_classes(cur)

    count = 0
    total = len(rklist)

    for rk in rklist:
        dhlist = []

        # get rprev omega and phi dihedral angles
        rpk = pqry1(cur, f"select rprev from residue where res_key = {rk}")
        if rpk is not None:
            # ak_str for easier debugging
            pqry(
                cur,
                "select d.ak_str, d.len14, d.chirality from dihedron d, dihedron_class dc where"
                f" d.res_key = {rpk}"
                f" and (d.class_key = {omgClassKey} or d.class_key = {phiClassKey})"
                " and d.class_key = dc.dc_key"
                " order by dc.d_class",  # omg, phi
            )
            for dh in cur.fetchall():
                dhlist += dh[1:]
                # print(dh[0], dh[1:])
        else:
            # null state with chirality = 0 for prev omg, phi
            dhlist = [avgLen14, 0, avgLen14, 0]

        # get this residue all dihedral angles except phi, omg (use rprev above)
        pqry(
            cur,
            "select d.ak_str, d.len14, d.chirality from dihedron d where"
            f" d.res_key = {rk}"
            f" and d.class_key != {phiClassKey} and d.class_key != {omgClassKey}",
        )
        dhdict = {}
        for dh in cur.fetchall():
            dhdict[Edron.gen_tuple(dh[0])] = dh[1:]

        for k in sorted(dhdict):
            dhlist += dhdict[k]
            # print(k, dhdict[k])

        # separate chirality from distance values to enable sigmoid on chirality
        dhLenArr = np.array(dhlist[::2], dtype=np.float64)
        dhChrArr = np.array(dhlist[1::2], dtype=np.float64)

        # print(dhLenArr)
        # print(dhChrArr)

        # normalise dihedra len to -1/+1 (chirality already on -1/+1)
        dhLenArrNorm = dhlNorm(dhLenArr)

        # get hedra
        hlist = []

        # get rprev hedra to complete r phi and omega dihedral angles

        if rpk is not None:
            pqry(
                cur,
                "select h.ak_str, h.len12, h.len23, h.len13 from hedron h, hedron_class hc"
                f" where h.res_key = {rpk}"
                f" and (h.class_key = {omgH1key} or h.class_key = {phiH1key})"
                " and h.class_key = hc.hc_key"
                " order by hc.h_class",  # omg, phi
            )
            for h in cur.fetchall():
                hlist += [h[1:]]
                # print(h[0], h[1:])
        else:
            hlist = [hAvgArr, hAvgArr]

        # exclude phiH1 as use from rprev, but r omgH1 is CACN to get psi 2nd N
        pqry(
            cur,
            "select h.ak_str, h.len12, h.len23, h.len13 from hedron h"
            f" where h.res_key = {rk} and h.class_key != {phiH1key}",
        )
        hdict = {}
        for h in cur.fetchall():
            hdict[Edron.gen_tuple(h[0])] = h[1:]

        for k in sorted(hdict):
            hlist += [hdict[k]]
            # print(k, hdict[k])

        hArr = np.array(hlist, dtype=np.float64)
        # print(hArr)
        # normalise hedra len to -1/+1
        hArrNorm = harrNorm(hArr)

        if True:
            r = pqry1(cur, f"select 1 from dhlen where res_key = {rk}")
            if r is None:
                cur.execute(
                    "insert into dhlen(res_key, bytes) values (%s, %s)",
                    (rk, pickle.dumps(dhLenArrNorm)),
                )

            r = pqry1(cur, f"select 1 from dhchirality where res_key = {rk}")
            if r is None:
                cur.execute(
                    "insert into dhchirality(res_key, bytes) values (%s, %s)",
                    (rk, pickle.dumps(dhChrArr)),
                )

            r = pqry1(cur, f"select 1 from hlen where res_key = {rk}")
            if r is None:
                cur.execute(
                    "insert into hlen(res_key, bytes) values (%s, %s)",
                    (rk, pickle.dumps(hArrNorm)),
                )

            count += 1
            if (count % 100000) == 0:
                print("gnoLengths", count, "of", total)
                conn.commit()
        else:
            # test codeode
            dbdhl = pqry1p(cur, f"select bytes from dhlen where res_key = {rk}")
            if dbdhl is not None and (dbdhl != dhLenArrNorm).any():
                print(f"dhl fail: {rk} db: {dbdhl} != {dhLenArrNorm}")
            dbchr = pqry1p(cur, f"select bytes from dhchirality where res_key = {rk}")
            if dbchr is not None and (dbchr != dhChrArr).any():
                print(f"chr fail: {rk} db: {dbchr} != {dhChrArr}")
            dbha = pqry1p(cur, f"select bytes from hlen where res_key = {rk}")
            if dbha is not None and (dbha != hArrNorm).any():
                print(f"har fail: {rk} db: {dbha} != {hArrNorm}")
            count += 1
            if (count % 100000) == 0:
                print("gnoLengths", count, "of", total)

    conn.commit()
    conn.close()


def gnoCoords(target):
    """populates dhcoords table of atom coords for each dihedron"""

    def gnoc_loadResidues(cur, cic, chain_key):
        # dcDict = get_dcDict(cur)
        # print(cic.chain.full_id)
        for ric in cic.ordered_aa_ic_list:
            rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")

            pqry(
                cur,
                f"select res_key, std_all_angles from residue where res_seqpos = '{rsp}' and chain_key = {chain_key}",
            )
            rk, std = cur.fetchall()[0]

            if not std:
                continue

            if len(ric.rprev) > 0:
                omg = ric.pick_angle("omg")
                phi = ric.pick_angle("phi")
                dlst = [omg, phi]
            else:
                # omg = phi = ric.pick_angle("psi")
                dlst = []
            for k in sorted(ric.dihedra):
                d = ric.dihedra[k]
                # filter next omg, phi and alt cb path
                if d.e_class not in ["CNCAC", "CACNCA", "CNCACB"]:
                    dlst.append(d)
            # dlst.sort(key=lambda x: dcDict[x.e_class])
            dlst2 = dlst  # [omg, phi] + dlst

            # print(ric)
            dndxs = [d.ndx for d in dlst2]
            datoms = np.array([cic.dAtoms[n] for n in dndxs])

            r = pqry1(cur, f"select 1 from dhcoords where res_key = {rk}")
            if r is None:
                cur.execute(
                    "insert into dhcoords(res_key, bytes) values (%s, %s)",
                    (rk, pickle.dumps(datoms)),
                )

    def gnoc_loadPdbChain(cur, pdb_chain, prot_id, chainID, filename):
        # global conn
        chain_name = prot_id + chainID
        # print(chain_name)
        # pdb_key = None
        # cur = conn.cursor()

        pqry(
            cur,
            f"select rebuild, chain_key from chain where chain_name = '{chain_name}'",
        )
        rslt = cur.fetchall()[0]
        rebuild, chain_key = rslt[0], rslt[1]

        if rebuild:
            if not pdb_chain.internal_coord:
                pdb_chain.atom_to_internal_coordinates()
            cic = pdb_chain.internal_coord
            cic.internal_to_atom_coordinates()
            gnoc_loadResidues(cur, cic, chain_key)  # need for atomDict on sdPass

    (pdb_structure, prot_id, chainID, filename) = getPDB(target)
    # print(prot_id, chainID, filename)
    if pdb_structure is not None:
        conn = openDb()
        cur = conn.cursor()

        if isinstance(pdb_structure, Chain):
            gnoc_loadPdbChain(cur, pdb_structure, prot_id, chainID, filename)
        else:
            for chn in pdb_structure.get_chains():
                gnoc_loadPdbChain(cur, chn, pdb_structure.id, chn.id, filename)

        cur.close()
        conn.commit()
        conn.close()
    # return f"{prot_id}-{chainID}"


def genNNout(targList):
    global procResidues
    # getNorms()
    get_dcDict(cur0)  # pre-load
    rkList = genRkList(targList)

    print(f"starting gnoLengths for {len(rkList)} residues")
    if PROCESSES > 0:
        lol = splitList(rkList, PROCESSES)
        with Pool(PROCESSES) as p:
            rslts = [p.apply_async(gnoLengths, (j,)) for j in lol]
            for i in rslts:
                i.get()
                # print(i.get())
        print("genNNout() gnoLengths() pool closed.")
    else:
        gnoLengths(rkList)

    print(f"starting gnoCoords for {len(targList)} targets")
    if PROCESSES > 0:
        with Pool(PROCESSES) as p:
            rslts = [p.apply_async(gnoCoords, (j,)) for j in targList]
            for i in rslts:
                i.get()
                # print(i.get())
        print("genNNout() gnoCoords() pool closed.")
    else:
        for targ in targList:
            gnoCoords(targ)


def gni(
    rklist,
    gridStep,
    resChar,
):

    conn = openDb()
    cur = conn.cursor()
    curj = conn.cursor(cursor_factory=DictCursor)

    hydrogens = False

    if hydrogens:
        crMap = crMapH
    else:
        crMap = crMapNoH
    NoAtom = crMap[NoAtomCR]

    # get grid points for stepSize and resChar
    rc = "X" if resChar is None else resChar
    gref = pqry1(
        cur,
        "select grid_ref from grid where" f" res_char='{rc}' and step={gridStep}",
    )
    pqry(
        cur,
        f"select index, x, y, z from grid_point where grid_ref = {gref} order by index",
    )

    # grid needed as dict because grid_point data is only used voxels, not all voxels
    gridDict = {gp[0]: [gp[1], gp[2], gp[3]] + NoAtom for gp in cur.fetchall()}

    # prepare grid normalisation parameters
    gridMinArr = (
        np.array([gp[0:3] for gp in gridDict.values()], dtype=np.float32) - gridStep
    )

    count = 0
    total = len(rklist)
    # print(f"{total} residues ({rc}) to process. ")
    for rk in rklist:
        # get environments for input
        qry = (
            "select gp.index, ea.x, ea.y, ea.z, a.crc_key from"
            " env_atoms ea, atom a, ea_grid eg, grid_point gp"
            f" where ea.res_key = {rk}"
            " and a.atom_key = ea.env_atom"
            " and ea.ea_key = eg.ea_key"
            " and eg.gp_key = gp.gp_key"
            f" and gp.grid_ref = {gref}"
            " order by gp.index"
        )
        pqry(
            cur,
            qry,
        )
        rowcount = cur.rowcount
        if not rowcount > 0:
            raise ValueError(f"no rows returned for rk {rk} gref {gref}")
            # print(f"no rows returned for rk {rk} gref {gref}")
            # continue

        ea_grid = {
            kv[0]: ([kv[1], kv[2], kv[3]] + crMap[kv[4]]) for kv in cur.fetchall()
        }

        gridDictCopy = copy.deepcopy(gridDict)
        for gNdx in ea_grid:
            # replace grid voxels with any voxels populated around this residue
            # this is only populated voxels, so index is only relevant to dict
            gridDictCopy[gNdx] = ea_grid[gNdx]

        gridArr = np.array([gp for gp in gridDictCopy.values()], dtype=np.float32)
        # normalise coords to -1/+1 - so all unpopulated grid points go to 0,0,0 atom type 0
        gridArr[:, 0:3] = (2 * ((gridArr[:, 0:3] - gridMinArr) / (2 * gridStep))) - 1

        ndx = 0
        for gp in ea_grid:
            while gridArr[ndx][3] == 1.0:
                ndx += 1
            ea_grid[gp] = gridArr[ndx].tolist()
            ndx += 1

        r = pqry1(cur, f"select 1 from eagn where res_key = {rk} and grid_ref = {gref}")
        if r is None:
            try:
                curj.execute(
                    "insert into eagn(res_key, grid_ref, jdict) values(%s, %s, %s)",
                    (rk, gref, Json(ea_grid)),
                )
            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                raise

        # gdc2 = copy.deepcopy(gridDict)
        # for gp in gdc2:
        #    gdc2[gp][0:3] = 0.0, 0.0, 0.0

        # curj.execute(
        #    f"select jdict from eagn where res_key = {rk} and grid_ref = {gref}"
        # )
        # eag2 = curj.fetchone()[0]
        # for gNdx in eag2:
        #    gdc2[int(gNdx)] = eag2[gNdx]
        # ga2 = np.array([gp for gp in gdc2.values()], dtype=np.float32)
        # assert np.array_equal(gridArr, ga2)

        count += 1
        if (count % 100000) == 0:
            print(count, "/", total)
            conn.commit()

    cur.close()
    conn.commit()
    conn.close()


def genNNin(targList, gridStep, resChar=None):
    if resChar in resList:
        rkList = genRkList(targList, resChar)
        procRkList = splitList(rkList, PROCESSES)
        total = len(rkList)
    elif resChar:
        # all distinct res chars
        procRkList = [genRkList(targList, resch) for resch in resList]
        total = len(procRkList)
    else:
        rkList = genRkList(targList)
        procRkList = splitList(rkList, PROCESSES)
        total = len(rkList)

    # total = len(list(itertools.chain.from_iterable(procRkList)))
    print(f"starting genNNin for {total} residues")

    if PROCESSES > 0:
        with Pool(PROCESSES) as p:
            if resChar and resChar not in resList:
                rslts = [
                    p.apply_async(gni, (j, gridStep, rc))
                    for j, rc in zip(procRkList, resList)
                ]
            else:
                rslts = [p.apply_async(gni, (j, gridStep, resChar)) for j in procRkList]
            for i in rslts:
                i.get()
                # print(i.get())
        print("genNNin() pool closed.")

    else:
        if resChar and resChar not in resList:
            for j, rc in zip(procRkList, resList):
                gni(j, gridStep, rc)
        else:
            for j in procRkList:
                gni(j, gridStep, resChar)


def gridProtein(targList):
    global args
    global gridSearchMulti
    global gridStep
    gProt = gridProt(targList, args.pr, limit=args.limit_count)
    gProt.quiet = True
    if args.gridUG:
        gProt.updateDbGlobal = True
    print(f"starting gridStep for {len(gridProt.resDict)} residues")
    gProt.search_multi = gridSearchMulti
    gProt.updatedb = not args.gridNU
    gProt.gridStep(gridStep)


def gridNlp():
    conn = openDb()
    cur = conn.cursor()
    pqry(cur, "delete from grid_point_nlp")
    gcrstr, crstr = "", ""
    started = False
    for k in covalent_radii.keys():
        if started:
            crstr += f", {k.lower()}"
            gcrstr += f", g.{k.lower()}"
        else:
            crstr += f"{k.lower()}"
            gcrstr += f"g.{k.lower()}"
            started = True

    pqry(cur, f"select g.grid_ref, {crstr} from grid_atom_counts g")
    gacTbl = cur.fetchall()
    gacDict = {}
    for r in gacTbl:
        gacDict[r[0]] = np.array(r[1::], dtype=np.float64)
    for gref, gac in gacDict.items():
        pqry(
            cur,
            f"select g.gp_key, g.res_char, {gcrstr} from grid_point_counts g, grid_point gp where gp.grid_ref = {gref} and g.gp_key = gp.gp_key",
        )
        gpc = cur.fetchall()
        gpck = [g[0] for g in gpc]
        gpcr = [g[1] for g in gpc]
        gpc2 = [g[2::] for g in gpc]
        gpca = np.array(gpc2, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            rslt = -np.log(gpca / gac)
        rslt[np.isnan(rslt)] = 100
        rslt[np.isinf(rslt)] = 100

        # print(np.amin(rslt))
        # [','.join(v) for v in r[1]]
        for r in zip(gpck, gpcr, rslt):
            pqry(
                cur,
                f"insert into grid_point_nlp (gp_key, res_char, {crstr}) values ({r[0]}, '{r[1]}', { ', '.join([str(v) for v in r[2]]) })",
            )
    conn.commit()
    closeDb(conn)


def tstNNout(rklist, resChar=None):
    conn = openDb()
    cur = conn.cursor()

    count = 0
    total = len(rklist)
    print(f"starting tstNNout for {total} residues")

    (
        phiClassKey,
        omgClassKey,
        acbClassKey,
        psiH1key,
        phiH1key,
        omgH1key,
    ) = get_phi_omg_classes(cur)
    hedra_counts, dihedra_counts = get_dh_counts(hydrogens=False)

    for rk in rklist:
        count += 1
        problems = 0
        pqry(
            cur,
            "select r1.rprev, r1.res_char, r1.res_seqpos, c.chain_name from"
            f" residue r1, chain c where r1.res_key = {rk}"
            " and r1.chain_key = c.chain_key"
            " and r1.std_all_angles",
        )
        rslt = cur.fetchall()
        if rslt == []:
            continue  # skip no std_all_agles

        rpk, rc, rsp, chainName = rslt[0]
        dhLenArr1 = pqry1p(cur, f"select bytes from dhlen where res_key = {rk}")
        dhChrArr1 = pqry1p(cur, f"select bytes from dhchirality where res_key = {rk}")
        hArr1 = pqry1p(cur, f"select bytes from hlen where res_key = {rk}")

        # pad each array to max if doing all residues
        if resChar == "X":
            dhLenArr0 = np.pad(dhLenArr1, (0, MaxDihedron - len(dhLenArr1)))
            dhChrArr0 = np.pad(dhChrArr1, (0, MaxDihedron - len(dhChrArr1)))
            hArr0 = np.pad(hArr1, [(0, MaxHedron - len(hArr1)), (0, 0)])
        else:
            dhLenArr0, dhChrArr0, hArr0 = dhLenArr1, dhChrArr1, hArr1

        # output ready
        outputArr = np.concatenate((dhChrArr0, dhLenArr0, hArr0.flatten()))

        # end of aa0_dataset

        dhChrArr, dhLenArr, hArr, dhLenArrNorm, hArrNorm = outputArr2values(
            rc, outputArr
        )  # rc

        """
        assert np.array_equal(dhLenArr1, dhLenArr)
        assert np.array_equal(dhChrArr1, dhChrArr)
        assert np.array_equal(hArr1, hArr)
        """

        hedraAngle, hedraAngleRads, dihedraAngle, dihedraAngleRads = len2ang(
            rc, dhChrArr, dhLenArr, hArr  # rc
        )
        if rpk is not None:
            pqry(
                cur,
                "select d.angle, d.len14, d.chirality, d.class_key from dihedron d,"
                " dihedron_class dc"
                f" where d.res_key = {rpk}"
                f" and (d.class_key = {phiClassKey} or d.class_key = {omgClassKey})"
                " and d.class_key = dc.dc_key"
                " order by dc.d_class",  # omg, phi
            )
            rsltlst = cur.fetchall()
            dhndx = 0
        else:
            rsltlst = [(0, 0, 0), (0, 0, 0)]
            dhndx = 2

        q = (
            f"select d.ak_str, d.angle, d.len14, d.chirality, d.class_key from dihedron d"
            f" where d.res_key = {rk}"
            f" and d.class_key != {phiClassKey} and d.class_key != {omgClassKey}"
        )
        if acbClassKey is not None:
            q += f" and d.class_key != {acbClassKey}"
        pqry(cur, q)

        dhdict = {}
        for dh in cur.fetchall():
            dhdict[Edron.gen_tuple(dh[0])] = dh[1:]

        for k in sorted(dhdict):
            rsltlst.append(dhdict[k])
            # print(k, dhdict[k])

        dbDhArr = np.array([a[0] for a in rsltlst], dtype=float)
        dbDhLArr = np.array([a[1] for a in rsltlst], dtype=float)
        dbDhCArr = np.array([a[2] for a in rsltlst], dtype=float)

        """
        clist = [a[2] for a in rsltlst]
        for a in clist:
            c = pqry1(cur, f"select d_class from dihedron_class where dc_key = {a}")
            print(a, c)
        """

        if not np.allclose(
            dihedraAngle[dhndx : dihedra_counts[rc]],
            dbDhArr[dhndx:],
            rtol=1e-03,
            atol=1e-05,
        ):
            problems += 1
            print(chainName, "dihedra", rk, rsp, count)
            """"""
            # print(dihedraAngle)
            # print(dbDhArr)
            # print(np.column_stack((dihedraAngle, dbDhArr)))
            # print(dihedraAngle)
            print("gno lengths    :", dhLenArr[dhndx : dihedra_counts[rc]])
            print("db lengths     :", dbDhLArr[dhndx:])
            print("gno chirality  :", dhChrArr[dhndx : dihedra_counts[rc]])
            print("db chirality   :", dbDhCArr[dhndx:])
            print("gno len2angles :", dihedraAngle[dhndx : dihedra_counts[rc]])
            print("db angles      :", dbDhArr[dhndx:])
            # print(dhLenArr)
            print()
            """"""
        if rpk is not None:
            pqry(
                cur,
                "select h.angle, h.len12, h.len23, h.len13 from hedron h, hedron_class hc"
                f" where h.res_key = {rpk}"
                f" and (h.class_key = {phiH1key} or h.class_key = {omgH1key})"
                " and h.class_key = hc.hc_key"
                " order by hc.h_class",  # omg, phi
            )
            rsltlst = cur.fetchall()
            hndx = 0
        else:
            rsltlst = [(0, 0, 0, 0), (0, 0, 0, 0)]
            hndx = 2
        pqry(
            cur,
            f"select ak_str, angle, len12, len23, len13 from hedron where res_key = {rk}"
            f" and class_key != {phiH1key}",
        )
        hdict = {}
        for h in cur.fetchall():
            hdict[Edron.gen_tuple(h[0])] = h[1:]

        for k in sorted(hdict):
            rsltlst += [hdict[k]]
            # print(k, hdict[k])

        dbHArr = np.array([a[0] for a in rsltlst], dtype=float)
        if not np.allclose(
            hedraAngle[hndx : hedra_counts[rc]], dbHArr[hndx:], rtol=1e-03, atol=1e-08
        ):
            problems += 1
            print(chainName, "hedra", rk, rsp, count)
            """"""
            print("gno len2angles :", hedraAngle[hndx : hedra_counts[rc]])
            print("db angles      :", dbHArr[hndx:])
            print()
            """"""
        dAtoms = lenAng2coords(rc, hedraAngleRads, hArr, dihedraAngleRads)  # rc

        dbdAtoms = pqry1p(cur, f"select bytes from dhcoords where res_key = {rk}")
        # print(rc)
        # if dAtoms is None or dbdAtoms is None:
        #    print(rpk, rk, rc)
        if not np.allclose(
            dAtoms[dhndx : dihedra_counts[rc]], dbdAtoms, rtol=1e-02, atol=1e-07
        ):
            problems += 1
            print(chainName, "coords", rk, rsp, count)
            """"""
            zi = zip(dAtoms[dhndx:], dbdAtoms)
            for z in zi:
                if True or not np.allclose(z[0], z[1], rtol=1e-02, atol=1e-07):
                    print("calc:", z[0].flatten())
                    print("db  :", z[1].flatten())
                    print("----")
            """"""
            # print(dAtoms)
            #
            # print(dbdAtoms)
            # print()

        if problems > 0:
            print(f"chain {chainName} {rc} res key {rk} has {problems} problems")
            pqry(cur, f"update residue set std_all_angles = FALSE where res_key = {rk}")
            conn.commit()
            print()

        if (count % 100000) == 0:
            print(count, "/", total)

    #
    # print(f"tstNNout finished {count} residues")
    cur.close()
    conn.close()


def testNNoutput(targList, resChar="X"):
    rkList = genRkList(targList)
    # print(f"starting tstNNoutput for {len(rkList)} residues")
    if PROCESSES > 0:
        procRkList = splitList(rkList, PROCESSES)
        with Pool(PROCESSES) as p:
            rslts = [p.apply_async(tstNNout, (j, resChar)) for j in procRkList]
            for i in rslts:
                i.get()
                # print(i.get())
        print("testNNoutput() pool closed.")

    else:
        rklist = genRkList(targList, resChar)
        tstNNout(rklist, resChar)


def chainCheck(targList):
    if not targList:
        print("chain check require input files")
        return
    for chn in targList:
        if not pqry1(cur0, f"select 1 from chain where chain_name = '{chn}'"):
            print(f"{chn} not found")


if __name__ == "__main__":
    parseArgs()

    np.set_printoptions(precision=8, linewidth=300, suppress=True)

    toProcess = args.file
    if args.filelist:
        flist = open(args.filelist, "r")
        for aline in flist:
            fields = aline.split()
            pdbidMatch = pdbidre.match(fields[0])
            if pdbidMatch:
                toProcess.append(pdbidMatch.group(0))

    if len(toProcess) == 0 and not procResidues and not args.gnlp and not args.norm:
        print("no files to process. use '-h' for help")
        sys.exit(0)

    targList = []
    fileNo = 1
    for target in toProcess:
        if args.skip_count and fileNo <= args.skip_count:
            fileNo += 1
            continue
        if args.limit_count is not None:
            if args.limit_count <= 0:
                # sys.exit(0)
                break
            args.limit_count -= 1
        targList.append(target.upper())
        fileNo += 1

    if args.delete:
        for targ in targList:
            if args.delete:
                delPDB(targ)

    if args.dbl:
        dbLoadProtein(targList)
    if args.norm:
        genNormFactors()
    if args.sd:
        sdLoadProtein()
    if args.gno:
        genNNout(targList)
    if args.grid:
        gridProtein(targList)
    if args.gnlp:
        gridNlp()
        print("gridNlp done.")
    if args.gni:
        if args.rc:
            genNNin(targList, gridStep, resChar)
        else:
            genNNin(targList, gridStep)
    if args.gnir:
        genNNin(targList, gridStep, resChar=True)
    if args.tst:
        testNNoutput(targList, resChar)
    if args.chnchk:
        chainCheck(targList)

conn0.close()
