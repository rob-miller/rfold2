#!/usr/local/bin/python3 -u

import argparse
import sys
import numpy as np

from utils import parse_configuration
from datasets import get_dataset

# from Bio.PDB.internal_coords import IC_Residue, AtomKey, IC_Chain

from utils.rpdb import PDB_repository_base, getPDB, pdbidre, resList  # , get_dh_counts
from utils.rpdb import outputArr2values, len2ang, lenAng2coords

from utils.rdb import openDb, pqry, pqry1, pqry1p  # pgetset

from utils.rpdbT import rpdbT


dftlTarg = "2V9LA"  # "7RSAA"

lrtol = 0.1
latol = 0.1


def parseArgs():
    global args
    global PROCESSES
    # global sdPass
    arg_parser = argparse.ArgumentParser(
        description="test building local atom coords from dataset output"
    )

    arg_parser.add_argument("configFile", help="path to NN training configfile")

    arg_parser.add_argument(
        "-pf",
        dest="protFile",
        # nargs="*",
        help="a .cif path/protFile to read, or a PDB idCode with "
        "optional chain ID to read from {0} as .cif.gz (default {1})".format(
            (
                PDB_repository_base
                or "[PDB resource not defined - please configure before use]"
            ),
            dftlTarg,
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
        "-skip", dest="skip_count", help="count of pdb ID list entries to skip"
    )
    arg_parser.add_argument(
        "-limit",
        dest="limit_count",
        help="stop after processing this many pdb ID list entries",
    )

    args = arg_parser.parse_args()
    if args.limit_count is not None:
        args.limit_count = int(args.limit_count)


def getResAngleMap(cur):
    dihedraMap = {}
    hedraMap = {}
    for resChar in resList:
        rk = pqry1(
            cur,
            f"select res_key from residue where res_char = '{resChar}' limit 1",
        )
        pqry(
            cur,
            "select dc.d_class from dihedron d, dihedron_class dc"
            f" where d.res_key = {rk} and d.class_key = dc.dc_key"
            " order by dc.d_class",
        )
        dihedraMap[resChar] = [x[0] for x in cur.fetchall()]
        pqry(
            cur,
            "select hc.h_class from hedron h, hedron_class hc"
            f" where h.res_key = {rk} and h.class_key = hc.hc_key"
            " order by hc.h_class",
        )
        hedraMap[resChar] = [x[0] for x in cur.fetchall()]
    return dihedraMap, hedraMap


def find_min_rtol_atol(arr1, arr2):
    # initialize rtol and atol with large values
    rtol = 1.0
    atol = 1.0
    # loop through each pair of elements in arr1 and arr2
    for x, y in zip(arr1.flat, arr2.flat):
        # calculate the absolute difference between x and y
        diff = abs(x - y)
        # if diff is zero, skip this pair
        if diff == 0:
            continue
        # calculate the relative difference between x and y
        rel_diff = diff / abs(y)
        # update rtol and atol with smaller values if possible
        rtol = min(rtol, rel_diff)
        atol = min(atol, diff)
    # return rtol and atol
    return rtol, atol


def min_tol_for_allclose(a, b):
    """
    Compute the minimum values of atol and rtol required for a and b to be considered equal by numpy.allclose.
    """
    # Start with large tolerances and gradually decrease until arrays are considered equal
    # global lrtol, latol
    atol = 1  # latol
    rtol = 1  # lrtol
    while np.allclose(a, b, atol=atol, rtol=rtol):
        atol /= 10
        rtol /= 10
        if atol < np.finfo(float).eps and rtol < np.finfo(float).eps:
            # Arrays are very different, no tolerance level will make them equal
            return None, None
    return atol * 10, rtol * 10


def pmt(a, b):
    print("required rtol, atol = ", min_tol_for_allclose(a, b))


# @profile
def getData(outp, rpdbt):
    # cuda rpdbT
    coutp = outp.to(device)
    ctargLens = rpdbt.outputArr2values(rc, coutp)
    ctargAngs = rpdbt.len2ang(rc, ctargLens[0], ctargLens[1], ctargLens[2])
    ctargAtoms = rpdbt.lenAng2coords(rc, ctargAngs[1], ctargLens[2], ctargAngs[3])

    # numpy rpdb
    noutp = outp.detach().cpu().numpy()
    ntargLens = outputArr2values(rc, noutp)
    ntargAngs = len2ang(rc, ntargLens[0], ntargLens[1], ntargLens[2])
    ntargAtoms = lenAng2coords(rc, ntargAngs[1], ntargLens[2], ntargAngs[3])

    return ctargLens, ctargAngs, ctargAtoms, ntargLens, ntargAngs, ntargAtoms


if __name__ == "__main__":
    # read instructions
    parseArgs()
    if not args.configFile:
        print("no files to process. use '-h' for help")
        sys.exit(0)

    config = parse_configuration(args.configFile)
    nn_dataset = get_dataset(config["dataset"])
    device = config["model"]["devlist"][-1]
    rpdbt = rpdbT(device)

    conn = openDb()
    cur = conn.cursor()
    dihedraMap, hedraMap = getResAngleMap(cur)
    targList = []

    if args.protFile:
        targList.append(args.protFile)
    if args.filelist:
        flist = open(args.filelist, "r")
        for aline in flist:
            fields = aline.split()
            pdbidMatch = pdbidre.match(fields[0])
            if pdbidMatch:
                targList.append(pdbidMatch.group(0))
    if targList == []:
        targList.append(dftlTarg)

    fileNo = 0
    resCount = 0
    for targ in targList:
        if args.skip_count and fileNo <= args.skip_count:
            fileNo += 1
            continue
        if args.limit_count is not None:
            if args.limit_count <= 0:
                # sys.exit(0)
                break
            args.limit_count -= 1

        (pdb_structure, prot_id, chainID, protFile) = getPDB(targ)
        if not pdb_structure:
            print(
                f"unable to load {config['target']['protein']} as cif/pdb structure;use -h for help."
            )
            continue  # sys.exit(0)
        else:
            fileNo += 1
            print(f"{fileNo} loaded {prot_id} chain {chainID} from {protFile}")

        pdb_structure.atom_to_internal_coordinates()
        pdb_structure.internal_to_atom_coordinates()
        cic = pdb_structure.internal_coord
        rdndx = {v: k for k, v in cic.dihedraNdx.items()}
        pdbkey = pqry1(cur, f"select pdb_key from pdb where pdb_id='{prot_id}'")
        if pdbkey is None:
            print(f"{prot_id} not found in pdb table")
            continue
        chain_name = prot_id + chainID
        chain_key = pqry1(
            cur, f"select chain_key from chain where chain_name = '{chain_name}'"
        )
        if chain_key is None:
            print(f"{chain_name} not found in chain table")
            continue
        for ric in cic.ordered_aa_ic_list:
            rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")
            rkey = pqry1(
                cur,
                f"select res_key from residue where res_seqpos = '{rsp}' and chain_key = {chain_key}",
            )
            if rkey is None:
                print(f"{rsp} not found in residue table for chain {chain_name}")
                continue
            rc = ric.rbase[2]

            # dict of angles from ric - this is base truth
            # from NNOnly_mfn NNOnlyMfn move
            dhndx = 0
            dhs = sorted(ric.dihedra)  # AtomKey sort on tuples
            dndxlst = []
            if len(ric.rprev) != 0:
                # omg, phi only exist if rprev exists, get up front
                try:
                    dndxlst = [
                        ric.pick_angle("omg").ndx,
                        ric.pick_angle("phi").ndx,
                    ]
                except AttributeError:
                    print(f"fail getting rprev for {chain_name} {rsp}")
                    continue
            else:
                dhndx = 2
            dndxlst += [ric.dihedra[x].ndx for x in dhs if x[2] in ric]

            rclassdict = {
                cic.dihedra[rdndx[ndx]].e_class: cic.dAtoms[ndx] for ndx in dndxlst
            }

            # dhsec = set of dh to predict for this residue
            dhs = sorted(ric.dihedra)
            dhsec = [ric.dihedra[dh].e_class for dh in dhs]
            dhsec = [ec for ec in dhsec if ec not in ["CNCAC", "CACNCA", "CNCACB"]]
            if len(ric.rprev) > 0:
                dhsec.insert(0, "CNCAC")
                dhsec.insert(0, "CACNCA")

            # database dhcoords
            dbdhcoords = pqry1p(
                cur, f"select bytes from dhcoords where res_key = {rkey}"
            )

            if dbdhcoords is None:
                print(f"no data for {chain_name} {rsp}")
                continue
            dbdhcdict = {
                dhec: dbdhcoords[ndx] for dhec, ndx in zip(dhsec, range(len(dhsec)))
            }

            dbdhlens = pqry1p(cur, f"select bytes from dhlen where res_key = {rkey}")

            if dbdhlens is None:
                print(f"no length data for {chain_name} {rsp}")
                continue

            dbdhldict = {
                dhec: dbdhlens[ndx] for dhec, ndx in zip(dhsec, range(len(dhsec)))
            }

            dbhlens = pqry1p(cur, f"select bytes from hlen where res_key = {rkey}")

            dbhldict = {
                dhec: dbhlens[ndx] for dhec, ndx in zip(dhsec, range(len(dhsec)))
            }

            dbdhchi = pqry1p(
                cur, f"select bytes from dhchirality where res_key = {rkey}"
            )

            dbdhchdict = {
                dhec: dbdhchi[ndx] for dhec, ndx in zip(dhsec, range(len(dhsec)))
            }

            # now get NN output
            inp, outp = nn_dataset.getRkInOut(rkey, rc)

            (
                ctargLens,
                ctargAngs,
                ctargAtoms,
                ntargLens,
                ntargAngs,
                ntargAtoms,
            ) = getData(outp, rpdbt)

            ctargLens = tuple(x.detach().cpu().numpy() for x in ctargLens)
            ctargAngs = tuple(x.detach().cpu().numpy() for x in ctargAngs)
            ctargAtoms = ctargAtoms.detach().cpu().numpy()

            cttadict = {
                dhec: ctargAtoms[ndx + dhndx]
                for dhec, ndx in zip(dhsec, range(len(dhsec)))
            }

            # for a in ctargAtoms[:3]:
            #    print()

            nttadict = {
                dhec: ntargAtoms[ndx + dhndx]
                for dhec, ndx in zip(dhsec, range(len(dhsec)))
            }

            lenOk = True
            lenList = []
            for i in range(len(ntargLens)):
                rslt = np.allclose(ntargLens[i], ctargLens[i], rtol=lrtol, atol=latol)
                lenList.append(rslt)
                if not rslt:
                    lenOk = False
            if not lenOk:
                print(f"{chain_name} {rsp} length data match fail numpy vs cuda: ")
                for i in range(len(ntargLens)):
                    print(i, lenList[i])
                    if not lenList[i]:
                        pmt(ntargLens[i], ctargLens[i])

            angOk = True
            angList = []
            for i in range(len(ntargAngs)):
                rslt = np.allclose(ntargAngs[i], ctargAngs[i], rtol=lrtol, atol=latol)
                angList.append(rslt)
                if not rslt:
                    angOk = False
            if not angOk:
                print(f"{chain_name} {rsp} angle data match fail numpy vs cuda:")
                for i in range(len(ntargAngs)):
                    print(i, angList[i])
                    if not angList[i]:
                        pmt(ntargAngs[i], ctargAngs[i])

            atomOk = True
            atomList = []
            for i in range(len(ntargAtoms)):
                rslt = np.allclose(ntargAtoms[i], ctargAtoms[i], rtol=lrtol, atol=latol)
                atomList.append(rslt)
                if not rslt:
                    atomOk = False
            if not atomOk:
                print(f"{chain_name} {rsp} atom data match fail numpy vs cuda:")
                for i in range(len(ntargAtoms)):
                    print(i, atomList[i])
                    if not atomList[i]:
                        pmt(ntargAtoms[i], ctargAtoms[i])

            if not (
                np.allclose(ntargLens[0], dbdhchi)
                or np.allclose(ntargLens[3], dbdhlens)
                or np.allclose(ntargLens[4], dbhlens)
            ):
                print(f"{chain_name} {rsp} length data match fail numpy vs database: ")
                chrslt = np.allclose(ntargLens[0], dbdhchi)
                print("chirality: ", chrslt)
                if not chrslt:
                    pmt(ntargLens[0], dbdhchi)
                dlrslt = np.allclose(ntargLens[3], dbdhlens)
                print("dlens: ", dlrslt)
                if not dlrslt:
                    pmt(ntargLens[3], dbdhlens)
                hlrslt = np.allclose(ntargLens[4], dbhlens)
                print("hlens: ", hlrslt)
                if not hlrslt:
                    pmt(ntargLens[4], dbhlens)

            allgood = True
            allpass = True
            for k in rclassdict.keys():
                crslt = np.allclose(rclassdict[k], cttadict[k], rtol=lrtol, atol=latol)
                nrslt = np.allclose(rclassdict[k], nttadict[k], rtol=lrtol, atol=latol)
                dbrslt = np.allclose(rclassdict[k], dbdhcdict[k])
                allgood = crslt and nrslt and dbrslt
                if not allgood:
                    allpass = False
                    print(
                        f"{chain_name} {rsp} datoms {k} cuda: {crslt} np: {nrslt} db: {dbrslt}"
                    )
                    print(rclassdict[k])
                    if not crslt:
                        print("cuda:")
                        print(cttadict[k])
                        pmt(rclassdict[k], cttadict[k])
                        found = False
                        for subk in cttadict.keys():
                            if np.allclose(
                                rclassdict[k], cttadict[subk], rtol=lrtol, atol=latol
                            ):
                                print(f"matches cuda dict {subk}")
                                found = True
                        if not found:
                            print("   no other match")
                    if not nrslt:
                        print(
                            " numpy:",
                            ("matches cuda" if atomOk else "does not match cuda"),
                        )
                        if not atomOk:
                            print(nttadict[k])
                            print(
                                "required rtol, atol = ",
                                min_tol_for_allclose(cttadict[k], nttadict[k]),
                            )
                    if not dbrslt:
                        print(" database")
                        print(dbdhcdict[k])
                    print("---")

            if not allpass:
                print("===========================")
            resCount += 1

    print(f"finished {fileNo} targets ({resCount} residues) of {len(targList)}.")
