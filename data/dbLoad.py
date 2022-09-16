#!/usr/local/bin/python3


from multiprocessing import Pool
import sys
import argparse
import signal
import numpy as np
import math
import warnings

# import os

from utils.rpdb import PDB_repository_base, getPDB, pdbidre, get_dh_counts
from utils.rdb import openDb, pqry, pqry1, pgetset

# , closeDb

# import rdb

from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# from Bio.PDB.internal_coords import IC_Chain, IC_Residue, Hedron, Dihedron
from Bio.PDB.internal_coords import IC_Residue, AtomKey, IC_Chain
from Bio.PDB.ic_data import (
    covalent_radii,
    ic_data_backbone,
    ic_data_sidechains,
    residue_atom_bond_state,
)
from Bio.PDB.ic_rebuild import structure_rebuild_test
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBExceptions import PDBException

warnings.simplefilter("ignore", PDBConstructionWarning)

# IC_Residue.accept_atoms += IC_Residue.accept_deuteriums
# only mainchain atoms, no hydrogens or deuterium
IC_Residue.accept_atoms = IC_Residue.accept_mainchain
IC_Residue.no_altloc = True
IC_Chain.MaxPeptideBond = 1.5
# AtomKey.d2h = True

# multiprocessing support
PROCESSES = 7

# db support
# conn = openDb()


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
        "-skip", dest="skip_count", help="count of pdb ID list entries to skip"
    )
    arg_parser.add_argument(
        "-limit",
        dest="limit_count",
        help="stop after processing this many pdb ID list entries",
    )
    arg_parser.add_argument(
        "-p", dest="PROCESSES", help=f"number of subprocesses, default {PROCESSES}"
    )

    args = arg_parser.parse_args()

    if args.skip_count:
        args.skip_count = int(args.skip_count)
    if args.limit_count:
        args.limit_count = int(args.limit_count)
    if args.PROCESSES:
        PROCESSES = int(args.PROCESSES)
    # if args.sd:
    #    sdPass = True


class dbLoad:
    # number of environment atoms to log in env_atoms table with xyz coords
    envAtomsMax = 6
    # number of env atom distances to track for non_bond_dists table
    nbdReport = 6

    # environment atom must be in envAtomsMax for at least this many center atoms:
    # rtm try 1 next time
    minEnvAtomCount = 2

    # second pass to compute standard deviation
    sdPass = False

    # dicts for covalent radii classes and non-bond distance classes
    acr = {}
    acrp = {}
    nbds = {}
    nbdsAvg = {}

    def __init__(self, sdp=False):
        self.openDb()
        self.load_classes()
        self.sdPass = sdp

    def __del__(self):
        self.closeDb()

    def openDb(self):
        self.conn = openDb()
        self.cur = self.conn.cursor()

    def commit(self):
        self.conn.commit()

    def closeDb(self):
        if self.conn:
            self.conn.close()
        self.cur = self.conn = None

    def dlq(self, qry):
        pqry(self.cur, qry)

    def dlq1(self, qry):
        return pqry1(self.cur, qry)

    def load_covalent_radii(self):
        for str, rad in covalent_radii.items():
            self.dlq(
                "insert into atom_covalent_radii(cr_class, cr_radius)"
                f" values('{str}', {rad}) on conflict do nothing",
            )

    def load_cr_pairs(self):
        self.dlq("select acr_key from atom_covalent_radii;")
        crs = [cr[0] for cr in self.cur.fetchall()]
        for cr1 in crs:
            for cr2 in crs:
                self.dlq(
                    "insert into acr_pairs(acr1_key, acr2_key)"
                    f" values({cr1}, {cr2}) on conflict do nothing;",
                )

    def load_nbds(self):
        for i in range(self.nbdReport):
            self.dlq(
                "insert into non_bond_dists(chain_key, acrp_key, dist_num)"
                f" values (NULL, NULL, {i}) on conflict do nothing",
            )

        self.dlq("select acrp_key from acr_pairs;")
        crps = [crp[0] for crp in self.cur.fetchall()]

        for crp in crps:
            for i in range(self.nbdReport):
                self.dlq(
                    "insert into non_bond_dists(chain_key, acrp_key, dist_num)"
                    f" values (NULL, {crp}, {i}) on conflict do nothing",
                )

    def load_edra_classes(self):
        def gen_classes(edron, res):
            if len(edron) == 4:
                ec = "dihedron"
                el = "d"
            else:
                ec = "hedron"
                el = "h"
            e_class = "".join(edron)
            self.dlq(
                f"insert into {ec}_class ({el}_class) values('{e_class}')"
                " on conflict do nothing",
            )
            if res == "X":
                for r in "GAVLIMFPSTCNQYWDEHKR":
                    er_class = "".join(r + a for a in edron)
                    self.dlq(
                        f"insert into {ec}_residue_class ({el}r_class) values('{er_class}')"
                        " on conflict do nothing",
                    )
            else:
                er_class = "".join(res + a for a in edron)
                self.dlq(
                    f"insert into {ec}_residue_class ({el}r_class) values('{er_class}')"
                    " on conflict do nothing",
                )

            rabs = residue_atom_bond_state[res]
            rabsx = residue_atom_bond_state["X"]
            e_cr_class = "".join(rabs.get(a, rabsx.get(a, "Hsb")) for a in edron)
            self.dlq(
                f"insert into {ec}_cr_class ({el}cr_class) values('{e_cr_class}')"
                " on conflict do nothing",
            )

        for edron in ic_data_backbone:  # note fixed order as we sort on this key later
            gen_classes(edron, "X")
        for res in ic_data_sidechains:
            for edron in ic_data_sidechains[res]:
                gen_classes(edron[0:3], res)

    def load_classes(self):
        # global acr, acrp, nbds
        data = False
        if not self.dlq1("select exists (select * from atom_covalent_radii)"):
            self.load_covalent_radii()
            data = True

        if not self.dlq1("select exists (select * from hedron_cr_class)"):
            self.load_edra_classes()
            data = True

        if not self.dlq1("select exists (select * from acr_pairs)"):
            self.load_cr_pairs()
            data = True

        if not self.dlq1("select exists (select * from non_bond_dists)"):
            self.load_nbds()
            data = True

        self.dlq("select cr_class, acr_key from atom_covalent_radii;")
        self.acr = {k: v for (k, v) in self.cur.fetchall()}

        self.dlq("select acr1_key, acr2_key, acrp_key from acr_pairs;")
        for acr1k, acr2k, acrpk in self.cur.fetchall():
            if acr1k not in self.acrp:
                self.acrp[acr1k] = {}
            self.acrp[acr1k][acr2k] = acrpk

        self.dlq(
            "select dist_num, nbd_key from non_bond_dists where chain_key is NULL and acrp_key is NULL",
        )
        self.nbds["all"] = {k: v for (k, v) in self.cur.fetchall()}

        self.dlq(
            "select acrp_key, dist_num, nbd_key from non_bond_dists where chain_key is NULL and acrp_key is not NULL",
        )
        for acrpk, dn, nbdk in self.cur.fetchall():
            if acrpk not in self.nbds:
                self.nbds[acrpk] = {}
            self.nbds[acrpk][dn] = nbdk

        if data:
            self.commit()

    def loadResidueEnvirons(
        self,
        cic: IC_Chain,
        distPlot: np.array,
        akDict: dict = None,
        chain_key: int = None,
    ):

        # global acr, acrp, nbds, nbdsAvg, sdPass
        dpSorted = np.argsort(distPlot)  # get indexes of each row so closest atom first
        aaNdx = cic.atomArrayIndex
        ndxAA = {ndx: ak for ak, ndx in aaNdx.items()}
        aa = cic.atomArray
        # aaCopy = aa.copy()
        aNameX = AtomKey.fields.atm
        rPosX = AtomKey.fields.respos
        rNameX = AtomKey.fields.resname
        if akDict is not None:
            nbdDict = {}
            for i in self.acrp:
                for j in self.acrp[i]:
                    nbdDict[self.acrp[i][j]] = {}
                    nbdDict[self.acrp[i][j]]["min"] = [1000] * self.nbdReport
                    nbdDict[self.acrp[i][j]]["max"] = [0] * self.nbdReport
                    nbdDict[self.acrp[i][j]]["sum"] = [0] * self.nbdReport
                    nbdDict[self.acrp[i][j]]["count"] = [0] * self.nbdReport

        for ric in cic.ordered_aa_ic_list:
            # global minEnvAtomCount
            envAtoms = {}
            envAtomCounts = {}
            envAtomMin = {}
            envAtomMax = {}
            envAtomMinP = {}
            envAtomMaxP = {}
            resKey = None
            prevSet = set()
            envCoords = {}
            if 0 < len(ric.rprev):
                for r in ric.rprev:
                    prevSet.update(r.ak_set)
            for ak in ric.ak_set:
                if ak.ric != ric:
                    continue  # skip adjacent residue as centers for dist check
                if akDict is not None and ak.akl[aNameX] == "CA":
                    resKey = akDict[ak][1]
                dpa = dpSorted[aaNdx[ak]]  # get sorted dp-indexes row for this ak
                envCount = 0
                dpaNdx = 0
                for envAtmNdx in dpa:
                    eak = ndxAA[envAtmNdx]
                    if eak in ric.ak_set:
                        # now skip close atoms if they are this ic_residue
                        # or in next ic_residue backbone
                        continue
                    if eak in prevSet and ak in prevSet:
                        # skip close atoms if in prev residue and center atom is bacbone
                        continue
                    rPosDiff = int(ak.akl[rPosX]) - int(eak.akl[rPosX])
                    if abs(rPosDiff) == 1:  # adjacent residues
                        if rPosDiff < 0:  # this residue to next residue
                            if ak.akl[aNameX] == "C":
                                if eak.akl[aNameX] == "CB":
                                    continue  # dihedron C(i-1)-CB(i)
                                if eak.akl[rNameX] == "P" and eak.akl[aNameX] == "CD":
                                    continue  # hedron X(C)-Pro(CD)
                            if (
                                ak.akl[aNameX] == "CA"
                                and eak.akl[rNameX] == "P"
                                and eak.akl[aNameX] == "CD"
                            ):
                                continue  # dihedron X(CA)-Pro(CD)
                        else:  # previous residue to this residue
                            if eak.akl[aNameX] == "C":
                                if ak.akl[aNameX] == "CB":
                                    continue  # dihedron C(i-1)-CB(i)
                                if ak.akl[rNameX] == "P" and ak.akl[aNameX] == "CD":
                                    continue  # hedron X(C)-Pro(CD)
                            if (
                                eak.akl[aNameX] == "CA"
                                and ak.akl[rNameX] == "P"
                                and ak.akl[aNameX] == "CD"
                            ):
                                continue  # dihedron X(CA)-Pro(CD)
                    envAtoms[envAtmNdx] = eak
                    envAtmDistance = distPlot[aaNdx[ak]][aaNdx[eak]]
                    if akDict is not None:
                        if envAtmDistance < 1.3:
                            continue  # rtm arbitrary cutoff
                        if envAtmNdx in envAtomCounts:
                            envAtomCounts[envAtmNdx] += 1
                        else:
                            envAtomCounts[envAtmNdx] = 1
                            envAtomMin[envAtmNdx] = envAtmDistance
                            envAtomMax[envAtmNdx] = envAtmDistance
                            envAtomMinP[envAtmNdx] = ak
                            envAtomMaxP[envAtmNdx] = ak
                        if dpaNdx < self.nbdReport:
                            acr1 = self.acr[ak.cr_class()]
                            acr2 = self.acr[eak.cr_class()]
                            acrpk = self.acrp[acr1][acr2]
                            if self.sdPass:
                                nbdsk = self.nbds[acrpk][dpaNdx]
                                sdv = envAtmDistance - self.nbdsAvg[nbdsk]
                                nbdDict[acrpk]["sum"][dpaNdx] += sdv * sdv
                            else:
                                if envAtmDistance < envAtomMin[envAtmNdx]:
                                    envAtomMin[envAtmNdx] = envAtmDistance
                                    envAtomMinP[envAtmNdx] = ak
                                if envAtmDistance > envAtomMax[envAtmNdx]:
                                    envAtomMax[envAtmNdx] = envAtmDistance
                                    envAtomMaxP[envAtmNdx] = ak
                                if envAtmDistance < nbdDict[acrpk]["min"][dpaNdx]:
                                    nbdDict[acrpk]["min"][dpaNdx] = envAtmDistance
                                if envAtmDistance > nbdDict[acrpk]["max"][dpaNdx]:
                                    nbdDict[acrpk]["max"][dpaNdx] = envAtmDistance
                                nbdDict[acrpk]["sum"][dpaNdx] += envAtmDistance
                            nbdDict[acrpk]["count"][dpaNdx] += 1
                    else:
                        # for eval only, keep track of envAtomCounts,
                        # do not ignore atoms closer that 1.3 angstrom
                        if envAtmNdx in envAtomCounts:
                            envAtomCounts[envAtmNdx] += 1
                        else:
                            envAtomCounts[envAtmNdx] = 1

                    envCount += 1
                    dpaNdx += 1
                    if envCount > self.envAtomsMax:
                        break

                if not self.sdPass:
                    psiO = ric.pick_angle("N:CA:C:O")
                    if psiO is None:
                        continue
                    if len(envAtoms) == 0:
                        continue
                    cst = np.transpose(psiO.cst)
                    # rtm here we reject env-atoms if not in env of at least 2 center-atoms:
                    asel = np.array(
                        sorted(
                            [
                                eai
                                for eai in envAtoms
                                if envAtomCounts[eai] >= self.minEnvAtomCount
                            ]
                        )
                    )
                    transformed = aa[asel].dot(cst)
                    if akDict is not None:
                        i = 0
                        for ndx in asel:
                            # print(envAtoms[ndx], transformed[i])
                            aKey = akDict[envAtoms[ndx]][0]
                            minp = akDict[envAtomMinP[ndx]][0]
                            maxp = akDict[envAtomMaxP[ndx]][0]
                            t = transformed[i]
                            self.dlq(
                                "insert into env_atoms (res_key, env_atom, x, y, z, min, min_partner, max, max_partner)"
                                f" values ({resKey}, {aKey}, {t[0]}, {t[1]}, {t[2]}, {envAtomMin[ndx]}, {minp}, {envAtomMax[ndx]}, {maxp})"
                                # " on conflict do nothing",
                            )
                            i += 1
                    else:
                        # log transformed env atoms for efn
                        envCoords[ric] = transformed

        if akDict is not None:
            for acrpk in nbdDict:
                for i in range(self.nbdReport):
                    if nbdDict[acrpk]["count"][i] == 0:
                        next
                    if self.sdPass:
                        self.dlq(
                            "update non_bond_dists set (sum_sd_dev, sd_count) ="
                            f" ({nbdDict[acrpk]['sum'][i]}, {nbdDict[acrpk]['count'][i]})"
                            f" where chain_key = {chain_key} and acrp_key = {acrpk}"
                            f" and dist_num = {i}",
                        )
                    else:
                        self.dlq(
                            "insert into non_bond_dists(chain_key, acrp_key, dist_num, min, max, sum, count) values"
                            f" ({chain_key}, {acrpk}, {i}, {nbdDict[acrpk]['min'][i]}, {nbdDict[acrpk]['max'][i]}, {nbdDict[acrpk]['sum'][i]}, {nbdDict[acrpk]['count'][i]})"
                            " on conflict (chain_key, acrp_key, dist_num) do update set (min, max, sum, count) ="
                            f" ({nbdDict[acrpk]['min'][i]}, {nbdDict[acrpk]['max'][i]}, {nbdDict[acrpk]['sum'][i]}, {nbdDict[acrpk]['count'][i]})",
                        )
        else:
            return envCoords

    def loadResidues(self, cic, chain_key, atomDict):
        for ric in cic.ordered_aa_ic_list:
            rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")
            dssp = ric.residue.xtra.get("SS_DSSP", "")
            asa = ric.residue.xtra.get("EXP_DSSP_ASA", "NULL")
            rasa = ric.residue.xtra.get("EXP_DSSP_RASA", "NULL")
            if rasa == "NA":
                rasa = "NULL"  # 4FZVB
            resKey = pgetset(
                self.cur,
                f"select res_key from residue where res_seqpos = '{rsp}' and chain_key = {chain_key}",
                "insert into residue (chain_key, res_char, res_seqpos, dssp, asa, rasa)"
                f" values ({chain_key}, '{ric.rbase[2]}', '{rsp}', '{dssp}', {asa}, {rasa})"
                " returning res_key",
            )
            if len(ric.rprev) > 0:
                # set prev and next residues
                # ignore cases where multiple as hopefully we only look at clean entries
                for rp in ric.rprev:
                    rpsp = rp.rbase[2] + str(rp.rbase[0]) + (rp.rbase[1] or "")
                    rprk = self.dlq1(
                        f"select res_key from residue where res_seqpos = '{rpsp}' and chain_key = {chain_key}",
                    )
                    self.dlq(
                        f"update residue set rnext = {resKey} where res_key = {rprk}"
                    )
                    self.dlq(
                        f"update residue set rprev = {rprk} where res_key = {resKey}"
                    )
                    break  # only take first as preferred

            for ak in ric.ak_set:
                if ak.ric == ric:
                    crck = self.acr[ak.cr_class()]
                    # self.dlq1(
                    #    cur,
                    #    f"select acr_key from atom_covalent_radii where cr_class = '{ak.cr_class()}'",
                    # )
                    atmKey = pgetset(
                        self.cur,
                        f"select atom_key from atom where res_key = {resKey} and atom_name = '{ak}'",
                        f"insert into atom (res_key, atom_name, crc_key) values ({resKey}, '{ak}', {crck})"
                        " returning atom_key",
                    )
                    atomDict[ak] = (atmKey, resKey)

    def loadHedra(self, cic, atomDict):
        for key, hedron in cic.hedra.items():
            akl = [0, 0, 0]
            # print(key)
            for i in range(3):
                akl[i] = atomDict[key[i]][0]
            rk = atomDict[key[0]][1]
            hk = self.dlq1(
                f"select hedron_key from hedron where res_key = {rk} and ak_str = '{hedron.id}'",
            )
            if hk is not None:
                continue  # skip if this hedron already loaded, don't wait for on conflict
            # print(f"loading {hedron.id}")
            # if insert, might race with other processes so skip on conflict
            hck = pgetset(
                self.cur,
                f"select hc_key from hedron_class where h_class = '{hedron.e_class}'",
                f"insert into hedron_class (h_class) values ('{hedron.e_class}')"
                " on conflict do nothing returning hc_key",
            )
            hcrck = pgetset(
                self.cur,
                f"select hcrc_key from hedron_cr_class where hcr_class = '{hedron.cre_class}'",
                f"insert into hedron_cr_class (hcr_class) values ('{hedron.cre_class}')"
                " on conflict do nothing returning hcrc_key",
            )
            # if above inserted then this does too, supply conn for commit
            hrck = pgetset(
                self.cur,
                f"select hrc_key from hedron_residue_class where hr_class = '{hedron.re_class}'",
                f"insert into hedron_residue_class (hr_class) values ('{hedron.re_class}')"
                " on conflict do nothing returning hrc_key",
                self.conn,
            )

            if hck is None or hcrck is None or hrck is None:
                print(
                    f"fail {rk} {hedron.hedron2.id} {hedron.e_class} {hck} {hedron.cre_class} {hcrck} {hedron.re_class} {hrck}"
                )
                sys.exit()

            self.dlq(
                "insert into hedron (res_key, ak_str, ak1, ak2, ak3, class_key,"
                " r_class_key, cr_class_key, angle, len12, len23, len13)"
                f" values ({rk}, '{hedron.id}', {akl[0]}, {akl[1]}, {akl[2]},"
                f" {hck}, {hrck}, {hcrck}, {hedron.angle}, {hedron.len12}, {hedron.len23},"
                f" {cic.hedraL13[hedron.ndx]}) on conflict do nothing",
            )
            # hk = self.dlq1(
            #    cur,
            #    f"select hedron_key from hedron where res_key = {rk} and ak_str = '{hedron.id}'",
            # )

    def loadDihedra(self, cic, atomDict, dhSigns):
        for key, dihedron in cic.dihedra.items():
            # print(key)
            rk = atomDict[key[0]][1]

            dk = self.dlq1(
                f"select dihedron_key from dihedron where res_key = {rk} and ak_str = '{dihedron.id}'",
            )
            if dk is not None:
                continue  # skip if this hedron already loaded, don't wait for on conflict
            # print(f"loading {dihedron.id}")

            hk1 = self.dlq1(
                "select hedron_key from hedron where res_key ="
                f" {rk} and ak_str = '{dihedron.hedron1.id}'",
            )

            rev = dihedron.reverse

            rk2 = atomDict[dihedron.hedron2.atomkeys[0]][1]
            hk2 = self.dlq1(
                "select hedron_key from hedron where res_key ="
                f" {rk2} and ak_str = '{dihedron.hedron2.id}'",
            )

            dck = pgetset(
                self.cur,
                f"select dc_key from dihedron_class where d_class = '{dihedron.e_class}'",
                f"insert into dihedron_class (d_class) values ('{dihedron.e_class}')"
                " on conflict do nothing returning dc_key",
            )
            dcrck = pgetset(
                self.cur,
                f"select dcrc_key from dihedron_cr_class where dcr_class = '{dihedron.cre_class}'",
                f"insert into dihedron_cr_class (dcr_class) values ('{dihedron.cre_class}')"
                " on conflict do nothing returning dcrc_key",
            )
            drck = pgetset(
                self.cur,
                f"select drc_key from dihedron_residue_class where dr_class = '{dihedron.re_class}'",
                f"insert into dihedron_residue_class (dr_class) values ('{dihedron.re_class}')"
                " on conflict do nothing returning drc_key",
                self.conn,
            )
            if dck is None or dcrck is None or drck is None:
                print(
                    f"fail {rk} {rk2} {dihedron.hedron2.id} {dihedron.e_class} {dck} {dihedron.cre_class} {dcrck} {dihedron.re_class} {drck}"
                )
                sys.exit()

            self.dlq(
                "insert into dihedron (res_key, ak_str, hk1, hk2, reverse, class_key,"
                " r_class_key, cr_class_key, angle, len14, chirality)"
                f" values ({rk}, '{dihedron.id}', {hk1}, {hk2}, {rev},"
                f" {dck}, {drck}, {dcrck}, {dihedron.angle},"
                f" {cic.dihedraL14[dihedron.ndx]}, {dhSigns[dihedron.ndx]})"
                " on conflict do nothing",
            )

    def loadChainResidues(self, chain_key, pdb_chain):
        # global conn
        # global sdPass
        if not pdb_chain.internal_coord:
            # with warnings.catch_warnings():
            #    warnings.simplefilter("ignore", PDBConstructionWarning)
            pdb_chain.atom_to_internal_coordinates()
        cic = pdb_chain.internal_coord
        atomDict = {}
        ric = cic.ordered_aa_ic_list[0]
        rsp = ric.rbase[2] + str(ric.rbase[0]) + (ric.rbase[1] or "")
        r0k = self.dlq1(
            f"select res_key from residue where res_seqpos = '{rsp}' and chain_key = {chain_key}",
        )
        if r0k is not None and not self.sdPass:
            return  # short circuit if already loaded

        # compute hedra L13 and dihedra L14
        dhSigns = (cic.dihedral_signs()).astype(int)
        distPlot = cic.distance_plot()
        cic.distplot_to_dh_arrays(distPlot, dhSigns)

        self.loadResidues(cic, chain_key, atomDict)  # need atomDict on sdPass

        if not self.sdPass:
            self.loadHedra(cic, atomDict)
            self.loadDihedra(cic, atomDict, dhSigns)

        self.loadResidueEnvirons(cic, distPlot, atomDict, chain_key)
        # conn.commit()

    def loadPdbChain(self, pdb_chain, prot_id, chainID, filename, sourceListName):
        # global conn
        # global sdPass
        # sd = "sd:" if self.sdPass else ""
        chain_name = prot_id + chainID
        # print(sd + chain_name)
        pdb_key = None
        # cur = conn.cursor()
        rebuild = self.dlq1(
            f"select rebuild from chain where chain_name = '{chain_name}'"
        )
        if rebuild is None:
            # print(f"rebuild {chain_name}")
            pdb_key = pgetset(
                self.cur,
                f"select pdb_key from pdb where pdb_id='{prot_id}'",
                f"insert into pdb(pdb_id, filename) values('{prot_id}', '{filename}')"
                " on conflict do nothing returning pdb_key",
            )
            r = structure_rebuild_test(
                pdb_chain, quick=True
            )  # quick because mainchain only
            cbreaks = len(pdb_chain.internal_coord.initNCaCs) - 1
            slnStr = "NULL" if sourceListName is None else f"'{sourceListName}'"
            self.dlq(
                "insert into chain(pdb_key, chain_name, chain_id, residues, atoms, chain_breaks, disordered_atoms, rebuild, source_list)"
                f" values({pdb_key}, '{chain_name}', '{chainID}', {r['residues']}, {r['aCount']}, {cbreaks}, {r['disAtmCount']}, {r['pass']}, {slnStr})"
                " on conflict do nothing"
                # f" on conflict(chain_name) do update set (residues, atoms, chain_breaks, disordered_atoms, rebuild) = ( {r['residues']}, {r['aCount']}, {cbreaks}, {r['disAtmCount']}, {r['pass']})",
            )
            rebuild = r["pass"]
        if rebuild:
            chain_key = self.dlq1(
                f"select chain_key from chain where chain_name = '{chain_name}'"
            )
            self.loadChainResidues(chain_key, pdb_chain)

            if not self.sdPass:
                # set std_all_angles
                self.dlq(
                    "select r.res_key, r.res_char,"
                    " (select count(*) from hedron h where h.res_key = r.res_key),"
                    " (select count(*) from dihedron d where d.res_key = r.res_key)"
                    f" from residue r where r.chain_key = {chain_key} and"
                    " length(r.res_char) = 1 order by r.res_key",
                )
                rkchdList = [(r[0], r[1], r[2], r[3]) for r in self.cur.fetchall()]
                hedra_counts, dihedra_counts = get_dh_counts(hydrogens=False)
                # hedra_counts -1 for extra omg H1
                rkclist = [
                    (r[0], r[1])
                    for r in rkchdList
                    if r[2] == hedra_counts[r[1]] - 1 and r[3] == dihedra_counts[r[1]]
                ]
                for rk, rc in rkclist:
                    self.dlq(
                        f"update residue set std_all_angles = True where res_key = {rk}"
                    )

    def doDSSP(self, model, filename):
        # global sdPass
        if self.sdPass:
            return
        try:
            # DSSP(model, filename, file_type="MMCIF")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                DSSP(
                    model,
                    filename,
                    file_type="MMCIF",
                )
        except PDBException as e:
            print(e, "filename= ", filename)

    def loadPDB(self, target, sourceListName=None):
        # global conn
        (pdb_structure, prot_id, chainID, filename) = getPDB(target)
        # print(prot_id, chainID, filename)
        if pdb_structure is not None:
            if isinstance(pdb_structure, Chain):
                self.doDSSP(pdb_structure.parent, filename)
                self.loadPdbChain(
                    pdb_structure, prot_id, chainID, filename, sourceListName
                )
            else:
                if isinstance(pdb_structure, Structure):
                    self.doDSSP(pdb_structure[0], filename)
                elif isinstance(pdb_structure, Model):
                    self.doDSSP(pdb_structure, filename)
                for chn in pdb_structure.get_chains():
                    self.loadPdbChain(
                        chn, pdb_structure.id, chn.id, filename, sourceListName
                    )

            self.commit()
        # return f"{prot_id}-{chainID}"

    def nbdMinMaxAverages(self):
        for acr1 in self.acrp:
            for acr2 in self.acrp[acr1]:
                acrpk = self.acrp[acr1][acr2]
                for dist in range(self.nbdReport):
                    self.dlq(
                        "select min(min) as min, max(max) as max, sum(sum) as sum, sum(count) as count from"
                        " non_bond_dists where chain_key is not NULL and "
                        f" acrp_key = {acrpk} and dist_num = {dist};",
                    )
                    rslt = self.cur.fetchone()
                    if rslt[2] is not None and rslt[3] is not None and rslt[3] != 0:
                        nbdsk = self.nbds[acrpk][dist]
                        avg = rslt[2] / rslt[3]
                        self.dlq(
                            "update non_bond_dists set (min, max, sum, count, avg) ="
                            f" ({rslt[0]}, {rslt[1]}, {rslt[2]}, {rslt[3]}, {avg}) where nbd_key = {nbdsk}",
                        )
                        self.nbdsAvg[nbdsk] = avg

        self.commit()

    def nbdStdDevs(self):
        for acr1 in self.acrp:
            for acr2 in self.acrp[acr1]:
                acrpk = self.acrp[acr1][acr2]
                for dist in range(self.nbdReport):
                    self.dlq(
                        "select sum(sum_sd_dev) as sum, sum(sd_count) as count from"
                        " non_bond_dists where chain_key is not NULL and "
                        f" acrp_key = {acrpk} and dist_num = {dist};",
                    )
                    rslt = self.cur.fetchone()
                    if rslt[0] is not None and rslt[1] is not None and rslt[1] != 0:
                        nbdsk = self.nbds[acrpk][dist]
                        sd = math.sqrt(rslt[0] / rslt[1])
                        self.dlq(
                            "update non_bond_dists set (sum_sd_dev, sd_count, std_dev) ="
                            f" ({rslt[0]}, {rslt[1]}, {sd}) where nbd_key = {nbdsk}",
                        )

        self.commit()

    def nbdAll(self):
        for dist in range(self.nbdReport):
            self.dlq(
                "select min(min) as min, max(max) as max, sum(sum) as sum, sum(count) as count, sum(sum_sd_dev) as ssd, sum(sd_count) as sdc from"
                " non_bond_dists where chain_key is NULL and "
                f" dist_num = {dist};",
            )
            rslt = self.cur.fetchone()
            if rslt[2] is not None and rslt[3] is not None and rslt[3] != 0:
                avg = rslt[2] / rslt[3]
            if rslt[4] is not None and rslt[5] is not None and rslt[5] != 0:
                nbdsk = self.nbds["all"][dist]
                sd = math.sqrt(rslt[4] / rslt[5])
                self.dlq(
                    "update non_bond_dists set (min, max, sum, count, avg, sum_sd_dev, sd_count, std_dev) ="
                    f" ({rslt[0]}, {rslt[1]}, {rslt[2]}, {rslt[3]}, {avg},"
                    f" {rslt[4]}, {rslt[5]}, {sd}) where nbd_key = {nbdsk}",
                )

        self.commit()


if __name__ == "__main__":
    parseArgs()
    dbl = dbLoad(args.sd)

    toProcess = args.file
    if args.filelist:
        flist = open(args.filelist, "r")
        for aline in flist:
            fields = aline.split()
            pdbidMatch = pdbidre.match(fields[0])
            if pdbidMatch:
                toProcess.append(pdbidMatch.group(0))

    if len(toProcess) == 0 and not dbl.sdPass:
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
        targList.append(target)
        fileNo += 1

    for i in range(1 if dbl.sdPass else 2):
        if dbl.sdPass:
            dbl.nbdMinMaxAverages()

        if PROCESSES > 0:
            with Pool(PROCESSES) as p:
                rslts = [p.apply_async(dbl.loadPDB, (i,)) for i in targList]
                for i in rslts:
                    i.get()
                    # print(i.get())
            print("Now the pool is closed and no longer available")
        else:
            for i in targList:
                dbl.loadPDB(i)

        if dbl.sdPass:
            dbl.nbdStdDevs()
            dbl.nbdAll()

        dbl.sdPass = True

    # conn.commit()
    # closeDb(conn)
