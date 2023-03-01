#!/usr/local/bin/python3

import os
import re
import gzip

from Bio.PDB import ic_data
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.internal_coords import Edron

from .rdb import openDb, pqry1, pqry

# import numpy as np
import torch


class rpdbT:
    PDB_repository_base = None

    pdbDirs = [
        "/Users/rob/data/pdb/",
        "/home/rob/data/pdb/",
        "/media/data/structures/divided/",
        "/Volumes/data/structures/divided/",
    ]
    for d in pdbDirs:
        if os.path.isdir(d):
            PDB_repository_base = d
            break

    resList = [
        "G",
        "A",
        "V",
        "L",
        "I",
        "M",
        "F",
        "P",
        "S",
        "T",
        "C",
        "N",
        "Q",
        "Y",
        "W",
        "D",
        "E",
        "H",
        "K",
        "R",
    ]

    resMap = {
        "G": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "A": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "V": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "L": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "I": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "M": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "F": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "P": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "S": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "T": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "C": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "N": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "Q": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "W": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "D": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        "E": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "H": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "R": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    }

    crMapH = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no atom
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Csb
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Cres
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Cdb
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Osb
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Ores
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Odb
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Nsb
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Nres
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Ndb
        10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Ssb
        11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Hsb
    }

    crMapNoH = {
        0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # no atom
        1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Csb
        2: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Cres
        3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Cdb
        4: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # Osb
        5: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Ores
        6: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Odb
        7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # Nsb
        8: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # Nres
        9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # Ndb
        10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Ssb
    }

    NoAtomCR = 0  # covalent radii class for no atom

    MaxHedron = 16  # Trp + 1 extra for 2x omg H1 * rtm wrong for H=True!
    MaxDihedron = 14  # Trp * rtm wrong for H=True!

    hydrogens = False

    hedra_countsG = None
    dihedra_countsG = None

    pdbidre = re.compile(r"(^\d(\w\w)\w)(\w)?$")

    PDB_parser = PDBParser(PERMISSIVE=True, QUIET=True)
    CIF_parser = MMCIFParser(QUIET=True)

    def __init__(self, device, hydrogens=False):
        self.device = device
        self.hydrogens = hydrogens

    def cif_load(self, filename, prot_id):
        """Read .cif file optionally gzipped."""
        return self.CIF_parser.get_structure(
            prot_id,
            gzip.open(filename, mode="rt") if filename.endswith(".gz") else filename,
        )

    def pdb_load(self, filename, prot_id):
        """Read .pdb file optionally gzipped."""
        return self.PDB_parser.get_structure(
            prot_id,
            gzip.open(filename, mode="rt") if filename.endswith(".gz") else filename,
        )

    def getPDB(self, target):
        prot_id = ""
        # ciffilename = None
        # pdbfilename = None
        pdb_structure = None
        pdbidMatch = self.pdbidre.match(target)

        if pdbidMatch is not None:
            assert self.PDB_repository_base, "PDB repository base directory missing, "
            "please configure for this host"

            pdbfilename = (
                self.PDB_repository_base
                + "pdb/"
                + pdbidMatch.group(2).lower()
                + "/pdb"
                + pdbidMatch.group(1).lower()
                + ".ent.gz"
            )
            ciffilename = (
                self.PDB_repository_base
                + "mmCIF/"
                + pdbidMatch.group(2).lower()
                + "/"
                + pdbidMatch.group(1).lower()
                + ".cif.gz"
            )
            prot_id = pdbidMatch.group(1)
        else:
            pdbfilename = target
            ciffilename = target
            pdbidre = re.compile(r"(\d\w\w\w)(\w)?")  # find anywhere in string
            pdbidMatch2 = pdbidre.search(target)
            if pdbidMatch2:
                prot_id = pdbidMatch2.group(0)
            else:
                prot_id = target

        try:  # cif preferred
            pdb_structure = self.cif_load(ciffilename, prot_id)
            filename = ciffilename

        except Exception:
            pass  # continue  # try as pdb below

        if pdb_structure is None or pdb_structure.child_dict == {}:
            try:
                pdb_structure = self.pdb_load(pdbfilename, prot_id)
                filename = pdbfilename
            except Exception:
                print(f"unable to open {target} as PDB, MMCIF or PIC file format.")
                return (None, None, None, None)

        chainID = ""
        # get specified chain if given

        if pdbidMatch is not None and pdbidMatch.group(3) is not None:
            # have chain specifier for PDBid
            # if pdb_structure[0][pdbidMatch.group(3)] is not None:
            chainID = pdbidMatch.group(3)
            if chainID in pdb_structure[0]:
                pdb_chain = pdb_structure[0][chainID]
                pdb_structure = pdb_chain
            elif chainID.upper() in pdb_structure[0]:
                pdb_chain = pdb_structure[0][chainID.upper()]
                pdb_structure = pdb_chain
            else:
                print("chain " + chainID + " not found in " + filename)
                return (None, None, None, None)

        return (pdb_structure, prot_id, chainID, filename)

    def get_dh_counts(self):
        if self.hedra_countsG is None or self.dihedra_countsG is None:

            # ic_data tables do not include next or prev residues so that
            # includes psi, phi, omg.  That's 3 dihedra and works out to 3 hedra,
            # plus we use omega H1 from both residue i (so prev) and i+1 (self)
            hedra_counts = {rc: 4 for rc in self.resList}  # not in ic_data, plus 1
            dihedra_counts = {rc: 3 for rc in self.resList}  # psi, phi, omg

            def _tstAngle(ang):
                leng = 0
                for a in ang:
                    if not self.hydrogens and a[0] == "H":  # skip H's as directed
                        return 0
                    if a == "OXT":  # ignore OXT as it is synonym for O
                        return 0
                    if leng == 3:  # must be dihed; ignore chi1 etc labels
                        return 4
                    leng += 1
                return leng

            # count accepted hedra, dihedra in backbone
            for ang in ic_data.ic_data_backbone:
                c = _tstAngle(ang)
                if c == 3:
                    for rc in self.resList:
                        hedra_counts[rc] += 1
                elif c == 4:
                    for rc in self.resList:
                        dihedra_counts[rc] += 1

            # count accepted hedra, dihedra in sidechain
            for rc, angList in ic_data.ic_data_sidechains.items():
                for ang in angList:
                    c = _tstAngle(ang)
                    if c == 3:
                        hedra_counts[rc] += 1
                    elif c == 4:
                        dihedra_counts[rc] += 1

            # fix dict Gly because ic_data_backbone is for Ala
            if self.hydrogens:
                hedra_counts["G"] -= 6
                dihedra_counts["G"] -= 5
            else:
                hedra_counts["G"] -= 2
                dihedra_counts["G"] -= 1

            self.hedra_countsG, self.dihedra_countsG = hedra_counts, dihedra_counts
        return self.hedra_countsG, self.dihedra_countsG

    normDict = None

    def getNormValues(self):
        if self.normDict is None:
            conn = openDb()
            cur = conn.cursor()
            rslt = None
            pqry(cur, "select name, min, range from len_normalization")
            rslt = cur.fetchall()
            self.normDict = {key: (v1, v2, 0) for (key, v1, v2) in rslt}
            conn.close()
            for x in ["len12", "len23", "len13", "len14"]:
                self.normDict[x][2] = self.normDict[x][0] + (self.normDict[x][1] / 2)

            # hMinLenArr, hRangeArr
            # avg defined as min + 1/2 range
            # so normalized avg is 0
            self.normDict["hMinLenArr"] = torch.tensor(
                [self.normDict[x][0] for x in ["len12", "len23", "len13"]],
                dtype=torch.float,
                device=torch.device(self.device),
                requires_grad=True,
            )
            self.normDict["hRangeArr"] = torch.tensor(
                [self.normDict[x][1] for x in ["len12", "len23", "len13"]],
                dtype=torch.float,
                device=torch.device(self.device),
                requires_grad=True,
            )
            # max len14, don't need for hedra; min + range
            self.normDict["maxLen14"] = self.normDict["len14"][0] + self.normDict["len14"][1]

        return self.normDict

    def dhlDenorm(self, dhLenArrNorm):
        # denormalize from dbmng.py:gnoLengths() code:
        normDict = self.getNormValues()

        # "normalise dihedra len to -1/+1"
        # dhLenArr = len14min + (((dhLenArrNorm + 1) / 2) * len14range)
        dhLenArr = normDict["len14"][0] + (((dhLenArrNorm + 1) / 2) * normDict["len14"][1])
        return dhLenArr

    def harrDenorm(self, hArrNorm):
        normDict = self.getNormValues()
        # "normalise hedra len to -1/+1"
        hArr = normDict["hMinLenArr"] + (((hArrNorm + 1) / 2) * normDict["hRangeArr"])
        return hArr

    def dhlNorm(self, dhLenArr):
        normDict = self.getNormValues()
        # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
        dhLenArrNorm = (2 * ((dhLenArr - normDict["len14"][0]) / normDict["len14"][1])) - 1
        return dhLenArrNorm

    def harrNorm(self, hArr):
        normDict = self.getNormValues()
        hArrNorm = (2 * ((hArr - normDict["hMinLenArr"]) / normDict["hRangeArr"])) - 1
        return hArrNorm

    def outputArr2values(self, rc, outputArr):
        """Split output array into dhChirality, dhLen, hLen arrays"""
        hedra_counts, dihedra_counts = self.get_dh_counts()
        if rc == "X":
            rc = "W"
        rhc, rdhc = hedra_counts[rc], dihedra_counts[rc]

        maxOutputLen = (3 * self.MaxHedron) + (2 * self.MaxDihedron)

        # force dhChrArr to +/-1
        dhChrArr = outputArr[0:rdhc].clone().detach()
        dhChrArr[dhChrArr < 0] = -1.0
        dhChrArr[dhChrArr >= 0] = 1.0
        
        if len(outputArr) != maxOutputLen:
            # per residue net
            ndx = 2 * rdhc
            dhLenArrNorm = outputArr[rdhc:ndx]
            hArr = outputArr[ndx:].reshape(1, -1)
        else:
            # all residues net
            # split by MaxDihedron, then select rdhc, rhc regions
            dhLenArrNorm = outputArr[self.MaxDihedron : self.MaxDihedron + rdhc]
            ndx = 2 * self.MaxDihedron
            hArrNorm = outputArr[ndx : ndx + (3 * rhc)]
            hArrNorm = hArrNorm.reshape(-1, 3)

        dhLenArr = self.dhlDenorm(dhLenArrNorm)
        hArr = self.harrDenorm(hArrNorm)

        return dhChrArr, dhLenArr, hArr, dhLenArrNorm, hArrNorm

    phiClassKey = None
    omgClassKey = None
    acbClassKey = None

    psiH1key = None
    phiH1key = None
    omgH1key = None

    def get_phi_omg_classes(self, cur):
        """Get di/hedra class keys for filtering including alt cbeta"""

        if None in [
            self.phiClassKey,
            self.omgClassKey,
            self.acbClassKey,
            self.phiH1key,
            self.omgH1key,
        ]:
            self.phiClassKey = pqry1(
                cur, "select dc_key from dihedron_class where d_class = 'CNCAC'"
            )
            self.omgClassKey = pqry1(
                cur, "select dc_key from dihedron_class where d_class = 'CACNCA'"
            )
            self.acbClassKey = pqry1(
                cur, "select dc_key from dihedron_class where d_class = 'CNCACB'"
            )

            self.psiH1key = pqry1(
                cur, "select hc_key from hedron_class where h_class = 'NCAC'"
            )
            self.phiH1key = pqry1(
                cur, "select hc_key from hedron_class where h_class = 'CNCA'"
            )
            self.omgH1key = pqry1(
                cur, "select hc_key from hedron_class where h_class = 'CACN'"
            )

        return (
            self.phiClassKey,
            self.omgClassKey,
            self.acbClassKey,
            self.psiH1key,
            self.phiH1key,
            self.omgH1key,
        )

    dcDict = None

    def get_dcDict(self, cur):
        if self.dcDict is None:
            pqry(cur, "select d_class, dc_key from dihedron_class")
            rsltlst = cur.fetchall()
            self.dcDict = {a[0]: a[1] for a in rsltlst}
        return self.dcDict

    hedraMap = None

    def getHedraMap(self):
        if self.hedraMap is None:

            self.hedraMap = {}

            conn = openDb()
            cur = conn.cursor()

            (
                phiClassKey,
                omgClassKey,
                acbClassKey,
                psiH1key,
                phiH1key,
                omgH1key,
            ) = self.get_phi_omg_classes(cur)

            for rc in self.resList:
                pqry(
                    cur,
                    "select r1.res_key, r2.res_key from residue r1, residue r2"
                    " where r2.rprev = r1.res_key and r1.std_all_angles"
                    f" and r2.std_all_angles and r2.res_char = '{rc}' limit 1",
                )
                rpk, rk = cur.fetchall()[0]

                pqry(
                    cur,
                    "select d.hk1, d.hk2, d.reverse from dihedron d, dihedron_class dc"
                    f" where d.res_key = {rpk}"
                    f" and (d.class_key = {phiClassKey} or d.class_key = {omgClassKey})"
                    " and d.class_key = dc.dc_key"
                    " order by dc.d_class",
                )
                rsltList = cur.fetchall()

                q = (
                    f"select ak_str, hk1, hk2, reverse from dihedron where res_key = {rk}"
                    f" and class_key != {phiClassKey} and class_key != {omgClassKey}"
                )
                if acbClassKey is not None:
                    q += f" and class_key != {acbClassKey}"
                    pqry(cur, q)

                dhdict = {}
                for dh in cur.fetchall():
                    dhdict[Edron.gen_tuple(dh[0])] = dh[1:]

                for k in sorted(dhdict):
                    rsltList += [dhdict[k]]

                hk1LocalArr = torch.tensor(
                    [r[0] for r in rsltList],
                    dtype=torch.float,
                    device=torch.device(self.device),
                    requires_grad=True,
                )
                hk2LocalArr = torch.tensor(
                    [r[1] for r in rsltList],
                    dtype=torch.float,
                    device=torch.device(self.device),
                    requires_grad=True,
                )
                revArr = torch.tensor(
                    [r[2] for r in rsltList],
                    dtype=torch.float,
                    device=torch.device(self.device),
                    requires_grad=True,
                )

                pqry(
                    cur,
                    "select h.hedron_key from hedron h, hedron_class hc"
                    f" where h.res_key = {rpk}"
                    f" and (h.class_key = {phiH1key} or h.class_key = {omgH1key})"
                    " and h.class_key = hc.hc_key"
                    " order by hc.h_class",
                )
                hkl = cur.fetchall()
                pqry(
                    cur,
                    f"select ak_str, hedron_key from hedron where res_key = {rk}"
                    f"and class_key != {phiH1key}",
                )
                # hkl += cur.fetchall()
                hdict = {}
                for h in cur.fetchall():
                    hdict[Edron.gen_tuple(h[0])] = h[1:]

                for k in sorted(hdict):
                    hkl += [hdict[k]]

                hkLocalArr = torch.tensor(
                    [hk[0] for hk in hkl],
                    dtype=torch.float,
                    device=torch.device(self.device),
                    requires_grad=True,
                )

                # https://stackoverflow.com/a/8251757/2783487
                xsorted = torch.argsort(hkLocalArr)
                ypos = torch.searchsorted(hkLocalArr[xsorted], hk1LocalArr)
                hk1Arr = xsorted[ypos]
                ypos = torch.searchsorted(hkLocalArr[xsorted], hk2LocalArr)
                hk2Arr = xsorted[ypos]

                self.hedraMap[rc] = [hk1Arr, hk2Arr, revArr]

        return self.hedraMap

    def multi_rot_Z(self, angle_rads):
        """Create [entries] numpy Z rotation matrices for [entries] angles.

        :param entries: int number of matrices generated.
        :param angle_rads: numpy array of angles
        :returns: entries x 4 x 4 homogeneous rotation matrices
        """
        rz = torch.empty(
            (angle_rads.shape[0], 4, 4),
            dtype=torch.float,
            device=torch.device(self.device),
        )
        rz[...] = torch.eye(4)
        rz[:, 0, 0] = rz[:, 1, 1] = torch.cos(angle_rads)
        rz[:, 1, 0] = torch.sin(angle_rads)
        rz[:, 0, 1] = -rz[:, 1, 0]
        return rz

    def len2ang(self, rc, dhChrArr, dhLenArr, hArr):
        """Compute di/hedra from from distance and chirality data.

        * oa = hedron1 L12 if reverse else hedron1 L23
        * ob = hedron1 L23 if reverse else hedron1 L12
        * ac = hedron2 L12 if reverse else hedron2 L23
        * ab = hedron1 L13 = law of cosines on OA, OB (hedron1 L12, L23)
        * oc = hedron2 L13 = law of cosines on OA, AC (hedron2 L12, L23)
        * bc = dihedron L14

        target is OA, the dihedral angle along edge oa
        """
        hedraL12 = hArr[:, 0]
        hedraL23 = hArr[:, 1]
        hedraL13 = hArr[:, 2]

        hedraMap = self.getHedraMap()
        if rc == "X":
            rc = "W"
        dH1ndx = hedraMap[rc][0]
        dH2ndx = hedraMap[rc][1]
        dFwd = torch.logical_not(hedraMap[rc][2])

        oa = hedraL12[dH1ndx]
        oa[dFwd] = hedraL23[dH1ndx][dFwd]

        ob = hedraL23[dH1ndx]
        ob[dFwd] = hedraL12[dH1ndx][dFwd]
        ac = hedraL12[dH2ndx]
        ac[dFwd] = hedraL23[dH2ndx][dFwd]
        ab = hedraL13[dH1ndx]
        oc = hedraL13[dH2ndx]
        bc = dhLenArr

        # Ws = (ab + ac + bc) / 2
        # Xs = (ob + bc + oc) / 2
        Ys = (oa + ac + oc) / 2
        Zs = (oa + ob + ab) / 2
        # Wsqr = Ws * (Ws - ab) * (Ws - ac) * (Ws - bc)
        # Xsqr = Xs * (Xs - ob) * (Xs - bc) * (Xs - oc)

        # rtm force positive else get nan below
        Ysqr = torch.abs(Ys * (Ys - oa) * (Ys - ac) * (Ys - oc))
        Zsqr = torch.abs(Zs * (Zs - oa) * (Zs - ob) * (Zs - ab))
        Hsqr = (
            4 * oa * oa * bc * bc
            - torch.square((ob * ob + ac * ac) - (oc * oc + ab * ab))
        ) / 16
        """
        Jsqr = (
            4 * ob * ob * ac * ac
            - torch.square((oc * oc + ab * ab) - (oa * oa + bc * bc))
        ) / 16
        Ksqr = (
            4 * oc * oc * ab * ab
            - torch.square((oa * oa + bc * bc) - (ob * ob + ac * ac))
        ) / 16
        """

        Y = torch.sqrt(Ysqr)
        Z = torch.sqrt(Zsqr)
        # X = torch.sqrt(Xsqr)
        # W = torch.sqrt(Wsqr)

        cosOA = (Ysqr + Zsqr - Hsqr) / (2 * Y * Z)
        # cosOB = (Zsqr + Xsqr - Jsqr) / (2 * Z * X)
        # cosOC = (Xsqr + Ysqr - Ksqr) / (2 * X * Y)
        # cosBC = (Wsqr + Xsqr - Hsqr) / (2 * W * X)
        # cosCA = (Wsqr + Ysqr - Jsqr) / (2 * W * Y)
        # cosAB = (Wsqr + Zsqr - Ksqr) / (2 * W * Z)

        # OA =
        # compute dihedral angles
        # ensure cosOA is in range [-1,1] for arccos
        cosOA[cosOA < -1.0] = -1.0
        cosOA[cosOA > 1.0] = 1.0
        # without torch.longdouble here a few OCCACB angles lose last digit match
        dihedraAngleRads = torch.arccos(cosOA)
        dihedraAngleRads *= dhChrArr
        dihedraAngle = torch.rad2deg(dihedraAngleRads)
        # OB = torch.rad2deg(torch.arccos(cosOB))
        # OC = torch.rad2deg(torch.arccos(cosOC))
        # BC = torch.rad2deg(torch.arccos(cosBC))
        # CA = torch.rad2deg(torch.arccos(cosCA))
        # AB = torch.rad2deg(torch.arccos(cosAB))

        # law of cosines for hedra angles
        cosHar = (
            torch.square(hedraL12) + torch.square(hedraL23) - torch.square(hedraL13)
        ) / (2 * hedraL12 * hedraL23)
        cosHar[cosHar < -1.0] = -1.0
        cosHar[cosHar > 1.0] = 1.0
        hedraAngleRads = (torch.arccos(cosHar),)[0]

        hedraAngle = torch.rad2deg(hedraAngleRads)
        # fail = torch.isnan(torch.sum(hedraAngle)) or torch.isnan(
        #    torch.sum(dihedraAngle)
        # )
        return hedraAngle, hedraAngleRads, dihedraAngle, dihedraAngleRads

    def lenAng2coords(self, rc, hedraAngleRads, hArr, dihedraAngleRads):
        hedraMap = self.getHedraMap()
        if rc == "X":
            rc = "W"
        dRev = (hedraMap[rc][2]).long()
        dFwd = torch.logical_not(dRev)
        dH1ndx = hedraMap[rc][0]
        dH2ndx = hedraMap[rc][1]
        hedraL12 = hArr[:, 0].float()
        hedraL23 = hArr[:, 1].float()
        # hedraL13 = hArr[:, 2]

        hedraCount = len(hArr)
        hAtoms: torch.ndarray = torch.zeros(
            (hedraCount, 3, 4), dtype=torch.float, device=torch.device(self.device)
        )
        hAtoms[:, :, 3] = 1.0  # homogeneous
        hAtomsR: torch.ndarray = hAtoms.detach().clone()

        # hedra inital coords

        # sar = supplementary angle radian: angles which add to 180
        sar = torch.subtract(torch.pi, hedraAngleRads)
        sinSar = torch.sin(sar)
        cosSarN = torch.cos(sar) * -1
        """
        if dbg:
            print("sar", sar[0:10])
        """
        # a2 is len3 up from a2 on Z axis, X=Y=0
        hAtoms[:, 2, 2] = hedraL23

        # a0 X is sin( sar ) * len12
        hAtoms[:, 0, 0] = sinSar * hedraL12

        # a0 Z is -(cos( sar ) * len12)
        # (assume angle always obtuse, so a0 is in -Z)
        hAtoms[:, 0, 2] = cosSarN * hedraL12
        """
        if dbg:
            print("hAtoms_needs_update", hAtoms_needs_update[0:10])
            print("hAtoms", hAtoms[0:10])
        """
        # same again but 'reversed' : a0 on Z axis, a1 at origin, a2 in -Z

        # a0r is len12 up from a1 on Z axis, X=Y=0
        hAtomsR[:, 0, 2] = hedraL12
        # a2r X is sin( sar ) * len23
        hAtomsR[:, 2, 0] = sinSar * hedraL23
        # a2r Z is -(cos( sar ) * len23)
        hAtomsR[:, 2, 2] = cosSarN * hedraL23
        """
        if dbg:
            print("hAtomsR", hAtomsR[0:10])
        """

        # dihedra parts other than dihedral angle

        dihedraCount = len(dihedraAngleRads)
        a4_pre_rotation = torch.empty(
            (dihedraCount, 4), dtype=torch.float, device=torch.device(self.device)
        )

        # only 4th atom takes work:
        # pick 4th atom based on rev flag
        a4_pre_rotation[dRev] = hAtoms[dH2ndx, 0][dRev]
        a4_pre_rotation[dFwd] = hAtomsR[dH2ndx, 2][dFwd]

        """
        if dbg:
            npa4pr = a4_pre_rotation.cpu().detach().numpy()
            nph0 = (hAtoms[dH2ndx, 0][dRev]).cpu().detach().numpy()
            nph1 = (hAtomsR[dH2ndx, 2][dFwd]).cpu().detach().numpy()

            # numpy multiply, add operations below intermediate array but out=
            # not working with masking:
            a4_pre_rotation[:, 2] = torch.multiply(a4_pre_rotation[:, 2], -1)  # a4 to +Z

            npa4pr2 = a4_pre_rotation.cpu().detach().numpy()

            a4shift = torch.empty(
                dihedraCount, device=torch.device(self.device), dtype=torch.float
            )
            a4shift[dRev] = hedraL23[dH2ndx][dRev]  # len23
            a4shift[dFwd] = hedraL12[dH2ndx][dFwd]  # len12

            npa4s = a4shift.cpu().detach().numpy()

            a4_pre_rotation[:, 2] = torch.add(
                a4_pre_rotation[:, 2],
                a4shift,
            )  # so a2 at origin

            npa4pr3 = a4_pre_rotation.cpu().detach().numpy()

            print("dihedraCount", dihedraCount)
            print("a4shift", a4shift[0:10])
            print("a4_pre_rotation", a4_pre_rotation[0:10])
        """
        # now build dihedra initial coords

        dH1atoms = hAtoms[dH1ndx]  # fancy indexing so
        dH1atomsR = hAtomsR[dH1ndx]  # these copy not view

        dAtoms = torch.empty(
            (dihedraCount, 4, 4), dtype=torch.float, device=torch.device(self.device)
        )

        dAtoms[:, :3][dFwd] = dH1atoms[dFwd]
        # dAtoms[:, :3][dRev] = dH1atomsR[:, 2::-1][dRev]
        dH1atomsRF = torch.flip(dH1atomsR, [1])
        dAtoms[:, :3][dRev] = dH1atomsRF[dRev]

        """
        if dbg:
            print("dH1atoms", dH1atoms[0:10])
            print("dH1atosR", dH1atomsR[0:10])
            print("dAtoms", dAtoms[0:10])
        """

        # build rz rotation matrix for dihedral angle
        """
        if dbg:
            print("dangle-rads", dihedraAngleRads[0:10])
        """
        rz = self.multi_rot_Z(dihedraAngleRads)

        a4rot = torch.matmul(
            rz,
            a4_pre_rotation[:].reshape(-1, 4, 1),
        ).reshape(-1, 4)

        dAtoms[:, 3][dFwd] = a4rot[dFwd]
        dAtoms[:, 3][dRev] = a4rot[dRev]

        # dA = dAtoms.cpu().detach().numpy()
        # paErr = np.any((dA < -5) | (dA > 5))

        return dAtoms
