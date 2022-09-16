#!/usr/local/bin/python3


from multiprocessing import Pool
import sys
import argparse
import signal
import numpy as np
import math
import warnings
import gzip

from utils import parse_configuration
from models import get_model, checkpoint_load
from torchinfo import summary
from datasets import get_dataset

from energyfns import get_efn

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

from Bio.PDB.PICIO import read_PIC_seq
from Bio import SeqIO

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
        description="simulate folding of supplied protein "
    )

    arg_parser.add_argument("configFile", help="path to the configfile")

    arg_parser.add_argument(
        "-pf",
        dest="protFile",
        # nargs="*",
        help="a .cif path/protFile to read, or a PDB idCode with "
        "optional chain ID to read from {0} as .cif.gz".format(
            (
                PDB_repository_base
                or "[PDB resource not defined - please configure before use]"
            )
        ),
    )
    """
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
    """

    arg_parser.add_argument(
        "-p", dest="PROCESSES", help=f"number of subprocesses, default {PROCESSES}"
    )

    arg_parser.add_argument(
        "-nncf", dest="nnConfigFile", help="override NN config file for model"
    )
    arg_parser.add_argument(
        "-cu", dest="cuda", help="override cuda device in NN config file"
    )
    arg_parser.add_argument("-cpu", help="load NN on cpu not cuda", action="store_true")
    # arg_parser.add_argument("-fcf", dest="foldConfigFile", help="folding process config file")

    args = arg_parser.parse_args()

    """
    if args.skip_count:
        args.skip_count = int(args.skip_count)
    if args.limit_count:
        args.limit_count = int(args.limit_count)
    """

    if args.PROCESSES:
        PROCESSES = int(args.PROCESSES)
    # if args.sd:
    #    sdPass = True


def getNN(configDict):
    global args
    print(f"Reading config file {configDict['nn']['netconfig']}...")
    config = parse_configuration(configDict["nn"]["netconfig"])

    if "cuda" in configDict["nn"]:
        config["process"]["device"] = f"cuda:{configDict['nn']['cuda']}"
    if args.cuda:
        config["process"]["device"] = f"cuda:{args.cuda}"
    if "cpu" in configDict["nn"] and configDict["nn"]["cpu"]:
        config["process"]["device"] = "cpu"
    if args.cpu:
        config["process"]["device"] = "cpu"

    print(f"device: {config['process']['device']}")
    full_dataset = get_dataset(config["dataset"])  # need for in, out dims
    if hasattr(full_dataset, "inputDim"):
        config["model"]["input_dim"] = full_dataset.inputDim
        config["model"]["output_dim"] = full_dataset.outputDim

    print(f"input dimension {config['model']['input_dim']}")
    print(f"output dimension {config['model']['output_dim']}")

    print("Initializing model...")
    model = get_model(config["model"]).to(config["process"]["device"])

    summary(model)
    epochs = config["process"]["epochs"]
    load_epoch = 0
    if config["checkpoint"]["load"] != -1:  # can use "last" instead of number
        if checkpoint_load(model, config):
            load_epoch = config["checkpoint"]["load"]
            print(f"re-loaded model from epoch {load_epoch}...")
            epochs += load_epoch
    return model


if __name__ == "__main__":
    parseArgs()
    if not args.configFile:
        print("no files to process. use '-h' for help")
        sys.exit(0)

    config = parse_configuration(args.configFile)

    if args.protFile:
        config["target"]["protein"] = args.protFile

    (pdb_structure, prot_id, chainID, protFile) = getPDB(config["target"]["protein"])
    if not pdb_structure:
        print(
            f"unable to load {config['target']['protein']} as cif/pdb structure;use -h for help."
        )
        sys.exit(0)

    for record in SeqIO.parse(
        gzip.open(protFile, mode="rt") if protFile.endswith(".gz") else protFile,
        "cif-atom",  # assume cif file, otherwise need pdb-atom
    ):
        print(f">{record.id}")
        out = [(record.seq[i : i + 80]) for i in range(0, len(record.seq), 80)]
        for lin in out:
            print(lin)

        pdb_structure2 = read_PIC_seq(record)
        pdb_structure2.internal_to_atom_coordinates()
        break  # only take first chain

    # toProcess = args.file
    # if config["nn"]["netconfig"]:
    #    nn = getNN(config)

    environE = get_efn(config["energyfn"]["environ"])

    targGlobalE, targSeqE = environE.evaluate(pdb_structure)
    predGlobalE, predSeqE = environE.evaluate(pdb_structure2)

    print("hello")
