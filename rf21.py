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

from energyfns import get_efn
from movefns import get_mfn

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
from Bio.PDB.ic_rebuild import structure_rebuild_test, compare_residues
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBExceptions import PDBException

from Bio.PDB.PICIO import read_PIC_seq
from Bio import SeqIO

# dash stuff
import threading
import socket

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import time
import plotly.express as px
import logging
from flask import Flask

# import pandas as pd
import plotly.graph_objs as go

server = Flask(__name__)
# Set the logging level to exclude messages with severity level "POST"
logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = dash.Dash(__name__, server=server)

# pd.options.plotting.backend = "plotly"
# end dash setup

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
    global dash_thread
    print("KeyboardInterrupt is caught")
    dash_thread.stop()
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
        "-cu",
        dest="cudalist",
        help="list of int: 0,1,2 - override cuda device in NN config file",
        type=lambda s: [int(item) for item in s.split(",")],
    )
    arg_parser.add_argument("-cpu", help="load NN on cpu not cuda", action="store_true")
    # arg_parser.add_argument("-fcf", dest="foldConfigFile", help="folding process config file")

    arg_parser.add_argument("-fake", help="use db instead of NN", action="store_true")
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


# dash code

distplot = None
seqEplot = None
globEplot = None
target = None
iter = 0
maxit = None
config = None
dash_thread = None


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def getDashPort():
    tport = 8050
    while is_port_in_use(tport):
        tport += 1
    return tport


def startDash():
    global app, config
    app.layout = html.Div(
        [
            html.H1("CA Distplot Monitor"),
            # html.Div(id="output"),
            html.H2(id="mystring"),
            dcc.Graph(id="distplot"),
            dcc.Graph(id="seqEplot"),
            dcc.Graph(id="globEplot"),
            html.H3("Configuration:"),
            dcc.Markdown(id="configData", dangerously_allow_html=True),
            dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
        ]
    )
    dport = getDashPort()
    print(f"Dash interface starting on port {dport}")
    app.run(host="0.0.0.0", port=dport, debug=False)


@app.callback(
    [
        Output("distplot", "figure"),
        Output("seqEplot", "figure"),
        Output("globEplot", "figure"),
        # . Output("mystring", "children"),
    ],
    [Input("interval-component", "n_intervals")],
)
def update_output(n):
    global distplot, seqEplot, globEplot
    fig = px.imshow(distplot) if distplot is not None else None
    fig2 = seqEplot  # px.imshow(seqEplot)
    fig3 = globEplot
    return fig, fig2, fig3  # , f"target: {target}"


@app.callback(
    Output("mystring", "children"),
    [Input("interval-component", "n_intervals")],
)
def update_target(n):
    global target, iter, maxit
    targStr = f"target: {target}  iteration {iter}/{maxit}"
    return targStr


@app.callback(
    Output("configData", "children"),
    [Input("interval-component", "n_intervals")],
)
def update_config(n):
    global config

    configStr = "<table>\n<tbody>\n"
    for tkey in config.keys():
        if isinstance(config[tkey], dict):
            configStr += f"<tr><td>{tkey}</td></tr>\n<tr><td>\n<table>\n<tbody>\n"
            for k, v in config[tkey].items():
                configStr += f"<tr><td> </td><td>{k}</td><td>{v}</td></tr>\n"
            configStr += "</tbody></table></td></tr>\n"
        else:
            configStr += (
                f"<tr><td>{tkey}</td><tr><td> </td><td>{config[tkey]}</td></tr></tr>\n"
            )
    configStr += "</tbody></table>\n"
    return configStr


# end dash code


def make_extended(cic):
    for d in cic.dihedra.values():
        if d.e_class == "NCACN":  # psi
            d.angle = 123
        elif d.e_class == "CNCAC":  # phi
            d.angle = -104


if __name__ == "__main__":
    # read instructions
    parseArgs()
    if not args.configFile:
        print("no files to process. use '-h' for help")
        sys.exit(0)

    config = parse_configuration(args.configFile)

    if args.protFile:
        config["target"]["protein"] = args.protFile

    # load target structure
    (pdb_structure, prot_id, chainID, protFile) = getPDB(config["target"]["protein"])
    if not pdb_structure:
        print(
            f"unable to load {config['target']['protein']} as cif/pdb structure;use -h for help."
        )
        sys.exit(0)
    else:
        print(f"loaded {prot_id} chain {chainID} from {protFile}")
    target = prot_id
    maxit = config["iterations"]
    dash_thread = threading.Thread(target=startDash)
    dash_thread.start()

    pdb_structure.atom_to_internal_coordinates()
    """
    # move struct to origin for coordinate comparisons
    cic = pdb_structure.internal_coord
    incacs = cic.initNCaCs
    cic.atomArrayValid[cic.atomArrayIndex[incacs[0][0]]] = False
    """

    pdb_structure.internal_to_atom_coordinates()
    # need if resetting struct to origin above:
    # pdb_structure.atom_to_internal_coordinates()
    # for debug below
    # dp1 = pdb_structure.internal_coord.distance_plot()

    if isinstance(pdb_structure, Chain):
        chn = pdb_structure
    else:
        for chn in pdb_structure.get_chains():
            break
    cic = chn.internal_coord

    atmNameNdx = AtomKey.fields.atm
    atomArrayIndex = cic.atomArrayIndex
    CaSelect = [
        atomArrayIndex.get(k)
        for k in atomArrayIndex.keys()
        if k.akl[atmNameNdx] == "CA"
    ]

    distplot = dp1 = cic.distance_plot(CaSelect)
    pdb_structure.internal_to_atom_coordinates()
    # all done target structure

    # load or generate predicted structure
    if config["target"]["copy_struct"]:
        (pdb_structure2, prot_id2, chainID2, protFile2) = getPDB(
            config["target"]["protein"]
        )
        pdb_structure2.atom_to_internal_coordinates()
        pdb_structure2.internal_to_atom_coordinates()

        if isinstance(pdb_structure2, Chain):
            chn = pdb_structure2
        else:
            for chn in pdb_structure2.get_chains():
                break
        cic = chn.internal_coord

    else:
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

            if isinstance(pdb_structure2, Chain):
                chn = pdb_structure2
            else:
                for chn in pdb_structure2.get_chains():
                    break
            cic = chn.internal_coord
            make_extended(cic)

            pdb_structure2.atom_to_internal_coordinates()
            break  # only take first chain

    """
    # debug
    (pdb_structure2, prot_id2, chainID2, protFile2) = getPDB(
        config["target"]["protein"]
    )
    pdb_structure2.atom_to_internal_coordinates()
    pdb_structure2.internal_to_atom_coordinates()
    # end debug
    """

    atmNameNdx = AtomKey.fields.atm
    atomArrayIndex = cic.atomArrayIndex
    CaSelect = [
        atomArrayIndex.get(k)
        for k in atomArrayIndex.keys()
        if k.akl[atmNameNdx] == "CA"
    ]

    dp2 = cic.distance_plot(CaSelect)

    config["movefn"]["args"] = args  # so can pass cpu/cuda override, fake
    moveFn = get_mfn(config["movefn"])

    environE = get_efn(config["energyfn"])

    targGlobalE, targSeqE = environE.evaluate(pdb_structure)
    predGlobalE, predSeqE = environE.evaluate(pdb_structure2)

    seqEplotXvals = np.arange(len(targSeqE))
    seqEplot = go.Figure()
    seqEplot.add_trace(
        go.Scatter(x=seqEplotXvals, y=targSeqE, mode="lines", name="targ")
    )
    seqEplot.add_trace(
        go.Scatter(x=seqEplotXvals, y=predSeqE, mode="lines", name="pred")
    )

    globEplotXvals = np.arange(maxit)
    globEplotTarg = np.full((maxit), targGlobalE)
    globEplotPred = np.full((maxit), targGlobalE)
    globEplotPred[0] = predGlobalE
    globEplot = go.Figure()
    globEplot.add_trace(
        go.Scatter(x=globEplotXvals, y=globEplotTarg, mode="lines", name="targ")
    )
    globEplot.add_trace(
        go.Scatter(x=globEplotXvals, y=globEplotPred, mode="lines", name="pred")
    )

    # print(f"start: {targGlobalE} pred: {predGlobalE}")
    for iter in range(maxit):
        (globAvg, resArr, pdb_structure2) = moveFn.move(pdb_structure2)
        print(f"{iter} targ : {targGlobalE} pred: {globAvg}")

        dp2 = cic.distance_plot(CaSelect)
        distplot = np.triu(dp2) + np.tril(dp1, k=-1)

        seqEplot = go.Figure()
        seqEplot.add_trace(
            go.Scatter(x=seqEplotXvals, y=targSeqE, mode="lines", name="targ")
        )
        seqEplot.add_trace(
            go.Scatter(x=seqEplotXvals, y=resArr, mode="lines", name="pred")
        )

        globEplotPred[iter] = globAvg
        globEplot = go.Figure()
        globEplot.add_trace(
            go.Scatter(x=globEplotXvals, y=globEplotTarg, mode="lines", name="targ")
        )
        globEplot.add_trace(
            go.Scatter(x=globEplotXvals, y=globEplotPred, mode="lines", name="pred")
        )

        """
        dp2 = pdb_structure2.distance_plot()
        dpdiff = np.abs(dp1 - dp2)
        print(np.amax(dpdiff))
        """
        """
        d = compare_residues(pdb_structure, pdb_structure2.chain, verbose=True)
        print(d)
        """
        """
        pdb_structure2 = pdb_structure2.chain
        pdb_structure2.atom_to_internal_coordinates()
        pdb_structure2.internal_to_atom_coordinates()
        pdb_structure2 = pdb_structure2.internal_coord
        """

        """
        (pdb_structure2, prot_id2, chainID2, protFile2) = getPDB(
            config["target"]["protein"]
        )
        pdb_structure2.atom_to_internal_coordinates()
        pdb_structure2.internal_to_atom_coordinates()
        pdb_structure2 = pdb_structure2.internal_coord
        """

    print("finished.")
