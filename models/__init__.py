import importlib

# from torch import nn
import torch
import glob
import re
import os

from collections import OrderedDict


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py"."""
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace("_", "") + "model"
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(
            cls, torch.nn.Module
        ):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase."
            % (model_filename, target_model_name)
        )
        exit(0)

    return model


def get_model(configuration):
    """Create a model given the configuration."""
    model = find_model_using_name(configuration["name"])
    instance = model(configuration)
    print("model [{0}] was created".format(type(instance).__name__))
    return instance


def checkpoint_name(epoch, config):
    # loss and optimizer extraneous to model but identify different experiments
    mname = config["model"]["name"]
    reschar = config["dataset"]["res_char"]
    gridRes = config["dataset"]["grid_resolution"]
    loss = config["loss"]["name"]
    optim = config["optimizer"]["name"]
    layers = config["model"]["layers"]
    path = config["checkpoint"]["path"]
    if epoch == "last":
        # https://stackoverflow.com/questions/2225564/get-a-filtered-list-of-files-in-a-directory
        pat = f"{mname}_{reschar}_{gridRes}_{loss}_{optim}_{layers}_net_*.pt*"
        pat_path = os.path.join(path, pat)
        file_list = glob.glob(pat_path)
        # https://stackoverflow.com/questions/43074685/find-file-in-directory-with-the-highest-number-in-the-filename

        def get_num(f):
            s = re.findall(r"_(\d+).pt", f)
            return (int(s[0]) if s else -1, f)

        if not file_list:
            return pat_path
        file_path = max(file_list, key=get_num)
        file_path = file_path[:-1]  # remove device number
        # print(f"found {file_path}")
        config["checkpoint"]["load"] = get_num(file_path)[0]
    else:
        filename = f"{mname}_{reschar}_{gridRes}_{loss}_{optim}_net_{epoch}.pt"
        file_path = os.path.join(path, filename)
    return file_path


def checkpoint_save(model, epoch, config):
    save_path = checkpoint_name(epoch, config)
    for d in range(len(config["model"]["devlist"])):
        spd = save_path + str(d)
        dev = config["model"]["devlist"][d]
        if dev[1] == "u":  # c[u]da:n or c[p]u
            torch.save(model.netlist[d].cpu().state_dict(), spd)
            model.netlist[d].to(dev)
        else:
            torch.save(model.netlist[d].cpu().state_dict(), spd)


def checkpoint_load(model, config):
    epoch = config["checkpoint"]["load"]
    load_path = checkpoint_name(epoch, config)
    for d in range(len(config["model"]["devlist"])):
        lpd = load_path + str(d)
        if not os.path.exists(lpd):
            print(f"no checkpoint file found: {lpd}")
            return False
        dev = config["model"]["devlist"][d]
        state_dict = torch.load(lpd, map_location=dev)
        try:
            model.netlist[d].load_state_dict(state_dict)
        except RuntimeError:
            nsd = OrderedDict()
            for k, v in state_dict.items():
                nk = k.replace("shortcut_linear_relu_stack.", "")
                nsd[nk] = v
            model.netlist[d].load_state_dict(nsd)
        print(f"loaded checkpoint file: {lpd}")
    model.eval()
    return True
