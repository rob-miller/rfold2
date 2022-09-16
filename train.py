#!/usr/local/bin/python3 -u

import argparse
import os
import time

from utils import parse_configuration
from datasets import get_dataset
from models import get_model, checkpoint_name, checkpoint_save, checkpoint_load
from losses import get_loss
from optimizers import get_optimizer

import torch
from torch.utils.data import random_split, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

"""Performs training of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
    export: Whether to export the final model (default=True).
"""


def get_dataloaders(config):
    print("Initializing dataset...")
    full_dataset = get_dataset(config["dataset"])
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    train_size = int(config["process"]["train_fraction"] * len(full_dataset))
    test_size = len(full_dataset) - train_size
    if config["dataset"]["random_split"]:
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size]
        )
    else:
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
    train_dataloader = DataLoader(train_dataset, **config["loader"])
    test_dataloader = DataLoader(test_dataset, **config["loader"])

    if hasattr(full_dataset, "inputDim"):
        config["model"]["input_dim"] = full_dataset.inputDim
        config["model"]["output_dim"] = full_dataset.outputDim

    return train_dataloader, test_dataloader


def train_epoch(dataloader, model, loss_fn, optimizer, writer, epcb, config):
    size = len(dataloader.dataset)
    dev = config["process"]["device"]
    model.train()
    for batch, (inp, outp) in enumerate(dataloader):
        inp, outp = inp.to(dev), outp.to(dev)

        # Compute prediction error
        pred = model(inp)
        loss = loss_fn(pred, outp)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % config["report"]["batch_freq"] == 0 or os.path.exists(
            config["process"]["batchStop"]
        ):
            loss, current = loss.item(), batch * len(inp)
            # if writer is None or config["visualization"]["name"] == "text":
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # el
            if config["visualization"]["name"] == "tensorboard":
                writer.add_scalar("loss/train", loss, epcb + batch)

        if os.path.exists(config["process"]["batchStop"]):
            print(f"stopping: found {config['process']['batchStop']}")
            break


def test(dataloader, model, loss_fn, writer, epcb, config):
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    dev = config["process"]["device"]
    model.eval()
    test_loss = 0  # , correct = 0, 0
    with torch.no_grad():
        for inp, outp in dataloader:
            inp, outp = inp.to(dev), outp.to(dev)
            pred = model(inp)
            test_loss += loss_fn(pred, outp).item()
    test_loss /= num_batches
    # if writer is None or config["visualization"]["name"] == "text":
    print(f"Test Error: -- Avg loss: {test_loss:>8f} \n")
    # el
    if config["visualization"]["name"] == "tensorboard":
        writer.add_scalar("avg_loss/test", test_loss, epcb)


def train(config_file, export=True):
    print(f"Reading config file {config_file}...")
    config = parse_configuration(config_file)
    train_dataloader, test_dataloader = get_dataloaders(config)

    print(f"device: {config['process']['device']}")
    print(f"{len(train_dataloader.dataset)} training samples")
    print(f"{len(test_dataloader.dataset)} test samples")

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

    # loss_fn = get_loss(config["loss"])
    loss_fn = get_loss(config)
    optimizer = get_optimizer(model, config["optimizer"])
    writer = None
    if config["visualization"]["name"] == "tensorboard":
        name = config_file[:-4] + "tb"  # replace '.json' with '.tb'
        writer = SummaryWriter(f'{config["visualization"]["path"]}/{name}')

    checkpoint_freq = config["checkpoint"]["epoch_freq"]

    stopFile = checkpoint_name(0, config)
    config["process"]["batchStop"] = stopFile.replace(".pt", ".batchStop")
    config["process"]["epochStop"] = stopFile.replace(".pt", ".epochStop")

    print(f"starting {epochs} epochs")
    print(f"   touch {config['process']['batchStop']}")
    print("to stop after current batch, or")
    print(f"   touch {config['process']['epochStop']}")
    print("to stop after current epoch")

    num_batches = len(train_dataloader)
    for t in range(load_epoch, epochs):
        epoch_start_time = time.time()
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(
            train_dataloader, model, loss_fn, optimizer, writer, t * num_batches, config
        )

        if os.path.exists(config["process"]["batchStop"]) or os.path.exists(
            config["process"]["epochStop"]
        ):
            if os.path.exists(config["process"]["epochStop"]):
                print(f"stopping: found {config['process']['epochStop']}")
                os.remove(config["process"]["epochStop"])
            else:
                os.remove(config["process"]["batchStop"])
            break
        else:
            print(
                f"finished epoch {t+1} / {epochs} \t time: {(time.time() - epoch_start_time):.2f} secs"
            )
            test(test_dataloader, model, loss_fn, writer, (t + 1) * num_batches, config)

            if t % checkpoint_freq == 0:
                checkpoint_save(model, t, config)

    print("finishing, writing final checkpoint file...")
    # possibly redundant re-write of last checkpoint
    checkpoint_save(model, t, config)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn", True)

    parser = argparse.ArgumentParser(description="Perform model training.")
    parser.add_argument("configfile", help="path to the configfile")

    args = parser.parse_args()
    train(args.configfile)
    print("finished, exiting.")
