import importlib
from torch import nn


def find_loss_using_name(loss_name):
    """Import the module "losses/[loss_name]_loss.py"."""
    loss_filename = "losses." + loss_name + "_loss"
    losslib = importlib.import_module(loss_filename)
    loss = None
    target_loss_name = loss_name.replace("_", "") + "loss"
    for name, cls in losslib.__dict__.items():
        if name.lower() == target_loss_name.lower() and issubclass(cls, nn.Module):
            loss = cls

    if loss is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase."
            % (loss_filename, target_loss_name)
        )
        exit(0)

    return loss


def get_loss(configuration):
    """Create a model given the configuration."""
    name = configuration["loss"]["name"]
    if name == "MSE":
        return nn.MSELoss()
    else:
        model = find_loss_using_name(name)
        instance = model(configuration)
        # print("loss fn [{0}] created".format(type(instance).__name__))
        return instance
