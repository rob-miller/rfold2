"""This package includes all the modules related to energy functions.

    To add a custom energy function class called 'dummy', you need to add a file called 'dummy_efn.py' and define a subclass 'DummyEfn' inherited from BaseEfn.
"""
import importlib
from energyfns.base_efn import BaseEfn


def find_efn_using_name(efn_name):
    """Import the module "energyfns/[efn_name]_efn.py".

    In the file, the class called efnNameEfn() will
    be instantiated. It has to be a subclass of BaseEfn,
    and it is case-insensitive.
    """
    efn_filename = "energyfns." + efn_name + "_efn"
    efnlib = importlib.import_module(efn_filename)

    efn = None
    target_efn_name = efn_name.replace("_", "") + "efn"
    for name, cls in efnlib.__dict__.items():
        if name.lower() == target_efn_name.lower() and issubclass(cls, BaseEfn):
            efn = cls

    if efn is None:
        raise NotImplementedError(
            "In {0}.py, there should be a subclass of BaseEfn with class name that matches {1} in lowercase.".format(
                efn_filename, target_efn_name
            )
        )

    return efn


def get_efn(config):
    """Return initialized efn class from configuration (loaded from the config file).

    Example:
        from efns import get_efn
        efn = get_efn(configuration)
    """
    ds_class = find_efn_using_name(config["name"])
    return ds_class(config)
