"""This package includes modules related to movement (next conformation) of a protein structure.

    To add a custom move function class called 'dummy', you need to add a file called 'dummy_mfn.py' and define a subclass 'DummyMfn' inherited from BaseMfn.
"""
import importlib
from movefns.base_mfn import BaseMfn


def find_mfn_using_name(mfn_name):
    """Import the module "energyfns/[mfn_name]_mfn.py".

    In the file, the class called mfnNameMfn() will
    be instantiated. It has to be a subclass of BaseMfn,
    and it is case-insensitive.
    """
    mfn_filename = "movefns." + mfn_name + "_mfn"
    mfnlib = importlib.import_module(mfn_filename)

    mfn = None
    target_mfn_name = mfn_name.replace("_", "") + "mfn"
    for name, cls in mfnlib.__dict__.items():
        if name.lower() == target_mfn_name.lower() and issubclass(cls, BaseMfn):
            mfn = cls

    if mfn is None:
        raise NotImplementedError(
            "In {0}.py, there should be a subclass of BaseMfn with class name that matches {1} in lowercase.".format(
                mfn_filename, target_mfn_name
            )
        )

    return mfn


def get_mfn(config):
    """Return initialized mfn class from configuration (loaded from the json file).

    Example:
        from mfns import get_mfn
        mfn = get_mfn(configuration)
    """
    ds_class = find_mfn_using_name(config["name"])
    return ds_class(config)
