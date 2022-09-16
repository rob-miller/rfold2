"""This module implements an abstract base class (ABC) 'BaseEfn' for efns. Also
    includes some transformation functions.
"""
from abc import ABC  # , abstractmethod


class BaseEfn(ABC):
    """This class is an abstract base class (ABC) for efns."""

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class."""
        self.configuration = configuration

    # @abstractmethod
    # def __len__(self):
    #    """Return the total number of images in the efn."""
    #    return 0

    def evaluate(self, chain):
        """Return global (avg) energy and array of energy per residue"""
        return 0, []


"""
def get_transform(opt, method=cv2.INTER_LINEAR):
    transform_list = []
    if 'preprocess' in opt:
        if 'resize' in opt['preprocess']:
            transform_list.append(Resize(opt['input_size'][0], opt['input_size'][1], method))

    if 'tofloat' in opt and opt['tofloat'] is True:
        transform_list.append(ToFloat())

    return Compose(transform_list)
"""
