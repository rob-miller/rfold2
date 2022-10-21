"""This module implements an abstract base class (ABC) 'BaseMfn' for mfns (move
protein functions).
"""
from abc import ABC  # , abstractmethod


class BaseMfn(ABC):
    """This class is an abstract base class (ABC) for mfns (move functions)."""

    def __init__(self, configuration):
        """Initialize the class; save the configuration in the class."""
        self.configuration = configuration

    # @abstractmethod
    # def __len__(self):
    #    """Return the total number of images in the mfn."""
    #    return 0

    def step(self, chain):
        """Return next conformation of chain"""
        return chain


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
