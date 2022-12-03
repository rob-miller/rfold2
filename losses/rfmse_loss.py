from torch import nn
from utils.rpdbT import rpdbT

import torch


class rfmseLoss(nn.Module):
    def __init__(self, config):
        """Initialize the model."""
        super().__init__()
        self.rc = config["dataset"]["res_char"]
        self.rpdbt = rpdbT(device=config["model"]["devlist"][-1])

    def forward(self, predictions, targets):
        predList = []
        for pred in predictions:
            # dhChrArr, dhLenArr, hArr
            predLens = self.rpdbt.outputArr2values(self.rc, pred)
            predList += [predLens[0], predLens[0], predLens[1], predLens[1]]
            predList += predLens[2]
        preds = torch.cat(predList)

        targList = []
        for targ in targets:
            # dhChrArr, dhLenArr, hArr
            targLens = self.rpdbt.outputArr2values(self.rc, targ)
            targList += [targLens[0], targLens[0], targLens[1], targLens[1]]
            targList += targLens[2]
        targs = torch.cat(targList)

        square_difference = torch.square(preds - targs)
        loss_value = torch.mean(square_difference)
        return loss_value
