from torch import nn
from utils.rpdbT import rpdbT

import numpy as np
import torch


class rmsdLoss(nn.Module):
    def __init__(self, config):
        """Initialize the model."""
        super().__init__()
        self.rc = config["dataset"]["res_char"]
        self.rpdbt = rpdbT(device=config["process"]["device"])

    def forward(self, predictions, targets):
        predList = []
        npp = predictions.cpu().detach().numpy()
        npperr = np.any((npp < -1) | (npp > 1))
        for pred in predictions:
            # dhChrArr, dhLenArr, hArr
            predLens = self.rpdbt.outputArr2values(self.rc, pred)
            pl0 = predLens[0].cpu().detach().numpy()
            pl1 = predLens[1].cpu().detach().numpy()
            pl2 = predLens[2].cpu().detach().numpy()
            plerr = (
                np.any((pl0 < -1) | (pl0 > 1))
                or np.any((pl1 < 1.5) | (pl1 > 5))
                or np.any((pl2 < 0.9) | (pl2 > 3))
            )
            # hedraAngle, hedraAngleRads, dihedraAngle, dihedraAngleRads
            predAngs = self.rpdbt.len2ang(
                self.rc, predLens[0], predLens[1], predLens[2]
            )
            pa0 = predAngs[0].cpu().detach().numpy()
            pa1 = predAngs[1].cpu().detach().numpy()
            pa2 = predAngs[2].cpu().detach().numpy()
            pa3 = predAngs[3].cpu().detach().numpy()
            paxerr = np.any((pa0 < 90) | (pa0 > 160)) or np.any(
                (pa2 < -180) | (pa2 > 180)
            )
            predAtoms = self.rpdbt.lenAng2coords(
                self.rc, predAngs[1], predLens[2], predAngs[3]
            )
            pa = predAtoms.cpu().detach().numpy()
            paErr = np.any((pa < -5) | (pa > 5))
            predList += predAtoms
            # print('hello')
        preds = torch.cat(predList)
        pd = preds.cpu().detach().numpy()

        targList = []
        for targ in targets:
            targLens = self.rpdbt.outputArr2values(self.rc, targ)
            targAngs = self.rpdbt.len2ang(
                self.rc, targLens[0], targLens[1], targLens[2]
            )
            targAtoms = self.rpdbt.lenAng2coords(
                self.rc, targAngs[1], targLens[2], targAngs[3]
            )
            targList += targAtoms
        targs = torch.cat(targList)
        tg = targs.cpu().detach().numpy()
        square_difference = torch.square(preds - targs)
        sqd = square_difference.cpu().detach().numpy()
        loss_value = torch.mean(square_difference)
        lv = loss_value.isinf()
        return loss_value
