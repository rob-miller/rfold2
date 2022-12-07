from torch import nn


# with ResNet ideas from
# https://stackoverflow.com/questions/57229054/how-to-implement-my-own-resnet-with-torch-nn-sequential-in-pytorch

# model parallel starting from
# https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class rnLinRelu(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()
        self.layer = ResNet(
            nn.Sequential(
                nn.Linear(idim, odim),
                nn.ReLU(),
                nn.Linear(idim, odim),
                nn.ReLU(),
            )
        )

    def forward(self, x):
        return self.layer(x)


class env2ang9Model(nn.Module):
    def __init__(self, config):
        """Initialize the model."""
        super().__init__()
        self.config = config
        idim, odim = config["input_dim"], config["output_dim"]
        lst = [rnLinRelu(idim, idim) for i in range(7)]
        lst2 = [rnLinRelu(idim, idim) for i in range(7)]
        self.netlist = nn.ModuleList(
            [
                # nn.ModuleList([rnLinRelu(idim, idim) for i in range(15)]),
                # nn.Sequential(*lst),
                nn.Sequential(*lst),
                nn.Sequential(*lst2, nn.Linear(idim, odim)),
            ]
        )

    def forward(self, x):
        for nd in range(len(self.config["devlist"]) - 1):
            x = self.netlist[nd](x.to(self.config["devlist"][nd]))
        return self.netlist[-1](x.to(self.config["devlist"][-1]))
