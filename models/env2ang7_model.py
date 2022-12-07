from torch import nn


# with ResNet ideas from
# https://stackoverflow.com/questions/57229054/how-to-implement-my-own-resnet-with-torch-nn-sequential-in-pytorch


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class env2ang7Model(nn.Module):
    def __init__(self, config):
        """Initialize the model."""
        super().__init__()
        self.config = config
        # shortcut_linear_relu_stack
        self.netlist = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config["input_dim"], config["input_dim"]),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                ResNet(
                    nn.Sequential(
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                        nn.Linear(config["input_dim"], config["input_dim"]),
                        nn.ReLU(),
                    )
                ),
                nn.Linear(config["input_dim"], config["output_dim"]),
            ),
        ])

    def forward(self, x):
        for nd in range(len(self.config["devlist"]) - 1):
            x = self.netlist[nd](x.to(self.config["devlist"][nd]))
        return self.netlist[-1](x.to(self.config["devlist"][-1]))
        # x = self.flatten(x)
        # logits = self.shortcut_linear_relu_stack(x)
        # return logits
