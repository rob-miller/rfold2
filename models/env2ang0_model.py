from torch import nn


class env2ang0Model(nn.Module):
    def __init__(self, config):
        """Initialize the model."""
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(config["input_dim"], config["input_dim"]),
            nn.ReLU(),
            nn.Linear(config["input_dim"], config["input_dim"]),
            nn.ReLU(),
            nn.Linear(config["input_dim"], config["output_dim"]),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
