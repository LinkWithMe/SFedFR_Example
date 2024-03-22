import torch.nn as nn
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer
class MLPSNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPSNN, self).__init__()
        self.layer=nn.Sequential(
          layer.Flatten(),
          layer.Linear(input_size, 64),
          neuron.IFNode(),
          layer.Linear(64, 30),
          neuron.IFNode(),
          layer.Linear(30, output_size),
          neuron.IFNode()
        )

    def forward(self, x):
        return self.layer(x)