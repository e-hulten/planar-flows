import torch
import torch.nn as nn
from transform import PlanarTransform


class PlanarFlow(nn.Module):
    def __init__(self, dim=2, K=6):
        super().__init__()
        self.layers = []

        for _ in range(K):
            self.layers.append(PlanarTransform(dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        log_det_J = 0

        for layer in self.layers:
            log_det_J += layer.log_det_J(z)
            z = layer(z)

        return z, log_det_J
