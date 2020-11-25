import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from planar_transform import PlanarTransform


class PlanarFlow(nn.Module):
    def __init__(self, dim: int = 2, K: int = 6):
        """Make a planar flow by stacking planar transformations in sequence.

        Args:
            dim: Dimensionality of the distribution to be estimated.
            K: Number of transformations in the flow. 
        """
        super().__init__()
        self.layers = [PlanarTransform(dim) for _ in range(K)]
        self.model = nn.Sequential(*self.layers)

    def forward(self, z: Tensor) -> Tuple[Tensor, float]:
        log_det_J = 0

        for layer in self.layers:
            log_det_J += layer.log_det_J(z)
            z = layer(z)

        return z, log_det_J
