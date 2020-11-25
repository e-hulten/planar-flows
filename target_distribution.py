import numpy as np
import torch
from torch import Tensor
from typing import Callable


class TargetDistribution:
    def __init__(self, name: str):
        """Define target distribution. 

        Args:
            name: The name of the target density to use. 
                  Valid choices: ["U_1", "U_2", "U_3", "U_4", "ring"].
        """
        self.func = self.get_target_distribution(name)

    def __call__(self, z: Tensor) -> Tensor:
        return self.func(z)

    @staticmethod
    def get_target_distribution(name: str) -> Callable[[Tensor], Tensor]:
        w1 = lambda z: torch.sin(2 * np.pi * z[:, 0] / 4)
        w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
        w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

        if name == "U_1":

            def U_1(z):
                u = 0.5 * ((torch.norm(z, 2, dim=1) - 2) / 0.4) ** 2
                u = u - torch.log(
                    torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
                    + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
                )
                return u

            return U_1
        elif name == "U_2":

            def U_2(z):
                u = 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
                return u

            return U_2
        elif name == "U_3":

            def U_3(z):
                u = -torch.log(
                    torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
                    + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
                    + 1e-6
                )
                return u

            return U_3
        elif name == "U_4":

            def U_4(z):
                u = -torch.log(
                    torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2)
                    + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2)
                    + 1e-6
                )
                return u

            return U_4
        elif str == "ring":

            def ring_density(z):
                exp1 = torch.exp(-0.5 * ((z[:, 0] - 2) / 0.8) ** 2)
                exp2 = torch.exp(-0.5 * ((z[:, 0] + 2) / 0.8) ** 2)
                u = 0.5 * ((torch.norm(z, 2, dim=1) - 4) / 0.4) ** 2
                u = u - torch.log(exp1 + exp2 + 1e-6)
                return u

            return ring_density
