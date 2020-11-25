import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch import Tensor
from target_distribution import TargetDistribution


class VariationalLoss(nn.Module):
    def __init__(self, distribution: TargetDistribution):
        super().__init__()
        self.distr = distribution
        self.base_distr = MultivariateNormal(torch.zeros(2), torch.eye(2))

    def forward(self, z0: Tensor, z: Tensor, sum_log_det_J: float) -> float:
        base_log_prob = self.base_distr.log_prob(z0)
        target_density_log_prob = -self.distr(z)
        return (base_log_prob - target_density_log_prob - sum_log_det_J).mean()
