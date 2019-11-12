import torch
import torch.nn as nn


class VariationalLoss(nn.Module):
    def __init__(self, distr):
        super().__init__()
        self.distr = distr
        self.base_distr = torch.distributions.MultivariateNormal(
            torch.zeros(2), torch.eye(2)
        )

    def forward(self, z0, z, sum_log_det_J):
        base_log_prob = self.base_distr.log_prob(z0)
        target_density_log_prob = -self.distr(z)
        return (base_log_prob - target_density_log_prob - sum_log_det_J).mean()

