import torch
import torch.nn as nn


class PlanarTransform(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, 0.1))

    def forward(self, z):
        if torch.mm(self.u, self.w.T) < -1:
            self.u.data = self.u_hat()
        zk = z + self.u * nn.Tanh()(torch.mm(z, self.w.T) + self.b)
        return zk

    def u_hat(self):
        # enforce w^T u > -1 to ensure invertibility
        # slows down the computation quite a bit, should always try to run without first
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.log(1 + torch.exp(wtu))
        return (
            self.u + (m_wtu - wtu) * self.w / torch.norm(self.w, p=2, dim=1) ** 2
        )  # torch.mul(self.w,self.w).sum()

    def log_det_J(self, z):
        if torch.mm(self.u, self.w.T) < -1:
            print("Normalising u to u_hat. Old w.T.dot(u)=", torch.mm(self.u, self.w.T))
            self.u.data = self.u_hat()
            print("New w.T.dot(u):", torch.mm(self.u, self.w.T))

        a = torch.mm(z, self.w.T) + self.b
        psi = (1 - nn.Tanh()(a) ** 2) * self.w
        abs_det = (1 + torch.mm(self.u, psi.T)).abs()
        log_det = torch.log(1e-4 + abs_det)
        # debugging
        if torch.isnan(log_det).sum() > 0:
            print("u:", self.u)
            print("w:", self.w)
            print("abs_det:", abs_det)
        return log_det
