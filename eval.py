import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os
from model import PlanarFlow
from loss import VariationalLoss
from utils import *

# ------------ parameters ------------
target_distr = "U_3"  # U_1, U_2, U_3, U_4, ring
flow_length = 32
dim = 2
n_samples = 500
batch_size = 128
lr = 7.5e-4
xlim = ylim = 5
# ------------------------------------

if not os.path.exists("results"):
    os.makedirs("results")

density = target_density(target_distr)
# load model
model = PlanarFlow(dim, flow_length)
checkpoint = torch.load("models/model_" + target_distr + ".pt")
model.load_state_dict(checkpoint["model_state_dict"])

# plot true density
ax = plot_density(density, xlim=xlim, ylim=ylim)
ax.text(
    0,
    ylim - 1,
    "True density, $" + target_distr + "$",
    horizontalalignment="center",
    size=14,
)
plt.savefig(
    "results/" + target_distr + "_true_density.png",
    bbox_inches="tight",
    pad_inches=0.5,
)
plt.close()

# plot estimated density
batch = torch.zeros(n_samples, dim).normal_(mean=0, std=1)
z = model(batch)[0].detach().numpy()
ax = plot_samples2(model, xlim=xlim, ylim=ylim)
ax.text(
    0,
    ylim - 1,
    "Estimated density ${}$ after {} iterations\n Learning rate: {}".format(
        target_distr, checkpoint["epoch"], lr
    ),
    horizontalalignment="center",
    size=12,
)

plt.savefig(
    "results/" + target_distr + "_estimated_density.png",
    bbox_inches="tight",
    pad_inches=0.5,
)
plt.close()
