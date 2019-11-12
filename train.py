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
continue_training = True
target_distr = "U_3"  # U_1, U_2, U_3, xU_4, ring
flow_length = 32
dim = 2
epochs = 50000
batch_size = 128
lr = 7.5e-4
xlim = ylim = 5
# ------------------------------------

if not os.path.exists("train_plots"):
    os.makedirs("train_plots")
if not os.path.exists("models"):
    os.makedirs("models")

density = target_density(target_distr)
model = PlanarFlow(dim, K=flow_length)
bound = VariationalLoss(density)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,momentum=0.9,alpha=0.90, eps=1e-6, weight_decay=1e-3)


if continue_training is True:
    checkpoint = torch.load("models/model_" + target_distr + ".pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    tot_epochs = checkpoint["epoch"]
    loss = checkpoint["loss"]
else:
    tot_epochs = 0

for epoch in range(tot_epochs + 1, tot_epochs + epochs + 1):
    batch = torch.zeros(batch_size, dim).normal_(mean=0, std=1)
    zk, log_jacobians = model(batch)
    loss = bound(batch, zk, log_jacobians)

    if torch.isnan(log_jacobians).sum() > 0:
        print("log_jacobians is nan")
        break

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print("(epoch {:05d}/{}) loss: {}".format(epoch, tot_epochs + epochs, loss))

    if epoch % 1000 == 0:
        ax = plot_samples2(model, xlim=xlim, ylim=xlim)
        ax.text(
            0,
            ylim - 2,
            "Flow length: {}\nDensity of one batch, iteration #{:06d}\nLearning rate: {}".format(
                flow_length, epoch, lr
            ),
            horizontalalignment="center",
        )
        plt.savefig(
            "train_plots/" + "iteration_{:06d}.png".format(epoch),
            bbox_inches="tight",
            pad_inches=0.5,
        )
        plt.close()

if torch.isnan(log_jacobians).sum() == 0:
    torch.save(
        {
            "epoch": tot_epochs + epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        "models/model_" + target_distr + ".pt",
    )
