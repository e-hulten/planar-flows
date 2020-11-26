import os
import torch
import torch.nn
import matplotlib.pyplot as plt
from planar_flow import PlanarFlow
from target_distribution import TargetDistribution
from loss import VariationalLoss
from utils.plot import plot_transformation

if __name__ == "__main__":
    # ------------ parameters ------------
    target_distr = "ring"  # U_1, U_2, U_3, U_4, ring
    flow_length = 32
    dim = 2
    num_batches = 20000
    batch_size = 128
    lr = 6e-4
    xlim = ylim = 7  # 5 for U_1 to U_4, 7 for ring
    # ------------------------------------

    density = TargetDistribution(target_distr)
    model = PlanarFlow(dim, K=flow_length)
    bound = VariationalLoss(density)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_batches + 1):
        # Get batch from N(0,I).
        batch = torch.zeros(batch_size, dim).normal_(mean=0, std=1)
        # Pass batch through flow.
        zk, log_jacobians = model(batch)
        # Compute loss under target distribution.
        loss = bound(batch, zk, log_jacobians)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if epoch % 1000 == 0:
            print(f"(epoch {epoch:05d}/{num_batches}) loss: {loss}")

        if epoch == 1 or epoch % 100 == 0:
            ax = plot_transformation(model, xlim=xlim, ylim=xlim)

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
                "epoch": num_batches,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "loss": loss,
            },
            "models/model_" + target_distr + "_K_" + str(flow_length) + ".pt",
        )
