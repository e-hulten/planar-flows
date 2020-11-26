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

    # Train model.
    for batch_num in range(1, num_batches + 1):
        # Get batch from N(0,I).
        batch = torch.zeros(size=(batch_size, 2)).normal_(mean=0, std=1)
        # Pass batch through flow.
        zk, log_jacobians = model(batch)
        # Compute loss under target distribution.
        loss = bound(batch, zk, log_jacobians)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if batch_num % 1000 == 0:
            print(f"(batch_num {batch_num:05d}/{num_batches}) loss: {loss}")

        if batch_num == 1 or batch_num % 100 == 0:
            # Save plots during training. Plots are saved to the 'train_plots' folder.
            plot_training(model, flow_length, batch_num, lr, axlim)

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
