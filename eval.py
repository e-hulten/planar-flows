import torch
import matplotlib.pyplot as plt
from planar_flow import PlanarFlow
from target_distribution import TargetDistribution
from utils.plot import plot_density, plot_transformation, plot_comparison
from utils.gif import make_gif_from_train_plots

# ------------ parameters ------------
target_distr = "ring"  # U_1, U_2, U_3, U_4, ring
flow_length = 32
dim = 2
n_samples = 500
batch_size = 128
lr = 6e-4
xlim = ylim = 7

FNAME_TRUE = f"results/{target_distr}_true_density.png"
FNAME_ESTIMATED = f"results/{target_distr}_K{flow_length}_estimated_density.png"
FNAME_GIF = f"{target_distr}.gif"
# ------------------------------------

# Load model.
model = PlanarFlow(dim, flow_length)
checkpoint = torch.load(f"models/model_{target_distr}_K_{flow_length}.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Plot and save true density.
density = TargetDistribution(target_distr)

ax = plot_density(density, xlim=xlim, ylim=ylim)
ax.text(
    0,
    ylim - 1,
    "True density, $" + target_distr + "$",
    horizontalalignment="center",
    size=14,
)
plt.savefig(
    FNAME_TRUE, bbox_inches="tight", pad_inches=0.5,
)
plt.close()
print(f"Success: Plot of true {target_distr} saved at 'results/{FNAME_TRUE}'.")

# Plot and save estimated density.
batch = torch.zeros(n_samples, dim).normal_(mean=0, std=1)
z = model(batch)[0].detach().numpy()
ax = plot_transformation(model, xlim=xlim, ylim=ylim)
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
    FNAME_ESTIMATED, bbox_inches="tight", pad_inches=0.5,
)
plt.close()
print(
    f"Success: Plot of estimated {target_distr} saved at 'results/{FNAME_ESTIMATED}'."
)

plot_comparison(model, density, target_distr, flow_length, xlim, ylim)
print(
    f"Success: Comparison of true vs. estimated {target_distr} saved at 'results/{target_distr}_K{flow_length}_comparison.pdf'."
)

make_gif_from_train_plots(fname=FNAME_GIF)
print(f"Success: Animation of the training process saved at 'gif/{FNAME_GIF}'.")
