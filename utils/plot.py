import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from target_distribution import TargetDistribution


def plot_density(density, xlim=4, ylim=4, ax=None, cmap="Blues"):
    x = y = np.linspace(-xlim, xlim, 300)
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = torch.from_numpy(np.concatenate([X_flatten, Y_flatten], 1))
    U = torch.exp(-density(Z))
    U = U.reshape(shape)
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_aspect(1)

    ax.pcolormesh(X, Y, U, cmap=cmap, rasterized=True)
    ax.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    return ax


def plot_samples(z):
    nbins = 250
    lim = 4
    # z = np.exp(-z)
    k = gaussian_kde([z[:, 0], z[:, 1]])
    xi, yi = np.mgrid[-lim : lim : nbins * 1j, -lim : lim : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.set_xlim(-5, 5)
    ax.set_aspect(1)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Blues", rasterized=True)
    return ax


def plot_transformation(model, n=500, xlim=4, ylim=4, ax=None, cmap="Blues"):
    base_distr = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    x = torch.linspace(-xlim, xlim, n)
    xx, yy = torch.meshgrid(x, x)
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    zk, sum_log_jacobians = model(zz)

    base_log_prob = base_distr.log_prob(zz)
    final_log_prob = base_log_prob - sum_log_jacobians
    qk = torch.exp(final_log_prob)
    if ax is None:
        fig = plt.figure(figsize=[7, 7])
        ax = fig.add_subplot(111)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect(1)
    ax.pcolormesh(
        zk[:, 0].detach().data.reshape(n, n),
        zk[:, 1].detach().data.reshape(n, n),
        qk.detach().data.reshape(n, n),
        cmap=cmap,
        rasterized=True,
    )

    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    if cmap == "Blues":
        ax.set_facecolor(plt.cm.Blues(0.0))
    elif cmap == "Reds":
        ax.set_facecolor(plt.cm.Reds(0.0))

    return ax


def plot_training(model, flow_length, batch_num, lr, axlim):
    ax = plot_transformation(model, xlim=axlim, ylim=axlim)
    ax.text(
        0,
        axlim - 2,
        "Flow length: {}\nDensity of one batch, iteration #{:06d}\nLearning rate: {}".format(
            flow_length, batch_num, lr
        ),
        horizontalalignment="center",
    )
    plt.savefig(
        f"train_plots/iteration_{batch_num:06d}.png",
        bbox_inches="tight",
        pad_inches=0.5,
    )
    plt.close()


def plot_comparison(model, target_distr, flow_length, dpi=400):
    xlim = ylim = 7 if target_distr == "ring" else 5
    fig, axes = plt.subplots(
        ncols=2, nrows=1, sharex=True, sharey=True, figsize=[10, 5], dpi=dpi
    )
    axes[0].tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )
    # Plot true density.
    density = TargetDistribution(target_distr)
    plot_density(density, xlim=xlim, ylim=ylim, ax=axes[0])
    axes[0].text(
        0,
        ylim - 1,
        "True density $\exp(-{})$".format(target_distr),
        size=14,
        horizontalalignment="center",
    )

    # Plot estimated density.
    batch = torch.zeros(500, 2).normal_(mean=0, std=1)
    z = model(batch)[0].detach().numpy()
    axes[1] = plot_transformation(model, xlim=xlim, ylim=ylim, ax=axes[1], cmap="Reds")
    axes[1].text(
        0,
        ylim - 1,
        "Estimated density $\exp(-{})$".format(target_distr),
        size=14,
        horizontalalignment="center",
    )
    fig.savefig(
        "results/" + target_distr + "_K" + str(flow_length) + "_comparison.pdf",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_available_distributions():
    target_distributions = ["U_1", "U_2", "U_3", "U_4", "ring"]
    cmaps = ["Reds", "Purples", "Oranges", "Greens", "Blues"]
    fig, axes = plt.subplots(1, len(target_distributions), figsize=(20, 5))
    for i, distr in enumerate(target_distributions):
        axlim = 7 if distr == "ring" else 5
        density = TargetDistribution(distr)
        plot_density(density, xlim=axlim, ylim=axlim, ax=axes[i], cmap=cmaps[i])
        axes[i].set_title(f"Name: '{distr}'", size=16)
        plt.setp(axes, xticks=[], yticks=[])
    plt.show()
