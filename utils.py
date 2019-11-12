import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from matplotlib.colors import ListedColormap


def target_density(str):
    w1 = lambda z: torch.sin(2 * np.pi * z[:, 0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
    w3 = lambda z: 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)

    if str == "U_1":

        def U_1(z):
            u = 0.5 * ((torch.norm(z, 2, dim=1) - 2) / 0.4) ** 2
            u = u - torch.log(
                torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
                + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
            )
            return u

        return U_1
    elif str == "U_2":

        def U_2(z):
            u = 0.5 * ((z[:, 1] - w1(z)) / 0.4) ** 2
            return u

        return U_2
    elif str == "U_3":

        def U_3(z):
            u = -torch.log(
                torch.exp(-0.5 * ((z[:, 1] - w1(z)) / 0.35) ** 2)
                + torch.exp(-0.5 * ((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2)
                + 1e-6
            )
            return u

        return U_3
    elif str == "U_4":

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


def plot_density(density, xlim=4, ylim=4):
    x = y = np.linspace(-xlim, xlim, 500)
    X, Y = np.meshgrid(x, y)
    shape = X.shape
    X_flatten, Y_flatten = np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))
    Z = torch.from_numpy(np.concatenate([X_flatten, Y_flatten], 1))
    U = torch.exp(-density(Z))
    U = U.reshape(shape)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
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
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.pcolormesh(X, Y, U, cmap="Purples")
    return ax


def plot_samples(z):
    nbins = 250
    lim = 4
    # z = np.exp(-z)
    k = kde.gaussian_kde([z[:, 0], z[:, 1]])
    xi, yi = np.mgrid[-lim : lim : nbins * 1j, -lim : lim : nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.set_xlim(-5, 5)
    ax.set_aspect(1)
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="Blues")
    return ax


def plot_samples2(model, n=1000, xlim=4, ylim=4):
    base_distr = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    x = torch.linspace(-xlim, xlim, n)
    xx, yy = torch.meshgrid(x, x)
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    zk, sum_log_jacobians = model(zz)

    base_log_prob = base_distr.log_prob(zz)
    final_log_prob = base_log_prob - sum_log_jacobians
    qk = torch.exp(final_log_prob)

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect(1)
    ax.pcolormesh(
        zk[:, 0].detach().data.reshape(n, n),
        zk[:, 1].detach().data.reshape(n, n),
        qk.detach().data.reshape(n, n),
        cmap="Blues",
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

    ax.set_facecolor(plt.cm.Blues(0.0))
    return ax


def plot_samples3(model, n=300, range_lim=4):
    base_dist = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
    x = torch.linspace(-range_lim, range_lim, n)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze()

    # plot posterior approx density
    zzk, sum_log_abs_det_jacobians = model(zz)
    log_q0 = base_dist.log_prob(zz)
    log_qk = log_q0 - sum_log_abs_det_jacobians
    qk = log_qk.exp()

    fig = plt.figure(figsize=[7, 7])
    ax = fig.add_subplot(111)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.pcolormesh(
        zzk[:, 0].view(n, n).data,
        zzk[:, 1].view(n, n).data,
        qk.view(n, n).data,
        cmap=plt.cm.Blues,
    )
    ax.set_facecolor(plt.cm.Blues(0.0))
    return ax
