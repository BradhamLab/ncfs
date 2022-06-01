"""
Generate results from this implementation to the original 2012 paper.

Author : Dakota Hawkins
"""


import os

import numpy as np
from matplotlib import pyplot as plt

from NCFS import NCFS


def plot_features(w, ax=None, title=""):
    if ax is None:
        ax = plt.subplot()
    idxs = np.where(w > 1)[0]
    ax.scatter(idxs, w[idxs], facecolors="none", edgecolors="red")
    ax.set_title(title, fontdict={"fontsize": "small"})
    ax.set_xlim(left=-0.05 * len(w), right=len(w))
    return ax


n_features = [100, 500, 1000, 5000]
fig, axes = plt.subplots(2, 4, figsize=(12, 8))
# Calculate feature weights using manhattan + euclidean distances
for i, n in enumerate(n_features):
    X, y = NCFS.toy_dataset(n)
    feature_select = NCFS.NCFS(eta=0.01)
    feature_select.fit(X, y)
    axes[0, i] = plot_features(
        feature_select.coef_,
        ax=axes[0, i],
        title="Manhattan distance, N = {}".format(n),
    )
    feature_select = NCFS.NCFS(eta=0.01)
    feature_select.fit(X, y, metric="euclidean")
    axes[1, i] = plot_features(
        feature_select.coef_,
        ax=axes[1, i],
        title="Euclidean distance, N = {}".format(n),
    )

fig.savefig(os.path.join("../", "images", "figure1_comp.png"))
plt.clf()

lambdas = [0.25, 0.5, 1, 1.5, 2]
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
X, y = NCFS.toy_dataset(1000)
for i, reg in enumerate(lambdas):
    feature_select = NCFS.NCFS(reg=reg, eta=0.01)
    feature_select.fit(X, y)
    axes[i] = plot_features(
        feature_select.coef_, ax=axes[i], title="$\lambda = {}$".format(reg)
    )

fig.savefig(os.path.join("../", "images", "figure2_comp.png"))
plt.clf()


sigmas = [0.25, 0.5, 1, 1.5, 2]
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for i, sigma in enumerate(sigmas):
    feature_select = NCFS.NCFS(sigma=sigma, eta=0.01)
    feature_select.fit(X, y)
    axes[i] = plot_features(
        feature_select.coef_, ax=axes[i], title="$\sigma = {}$".format(sigma)
    )

fig.savefig(os.path.join("../", "images", "figure3_comp.png"))
plt.clf()
