"""
Generate results from this implementation to the original 2012 paper.

Author : Dakota Hawkins
"""


import os

import numpy as np
from matplotlib import pyplot as plt

import ncfs


def plot_features(w, ax=None, title=""):
    if ax is None:
        ax = plt.subplot()
    idxs = np.where(w > 1)[0]
    ax.scatter(idxs, w[idxs], facecolors="none", edgecolors="red")
    ax.set_title(title, fontdict={"fontsize": "small"})
    ax.set_xlim(left=-0.05 * len(w), right=len(w))
    return ax


n_features = [100, 500, 1000, 5000]
fig, axes = plt.subplots(1, 4, figsize=(12, 6))
# Calculate feature weights using manhattan + euclidean distances
for i, n in enumerate(n_features):
    X, y = ncfs.toy_dataset(n)
    feature_select = ncfs.NCFS(eta=0.01, metric="manhattan")
    feature_select.fit(X, y)
    axes[i] = plot_features(
        feature_select.coef_,
        ax=axes[i],
        title="Manhattan distance, N = {}".format(n),
    )

figdir = os.path.join(
    os.path.basename(__file__),
    "../docs/images"
)
if not os.path.exists(figdir):
    os.makedirs(figdir)
fig.savefig(os.path.join(figdir, "figure1_comp.png"))
plt.clf()

lambdas = [0.25, 0.5, 1, 1.5, 2]
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
X, y = ncfs.toy_dataset(1000)
for i, reg in enumerate(lambdas):
    feature_select = ncfs.NCFS(reg=reg, eta=0.01)
    feature_select.fit(X, y)
    axes[i] = plot_features(
        feature_select.coef_, ax=axes[i], title="$\lambda = {}$".format(reg)
    )

fig.savefig(os.path.join(figdir, "figure2_comp.png"))
plt.clf()


sigmas = [0.25, 0.5, 1, 1.5, 2]
fig, axes = plt.subplots(1, 5, figsize=(12, 4))
for i, sigma in enumerate(sigmas):
    feature_select = ncfs.NCFS(sigma=sigma, eta=0.01)
    feature_select.fit(X, y)
    axes[i] = plot_features(
        feature_select.coef_, ax=axes[i], title="$\sigma = {}$".format(sigma)
    )

fig.savefig(os.path.join(figdir, "figure3_comp.png"))
plt.clf()
