"""Plots for the mutant phenotype score (see :mod:`pyrea.mps`)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .mps import DEFAULT_LR, lr_to_rl, relative_likelihood

__all__ = ["plot_mps_rank", "plot_rl_diagnostic"]

_MUT, _WT = "#d62728", "#1f77b4"


def plot_mps_rank(mps: pd.Series, mutated: pd.Series, lr: float = DEFAULT_LR, ax=None):
    """Samples rank-sorted by MPS, carriers marked (cf. Alvarez et al., Fig. 5a).

    Parameters
    ----------
    mps : pd.Series
        MPS per sample.
    mutated : pd.Series
        Binary (1 = carrier) per sample; aligned to ``mps`` by index.
    lr : float
        Likelihood ratio defining the shaded mutant / WT phenotype regions.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(2.5, 2))
    s = mps.dropna().sort_values()
    carriers = np.flatnonzero(mutated.reindex(s.index).to_numpy() == 1)
    t = lr_to_rl(lr)
    ax.axhspan(t, 1, color="#2ca02c", alpha=0.15, lw=0)
    ax.axhspan(-1, -t, color="#fa8072", alpha=0.25, lw=0)
    ax.plot(np.arange(len(s)), s.to_numpy(), color="k", lw=1)
    ax.vlines(carriers, -1, s.to_numpy()[carriers], color="#2ca02c", lw=0.6)
    ax.set(ylim=(-1.05, 1.05), xlim=(0, len(s)), xlabel="Samples (rank)", ylabel="MPS")
    return ax


def plot_rl_diagnostic(gene: str, mutations: pd.DataFrame, activity: pd.DataFrame,
                       rpt: pd.DataFrame | None = None, lr: float = DEFAULT_LR):
    """Per trait: mutant vs WT density (top) and the RL curve (bottom)."""
    traits = {"G-activity": activity}
    if rpt is not None:
        traits["RPT-activity"] = rpt
    samples = mutations.columns
    for t in traits.values():
        samples = samples.intersection(t.columns, sort=False)
    m = mutations.loc[gene, samples].to_numpy() == 1

    fig, axes = plt.subplots(2, len(traits), figsize=(4.5 * len(traits), 5.5),
                             sharex="col", squeeze=False)
    thr = lr_to_rl(lr)
    for j, (name, df) in enumerate(traits.items()):
        v = df.loc[gene, samples].to_numpy(dtype=float)
        grid = np.linspace(np.nanmin(v) - 0.5, np.nanmax(v) + 0.5, 200)
        top, bot = axes[0, j], axes[1, j]
        for vals, label, col in ((v[~m], "WT", _WT), (v[m], "Mutant", _MUT)):
            vals = vals[np.isfinite(vals)]
            if vals.size > 1 and np.ptp(vals) > 0:
                dens = gaussian_kde(vals)(grid)
                top.plot(grid, dens, color=col, label=label)
                top.fill_between(grid, dens, color=col, alpha=0.15)
        top.set_title(name)
        top.legend(frameon=False)
        bot.plot(grid, relative_likelihood(grid, v[m], v[~m]), color="purple")
        bot.axhline(0, color="0.5", ls="--")
        for y in (thr, -thr):
            bot.axhline(y, color="r", ls=":")
        bot.set(ylim=(-1.05, 1.05), xlabel=name)
    axes[0, 0].set_ylabel("Density")
    axes[1, 0].set_ylabel("RL")
    fig.suptitle(gene)
    fig.tight_layout()
    return fig
