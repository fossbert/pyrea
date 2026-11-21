"""Figure for a :class:`pyrea.MultiSpecificity`: networks x regulator sets.

Clean panel on an Axes that can be passed (``ax=``); the hits are marked afterwards with :func:`add_hits` (``ax=None`` uses the
current axes, ``**text_kwargs`` go to matplotlib).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_matrix", "add_hits"]


def plot_matrix(ms, value: str = "percentile", signs: str | None = None, ax=None, cmap: str = "Blues", title: str | None = None):
    """Networks (rows) x sets (columns) heatmap of one column of ``ms.table`` for one tail combination (``signs``: ``None`` = all
    members in the positive tail, else e.g. ``"+/-"``). ``value="percentile"`` (default): share of size-matched random sets of other
    regulators that overlap less (1 = more than all of them); ``"fold"``; ``"ext_percentile_max"`` (small = the members are among the
    best partners of each other). Sets that were not evaluated in a network stay empty."""
    mat = ms.matrix(value, signs)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.8 + 0.6 * mat.shape[1], 1.0 + 0.25 * mat.shape[0]))
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)), cmap=cmap, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7)
    label = signs if signs is not None else f"all {ms._default_sign()}"
    ax.set_title(f"{value}, tails {label}" if title is None else title, loc="left", fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    ax._pyrea_specificity = (mat, signs)
    return ax


def add_hits(ms, ax=None, q: float = 0.05, fold_min: float = 2.0, min_overlap: int = 3, marker: str = "*", **text_kwargs):
    """After :func:`plot_matrix`: put ``marker`` into the cells that are hits (see :meth:`MultiSpecificity.hits`)."""
    ax = plt.gca() if ax is None else ax
    info = getattr(ax, "_pyrea_specificity", None)
    if info is None:
        raise ValueError("Draw the matrix first (plot_matrix) on this axes")
    mat, signs = info
    col = {c: j for j, c in enumerate(mat.columns)}
    row = {r: i for i, r in enumerate(mat.index)}
    vals = mat.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    mid = (finite.min() + finite.max()) / 2 if len(finite) else 0
    for net, s in ms.hit_cells(signs, q=q, fold_min=fold_min, min_overlap=min_overlap):
        i, j = row.get(net), col.get(s)
        if i is None or j is None or not np.isfinite(vals[i, j]):
            continue
        props = dict(ha="center", va="center", fontsize=8, color="white" if vals[i, j] > mid else "black")
        props.update(text_kwargs)
        ax.text(j, i, marker, **props)
    return ax
