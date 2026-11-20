"""Figures for a :class:`pyrea.RegulonOverlap` (overlap of regulons by tail).

Same design as the other pyrea figure modules: the panels are clean, each is drawn on a matplotlib ``Axes`` that can be
passed (``ax=``) and returns it, and statistics are added afterwards, selectively, with the ``add_*`` functions
(``ax=None`` uses the current axes, ``**text_kwargs`` go to matplotlib).

=====================  ==============================  ====================================================
panel                  draw                            add statistics
=====================  ==============================  ====================================================
regulator x regulator  :func:`plot_heatmap`            :func:`add_heatmap_values`
shared targets (mor)   :func:`plot_scatter`            :func:`add_scatter_stats`
intersections          :func:`plot_upset`              :func:`add_upset_counts`
=====================  ==============================  ====================================================
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

__all__ = ["plot_heatmap", "add_heatmap_values", "plot_scatter", "add_scatter_stats", "plot_upset", "add_upset_counts"]

_CONCORDANT, _DISCORDANT, _ZERO = "#1f77b4", "#d62728", "#9e9e9e"
_PALETTE = ["#1f77b4", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#6a3d9a", "#8c564b"]


def _signs_label(signs) -> str:
    return f"{signs[0]} / {signs[1]}"


def plot_heatmap(ov, metric: str = "jaccard", signs: tuple = ("+", "+"), ax=None, cmap: str = "Blues",
                 title: str | None = None):
    """Regulator x regulator heatmap of one ``metric`` for one tail combination (``signs`` = tail of the row / column regulator).

    The diagonal is left empty. ``metric``: see :meth:`RegulonOverlap.matrix`. The numbers are added with :func:`add_heatmap_values`.
    """
    mat = ov.matrix(metric, signs)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 + 0.6 * len(ov.tfs), 1.2 + 0.6 * len(ov.tfs)))
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy()), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ov.tfs)), mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ov.tfs)), mat.index, fontsize=7)
    ax.set_title(f"{metric}, tails {_signs_label(signs)}" if title is None else title, loc="left", fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6)
    ax._pyrea_matrix = mat
    return ax


def add_heatmap_values(ov, ax=None, fmt: str = "{:.2f}", **text_kwargs):
    """Write the values of :func:`plot_heatmap` into its cells (``fmt``, e.g. ``"{:.1e}"`` for p-values)."""
    ax = plt.gca() if ax is None else ax
    mat = getattr(ax, "_pyrea_matrix", None)
    if mat is None:
        raise ValueError("Draw the heatmap first (plot_heatmap) on this axes")
    vals = mat.to_numpy()
    finite = vals[np.isfinite(vals)]
    mid = (finite.min() + finite.max()) / 2 if len(finite) else 0
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if np.isfinite(vals[i, j]):
                props = dict(ha="center", va="center", fontsize=6, color="white" if vals[i, j] > mid else "black")
                props.update(text_kwargs)
                ax.text(j, i, fmt.format(vals[i, j]), **props)
    return ax


def plot_scatter(ov, tf1: str, tf2: str, ax=None, value: str = "likelihood", label_top: int = 0, title: str | None = None):
    """Strength of the shared targets of two regulators: x = ``tf1``, y = ``tf2``.

    ``value="likelihood"`` (default): the likelihood (interaction confidence) of the two regulator-target interactions;
    ``value="mor"``: the mode of regulation, which is close to +-1 for most ARACNe targets (the points then pile up in the
    corners; the points are jittered slightly). Concordant targets (same sign of ``mor``) are blue, discordant ones red.
    ``label_top`` names that many targets with the highest product of the two likelihoods. The overlap statistics are added
    with :func:`add_scatter_stats`.
    """
    if value not in ("likelihood", "mor"):
        raise ValueError("value must be 'likelihood' or 'mor'")
    sh = ov.shared(tf1, tf2)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.2, 3.2))
    col = np.where(sh["concordant"], _CONCORDANT, np.where((sh["sign1"] == "0") | (sh["sign2"] == "0"), _ZERO, _DISCORDANT))
    x, y = sh[f"{value}1"].to_numpy(), sh[f"{value}2"].to_numpy()
    if value == "mor":
        jitter = np.random.default_rng(0).uniform(-0.012, 0.012, (len(sh), 2))
        x, y = x + jitter[:, 0], y + jitter[:, 1]
        ax.axhline(0, color="0.7", lw=0.5, ls=":")
        ax.axvline(0, color="0.7", lw=0.5, ls=":")
        lim = (-1.08, 1.08)
    else:
        top = float(max(np.max(x) if len(x) else 1, np.max(y) if len(y) else 1))
        lim = (-0.03 * top, top * 1.05)
    ax.scatter(x, y, s=9, c=col, alpha=0.7, lw=0)
    ax.plot(lim, lim, color="0.6", lw=0.5, ls=":")
    ax.set(xlim=lim, ylim=lim, xlabel=f"{value}, {tf1}", ylabel=f"{value}, {tf2}")
    ax.set_aspect("equal")
    ax.set_title(f"{len(sh)} shared targets" if title is None else title, loc="left", fontsize=8)
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", color=_CONCORDANT, ms=4, label="concordant"),
                       Line2D([0], [0], marker="o", ls="", color=_DISCORDANT, ms=4, label="discordant")],
              frameon=False, fontsize=6, loc="lower right", handletextpad=0.2)
    if label_top > 0 and len(sh):
        best = (sh["likelihood1"] * sh["likelihood2"]).sort_values(ascending=False).index[:label_top]
        pos = dict(zip(sh.index, zip(x, y)))
        for g in best:
            ax.annotate(g, pos[g], xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    return ax


def add_scatter_stats(ov, tf1: str, tf2: str, ax=None, loc: str = "upper left", **text_kwargs):
    """Overlap, expected number, fold and p (permutation if computed, else hypergeometric) of every tail combination of the
    pair, plus the sign agreement of the shared targets, as a small text box."""
    ax = plt.gca() if ax is None else ax
    t = ov.table
    swap = False
    rows = t[(t["tf1"] == tf1) & (t["tf2"] == tf2)]
    if rows.empty:
        rows = t[(t["tf1"] == tf2) & (t["tf2"] == tf1)]
        swap = True
    if rows.empty:
        raise ValueError(f"No comparison of {tf1!r} and {tf2!r} in the table")
    lines = []
    for _, r in rows.iterrows():
        s1, s2 = (r["sign2"], r["sign1"]) if swap else (r["sign1"], r["sign2"])
        p = r["p_perm"] if "p_perm" in r and np.isfinite(r.get("p_perm", np.nan)) else r["p_hyper"]
        pname = "p(perm)" if "p_perm" in r and np.isfinite(r.get("p_perm", np.nan)) else "p(hyper)"
        ptxt = "<0.001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"{s1}/{s2}: {int(r['overlap'])} of {int(r['n2'] if not swap else r['n1'])}, exp. {r['expected']:.1f}, "
                     f"{r['fold']:.1f}x, {pname} {ptxt}")
    pr_ = ov.pairs[((ov.pairs["tf1"] == tf1) & (ov.pairs["tf2"] == tf2)) | ((ov.pairs["tf1"] == tf2) & (ov.pairs["tf2"] == tf1))].iloc[0]
    lines.append(f"sign agreement {int(pr_['concordant'])}/{int(pr_['concordant'] + pr_['discordant'])}")
    vert, horiz = loc.split()
    props = dict(transform=ax.transAxes, ha=horiz, va="top" if vert == "upper" else "bottom", fontsize=5.5,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.4, alpha=0.9), zorder=5)
    props.update(text_kwargs)
    ax.text(0.03 if horiz == "left" else 0.97, 0.97 if vert == "upper" else 0.03, "\n".join(lines), **props)
    return ax


def _tail_sets(ov) -> dict:
    """Name -> set of genes for the tails of the analysis (``tail`` of the result)."""
    out = {}
    for tf in ov.tfs:
        m = ov.targets[tf]
        if ov.tail in ("positive", "both"):
            out[tf if ov.tail == "positive" else f"{tf}+"] = set(m.index[m > ov.mor_min])
        if ov.tail in ("negative", "both"):
            out[tf if ov.tail == "negative" else f"{tf}-"] = set(m.index[m < -ov.mor_min])
    return out


def plot_upset(ov, ax=None, sets=None, max_bars: int = 15, min_size: int = 1, title: str | None = None,
               colors: dict | None = None):
    """Intersections of the regulon tails (UpSet style) on one Axes: bars = number of targets that are in exactly these sets
    (membership dots below the axis). The tails are those of the analysis (``tail``); with ``tail="both"`` every regulator
    contributes a ``+`` and a ``-`` set. ``sets`` selects sets by name, ``max_bars`` / ``min_size`` limit the intersections,
    ``colors`` maps set names to colours. Counts: :func:`add_upset_counts`.
    """
    allsets = _tail_sets(ov)
    names = list(allsets) if sets is None else list(sets)
    for n in names:
        if n not in allsets:
            raise ValueError(f"Unknown set {n!r}; available: {list(allsets)}")
    genes = sorted(set().union(*[allsets[n] for n in names]))
    if not genes:
        raise ValueError("The selected sets are empty")
    patterns = {}
    for g in genes:
        key = tuple(g in allsets[n] for n in names)
        patterns[key] = patterns.get(key, 0) + 1
    ranked = sorted(((c, k) for k, c in patterns.items() if c >= min_size), reverse=True)[:max_bars]
    if not ranked:
        raise ValueError("No intersection reaches min_size")
    if ax is None:
        _, ax = plt.subplots(figsize=(0.55 * len(ranked) + 1.8, 2.6 + 0.2 * len(names)))
    palette = {n: _PALETTE[i % len(_PALETTE)] for i, n in enumerate(names)}
    palette.update(colors or {})
    top = max(c for c, _ in ranked)
    step = top * 0.14
    for i, (count, key) in enumerate(ranked):
        members = [j for j, m in enumerate(key) if m]
        ax.bar(i, count, width=0.6, color=palette[names[members[0]]] if len(members) == 1 else "0.3", lw=0)
        for j in range(len(names)):
            ax.plot(i, -(j + 1) * step, "o", ms=4.5, color="0.15" if key[j] else "0.85", zorder=3)
        if len(members) > 1:
            ax.plot([i, i], [-(members[0] + 1) * step, -(members[-1] + 1) * step], color="0.15", lw=1.2, zorder=2)
    for j, n in enumerate(names):
        ax.text(-0.75, -(j + 1) * step, f"{n}  ({len(allsets[n])})", ha="right", va="center", fontsize=6.5, color=palette[n],
                clip_on=False)
    ax.set_xlim(-0.6, len(ranked) - 0.4)
    ax.set_ylim(-(len(names) + 0.6) * step, top * 1.12)
    ax.set_xticks([])
    ax.set_yticks([t for t in ax.get_yticks() if 0 <= t <= top * 1.1])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_bounds(0, top * 1.1)
    ax.set_ylabel("targets", fontsize=7)
    ax.set_title("intersections of regulon tails" if title is None else title, loc="left", fontsize=8)
    ax._pyrea_upset = [(i, c) for i, (c, _) in enumerate(ranked)]
    return ax


def add_upset_counts(ov, ax=None, **text_kwargs):
    """Write the number of targets above each bar of :func:`plot_upset`."""
    ax = plt.gca() if ax is None else ax
    bars = getattr(ax, "_pyrea_upset", None)
    if bars is None:
        raise ValueError("Draw the intersections first (plot_upset) on this axes")
    props = dict(ha="center", va="bottom", fontsize=6)
    props.update(text_kwargs)
    for i, c in bars:
        ax.text(i, c, str(c), **props)
    return ax
