"""Figures for a :class:`pyrea.CoMutationResult` (co-mutation analysis).

Same design as :mod:`pyrea.phenotypes_plot`: the panels are clean, each is drawn on a matplotlib ``Axes`` that can be
passed (``ax=``) and returns it, and statistics are added afterwards, selectively, with the ``add_*`` functions
(``ax=None`` uses the current axes, ``**text_kwargs`` go to matplotlib).

=====================  ==============================  ====================================================
panel                  draw                            add statistics
=====================  ==============================  ====================================================
A score per cell       :func:`plot_cells`              :func:`add_cell_stats`
B interaction plot     :func:`plot_interaction`        :func:`add_interaction_stats`
C forest plot          :func:`plot_effects`            :func:`add_effect_p`
=====================  ==============================  ====================================================

``plot_comutation(res, stats=True)`` draws all three panels with all statistics.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, t as t_dist

from .comutation import _cliff

__all__ = ["plot_cells", "plot_interaction", "plot_effects", "plot_comutation", "cell_colors",
           "add_cell_stats", "add_interaction_stats", "add_effect_p"]

_CELL_COLORS = ["#9e9e9e", "#d55e00", "#0072b2", "#6a3d9a"]          # neither, D only, C only, D+C
_CONTRASTS = ["C within D", "C within not-D", "D within C", "D within not-C", "interaction C by D", "interaction D by C"]


def cell_colors(res, colors: dict | None = None) -> dict:
    """Colour of each cell label (neither grey, D only, C only, D+C); ``colors`` overrides single cells."""
    cols = dict(zip(res.cell_names, _CELL_COLORS))
    cols.update(colors or {})
    return cols


def _wrap(text, width: int = 10) -> str:
    text = str(text)
    if len(text) > width and "+" in text and " " not in text:           # "ERBB2amp+TP53" -> two lines at the plus
        head, tail = text.split("+", 1)
        return f"{head}+\n{tail}"
    return "\n".join(textwrap.wrap(text, width, break_long_words=False)) or text


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _title(res, extra: str = "") -> str:
    return f"{res.target}: {res.driver} x {res.partner}" + (f", {extra}" if extra else "")


def _check_score(res, score: str) -> None:
    if score not in res.scores.columns or score in ("cell",):
        raise ValueError(f"Unknown score {score!r}; available: {[c for c in ('G', 'RPT', 'expr') if c in res.scores.columns]}")


def _values(res, score: str, cell: str) -> np.ndarray:
    return res.scores.loc[res.scores["cell"] == cell, score].to_numpy(dtype=float)


def plot_cells(res, score: str = "G", ax=None, seed: int = 0, title: str | None = None, colors: dict | None = None):
    """Score of the target per cell (neither, D only, C only, D+C), one point per sample, boxplot behind.

    Statistics: :func:`add_cell_stats` (bracket between two cells).
    """
    _check_score(res, score)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 2.7))
    cols = cell_colors(res, colors)
    rng = np.random.default_rng(seed)
    for i, k in enumerate(res.cell_names):
        y = _values(res, score, k)
        if len(y) == 0:
            continue
        ax.boxplot([y], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor="0.3", lw=0.8), medianprops=dict(color="0.2", lw=1.2),
                   whiskerprops=dict(color="0.3", lw=0.8), capprops=dict(color="0.3", lw=0.8))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=max(4, min(18, 900 / len(y))), color=cols[k], alpha=0.75,
                   lw=0, zorder=3)
    ax.set_xticks(range(len(res.cell_names)),
                  [f"{_wrap(k)}\n(n={int(res.cells.loc[k, 'n'])})" for k in res.cell_names], fontsize=6.5)
    ax.set_ylabel(score)
    ax.set_title(_title(res, score) if title is None else title, loc="left", fontsize=8)
    return ax


def add_cell_stats(res, group_a: str, group_b: str, score: str = "G", ax=None, yoffset: float = 0.08,
                   exact_p: bool = False, line_kwargs: dict | None = None, **text_kwargs):
    """Bracket between two cells of :func:`plot_cells` with Cliff's delta (``group_a`` > ``group_b``) and the Mann-Whitney p.

    ``group_a`` / ``group_b`` are cell labels (``res.cell_names``). ``yoffset``: height above the higher cell as a fraction of
    the axis height.
    """
    ax = plt.gca() if ax is None else ax
    for g in (group_a, group_b):
        if g not in res.cell_names:
            raise ValueError(f"Unknown cell {g!r}; cells are {list(res.cell_names)}")
    a, b = _values(res, score, group_a), _values(res, score, group_b)
    if len(a) < 1 or len(b) < 1:
        raise ValueError("Both cells need at least one sample")
    delta = _cliff(a, b)
    p = float(mannwhitneyu(a, b).pvalue) if len(a) and len(b) else np.nan
    ia, ib = res.cell_names.index(group_a), res.cell_names.index(group_b)
    lo, hi = ax.get_ylim()
    height = hi - lo
    base = max(a.max(), b.max()) + yoffset * height
    lp = {"lw": 0.6, "color": "0.15"}
    lp.update(line_kwargs or {})
    tick = 0.02 * height
    ax.plot([ia, ia, ib, ib], [base - tick, base, base, base - tick], **lp)
    props = dict(fontsize=6, ha="center", va="bottom")
    props.update(text_kwargs)
    ptxt = f"p={p:1.1e}" if exact_p else _fmt_p(p)
    ax.text((ia + ib) / 2, base, f"delta {delta:.2f}, {ptxt}", **props)
    need = base + 0.12 * height
    if need > hi:
        ax.set_ylim(lo, need)
    return ax


def plot_interaction(res, score: str = "G", ax=None, title: str | None = None, colors: dict | None = None):
    """Interaction plot: mean of the score (``1 - alpha`` interval) of the four cells.

    x: driver absent / present; one line for the partner absent (grey) and one for the partner present. Parallel lines:
    no interaction (the partner shifts the score by the same amount with and without the driver); lines that diverge or cross:
    the partner changes what the driver does. The coefficient is added with :func:`add_interaction_stats`.
    """
    _check_score(res, score)
    if ax is None:
        _, ax = plt.subplots(figsize=(2.8, 2.7))
    cols = cell_colors(res, colors)
    nn, d_only, c_only, both = res.cell_names
    series = [(f"{res.partner} absent", (nn, d_only), cols[nn], -0.03), (f"{res.partner} present", (c_only, both), cols[both], 0.03)]
    for label, (c0, c1), col, shift in series:
        xs, ms, los, his = [], [], [], []
        for x, k in ((0, c0), (1, c1)):
            y = _values(res, score, k)
            if len(y) == 0:
                continue
            m = y.mean()
            half = t_dist.ppf(1 - res.alpha / 2, max(len(y) - 1, 1)) * (y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan)
            xs.append(x + shift), ms.append(m), los.append(m - half), his.append(m + half)
        ax.errorbar(xs, ms, yerr=[np.array(ms) - np.array(los), np.array(his) - np.array(ms)], color=col, marker="o", ms=4,
                    lw=1.2, capsize=2, label=label)
    ax.set_xticks([0, 1], [f"no {_wrap(res.driver, 8)}", _wrap(res.driver, 8)], fontsize=7)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(f"mean {score}")
    ax.set_title(_title(res, "interaction") if title is None else title, loc="left", fontsize=8)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    return ax


def add_interaction_stats(res, score: str = "G", ax=None, loc: str = "lower right", **text_kwargs):
    """Interaction coefficient ``D:C`` of the linear model with its interval and p-values (permutation and HC3) as a text box,
    plus the smallest interaction the design could detect. ``loc``: ``"upper left"``, ``"upper right"``, ``"lower left"`` or ``"lower right"``."""
    ax = plt.gca() if ax is None else ax
    term = f"{res.driver}:{res.partner}"
    if (score, term) not in res.model.index:
        raise ValueError(f"No model for score {score!r}")
    r = res.model.loc[(score, term)]
    lines = [f"interaction {term}", f"beta {r['beta']:.2f} [{r['low']:.2f}, {r['high']:.2f}]"]
    if np.isfinite(r["p_perm"]):
        lines.append(f"p(perm) {r['p_perm']:.3f}   p(HC3) {r['p_hc3']:.3f}")
    else:
        lines.append(f"p(HC3) {r['p_hc3']:.3f}")
    if np.isfinite(res.min_detectable_interaction):
        lines.append(f"detectable: >= {res.min_detectable_interaction:.1f} SD")
    vert, horiz = loc.split()
    props = dict(transform=ax.transAxes, ha=horiz, va="top" if vert == "upper" else "bottom", fontsize=5.5,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.4, alpha=0.9), zorder=5)
    props.update(text_kwargs)
    ax.text(0.03 if horiz == "left" else 0.97, 0.97 if vert == "upper" else 0.03, "\n".join(lines), **props)
    return ax


def plot_effects(res, ax=None, scores: list | None = None, contrasts: list | None = None, title: str | None = None):
    """Forest plot of the rank-based effects: Cliff's delta with its bootstrap interval, one row per contrast and score.

    Filled markers: simple effects (colour by contrast), diamonds: interactions. The vertical line is 0. ``scores``
    (``"G"``, ``"RPT"``, ``"expr"``) and ``contrasts`` (the index level of ``res.effects``) select the rows; the p-values
    of the simple effects are added with :func:`add_effect_p`.
    """
    avail = list(dict.fromkeys(res.effects.index.get_level_values("score")))
    scores = avail if scores is None else list(scores)
    for s in scores:
        if s not in avail:
            raise ValueError(f"Unknown score {s!r}; available: {avail}")
    contrasts = [c for c in _CONTRASTS if c in res.effects.index.get_level_values("contrast")] if contrasts is None else list(contrasts)
    rows = [(s, c) for c in contrasts for s in scores if (s, c) in res.effects.index]
    if ax is None:
        _, ax = plt.subplots(figsize=(3.8, 0.9 + 0.2 * len(rows)))
    palette = {"C within D": _CELL_COLORS[3], "C within not-D": _CELL_COLORS[2], "D within C": _CELL_COLORS[3], "D within not-C": _CELL_COLORS[1]}
    labels, pvals = [], []
    for y, (s, c) in enumerate(rows):
        r = res.effects.loc[(s, c)]
        inter = c.startswith("interaction")
        col = "0.15" if inter else palette.get(c, "0.4")
        if np.isfinite(r["delta"]):
            if np.isfinite(r["low"]):
                ax.hlines(y, r["low"], r["high"], color=col, lw=1.4)
            ax.plot(r["delta"], y, "D" if inter else "o", ms=4.5 if inter else 5, color=col, zorder=3)
        label = c.replace("C within", f"{res.partner} within").replace("D within", f"{res.driver} within") \
                 .replace("interaction C by D", f"interaction: {res.partner} by {res.driver}") \
                 .replace("interaction D by C", f"interaction: {res.driver} by {res.partner}")
        labels.append(f"{label} | {s}")
        pvals.append(np.nan if inter else r["p"])
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("Cliff's delta" + (" (95 % bootstrap interval)" if res.n_boot else ""), fontsize=7)
    ax.set_title(_title(res, "effects") if title is None else title, loc="left", fontsize=8)
    ax._pyrea_pvals = pvals
    return ax


def add_effect_p(res, ax=None, exact_p: bool = False, **text_kwargs):
    """Mann-Whitney p of every simple effect of :func:`plot_effects` at the right edge (blank for interactions)."""
    ax = plt.gca() if ax is None else ax
    pvals = getattr(ax, "_pyrea_pvals", None)
    if pvals is None:
        raise ValueError("Draw the forest plot first (plot_effects) on this axes")
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(len(pvals)), [(f"p={p:1.1e}" if exact_p else _fmt_p(p)) if np.isfinite(p) else "" for p in pvals])
    right.tick_params(length=0, labelsize=text_kwargs.pop("fontsize", 6), **text_kwargs)
    for sp in right.spines.values():
        sp.set_visible(False)
    return ax


def plot_comutation(res, score: str = "G", figsize=None, axes=None, stats: bool = False, colors: dict | None = None,
                    scores: list | None = None):
    """Three-panel overview: score per cell, interaction plot, forest plot of the effects.

    Parameters
    ----------
    axes : sequence of 3 Axes, optional
        Draw on these axes instead of a new figure; the list of axes is returned (otherwise the figure).
    stats : bool
        Add all statistics: bracket D only vs. D+C in A, the interaction coefficient in B, the p-values in C.
    scores : list of str, optional
        Rows of the forest plot (default: only ``score``; ``["G", "RPT", "expr"]`` shows all).
    """
    fig = None
    if axes is None:
        fig = plt.figure(figsize=figsize or (11.0, 3.3), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.8, 1.5])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes = list(axes)
    if len(axes) != 3:
        raise ValueError(f"Need 3 axes, got {len(axes)}")
    plot_cells(res, score, ax=axes[0], title=f"{score} of {res.target}", colors=colors)
    plot_interaction(res, score, ax=axes[1], title="interaction plot", colors=colors)
    plot_effects(res, ax=axes[2], scores=[score] if scores is None else scores, title="effects (Cliff's delta)")
    if stats:
        nn, d_only, c_only, both = res.cell_names
        add_cell_stats(res, both, d_only, score, ax=axes[0])          # D+C vs. D only: delta > 0 = higher with the partner
        add_interaction_stats(res, score, ax=axes[1])
        add_effect_p(res, ax=axes[2])
    for ax, letter in zip(axes, "ABC"):
        ax.annotate(letter, (-0.02, 1.13), xycoords="axes fraction", fontsize=10, fontweight="bold")
    if fig is None:
        return axes
    adj = f"; adjusted for {', '.join(res.covariates)}" if res.covariates else ""
    fig.suptitle(f"{res.target}: effect of {res.partner} with and without {res.driver}{adj}", fontsize=8, x=0.01, ha="left")
    return fig
