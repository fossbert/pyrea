"""Figures for a :class:`pyrea.PhenotypeComparison` (phenocopy of alteration groups).

The functions take the result of :func:`pyrea.compare_phenotypes`; they are also available as methods
of the result (``cmp.plot()``, ``cmp.plot_cross()``, ...). matplotlib only.

Every panel function draws on a matplotlib ``Axes`` that can be passed (``ax=``) and returns it, so axes,
limits, labels and further artists can be changed afterwards. The panels are clean: statistics are added
afterwards, selectively, with the ``add_*`` functions (``ax=None`` uses the current axes, the Axes is returned,
``**text_kwargs`` go to matplotlib), as in ``cbrviz.StripBox.add_pair_p``:

=====================  ==============================  ====================================================
panel                  draw                            add statistics
=====================  ==============================  ====================================================
A scores per group     :func:`plot_scores`             :func:`add_group_stats`, :func:`add_pair_stats`
B samples, two scales  :func:`plot_scale_scatter`      :func:`add_scatter_stats`
C cross-scoring        :func:`plot_cross`              :func:`add_cross_stats`
D forest plot          :func:`plot_auc`                :func:`add_auc_p`
=====================  ==============================  ====================================================

``plot_phenocopy(cmp, stats=True)`` draws all four panels with all statistics (for a figure that goes into a
manuscript). ``colors`` maps group names to colours.
"""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

__all__ = ["plot_scores", "plot_scale_scatter", "plot_cross", "plot_auc", "plot_phenocopy", "group_colors",
           "add_group_stats", "add_pair_stats", "add_scatter_stats", "add_cross_stats", "add_auc_p"]

_REF_COLOR = "#9e9e9e"
_PALETTE = ["#d55e00", "#0072b2", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]


def group_colors(cmp, colors: dict | None = None) -> dict:
    """Colour of every group (reference grey); the same in all panels. ``colors`` overrides single groups."""
    cols = {cmp.reference: _REF_COLOR}
    cols.update({g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(cmp.groups)})
    cols.update(colors or {})
    return cols


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.1 else f"p={p:.2f}"


def _title(cmp, extra: str = "") -> str:
    return f"{cmp.target} ({cmp.method})" + (f": {extra}" if extra else "")


def _wrap(text: str, width: int = 11) -> str:
    return "\n".join(textwrap.wrap(str(text), width, break_long_words=False)) or str(text)


def _label(cmp, g) -> str:
    n = int((cmp.scores["group"] == g).sum())
    return f"{_wrap(g)}\n(n={n})"


def _pick_groups(cmp, groups) -> list:
    """The groups to show (default: all, in the order of the comparison); validated."""
    names = list(cmp.groups) if groups is None else list(groups)
    for g in names:
        _check_group(cmp, g)
    return names


def _check_group(cmp, g) -> None:
    if g not in cmp.groups:
        raise ValueError(f"Unknown group {g!r}; groups are {list(cmp.groups)}")


def _pair_row(cmp, score: str, a: str, b: str):
    """Row of ``cmp.pairwise`` for the pair (either order) and whether it is stored as (a, b)."""
    if (score, a, b) in cmp.pairwise.index:
        return cmp.pairwise.loc[(score, a, b)], True
    if (score, b, a) in cmp.pairwise.index:
        return cmp.pairwise.loc[(score, b, a)], False
    raise ValueError(f"No pairwise comparison of {a!r} and {b!r} for {score!r}")


def plot_scores(cmp, score: str = "G", ax=None, seed: int = 0, title: str | None = None, colors: dict | None = None,
                groups: list | None = None, reference: bool = True):
    """Distribution of one score (``"G"``, ``"RPT"`` or ``"MPS[<group>]"``) per group, one point per sample.

    Parameters
    ----------
    score : str
        Column of ``cmp.scores``.
    ax : matplotlib Axes, optional
        Draw here (otherwise a new figure); the Axes is returned.
    colors : dict, optional
        Group name -> colour.
    groups : list of str, optional
        Which groups to show and in which order (default: all).
    reference : bool
        Show the reference group (default) or only the alteration groups.

    Statistics are added with :func:`add_group_stats` (AUC / interval / p of every group against the reference)
    and :func:`add_pair_stats` (bracket for two groups).
    """
    if score not in cmp.scores.columns:
        raise ValueError(f"Unknown score {score!r}; available: {[c for c in cmp.scores.columns if c != 'group']}")
    shown = _pick_groups(cmp, groups)
    order = ([cmp.reference] if reference else []) + shown
    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 + 0.9 * len(order), 2.6))
    cols = group_colors(cmp, colors)
    rng = np.random.default_rng(seed)
    for i, g in enumerate(order):
        y = cmp.scores.loc[cmp.scores["group"] == g, score].dropna().to_numpy(dtype=float)
        if len(y) == 0:
            continue
        ax.boxplot([y], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="none", edgecolor="0.3", lw=0.8), medianprops=dict(color="0.2", lw=1.2),
                   whiskerprops=dict(color="0.3", lw=0.8), capprops=dict(color="0.3", lw=0.8))
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=max(4, min(18, 900 / len(y))), color=cols[g],
                   alpha=0.75, lw=0, zorder=3)
    ax.set_xticks(range(len(order)), [_label(cmp, g) for g in order], fontsize=7)
    ax.set_ylabel(score)
    ax.set_title(_title(cmp, score) if title is None else title, loc="left", fontsize=8)
    ax._pyrea_order = order
    return ax


def _group_top(cmp, score: str, g) -> float:
    y = cmp.scores.loc[cmp.scores["group"] == g, score].dropna().to_numpy(dtype=float)
    return float(np.max(y)) if len(y) else np.nan


def _raise_top(ax, y: float, margin: float = 0.04) -> None:
    lo, hi = ax.get_ylim()
    need = y + margin * (hi - lo)
    if need > hi:
        ax.set_ylim(lo, need)


def add_group_stats(cmp, score: str = "G", ax=None, ci: bool = True, colors: dict | None = None,
                    exact_p: bool = False, **text_kwargs):
    """Write AUC (with its bootstrap interval if ``ci``) and the Mann-Whitney p of every group against the reference above its box.

    Use after :func:`plot_scores` with the same ``score``; the text sits just above the highest point of each group
    (offset in points), the upper y limit is raised if needed. ``exact_p=True`` prints ``p`` as ``1.2e-03``.
    """
    ax = plt.gca() if ax is None else ax
    cols = group_colors(cmp, colors)
    props = dict(fontsize=6, ha="center", va="bottom", linespacing=1.1)
    props.update(text_kwargs)
    color = props.pop("color", None)                      # one colour for all groups instead of the group colours
    order = getattr(ax, "_pyrea_order", [cmp.reference] + list(cmp.groups))
    top = -np.inf
    for i, g in enumerate(order):
        if g == cmp.reference or (g, score) not in cmp.summary.index:
            continue
        r = cmp.summary.loc[(g, score)]
        y = _group_top(cmp, score, g)
        if not (np.isfinite(r["auc"]) and np.isfinite(y)):
            continue
        lines = [f"AUC {r['auc']:.2f}"]
        if ci and np.isfinite(r.get("auc_low", np.nan)):
            lines.append(f"[{r['auc_low']:.2f}, {r['auc_high']:.2f}]")
        lines.append(f"p={r['p']:1.1e}" if exact_p else _fmt_p(r["p"]))
        ax.annotate("\n".join(lines), xy=(i, y), xytext=(0, 3), textcoords="offset points",
                    color=color or cols[g], **props)
        top = max(top, y)
    if np.isfinite(top):
        lo, hi = ax.get_ylim()
        _raise_top(ax, top + 0.30 * (hi - lo), margin=0.0)
    return ax


def add_pair_stats(cmp, group_a: str, group_b: str, score: str = "G", ax=None, yoffset: float = 0.08,
                   auc: bool = True, exact_p: bool = False, line_kwargs: dict | None = None, **text_kwargs):
    """Bracket between two groups with the AUC and the Mann-Whitney p of their comparison (``cmp.pairwise``).

    Use after :func:`plot_scores`. The bracket stands ``yoffset`` (fraction of the axis height) above the higher of the
    two groups; raise it when :func:`add_group_stats` texts are shown. ``auc=False`` prints the p only, ``exact_p=True``
    prints ``1.2e-03``. The AUC is for ``group_a`` > ``group_b``.
    """
    ax = plt.gca() if ax is None else ax
    for g in (group_a, group_b):
        _check_group(cmp, g)
    r, forward = _pair_row(cmp, score, group_a, group_b)
    a_val = r["auc"] if forward else 1 - r["auc"]
    order = getattr(ax, "_pyrea_order", [cmp.reference] + list(cmp.groups))
    if group_a not in order or group_b not in order:
        raise ValueError(f"{group_a!r} and {group_b!r} must both be shown in the panel (shown: {order})")
    ia, ib = order.index(group_a), order.index(group_b)
    lo, hi = ax.get_ylim()
    height = hi - lo
    base = np.nanmax([_group_top(cmp, score, group_a), _group_top(cmp, score, group_b)]) + yoffset * height
    lp = {"lw": 0.6, "color": "0.15"}
    lp.update(line_kwargs or {})
    tick = 0.02 * height
    ax.plot([ia, ia, ib, ib], [base - tick, base, base, base - tick], **lp)
    props = dict(fontsize=6, ha="center", va="bottom")
    props.update(text_kwargs)
    p = f"p={r['p']:1.1e}" if exact_p else _fmt_p(r["p"])
    ax.text((ia + ib) / 2, base, f"AUC {a_val:.2f}, {p}" if auc else p, **props)
    _raise_top(ax, base + 0.12 * height, margin=0.0)
    return ax


def plot_scale_scatter(cmp, scale_a: str | None = None, scale_b: str | None = None, ax=None,
                       title: str | None = None, colors: dict | None = None, groups: list | None = None):
    """Every sample on the MPS scale of group A (x) and of group B (y).

    A phenocopy of A by B puts the B samples with the A samples (upper right); the dotted lines mark the
    mutant / WT phenotype thresholds (likelihood ratio ``lr``) and the diagonal equal scores. The AUC of every group
    on the two scales is added with :func:`add_scatter_stats`. ``scale_a`` / ``scale_b`` are the groups whose MPS
    scales are the axes (default: the first two); ``groups`` restricts the groups that are drawn (default: all).
    """
    if len(cmp.groups) < 2:
        raise ValueError("Need at least two groups for a scale scatter")
    scale_a = cmp.groups[0] if scale_a is None else scale_a
    scale_b = cmp.groups[1] if scale_b is None else scale_b
    _check_group(cmp, scale_a), _check_group(cmp, scale_b)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 3.0))
    cols = group_colors(cmp, colors)
    sc = cmp.scores
    from .mps import lr_to_rl
    t = lr_to_rl(cmp.lr)
    for g in [cmp.reference] + _pick_groups(cmp, groups):
        s = sc[sc["group"] == g]
        ax.scatter(s[f"MPS[{scale_a}]"], s[f"MPS[{scale_b}]"], s=14 if g != cmp.reference else 7, color=cols[g],
                   alpha=0.8 if g != cmp.reference else 0.35, lw=0, label=f"{g} (n={len(s)})",
                   zorder=3 if g != cmp.reference else 2)
    ax.plot([-1, 1], [-1, 1], color="0.5", lw=0.6, ls=":")
    for v in (t, -t):
        ax.axvline(v, color="0.7", lw=0.5, ls=":")
        ax.axhline(v, color="0.7", lw=0.5, ls=":")
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel=f"RL, scale of {scale_a}", ylabel=f"RL, scale of {scale_b}")
    ax.set_title(_title(cmp, "samples on both scales") if title is None else title, loc="left", fontsize=8)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.2)
    ax._pyrea_scales = (scale_a, scale_b)
    return ax


def add_scatter_stats(cmp, ax=None, scale_a: str | None = None, scale_b: str | None = None, loc: str = "upper left",
                      **text_kwargs):
    """AUC (against the reference) of every group on the two scales of :func:`plot_scale_scatter`, as a small text box.

    ``scale_a`` / ``scale_b`` default to the scales of the scatter drawn on ``ax``. ``loc``: ``"upper left"``,
    ``"upper right"``, ``"lower left"`` or ``"lower right"`` (the legend is lower right by default).
    """
    ax = plt.gca() if ax is None else ax
    default = getattr(ax, "_pyrea_scales", tuple(cmp.groups[:2]))
    scale_a = default[0] if scale_a is None else scale_a
    scale_b = default[1] if scale_b is None else scale_b
    lines = []
    for axis, scale in (("x", scale_a), ("y", scale_b)):
        parts = [f"{g} {cmp.cross.loc[(scale, g), 'auc']:.2f}" for g in cmp.groups
                 if (scale, g) in cmp.cross.index and np.isfinite(cmp.cross.loc[(scale, g), "auc"])]
        lines.append(f"AUC on {axis}: " + ", ".join(parts))
    vert, horiz = loc.split()
    props = dict(transform=ax.transAxes, ha=horiz, va="top" if vert == "upper" else "bottom", fontsize=5.5,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", lw=0.4, alpha=0.9), zorder=5)
    props.update(text_kwargs)
    ax.text(0.02 if horiz == "left" else 0.98, 0.98 if vert == "upper" else 0.02, "\n".join(lines), **props)
    return ax


def plot_cross(cmp, ax=None, cmap: str = "RdBu_r", title: str | None = None, groups: list | None = None):
    """Matrix of the cross scoring (``cmp.cross``): AUC against the reference, with its interval and the share of samples with the mutant phenotype.

    Rows: the MPS scale defined by a group; columns: the group that is scored on it. Cells with a dashed frame are in-sample
    (scale and group are the same samples: optimistic benchmark). Phenocopy: the off-diagonal AUC is above 0.5
    with an interval clear of 0.5, in both directions. The numbers in the cells are added with :func:`add_cross_stats`.
    ``groups`` restricts the matrix to some groups (rows: their scales, columns: the groups scored; default: all).
    """
    names = _pick_groups(cmp, groups)
    if ax is None:
        _, ax = plt.subplots(figsize=(1.6 + 1.5 * len(names), 1.2 + 0.9 * len(names)))
    auc = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if (a, b) in cmp.cross.index:
                auc[i, j] = cmp.cross.loc[(a, b), "auc"]
    im = ax.imshow(np.ma.masked_invalid(auc), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            r = cmp.cross.loc[(a, b)] if (a, b) in cmp.cross.index else None
            if r is None or not np.isfinite(r["auc"]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.4")
                continue
            if a == b:
                ax.add_patch(Rectangle((j - 0.46, i - 0.46), 0.92, 0.92, fill=False, edgecolor="0.15", lw=1.3, ls="--"))
    ax._pyrea_cross = names
    ax.set_xticks(range(len(names)), [_label(cmp, b) for b in names], fontsize=7)
    ax.set_yticks(range(len(names)), [_wrap(f"scale of {a}", 12) for a in names], fontsize=7)
    ax.set_xlabel("group scored (dashed frame: in-sample)", fontsize=7)
    ax.set_title(_title(cmp, "AUC vs. reference on each scale") if title is None else title, loc="left", fontsize=8)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("AUC (0.5 = no difference)", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    return ax


def add_cross_stats(cmp, ax=None, ci: bool = True, share: bool = True, **text_kwargs):
    """Write the AUC (with its interval if ``ci``) and the share of samples with the mutant phenotype (``share``) into the cells of :func:`plot_cross`."""
    ax = plt.gca() if ax is None else ax
    names = getattr(ax, "_pyrea_cross", list(cmp.groups))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if (a, b) not in cmp.cross.index or not np.isfinite(cmp.cross.loc[(a, b), "auc"]):
                continue
            r = cmp.cross.loc[(a, b)]
            txt = f"{r['auc']:.2f}"
            if ci and np.isfinite(r["auc_low"]):
                txt += f"\n[{r['auc_low']:.2f}, {r['auc_high']:.2f}]"
            if share and np.isfinite(r["frac_mutant_phenotype"]):
                txt += f"\n{r['frac_mutant_phenotype']:.0%} mutant"
            props = dict(ha="center", va="center", fontsize=7, color="white" if abs(r["auc"] - 0.5) > 0.3 else "black")
            props.update(text_kwargs)
            ax.text(j, i, txt, **props)
    return ax


def plot_auc(cmp, ax=None, scores: list | None = None, pairwise: bool = False, title: str | None = None,
             colors: dict | None = None, groups: list | None = None):
    """Forest plot of the AUC with its bootstrap interval, one row per group and score.

    Filled markers: the group was not used to build the scale (G, RPT, or another group's MPS scale);
    hollow: in-sample. The vertical line is 0.5 (no difference from the reference). ``pairwise=True`` adds
    the group-against-group comparisons (grey). The Mann-Whitney p of every row is added with :func:`add_auc_p`.

    ``scores`` selects the rows by score (``"G"``, ``"RPT"``, ``"MPS[<group>]"``; see ``cmp.score_names``), in the order
    given; ``groups`` restricts the groups (default: all).
    """
    s = cmp.summary.drop(index=cmp.reference, level=0, errors="ignore")
    avail = list(dict.fromkeys(s.index.get_level_values("score")))
    scores = avail if scores is None else list(scores)
    for sc in scores:
        if sc not in avail:
            raise ValueError(f"Unknown score {sc!r}; available: {avail}")
    shown = _pick_groups(cmp, groups)
    rows = [(g, sc) for sc in scores for g in shown if (g, sc) in s.index and np.isfinite(s.loc[(g, sc), "auc"])]
    prow = []
    if pairwise and len(cmp.pairwise):
        prow = [(a, b, sc) for sc in scores for (sc2, a, b) in cmp.pairwise.index
                if sc2 == sc and a in shown and b in shown and np.isfinite(cmp.pairwise.loc[(sc, a, b), "auc"])]
    n = len(rows) + len(prow)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.6, 0.9 + 0.22 * n))
    cols = group_colors(cmp, colors)
    y = 0
    labels, pvals = [], []
    for g, sc in rows:
        r = s.loc[(g, sc)]
        insample = sc == f"MPS[{g}]"
        if np.isfinite(r.get("auc_low", np.nan)):
            ax.hlines(y, r["auc_low"], r["auc_high"], color=cols[g], lw=1.4)
        ax.plot(r["auc"], y, "o", ms=5, mfc="white" if insample else cols[g], mec=cols[g], mew=1.2, zorder=3)
        labels.append(f"{g} | {sc}")
        pvals.append(r["p"])
        y += 1
    for a, b, sc in prow:
        r = cmp.pairwise.loc[(sc, a, b)]
        if np.isfinite(r.get("auc_low", np.nan)):
            ax.hlines(y, r["auc_low"], r["auc_high"], color="0.5", lw=1.2)
        ax.plot(r["auc"], y, "s", ms=4, color="0.4", zorder=3)
        labels.append(f"{a} vs {b} | {sc}")
        pvals.append(r["p"])
        y += 1
    ax.axvline(0.5, color="0.4", lw=0.7, ls="--")
    ax.set_yticks(range(len(labels)), labels, fontsize=6.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUC (95 % bootstrap interval)" if cmp.n_boot else "AUC", fontsize=7)
    ax.set_title(_title(cmp, "AUC vs. reference") if title is None else title, loc="left", fontsize=8)
    ax._pyrea_pvals = pvals
    return ax


def add_auc_p(cmp, ax=None, exact_p: bool = False, **text_kwargs):
    """Write the Mann-Whitney p of every row of :func:`plot_auc` at the right edge (as tick labels of a twin axis)."""
    ax = plt.gca() if ax is None else ax
    pvals = getattr(ax, "_pyrea_pvals", None)
    if pvals is None:
        raise ValueError("Draw the forest plot first (plot_auc) on this axes")
    right = ax.twinx()
    right.set_ylim(ax.get_ylim())
    right.set_yticks(range(len(pvals)), [f"p={p:1.1e}" if exact_p else _fmt_p(p) for p in pvals])
    right.tick_params(length=0, labelsize=text_kwargs.pop("fontsize", 6), **text_kwargs)
    for sp in right.spines.values():
        sp.set_visible(False)
    return ax


def plot_phenocopy(cmp, scale_a: str | None = None, scale_b: str | None = None, score: str = "G", figsize=None,
                   axes=None, stats: bool = False, colors: dict | None = None, pairs: list | None = None,
                   scores: list | None = None, groups: list | None = None):
    """Four-panel overview: scores per group, samples on both scales, AUC matrix, AUC forest plot.

    With a single group the scatter and the matrix are left out (two panels).

    Parameters
    ----------
    axes : sequence of Axes, optional
        Draw the panels on these axes (4, or 2 with a single group) instead of a new figure; then the
        list of axes is returned, otherwise the figure.
    stats : bool
        Add all statistics (``add_group_stats`` and a bracket for the first two groups, ``add_scatter_stats``,
        ``add_cross_stats``, ``add_auc_p``). Default: clean panels; add selectively with the ``add_*`` functions.
    colors : dict, optional
        Group name -> colour.
    pairs : list of (group, group), optional
        Brackets for panel A when ``stats``; default: the first two groups shown.
    scores : list of str, optional
        Rows of the forest plot (panel D), e.g. ``["G", "MPS[RHOA mut]"]`` (default: all scores).
    groups : list of str, optional
        Groups to show in all panels (default: all).
    """
    shown = _pick_groups(cmp, groups)
    two = len(shown) >= 2
    need = 4 if two else 2
    fig = None
    if axes is None:
        if figsize is None:
            figsize = (10.0, 6.6) if two else (7.2, 3.0)
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        if two:
            gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        else:
            gs = fig.add_gridspec(1, 2)
            axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes = list(axes)
    if len(axes) != need:
        raise ValueError(f"Need {need} axes for {len(shown)} group(s), got {len(axes)}")
    if pairs is None and two:
        pairs = [tuple(shown[:2])]
    if two:
        scale_a = shown[0] if scale_a is None else scale_a
        scale_b = shown[1] if scale_b is None else scale_b
        plot_scores(cmp, score, ax=axes[0], title=f"{score} activity of {cmp.target}", colors=colors, groups=shown)
        plot_scale_scatter(cmp, scale_a, scale_b, ax=axes[1], title="samples on both MPS scales", colors=colors, groups=shown)
        plot_cross(cmp, ax=axes[2], title="cross scoring: AUC vs. reference", groups=shown)
        plot_auc(cmp, ax=axes[3], pairwise=True, title="AUC with bootstrap interval", colors=colors, scores=scores, groups=shown)
        if stats:
            add_group_stats(cmp, score, ax=axes[0], colors=colors)
            for a, b in pairs or []:
                add_pair_stats(cmp, a, b, score, ax=axes[0], yoffset=0.34)
            add_scatter_stats(cmp, ax=axes[1])
            add_cross_stats(cmp, ax=axes[2])
            add_auc_p(cmp, ax=axes[3])
        for ax, letter in zip(axes, "ABCD"):
            ax.annotate(letter, (-0.02, 1.13), xycoords="axes fraction", fontsize=10, fontweight="bold")
    else:
        plot_scores(cmp, score, ax=axes[0], title=f"{score} activity of {cmp.target}", colors=colors, groups=shown)
        plot_auc(cmp, ax=axes[1], title="AUC with bootstrap interval", colors=colors, scores=scores, groups=shown)
        if stats:
            add_group_stats(cmp, score, ax=axes[0], colors=colors)
            add_auc_p(cmp, ax=axes[1])
    if fig is None:
        return axes
    fig.suptitle(f"{cmp.target}: phenocopy of {', '.join(shown)}  (RL from {cmp.method}; "
                 f"reference = {cmp.reference})", fontsize=8, x=0.01, ha="left")
    return fig
