"""Does the overlap of two regulons drive the correlation of the regulators' activities?

The activity of a regulator is inferred from its targets. If two regulators share many targets, their activities correlate
simply because the same genes enter both. :func:`split_activity` therefore infers the activity of each regulator from three
parts of its regulon and correlates the parts of the two regulators across samples:

- ``full``: the whole regulon,
- ``shared``: the targets that the other regulator also has (in the chosen tail combination),
- ``unique``: the targets that the other regulator does **not** have at all.

Correlations between parts that contain the same genes (``target_overlap > 0``) are circular by construction (flagged
``circular``). A further caveat: VIPER ranks the genes within each sample (here over all genes of the expression matrix,
``eset_filter=False``), so when many genes shift strongly together (e.g. a large shared block), the ranks of all other genes shift with
them and parts without shared genes can correlate for that reason alone; the random-subset null (``n_random``) shows how far that carries. The informative cells are those without shared genes, above all ``unique`` x ``unique`` (do the regulators vary
together beyond the shared targets?) and ``shared`` x ``unique`` (do the shared targets behave like the other regulator's own
targets?). Needs R (``viper``).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

__all__ = ["split_activity", "ActivitySplit"]

_PARTS = ("full", "shared", "unique")


@dataclass
class ActivitySplit:
    """Result of :func:`split_activity`.

    Attributes
    ----------
    activity : pd.DataFrame
        Activity (VIPER NES) of every part, ``"<TF>|<part>"`` x samples.
    correlation : pd.DataFrame
        One row per part of ``tf1`` and part of ``tf2``: ``part1``, ``part2`` (full / shared / unique), ``n1``, ``n2`` (targets
        measured), ``target_overlap`` (genes in both parts), ``circular`` (``target_overlap > 0``), Spearman ``rho`` and ``p``.
    parts : pd.DataFrame
        Per regulator and part: ``n_regulon`` (targets in the regulon), ``n_measured`` (present in the expression matrix) and
        ``in_activity`` (the part had enough targets for VIPER).
    expression_corr : float
        Spearman correlation of the two regulators' own expression (NaN if not in the matrix).
    null : pd.DataFrame or None
        With ``n_random``: Spearman correlation of random subsets of one regulator's regulon (same size as the shared part) with the
        *unique* part of the other (columns ``"<TF1>|shared vs <TF2>|unique"`` and the reverse).
    tf1, tf2, signs, method, minsize, n_random, seed :
        The settings.

    Methods
    -------
    ``matrix()``, ``null_summary()``: tables; ``plot_correlations()``, ``plot_scatter()``: figures on an Axes that can be passed;
    ``add_correlation_values()``, ``add_scatter_stats()``: statistics added afterwards.
    """

    activity: pd.DataFrame
    correlation: pd.DataFrame
    parts: pd.DataFrame
    expression_corr: float
    null: Optional[pd.DataFrame]
    tf1: str
    tf2: str
    signs: tuple
    method: str
    minsize: int
    n_random: int
    seed: int

    def __repr__(self) -> str:
        return (f"ActivitySplit({self.tf1} x {self.tf2}, shared tails {self.signs[0]}/{self.signs[1]}, "
                f"{len(self.correlation)} correlations, expression rho {self.expression_corr:.2f})")

    def matrix(self, value: str = "rho") -> pd.DataFrame:
        """Parts of ``tf1`` (rows) x parts of ``tf2`` (columns); ``value``: rho, p, target_overlap or n1 / n2."""
        if value not in ("rho", "p", "target_overlap", "n1", "n2"):
            raise ValueError("value must be rho, p, target_overlap, n1 or n2")
        return self.correlation.pivot(index="part1", columns="part2", values=value).reindex(index=list(_PARTS), columns=list(_PARTS))

    def null_summary(self) -> pd.DataFrame:
        """Observed correlation of the shared part with the other regulator's unique part against the random subsets: ``observed``,
        ``null_mean``, ``null_sd`` and the one-sided ``p`` (share of subsets at least as high, with +1)."""
        if self.null is None:
            raise ValueError("No null was computed (n_random=0)")
        rows = {}
        for col in self.null.columns:
            a, b = col.split(" vs ")
            (t_a, _), (t_b, _) = a.split("|"), b.split("|")
            if t_a == self.tf1:
                obs = self.correlation.query("part1 == 'shared' and part2 == 'unique'")["rho"].iloc[0]
            else:
                obs = self.correlation.query("part1 == 'unique' and part2 == 'shared'")["rho"].iloc[0]
            v = self.null[col].dropna().to_numpy()
            rows[col] = {"observed": obs, "null_mean": v.mean(), "null_sd": v.std(ddof=1), "n_random": len(v),
                         "p": (1 + np.sum(v >= obs)) / (len(v) + 1)}
        return pd.DataFrame(rows).T

    # figures
    def plot_correlations(self, ax=None, cmap: str = "RdBu_r", title: Optional[str] = None):
        """Matrix of the Spearman correlations between the parts of ``tf1`` (rows) and ``tf2`` (columns). Cells that share genes
        (circular) are hatched; read the others. The numbers are added with :meth:`add_correlation_values`."""
        mat, ov_ = self.matrix("rho"), self.matrix("target_overlap")
        if ax is None:
            _, ax = plt.subplots(figsize=(3.6, 2.8))
        lim = max(0.3, float(np.nanmax(np.abs(mat.to_numpy()))))
        im = ax.imshow(np.ma.masked_invalid(mat.to_numpy()), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for i, a in enumerate(_PARTS):
            for j, b in enumerate(_PARTS):
                if np.isfinite(ov_.loc[a, b]) and ov_.loc[a, b] > 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor="0.35", lw=0))
        ax.set_xticks(range(3), [f"{self.tf2}\n{p}" for p in _PARTS], fontsize=7)
        ax.set_yticks(range(3), [f"{self.tf1} {p}" for p in _PARTS], fontsize=7)
        ax.set_title(f"activity correlation (hatched: shared genes)" if title is None else title, loc="left", fontsize=8)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("Spearman rho", fontsize=6)
        cb.ax.tick_params(labelsize=6)
        ax._pyrea_matrix = mat
        return ax

    def add_correlation_values(self, ax=None, fmt: str = "{:.2f}", **text_kwargs):
        """After ``plot_correlations``: write rho into the cells."""
        ax = plt.gca() if ax is None else ax
        mat = getattr(ax, "_pyrea_matrix", None)
        if mat is None:
            raise ValueError("Draw the matrix first (plot_correlations) on this axes")
        vals = mat.to_numpy()
        lim = max(0.3, float(np.nanmax(np.abs(vals))))
        for i in range(3):
            for j in range(3):
                if np.isfinite(vals[i, j]):
                    props = dict(ha="center", va="center", fontsize=7, color="white" if abs(vals[i, j]) > 0.6 * lim else "black")
                    props.update(text_kwargs)
                    ax.text(j, i, fmt.format(vals[i, j]), **props)
        return ax

    def plot_scatter(self, part1: str = "unique", part2: str = "unique", ax=None, title: Optional[str] = None):
        """Activity of ``tf1`` from ``part1`` against ``tf2`` from ``part2``, one point per sample."""
        for p in (part1, part2):
            if p not in _PARTS:
                raise ValueError(f"part must be one of {_PARTS}")
        a, b = f"{self.tf1}|{part1}", f"{self.tf2}|{part2}"
        if a not in self.activity.index or b not in self.activity.index:
            raise ValueError(f"No activity for {a!r} / {b!r} (too few targets for VIPER?); see .parts")
        if ax is None:
            _, ax = plt.subplots(figsize=(3.0, 3.0))
        ax.scatter(self.activity.loc[a], self.activity.loc[b], s=8, color="#1f77b4", alpha=0.6, lw=0)
        ax.set(xlabel=f"{self.tf1}, {part1} targets", ylabel=f"{self.tf2}, {part2} targets")
        ax.set_title(f"{part1} x {part2}" if title is None else title, loc="left", fontsize=8)
        return ax

    def add_scatter_stats(self, part1: str = "unique", part2: str = "unique", ax=None, loc: str = "upper left", **text_kwargs):
        """After ``plot_scatter``: Spearman rho, p and whether the two parts share genes, as a small text box."""
        ax = plt.gca() if ax is None else ax
        r = self.correlation[(self.correlation["part1"] == part1) & (self.correlation["part2"] == part2)]
        if r.empty:
            raise ValueError("No correlation for this pair of parts")
        r = r.iloc[0]
        p = "<0.001" if r["p"] < 0.001 else f"{r['p']:.3f}"
        lines = [f"rho {r['rho']:.2f}  p {p}", f"shared genes: {int(r['target_overlap'])}" + ("  (circular)" if r["circular"] else "")]
        vert, horiz = loc.split()
        props = dict(transform=ax.transAxes, ha=horiz, va="top" if vert == "upper" else "bottom", fontsize=6,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.4, alpha=0.9), zorder=5)
        props.update(text_kwargs)
        ax.text(0.03 if horiz == "left" else 0.97, 0.97 if vert == "upper" else 0.03, "\n".join(lines), **props)
        return ax


def split_activity(ov, expression: pd.DataFrame, tf1: str, tf2: str, signs: Optional[tuple] = None, method: str = "mad",
                   minsize: int = 25, n_random: int = 0, seed: int = 0, cores: int = 1) -> ActivitySplit:
    """Activity of two regulators from the full, shared and own parts of their regulons, and the correlations between the parts.

    Parameters
    ----------
    ov : RegulonOverlap
        Result of :func:`pyrea.regulon_overlap` containing both regulators (its sign threshold ``mor_min`` and likelihood filter apply).
    expression : pd.DataFrame
        Gene expression, genes x samples (the matrix the activity is inferred from).
    tf1, tf2 : str
        The two regulators.
    signs : tuple of two str, optional
        Tail combination of the *shared* part: targets with the sign ``signs[0]`` in ``tf1`` and ``signs[1]`` in ``tf2``
        (default ``("+", "+")``; ``("-", "-")`` for ``tail="negative"``). Targets the regulators share in other combinations belong to
        neither ``shared`` nor ``unique``.
    method, minsize, cores :
        As in :func:`pyrea.viper_activity` (parts with fewer than ``minsize`` measured targets get no activity, see ``parts``).
    n_random : int
        Random subsets of one regulator's regulon with the size of the shared part, correlated with the other's unique part
        (``null``, ``null_summary()``): do the shared targets correlate with the other regulator beyond what any targets of the
        first would?
    seed : int

    Returns
    -------
    ActivitySplit
    """
    from scipy.stats import rankdata  # noqa: F401  (kept for symmetry with other modules)
    from .rbackend.viper import viper_activity

    for t in (tf1, tf2):
        if t not in ov.tfs:
            raise ValueError(f"Unknown regulator {t!r}; analysed: {list(ov.tfs)}")
    if tf1 == tf2:
        raise ValueError("Need two different regulators")
    if signs is None:
        signs = ("-", "-") if ov.tail == "negative" else ("+", "+")
    s1, s2 = signs
    rng = np.random.default_rng(seed)
    expr_genes = pd.Index(expression.index.astype(str))

    mor = {t: ov.targets[t].dropna() for t in (tf1, tf2)}
    lik = {t: ov.likelihoods[t].reindex(mor[t].index) for t in (tf1, tf2)}
    in_tail = lambda m, s: (m > ov.mor_min) if s == "+" else (m < -ov.mor_min)
    shared = mor[tf1].index[in_tail(mor[tf1], s1)].intersection(mor[tf2].index[in_tail(mor[tf2], s2)])
    genes = {(tf1, "full"): mor[tf1].index, (tf2, "full"): mor[tf2].index,
             (tf1, "shared"): shared, (tf2, "shared"): shared,
             (tf1, "unique"): mor[tf1].index.difference(mor[tf2].index), (tf2, "unique"): mor[tf2].index.difference(mor[tf1].index)}

    def frame(name: str, tf: str, idx: pd.Index) -> pd.DataFrame:
        return pd.DataFrame({"source": name, "target": idx, "mor": mor[tf].reindex(idx).to_numpy(), "likelihood": lik[tf].reindex(idx).to_numpy()})

    frames, prow = [], []
    for (tf, part), idx in genes.items():
        frames.append(frame(f"{tf}|{part}", tf, idx))
        prow.append({"tf": tf, "part": part, "n_regulon": len(idx), "n_measured": int(idx.isin(expr_genes).sum())})
    random_names = {}
    if n_random > 0:
        k = int(genes[(tf1, "shared")].isin(expr_genes).sum())
        if k >= 2:
            for tf, other in ((tf1, tf2), (tf2, tf1)):
                pool = mor[tf].index[mor[tf].index.isin(expr_genes)]
                for i in range(n_random):
                    name = f"{tf}|random{i}"
                    frames.append(frame(name, tf, pd.Index(rng.choice(pool, k, replace=False))))
                random_names[tf] = [f"{tf}|random{i}" for i in range(n_random)]
    # eset_filter=False: rank all genes of the matrix, as in an analysis with the full network. The default filter would restrict the
    # signature to the targets of the few regulons given here, and the ranks of the parts would then depend on each other
    act = viper_activity(expression, pd.concat(frames, ignore_index=True), method=method, minsize=minsize, eset_filter=False, cores=cores)
    parts = pd.DataFrame(prow)
    parts["in_activity"] = [f"{t}|{p}" in act.index for t, p in zip(parts["tf"], parts["part"])]
    if not parts["in_activity"].all():
        import warnings
        missing = [f"{t}|{p}" for t, p, ok in zip(parts["tf"], parts["part"], parts["in_activity"]) if not ok]
        warnings.warn(f"No activity for {missing}: fewer than minsize={minsize} targets in the expression matrix", UserWarning, stacklevel=2)

    def rho(a: str, b: str):
        r, p = spearmanr(act.loc[a], act.loc[b])
        return float(r), float(p)

    crow = []
    for p1, p2 in product(_PARTS, _PARTS):
        a, b = f"{tf1}|{p1}", f"{tf2}|{p2}"
        overlap = int(len(set(genes[(tf1, p1)][genes[(tf1, p1)].isin(expr_genes)]) & set(genes[(tf2, p2)][genes[(tf2, p2)].isin(expr_genes)])))
        r, p = rho(a, b) if a in act.index and b in act.index else (np.nan, np.nan)
        crow.append({"part1": p1, "part2": p2, "n1": int(genes[(tf1, p1)].isin(expr_genes).sum()), "n2": int(genes[(tf2, p2)].isin(expr_genes).sum()),
                     "target_overlap": overlap, "circular": overlap > 0, "rho": r, "p": p})
    null = None
    if random_names:
        cols = {}
        if f"{tf2}|unique" in act.index:
            cols[f"{tf1}|shared vs {tf2}|unique"] = [rho(n, f"{tf2}|unique")[0] for n in random_names[tf1] if n in act.index]
        if f"{tf1}|unique" in act.index:
            cols[f"{tf2}|shared vs {tf1}|unique"] = [rho(n, f"{tf1}|unique")[0] for n in random_names[tf2] if n in act.index]
        null = pd.DataFrame({k: pd.Series(v) for k, v in cols.items()})
    act = act.loc[[i for i in act.index if "|random" not in i]]
    e1 = expression.loc[tf1] if tf1 in expression.index else None
    e2 = expression.loc[tf2] if tf2 in expression.index else None
    ecorr = float(spearmanr(e1, e2)[0]) if e1 is not None and e2 is not None else np.nan
    return ActivitySplit(activity=act, correlation=pd.DataFrame(crow), parts=parts, expression_corr=ecorr, null=null, tf1=tf1, tf2=tf2,
                         signs=(s1, s2), method=method, minsize=int(minsize), n_random=int(n_random), seed=int(seed))
