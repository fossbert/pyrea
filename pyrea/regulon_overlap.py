"""Overlap of the target sets (regulons) of several regulators, separated by the sign of the mode of regulation.

A regulon lists the targets of a regulator with a mode of regulation ``mor`` in [-1, 1] (``tfmode`` in ``viper``): > 0
activated, < 0 repressed targets. The overlap of two regulators is therefore measured per tail: the positive tails
(``+``/``+``: both activate), the negative tails (``-``/``-``: both repress) or, with ``tail="both"``, all four sign
combinations including the discordant ones (``+``/``-``: one activates what the other represses).

Each overlap is compared with two nulls:

- the hypergeometric distribution (``p_hyper``): the sets are random draws from the universe of targets;
- a degree-weighted permutation (``p_perm``, optional): the second set is redrawn with the same size, targets chosen with
  a probability proportional to one plus the number of the other regulators that have them in the same tail. ARACNe
  regulons share hub targets, which the hypergeometric null ignores, so ``p_hyper`` overstates the significance of
  overlaps between regulators that draw on the same hubs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, hypergeom, spearmanr

__all__ = ["regulon_overlap", "RegulonOverlap"]

_TAILS = ("positive", "negative", "both")
_SIGNS = {"positive": [("+", "+")], "negative": [("-", "-")],
          "both": [("+", "+"), ("-", "-"), ("+", "-"), ("-", "+")]}
_METRICS = ("overlap", "expected", "fold", "jaccard", "overlap_coef", "odds_ratio")


def _shared_frame(targets: pd.DataFrame, likelihoods: pd.DataFrame, tf1: str, tf2: str) -> pd.DataFrame:
    """Targets of both regulators with ``mor``, ``likelihood``, signs and concordance (index = gene)."""
    m = targets[[tf1, tf2]].dropna()
    lk = likelihoods.loc[m.index]
    out = pd.DataFrame({"mor1": m[tf1], "mor2": m[tf2], "likelihood1": lk[tf1], "likelihood2": lk[tf2]})
    sgn = lambda v: np.where(v > 0, "+", np.where(v < 0, "-", "0"))
    out["sign1"], out["sign2"] = sgn(out["mor1"]), sgn(out["mor2"])
    out["concordant"] = (out["sign1"] == out["sign2"]) & (out["sign1"] != "0")
    return out.sort_values("likelihood1", ascending=False)


@dataclass
class RegulonOverlap:
    """Result of :func:`regulon_overlap`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per regulator pair and sign combination (``sign1``, ``sign2``: ``"+"`` / ``"-"`` = tail of ``tf1`` /
        ``tf2``): ``n1``, ``n2`` (targets in the tails), ``overlap``, ``expected`` (hypergeometric), ``fold`` (overlap /
        expected), ``jaccard``, ``overlap_coef`` (overlap / smaller set), ``odds_ratio`` (with 0.5 added to the cells),
        ``p_hyper`` (one-sided enrichment) with BH ``q_hyper`` and, if ``n_perm`` > 0, the degree-weighted permutation
        ``perm_mean``, ``p_perm`` and ``q_perm``.
    pairs : pd.DataFrame
        One row per regulator pair, independent of ``tail``: targets of each regulator, ``shared`` targets (any
        sign), ``concordant`` / ``discordant`` (same / opposite sign of ``mor``), ``agreement`` (concordant / shared)
        and the Spearman correlation ``mor_corr`` of the two ``mor`` vectors over the shared targets.
    sizes : pd.DataFrame
        Per regulator: ``n_targets``, ``n_positive``, ``n_negative``.
    targets, likelihoods : pd.DataFrame
        Targets (rows) x regulators: ``mor`` and ``likelihood``; NaN where the gene is not a target. Targets of at
        least one regulator only.
    tfs, tail, mor_min, min_likelihood, universe_size, n_perm, seed :
        The settings of the analysis.
    shared_targets : pd.DataFrame
        **All shared targets of all pairs in one table** (long format, for export): ``tf1``, ``tf2``, ``gene``, ``mor1``,
        ``mor2``, ``likelihood1``, ``likelihood2``, ``sign1``, ``sign2``, ``concordant`` and ``in_tail`` (the sign combination is
        one of those of ``tail``), strongest first within a pair.
    universe_genes : pd.Index
        The genes that could be targets (used for the enrichment).

    Methods
    -------
    ``shared()``, ``matrix()``: tables. Collecting and exporting: ``gene_sets()``, ``core()``, ``collect()``,
    ``export()``; ``enrich()`` (over-representation of gene sets in the shared targets); ``split_activity()`` (activity of a
    regulator from its shared and its own targets). Figures on an Axes that can be passed: ``plot_heatmap()``,
    ``plot_scatter()``, ``plot_upset()``; statistics added afterwards: ``add_heatmap_values()``, ``add_scatter_stats()``,
    ``add_upset_counts()``; see :mod:`pyrea.regulon_overlap_plot`.
    """

    table: pd.DataFrame
    pairs: pd.DataFrame
    sizes: pd.DataFrame
    targets: pd.DataFrame
    likelihoods: pd.DataFrame
    tfs: tuple
    tail: str
    mor_min: float
    min_likelihood: float
    universe_size: int
    n_perm: int
    seed: int
    shared_targets: Optional[pd.DataFrame] = None
    universe_genes: Optional[pd.Index] = None

    def __repr__(self) -> str:
        return (f"RegulonOverlap(tfs={list(self.tfs)}, tail={self.tail!r}, universe={self.universe_size} targets, "
                f"{len(self.pairs)} pair(s), {len(self.table)} comparison(s))")

    def shared(self, tf1: str, tf2: str, signs: Optional[tuple] = None) -> pd.DataFrame:
        """Targets of both regulators with their ``mor`` and ``likelihood``.

        Columns ``mor1``, ``mor2``, ``likelihood1``, ``likelihood2``, ``sign1``, ``sign2`` (``"+"`` / ``"-"`` /
        ``"0"``) and ``concordant`` (same non-zero sign). ``signs=("+", "+")`` keeps the targets in that tail
        combination only.
        """
        for t in (tf1, tf2):
            if t not in self.tfs:
                raise ValueError(f"Unknown regulator {t!r}; analysed: {list(self.tfs)}")
        out = _shared_frame(self.targets, self.likelihoods, tf1, tf2)
        if signs is not None:
            out = out[(out["sign1"] == signs[0]) & (out["sign2"] == signs[1])]
        return out

    def matrix(self, metric: str = "jaccard", signs: tuple = ("+", "+")) -> pd.DataFrame:
        """Regulator x regulator matrix of one ``metric`` of ``table`` for one sign combination.

        Rows: the tail ``signs[0]`` of the row regulator; columns: the tail ``signs[1]`` of the column regulator. The
        diagonal is NaN. ``metric``: overlap, expected, fold, jaccard, overlap_coef, odds_ratio, p_hyper, q_hyper, p_perm, q_perm.
        """
        if metric not in self.table.columns:
            raise ValueError(f"Unknown metric {metric!r}; available: {[c for c in self.table.columns if c not in ('tf1', 'tf2', 'sign1', 'sign2', 'n1', 'n2')]}")
        if tuple(signs) not in [(a, b) for a, b in zip(self.table["sign1"], self.table["sign2"])]:
            raise ValueError(f"Sign combination {signs} not in the table (tail={self.tail!r})")
        mat = pd.DataFrame(np.nan, index=list(self.tfs), columns=list(self.tfs), dtype=float)
        for _, r in self.table.iterrows():
            if (r["sign1"], r["sign2"]) == tuple(signs):
                mat.loc[r["tf1"], r["tf2"]] = r[metric]
            if (r["sign2"], r["sign1"]) == tuple(signs):
                mat.loc[r["tf2"], r["tf1"]] = r[metric]
        return mat

    # ------------------------------------------------------------------ collecting, enrichment, export
    def _gene_set_info(self, signs: Optional[tuple] = None) -> dict:
        info = {}
        for _, r in self.table.iterrows():
            if signs is not None and (r["sign1"], r["sign2"]) != tuple(signs):
                continue
            if r["overlap"] <= 0:
                continue
            st = self.shared_targets
            sel = st[(st["tf1"] == r["tf1"]) & (st["tf2"] == r["tf2"]) & (st["sign1"] == r["sign1"]) & (st["sign2"] == r["sign2"])]
            rec = {"genes": list(sel["gene"]), "tfs": f"{r['tf1']},{r['tf2']}", "signs": f"{r['sign1']}/{r['sign2']}",
                   "tail": self.tail, "overlap": int(r["overlap"]), "expected": float(r["expected"]), "fold": float(r["fold"]),
                   "p_hyper": float(r["p_hyper"]), "universe_size": self.universe_size, "mor_min": self.mor_min,
                   "min_likelihood": self.min_likelihood}
            if "p_perm" in r.index:
                rec["p_perm"] = float(r["p_perm"])
            info[f"{r['tf1']}&{r['tf2']} ({r['sign1']}/{r['sign2']})"] = rec
        return info

    def gene_sets(self, signs: Optional[tuple] = None) -> dict:
        """The shared targets as gene lists: ``"TF1&TF2 (+/+)"`` -> genes, one per pair and tail combination of ``table``
        with at least one shared target (``signs=("+", "+")`` keeps one combination). Strongest interaction first."""
        return {k: v["genes"] for k, v in self._gene_set_info(signs).items()}

    def core(self, sign: str = "+", tfs: Optional[Sequence[str]] = None) -> list:
        """Targets that are in the ``sign`` tail (``"+"`` or ``"-"``) of **all** the given regulators (default: all analysed)."""
        if sign not in ("+", "-"):
            raise ValueError("sign must be '+' or '-'")
        tfs = list(self.tfs) if tfs is None else list(tfs)
        for t in tfs:
            if t not in self.tfs:
                raise ValueError(f"Unknown regulator {t!r}; analysed: {list(self.tfs)}")
        m = self.targets[tfs]
        keep = (m > self.mor_min).all(axis=1) if sign == "+" else (m < -self.mor_min).all(axis=1)
        return sorted(m.index[keep])

    def collect(self, into=None, label: Optional[str] = None, signs: Optional[tuple] = None):
        """Add the shared-target lists (:meth:`gene_sets`, plus the :meth:`core` of three or more regulators for the tails of
        the analysis) to a :class:`pyrea.GeneListCollection` for later export; ``label`` (e.g. the network) prefixes the names
        and is stored as ``source``. Returns the collection (a new one if ``into`` is not given)."""
        from .genelists import GeneListCollection
        col = GeneListCollection() if into is None else into
        col.add_overlap(self, label=label, signs=signs)
        if len(self.tfs) > 2 and signs is None:
            for sign in {"positive": "+", "negative": "-"}.get(self.tail, "+-"):
                genes = self.core(sign)
                if genes:
                    name = f"{'&'.join(self.tfs)} core ({sign})"
                    col.add(f"{label}: {name}" if label else name, genes, source=label or "", tfs=",".join(self.tfs), signs=sign,
                            tail=self.tail, overlap=len(genes), universe_size=self.universe_size, mor_min=self.mor_min)
        return col

    def enrich(self, genesets, signs: Optional[tuple] = None, universe: Optional[Iterable[str]] = None, min_size: int = 5,
               max_size: int = 500, min_overlap: int = 2, include_core: bool = True, species: str = "human") -> pd.DataFrame:
        """Over-representation of gene sets in the shared targets of every pair and tail combination.

        Parameters
        ----------
        genesets : str, sequence of str or mapping
            **The name of a bundled collection** (``"h"`` Hallmark, ``"reactome"``, ``"pid"``, ``"wikipathways"``; a list combines
            several; ``pr.available_genesets()`` lists them) or a mapping term -> genes of your own.
        signs : tuple, optional
            Test one tail combination only (e.g. ``("+", "+")``).
        universe : iterable of str, optional
            Default: the genes that could be targets of the network (``universe_genes``).
        min_size, max_size, min_overlap :
            Term size limits and the smallest overlap that is reported (see :func:`pyrea.gene_set_enrichment`).
        include_core : bool
            With three or more regulators also test the :meth:`core` of the tails.
        species : {"human", "mouse"}
            For the bundled collections.

        Each list of :meth:`gene_sets` (and the core) is tested with :func:`pyrea.gene_set_enrichment`. Returns the concatenated table
        with the column ``target_set`` first (``q`` is Benjamini-Hochberg within a ``target_set``) and, for bundled collections, ``collection``.
        """
        from .genelists import gene_set_enrichment, resolve_genesets
        resolved, coll = resolve_genesets(genesets, species)          # load the bundled collections once
        uni = self.universe_genes if universe is None else universe
        lists = dict(self.gene_sets(signs))
        if include_core and len(self.tfs) > 2 and signs is None:
            for sign in {"positive": "+", "negative": "-"}.get(self.tail, "+-"):
                genes = self.core(sign)
                if genes:
                    lists[f"{'&'.join(self.tfs)} core ({sign})"] = genes
        frames = []
        for name, genes in lists.items():
            e = gene_set_enrichment(genes, resolved, universe=uni, min_size=min_size, max_size=max_size, min_overlap=min_overlap)
            if coll is not None:
                e.insert(1, "collection", e["term"].map(coll))
            e.insert(0, "target_set", name)
            frames.append(e)
        cols = ["target_set", "term"] + (["collection"] if coll is not None else []) + ["n_genes", "n_term", "overlap", "expected", "fold", "p", "q", "genes"]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)

    def split_activity(self, expression: pd.DataFrame, tf1: str, tf2: str, signs: Optional[tuple] = None, method: str = "mad",
                       minsize: int = 25, n_random: int = 0, seed: int = 0):
        """Activity of two regulators from their full, shared and own targets, and how these activities correlate (needs R).

        See :func:`pyrea.split_activity` (returns an :class:`pyrea.ActivitySplit`)."""
        from .regulon_activity import split_activity
        return split_activity(self, expression, tf1, tf2, signs, method, minsize, n_random, seed)

    def export(self, path: Union[str, Path], collection=None, enrichment: Optional[pd.DataFrame] = None, split=None) -> Path:
        """Write the result to one Excel workbook (``.xlsx``), a GMT file (``.gmt``, gene lists only) or a folder of CSV files.

        Sheets / files: ``settings``, ``table``, ``pairs``, ``sizes``, ``shared_targets`` (all shared targets of all pairs, long),
        ``lists`` and ``genes`` (the gene lists of ``collection``, default ``self.collect()``), and, if given, ``enrichment`` (from
        :meth:`enrich`) and ``activity_correlation`` (the ``correlation`` table of an :class:`pyrea.ActivitySplit`).
        """
        col = self.collect() if collection is None else collection
        extra = {"settings": pd.DataFrame([{"tfs": ",".join(self.tfs), "tail": self.tail, "mor_min": self.mor_min,
                                            "min_likelihood": self.min_likelihood, "universe_size": self.universe_size,
                                            "n_perm": self.n_perm, "seed": self.seed}]),
                 "table": self.table, "pairs": self.pairs, "sizes": self.sizes, "shared_targets": self.shared_targets}
        if enrichment is not None:
            extra["enrichment"] = enrichment
        if split is not None:
            extra["activity_correlation"] = split.correlation
        return col.export(path, extra=extra)

    # figures: see pyrea.regulon_overlap_plot
    def plot_heatmap(self, metric: str = "jaccard", signs: tuple = ("+", "+"), ax=None, cmap: str = "Blues",
                     title: Optional[str] = None):
        """Regulator x regulator heatmap of one metric for one tail combination."""
        from .regulon_overlap_plot import plot_heatmap
        return plot_heatmap(self, metric, signs, ax, cmap, title)

    def plot_scatter(self, tf1: str, tf2: str, ax=None, value: str = "likelihood", label_top: int = 0,
                     title: Optional[str] = None):
        """Likelihood (or ``mor``) of the shared targets of two regulators (concordant blue, discordant red)."""
        from .regulon_overlap_plot import plot_scatter
        return plot_scatter(self, tf1, tf2, ax, value, label_top, title)

    def plot_upset(self, ax=None, sets: Optional[Sequence[str]] = None, max_bars: int = 15, min_size: int = 1,
                   title: Optional[str] = None, colors: Optional[dict] = None):
        """Intersections of the tails of several regulators (UpSet style: bars above, membership dots below)."""
        from .regulon_overlap_plot import plot_upset
        return plot_upset(self, ax, sets, max_bars, min_size, title, colors)

    def add_heatmap_values(self, ax=None, fmt: str = "{:.2f}", **text_kwargs):
        """After ``plot_heatmap``: write the values into the cells."""
        from .regulon_overlap_plot import add_heatmap_values
        return add_heatmap_values(self, ax, fmt, **text_kwargs)

    def add_scatter_stats(self, tf1: str, tf2: str, ax=None, loc: str = "upper left", **text_kwargs):
        """After ``plot_scatter``: overlap, expected, fold and p per tail combination and the sign agreement as a text box."""
        from .regulon_overlap_plot import add_scatter_stats
        return add_scatter_stats(self, tf1, tf2, ax, loc, **text_kwargs)

    def add_upset_counts(self, ax=None, **text_kwargs):
        """After ``plot_upset``: the number of targets above each bar."""
        from .regulon_overlap_plot import add_upset_counts
        return add_upset_counts(self, ax, **text_kwargs)


def _weighted_overlap_null(rng, weights: np.ndarray, fixed_mask: np.ndarray, size: int, n_perm: int) -> np.ndarray:
    """Overlap with ``fixed_mask`` of ``n_perm`` random sets of ``size`` genes drawn without replacement with probability
    proportional to ``weights`` (Gumbel top-k / Efraimidis-Spirakis)."""
    n = len(weights)
    pos = weights > 0
    logw = np.full(n, -np.inf)
    logw[pos] = np.log(weights[pos])
    out = np.empty(n_perm, dtype=int)
    chunk = max(1, min(n_perm, 4_000_000 // max(n, 1)))
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        keys = -np.log(-np.log(rng.random((b, n)))) + logw            # Gumbel-max trick: top-k of log w + Gumbel noise
        idx = np.argpartition(-keys, size - 1, axis=1)[:, :size]
        out[done:done + b] = fixed_mask[idx].sum(axis=1)
        done += b
    return out


def regulon_overlap(regulon: pd.DataFrame, tfs: Sequence[str], tail: str = "positive", mor_min: float = 0.0,
                    min_likelihood: float = 0.0, universe: Optional[Iterable[str]] = None, n_perm: int = 0,
                    seed: int = 0) -> RegulonOverlap:
    """Overlap of the regulons of several regulators, separated by the sign of the mode of regulation.

    Parameters
    ----------
    regulon : pd.DataFrame
        Long format as returned by :func:`pyrea.read_regulon_rds`: ``source`` (regulator), ``target``, ``mor`` (mode of
        regulation in [-1, 1]) and optionally ``likelihood`` (default 1).
    tfs : sequence of str
        Regulators to compare (at least two); every pair is analysed.
    tail : {"positive", "negative", "both"}
        Which tails of the regulons are overlapped. ``"positive"`` (default): the activated targets of both
        regulators (``mor > mor_min``); ``"negative"``: the repressed targets (``mor < -mor_min``); ``"both"``: all four
        combinations (``+``/``+``, ``-``/``-`` and the discordant ``+``/``-`` and ``-``/``+``).
    mor_min : float
        A target is positive if ``mor > mor_min`` and negative if ``mor < -mor_min``; the default 0 uses the sign only.
    min_likelihood : float
        Targets with a lower ``likelihood`` are dropped first.
    universe : iterable of str, optional
        Genes that can be targets (e.g. the genes of the expression matrix); default: all targets of ``regulon``. Targets
        outside are ignored. Enters the hypergeometric null and the permutation null.
    n_perm : int
        Degree-weighted permutations per comparison for ``p_perm`` (0: only the hypergeometric p-value).
    seed : int

    Returns
    -------
    RegulonOverlap
        See there for the tables (``table`` per pair and tail combination, ``pairs``, ``sizes``, ``targets``).

    Notes
    -----
    The p-values are one-sided (enrichment) and the q-values Benjamini-Hochberg over all rows of ``table``. The
    hypergeometric p-value ignores that ARACNe regulons share hub targets and then overstates significance; use
    ``n_perm`` for the degree-weighted null, in which the second regulator's tail is redrawn with probability
    proportional to one plus the number of the *other* regulators having each gene in the same tail (the two compared
    regulators are left out of the count). The smallest possible ``p_perm`` is ``1 / (n_perm + 1)``.

    Examples
    --------
    ::

        reg = pr.read_regulon_rds("FullReg.rds")
        ov = pr.regulon_overlap(reg, ["ZCCHC24", "ZEB1", "ZEB2"], tail="both", n_perm=2000)
        ov.table[["tf1", "tf2", "sign1", "sign2", "n1", "n2", "overlap", "expected", "p_hyper", "p_perm"]]
        ov.shared("ZCCHC24", "ZEB1", signs=("+", "+"))
        ov.plot_heatmap(signs=("+", "+"))
    """
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}")
    tfs = [str(t) for t in tfs]
    if len(tfs) < 2:
        raise ValueError("Need at least two regulators")
    if len(set(tfs)) != len(tfs):
        raise ValueError("Regulators must be unique")
    missing = [t for t in missing_names(regulon, tfs)]
    if missing:
        raise ValueError(f"Regulators not in the regulon: {missing}")
    need = {"source", "target", "mor"}
    if not need <= set(regulon.columns):
        raise ValueError(f"regulon needs the columns {sorted(need)} (long format, see read_regulon_rds)")
    df = regulon[[c for c in ("source", "target", "mor", "likelihood") if c in regulon.columns]].copy()
    if "likelihood" not in df:
        df["likelihood"] = 1.0
    df["source"], df["target"] = df["source"].astype(str), df["target"].astype(str)
    if df.duplicated(["source", "target"]).any():
        raise ValueError("The regulon has duplicated source-target pairs")

    if universe is None:
        genes = pd.Index(sorted(df["target"].unique()))
    else:
        genes = pd.Index(sorted(set(map(str, universe))))
        df = df[df["target"].isin(genes)]
    n_universe = len(genes)
    df = df[df["likelihood"] >= min_likelihood]
    gene_idx = pd.Series(np.arange(n_universe), index=genes)

    def tail_set(sub: pd.DataFrame, sign: str) -> np.ndarray:
        sel = sub["mor"] > mor_min if sign == "+" else sub["mor"] < -mor_min
        return gene_idx.reindex(sub.loc[sel, "target"]).dropna().to_numpy(dtype=int)

    deg = {s: np.zeros(n_universe) for s in "+-"}
    for s in "+-":
        sel = df["mor"] > mor_min if s == "+" else df["mor"] < -mor_min
        counts = df.loc[sel].groupby("target").size()
        deg[s][gene_idx.reindex(counts.index).to_numpy(dtype=int)] = counts.to_numpy()

    by_tf = {t: df[df["source"] == t] for t in tfs}
    sets = {(t, s): tail_set(by_tf[t], s) for t in tfs for s in "+-"}
    member = {}
    for key, idx in sets.items():
        mk = np.zeros(n_universe)
        mk[idx] = 1.0
        member[key] = mk
    sizes = pd.DataFrame({t: {"n_targets": len(by_tf[t]), "n_positive": len(sets[(t, "+")]), "n_negative": len(sets[(t, "-")])}
                          for t in tfs}).T
    rng = np.random.default_rng(seed)

    rows = []
    for t1, t2 in combinations(tfs, 2):
        for s1, s2 in _SIGNS[tail]:
            a, b = sets[(t1, s1)], sets[(t2, s2)]
            n1, n2 = len(a), len(b)
            mask = np.zeros(n_universe, dtype=bool)
            mask[a] = True
            k = int(mask[b].sum())
            exp = n1 * n2 / n_universe
            union = n1 + n2 - k
            rec = {"tf1": t1, "tf2": t2, "sign1": s1, "sign2": s2, "n1": n1, "n2": n2, "overlap": k, "expected": exp,
                   "fold": k / exp if exp > 0 else np.nan, "jaccard": k / union if union else np.nan,
                   "overlap_coef": k / min(n1, n2) if min(n1, n2) else np.nan,
                   "odds_ratio": ((k + .5) * (n_universe - n1 - n2 + k + .5)) / ((n1 - k + .5) * (n2 - k + .5)),
                   "p_hyper": float(hypergeom.sf(k - 1, n_universe, n1, n2)) if n1 and n2 else np.nan}
            if n_perm > 0:
                if n1 and n2:
                    # degree of every gene among the OTHER regulators (the two compared ones are left out, otherwise their own
                    # targets weigh more and the null overlaps too much); +1 keeps every gene drawable
                    weights = deg[s2] - member[(t1, s2)] - member[(t2, s2)] + 1.0
                    null = _weighted_overlap_null(rng, weights, mask, n2, n_perm)
                    rec["perm_mean"] = float(null.mean())
                    rec["p_perm"] = float((1 + np.sum(null >= k)) / (n_perm + 1))
                else:
                    rec["perm_mean"], rec["p_perm"] = np.nan, np.nan
            rows.append(rec)
    table = pd.DataFrame(rows)
    for p in [c for c in ("p_hyper", "p_perm") if c in table.columns]:
        q = pd.Series(np.nan, index=table.index)
        ok = table[p].notna()
        if ok.sum() > 1:
            q[ok] = false_discovery_control(table.loc[ok, p].to_numpy(), method="bh")
        elif ok.sum() == 1:
            q[ok] = table.loc[ok, p]
        table["q_" + p[2:]] = q
    cols = ["tf1", "tf2", "sign1", "sign2", "n1", "n2", "overlap", "expected", "fold", "jaccard", "overlap_coef",
            "odds_ratio", "p_hyper", "q_hyper"] + (["perm_mean", "p_perm", "q_perm"] if n_perm > 0 else [])
    table = table[cols]

    mor = pd.DataFrame({t: by_tf[t].set_index("target")["mor"] for t in tfs})
    lik = pd.DataFrame({t: by_tf[t].set_index("target")["likelihood"] for t in tfs})
    targets = mor.dropna(how="all").sort_index()
    likelihoods = lik.loc[targets.index]

    prow = []
    for t1, t2 in combinations(tfs, 2):
        both = targets[[t1, t2]].dropna()
        nz = both[(both[t1] != 0) & (both[t2] != 0)]
        conc = int((np.sign(nz[t1]) == np.sign(nz[t2])).sum())
        corr = float(spearmanr(nz[t1], nz[t2])[0]) if len(nz) > 2 and nz[t1].nunique() > 1 and nz[t2].nunique() > 1 else np.nan
        prow.append({"tf1": t1, "tf2": t2, "n_targets1": len(by_tf[t1]), "n_targets2": len(by_tf[t2]), "shared": len(both),
                     "concordant": conc, "discordant": len(nz) - conc, "agreement": conc / len(nz) if len(nz) else np.nan,
                     "mor_corr": corr})
    comb = {(s1, s2) for s1, s2 in _SIGNS[tail]}
    frames = []
    for t1, t2 in combinations(tfs, 2):
        f = _shared_frame(targets, likelihoods, t1, t2)
        f.insert(0, "gene", f.index)
        f.insert(0, "tf2", t2)
        f.insert(0, "tf1", t1)
        f["in_tail"] = [(a, b) in comb for a, b in zip(f["sign1"], f["sign2"])]
        frames.append(f.reset_index(drop=True))
    shared_targets = pd.concat(frames, ignore_index=True)
    return RegulonOverlap(table=table, pairs=pd.DataFrame(prow), sizes=sizes, targets=targets, likelihoods=likelihoods,
                          tfs=tuple(tfs), tail=tail, mor_min=float(mor_min), min_likelihood=float(min_likelihood),
                          universe_size=n_universe, n_perm=int(n_perm), seed=int(seed), shared_targets=shared_targets,
                          universe_genes=genes)


def missing_names(regulon: pd.DataFrame, tfs: Sequence[str]) -> list:
    """Regulators in ``tfs`` that are not a ``source`` of ``regulon``."""
    if "source" not in regulon.columns:
        return []
    have = set(regulon["source"].astype(str))
    return [t for t in tfs if t not in have]
