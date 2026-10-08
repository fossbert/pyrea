"""Mutant phenotype score (MPS) after Alvarez et al., Nat Genet 2016.

The MPS asks, per sample, how much its VIPER-inferred protein activity
resembles that of samples carrying a mutation in the gene versus wild-type
(WT) samples. For a trait value ``x`` (global activity "G" or residual
post-translational activity "RPT")::

    RL(x) = (p_M(x) - p_WT(x)) / (p_M(x) + p_WT(x))

``p_M`` / ``p_WT`` are Gaussian-kernel density estimates of the mutant and
WT samples. The MPS of a sample is the RL of the trait with the larger
absolute value. A likelihood ratio ``p_M / p_WT > 3`` corresponds to
``RL > 0.5`` (see :func:`lr_to_rl`).

Everything except :func:`mutation_association` is pure numpy/scipy;
the association test uses :func:`pyrea.aREA` and therefore needs R.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

__all__ = [
    "lr_to_rl",
    "relative_likelihood",
    "mutant_phenotype_score",
    "classify_phenotype",
    "mutation_association",
    "locus_specific_mps",
]

DEFAULT_LR = 3.0


def lr_to_rl(lr: float = DEFAULT_LR) -> float:
    """Convert a likelihood ratio ``p_M / p_WT`` to the RL scale: ``(lr-1)/(lr+1)``."""
    return (lr - 1.0) / (lr + 1.0)


def _usable(v: np.ndarray) -> bool:
    """A KDE needs at least two distinct finite values."""
    return v.size >= 2 and np.ptp(v) > 0


def relative_likelihood(x, mutant, wt) -> np.ndarray:
    """Relative likelihood of the mutant vs the WT phenotype at ``x``.

    Parameters
    ----------
    x : array-like
        Trait values at which RL is evaluated.
    mutant, wt : array-like
        Trait values of the mutant / WT samples, used to fit the KDEs.

    Returns
    -------
    np.ndarray
        RL in [-1, 1], positive = mutant-like. All NaN if either group has
        fewer than two distinct values (the KDE is then undefined).
    """
    x = np.asarray(x, dtype=float)
    mutant = np.asarray(mutant, dtype=float)
    wt = np.asarray(wt, dtype=float)
    mutant, wt = mutant[np.isfinite(mutant)], wt[np.isfinite(wt)]

    out = np.full(x.shape, np.nan)
    if not (_usable(mutant) and _usable(wt)):
        return out

    xs = x.reshape(1, -1)
    p_m = gaussian_kde(mutant).evaluate(xs)
    p_wt = gaussian_kde(wt).evaluate(xs)
    den = p_m + p_wt
    ok = np.isfinite(x.ravel()) & (den > 0)
    res = np.full(den.shape, np.nan)
    res[ok] = (p_m[ok] - p_wt[ok]) / den[ok]
    return res.reshape(x.shape)


def _larger_abs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise value with larger |.|, sign kept; a NaN side yields the other."""
    a_ok, b_ok = ~np.isnan(a), ~np.isnan(b)
    pick_a = a_ok & (~b_ok | (np.abs(np.nan_to_num(a)) >= np.abs(np.nan_to_num(b))))
    return np.where(pick_a, a, b)


def _gene_rl(gene: str, carriers: np.ndarray, wt: np.ndarray, traits: Sequence[pd.DataFrame],
             samples: pd.Index) -> np.ndarray:
    """RL of one gene over all ``samples``, combining the traits by larger |RL|."""
    rl = np.full(len(samples), np.nan)
    for tr in traits:
        if gene not in tr.index:
            continue
        v = tr.loc[gene, samples].to_numpy(dtype=float)
        rl = _larger_abs(rl, relative_likelihood(v, v[carriers], v[wt]))
    return rl


def _common_samples(*frames: pd.DataFrame) -> pd.Index:
    idx = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.columns, sort=False)
    return idx


def mutant_phenotype_score(mutations: pd.DataFrame,
                           activity: pd.DataFrame,
                           rpt: Optional[pd.DataFrame] = None,
                           min_mut: int = 10,
                           min_wt: int = 10) -> pd.DataFrame:
    """Mutant phenotype score for each gene in ``mutations``.

    Parameters
    ----------
    mutations : pd.DataFrame
        Binary (1 = mutated) genes x samples.
    activity : pd.DataFrame
        VIPER-inferred global activity (G), proteins x samples.
    rpt : pd.DataFrame, optional
        Residual post-translational activity (RPT), same layout. If given,
        the MPS takes the larger-|RL| of G and RPT per sample; otherwise it
        is the RL of G alone.
    min_mut, min_wt : int
        Minimum number of mutant / WT samples for a gene to be scored. Genes
        below the thresholds, or absent from ``activity``/``rpt``, are
        omitted. (The paper only states "at least two samples"; the KDE is
        shaky that low, hence the stricter default.)

    Returns
    -------
    pd.DataFrame
        MPS in [-1, 1], genes x samples (samples common to all inputs).
        Positive = mutant-like, negative = WT-like. A sample whose value
        is NaN in every trait gets NaN.
    """
    traits = [activity] if rpt is None else [activity, rpt]
    samples = _common_samples(mutations, *traits)
    rows = {}
    for gene in mutations.index:
        if gene not in activity.index:
            continue
        m = mutations.loc[gene, samples].to_numpy() == 1
        if m.sum() < min_mut or (~m).sum() < min_wt:
            continue
        rows[gene] = _gene_rl(gene, m, ~m, traits, samples)
    return pd.DataFrame.from_dict(rows, orient="index", columns=samples)


def classify_phenotype(mps, lr: float = DEFAULT_LR):
    """Label MPS values as ``"mutant"`` (RL > t), ``"wt"`` (RL < -t) or ``"intermediate"``.

    ``t = lr_to_rl(lr)``; the default ``lr=3`` gives t = 0.5. NaN stays NaN.
    Accepts a Series or DataFrame and returns the same shape of strings.
    """
    t = lr_to_rl(lr)
    labels = np.where(mps > t, "mutant", np.where(mps < -t, "wt", "intermediate")).astype(object)
    labels[np.isnan(np.asarray(mps, dtype=float))] = np.nan
    if isinstance(mps, pd.DataFrame):
        return pd.DataFrame(labels, index=mps.index, columns=mps.columns)
    if isinstance(mps, pd.Series):
        return pd.Series(labels, index=mps.index, name=mps.name)
    return labels


def mutation_association(mutations: pd.DataFrame,
                         traits: Mapping[str, pd.DataFrame],
                         integrate: Optional[Sequence[str]] = None,
                         min_mut: int = 2,
                         key: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Association of mutated samples with each trait via aREA (needs R).

    For every row of ``mutations`` the carriers form a one-set regulon that
    is tested for enrichment on the samples rank-sorted by the trait of the
    corresponding gene (Alvarez et al., Fig. 4 / 6). The NES sign tells the
    direction (positive = mutated samples have higher trait values).

    Parameters
    ----------
    mutations : pd.DataFrame
        Binary rows x samples (rows = genes or ``GENE:variant`` ids).
    traits : mapping name -> DataFrame
        Gene x sample matrices, e.g. ``{"G": vpres, "RPT": rpt, "expr": emat}``.
    integrate : sequence of names, optional
        Traits whose p-values are combined by taking the minimum
        (``p_min``, ``trait_min``), as in the paper's "integrated" bars for
        G and RPT. Defaults to all traits.
    min_mut : int
        Minimum number of carriers.
    key : mapping row id -> gene, optional
        Maps a row of ``mutations`` to the row of ``traits`` to use. Default:
        identity.

    Returns
    -------
    pd.DataFrame
        One row per tested id with ``n_mut``, and ``nes_<t>`` / ``p_<t>``
        per trait plus ``p_min`` and ``trait_min`` over ``integrate``.
        ``p`` is two-sided from the NES (standard normal).
    """
    from .rbackend.area import aREA
    from .utils import gene_sets_to_regulon

    integrate = list(traits) if integrate is None else list(integrate)
    samples = _common_samples(mutations, *traits.values())
    rows = {}
    for rid in mutations.index:
        carriers = list(samples[mutations.loc[rid, samples].to_numpy() == 1])
        if len(carriers) < min_mut:
            continue
        gene = rid if key is None else key[rid]
        reg = gene_sets_to_regulon({rid: carriers}, minsize=min_mut)
        rec = {"n_mut": len(carriers)}
        for name, tr in traits.items():
            if gene not in tr.index:
                rec[f"nes_{name}"] = rec[f"p_{name}"] = np.nan
                continue
            sig = tr.loc[gene, samples].astype(float).rename(name)
            nes = float(aREA(sig, reg, minsize=min_mut).iloc[0, 0])
            rec[f"nes_{name}"] = nes
            rec[f"p_{name}"] = 2 * norm.sf(abs(nes)) if np.isfinite(nes) else np.nan
        ps = pd.Series({n: rec[f"p_{n}"] for n in integrate}, dtype=float)
        rec["p_min"] = ps.min()
        rec["trait_min"] = ps.idxmin() if ps.notna().any() else np.nan
        rows[rid] = rec
    return pd.DataFrame.from_dict(rows, orient="index")


def locus_specific_mps(variants: pd.DataFrame,
                       activity: pd.DataFrame,
                       rpt: Optional[pd.DataFrame] = None,
                       expression: Optional[pd.DataFrame] = None,
                       sep: str = ":",
                       min_samples: int = 2,
                       min_mps: int = 10,
                       min_wt: int = 10):
    """Locus-specific analysis of individual variants (Alvarez et al., Fig. 6a).

    Parameters
    ----------
    variants : pd.DataFrame
        Binary variants x samples; row ids are ``"GENE<sep>variant"``.
    activity, rpt, expression : pd.DataFrame
        G-activity, RPT-activity and (optionally) mRNA, genes x samples.
    min_samples : int
        Minimum carriers for the aREA association (paper: 2).
    min_mps : int
        Minimum carriers for the per-variant MPS (paper: 10).
    min_wt : int
        Minimum WT samples for the per-variant MPS.

    Returns
    -------
    (assoc, mps) : tuple of DataFrame
        ``assoc``: output of :func:`mutation_association` (needs R) with the
        traits G, RPT and, if given, expr; ``p_min`` integrates G and RPT
        only, as in the paper. ``mps``: per-variant MPS, variants x samples.

    Notes
    -----
    For the per-variant MPS the "WT" group is the set of samples with *no*
    mutation in the gene (carriers of other variants are not counted as WT,
    since they would blur the WT density), while RL is still evaluated for
    every sample. This is a choice, not stated explicitly in the paper.
    """
    genes = pd.Series(variants.index.str.split(sep, n=1).str[0], index=variants.index)
    traits = {"G": activity}
    if rpt is not None:
        traits["RPT"] = rpt
    integrate = list(traits)
    if expression is not None:
        traits["expr"] = expression

    assoc = mutation_association(variants, traits, integrate=integrate,
                                 min_mut=min_samples, key=genes)

    samples = _common_samples(variants, activity, *( [rpt] if rpt is not None else []))
    mut_traits = [activity] if rpt is None else [activity, rpt]
    any_mut = variants.loc[:, samples].groupby(genes).max() > 0
    rows = {}
    for vid in variants.index:
        carriers = variants.loc[vid, samples].to_numpy() == 1
        gene = genes[vid]
        if carriers.sum() < min_mps or gene not in activity.index:
            continue
        wt = ~any_mut.loc[gene].to_numpy()
        if wt.sum() < min_wt:
            continue
        rows[vid] = _gene_rl(gene, carriers, wt, mut_traits, samples)
    mps = pd.DataFrame.from_dict(rows, orient="index", columns=samples)
    return assoc, mps
