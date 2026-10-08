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

import warnings
from typing import Mapping, Optional, Sequence, Union

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
    "mps_targets",
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
                           min_wt: int = 10,
                           key: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    """Mutant phenotype score for each row of ``mutations``.

    Parameters
    ----------
    mutations : pd.DataFrame
        Binary (1 = mutated, 0 = WT) genes x samples. NaN (e.g. gene not
        profiled in the sample) excludes the sample from both groups for
        that gene; its MPS is still evaluated.
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
    key : mapping row id -> gene, optional
        Rows of ``mutations`` that are not gene names (e.g. a gene fusion
        'CLDN18-ARHGAP6/26') are scored on the activity of the mapped gene.
        Rows not in the mapping are used as they are. The result is indexed
        by the row ids of ``mutations``.

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
    for rid in mutations.index:
        gene = rid if key is None else key.get(rid, rid)
        if gene not in activity.index:
            continue
        row = mutations.loc[rid, samples].to_numpy(dtype=float)
        m, wt = row == 1, row == 0  # NaN = not profiled: neither mutant nor WT
        if m.sum() < min_mut or wt.sum() < min_wt:
            continue
        rows[rid] = _gene_rl(gene, m, wt, traits, samples)
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
        row = mutations.loc[rid, samples].to_numpy(dtype=float)
        carriers = list(samples[row == 1])
        if len(carriers) < min_mut:
            continue
        keep = samples[~np.isnan(row)]  # drop samples not profiled for this row
        gene = rid if key is None else key[rid]
        reg = gene_sets_to_regulon({rid: carriers}, minsize=min_mut)
        rec = {"n_mut": len(carriers)}
        for name, tr in traits.items():
            if gene not in tr.index:
                rec[f"nes_{name}"] = rec[f"p_{name}"] = np.nan
                continue
            sig = tr.loc[gene, keep].astype(float).rename(name)
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
                       min_wt: int = 10,
                       exclude_other_variants: bool = True):
    """Locus-specific analysis of individual variants (Alvarez et al., Fig. 6).

    Every variant (e.g. ``KRAS:G12D``) is compared with the *wild-type* samples of its gene:
    its carriers are tested for their effect on the activity (G, RPT) and expression of the
    gene's protein (aREA association), and, if there are enough carriers, scored with the
    mutant phenotype score.

    Which samples count as wild type
    --------------------------------
    A variant matrix has a 0 for a variant in every sample that does not carry it, including
    samples with a *different* variant of the same gene. These are not wild type and would
    dilute the comparison. The "state" of a gene in a sample is therefore derived first:
    1 = any variant, 0 = no variant, NaN = not profiled. Then

    - MPS: the WT density is estimated on samples with gene state 0 only. RL is still
      evaluated for every sample, so carriers of other variants get a score, too.
    - association (``exclude_other_variants=True``, default): carriers of other variants are
      removed from the test (set to NaN) so that the variant is compared with WT samples
      only. Their number is reported in ``n_other_variants``. With ``False`` the variant is
      tested against all other samples, as a plain ``mutation_association`` would do.
    - not-profiled samples (NaN) never take part.

    For this to work ``variants`` must contain **all** variants of a gene, not only the
    frequent ones: carriers of a dropped variant would look like wild type. Filter with
    ``min_samples`` / ``min_mps`` here, not beforehand. Likewise it should include variants
    of unknown significance, not only the drivers.

    Parameters
    ----------
    variants : pd.DataFrame
        Binary variants x samples; row ids are ``"GENE<sep>variant"`` (e.g. ``"TP53:R175H"``).
        NaN = gene not profiled in the sample.
    activity, rpt, expression : pd.DataFrame
        G-activity, RPT-activity and (optionally) mRNA, genes x samples.
    sep : str
        Separator of gene and variant in the row ids.
    min_samples : int
        Minimum carriers for the aREA association (paper: 2).
    min_mps : int
        Minimum carriers for the per-variant MPS (paper: 10).
    min_wt : int
        Minimum WT samples (gene state 0) for the per-variant MPS.
    exclude_other_variants : bool
        Exclude carriers of other variants of the gene from the association test.

    Returns
    -------
    (assoc, mps) : tuple of DataFrame
        ``assoc``: one row per variant with ``n_mut``, ``n_other_variants``, and per trait
        (G, RPT, expr) ``nes_<t>`` (> 0: carriers have higher values) and ``p_<t>``, plus
        ``p_min`` / ``trait_min`` over G and RPT as in the paper (needs R).
        ``mps``: MPS of the variants with enough carriers, variants x samples.

    Examples
    --------
    From a cBioPortal "alterations across samples" export, with ``cbiokit`` (all variants,
    not only drivers; ``emat`` is the expression matrix the activity was computed from)::

        import cbiokit as cbk, pyrea as pr

        ex = cbk.read_alteration_export("alterations_across_samples.tsv")
        variants = cbk.alteration_matrix(ex, types=("MUT",), level="event", drivers_only=False,
                                         by="PATIENT_ID")[emat.columns]
        assoc, mps = pr.locus_specific_mps(variants, vpres, rpt, expression=emat)

        assoc.loc[assoc.index.str.startswith("TP53:")].sort_values("p_min").head()
        pr.plot_mps_rank(mps.loc["TP53:R175H"], variants.loc["TP53:R175H"])

    Compare with the gene-level view of the same data, where all variants are pooled::

        gene = cbk.alteration_matrix(ex, types=("MUT",), drivers_only=False, by="PATIENT_ID")[emat.columns]
        pr.mutant_phenotype_score(gene, vpres, rpt)
    """
    genes = pd.Series(variants.index.str.split(sep, n=1).str[0], index=variants.index)
    traits = {"G": activity}
    if rpt is not None:
        traits["RPT"] = rpt
    integrate = list(traits)
    if expression is not None:
        traits["expr"] = expression

    # gene state per sample: 1 any variant, 0 none, NaN not profiled; broadcast back to variant rows
    state = variants.groupby(genes.to_numpy()).max()
    per_variant = state.reindex(genes.to_numpy()).set_axis(variants.index)

    other = (per_variant == 1) & (variants != 1) & variants.notna()
    tested = variants.astype(float).mask(other) if exclude_other_variants else variants
    assoc = mutation_association(tested, traits, integrate=integrate, min_mut=min_samples, key=genes)
    n_other = other.sum(axis=1) if exclude_other_variants else pd.Series(0, index=variants.index)
    assoc.insert(1, "n_other_variants", n_other.reindex(assoc.index).to_numpy())

    samples = _common_samples(variants, activity, *([rpt] if rpt is not None else []))
    mut_traits = [activity] if rpt is None else [activity, rpt]
    rows = {}
    for vid in variants.index:
        carriers = variants.loc[vid, samples].to_numpy(dtype=float) == 1
        gene = genes[vid]
        if carriers.sum() < min_mps or gene not in activity.index:
            continue
        wt = (state.loc[gene, samples] == 0).to_numpy()
        if wt.sum() < min_wt:
            continue
        rows[vid] = _gene_rl(gene, carriers, wt, mut_traits, samples)
    mps = pd.DataFrame.from_dict(rows, orient="index", columns=samples)
    return assoc, mps


def mps_targets(mutations: pd.DataFrame,
                targets: Union[Sequence[str], Mapping[str, Sequence[str]]],
                activity: pd.DataFrame,
                rpt: Optional[pd.DataFrame] = None,
                expression: Optional[pd.DataFrame] = None,
                associate: bool = True,
                min_mut: int = 10,
                min_wt: int = 10,
                lr: float = DEFAULT_LR):
    """Test mutation rows (e.g. a gene fusion) against the activity of chosen target proteins.

    Typical use: a fusion or hotspot that is not itself a protein, tested for its effect on the
    activity of one or more candidate proteins, e.g. ``targets=["RHOA", "PTK2", "YAP1"]``.
    Each (row, target) pair is a separate test; the order of ``targets`` is kept, so a
    primary target can simply be listed first. P-values are not corrected for the number of
    targets.

    Parameters
    ----------
    mutations : pd.DataFrame
        Binary rows x samples, NaN = not profiled (see :func:`mutant_phenotype_score`).
    targets : sequence of str, or mapping row id -> sequence of str
        Target proteins for every row, or per row.
    activity, rpt, expression : pd.DataFrame
        G-activity, RPT-activity (optional) and mRNA (optional), genes x samples.
    associate : bool
        Add the aREA association with the traits (needs R); otherwise only the MPS summary.
    min_mut, min_wt : int
        As in :func:`mutant_phenotype_score`; pairs below the thresholds get no MPS (NaN).
    lr : float
        Likelihood ratio defining the mutant / WT phenotype (default 3, RL > 0.5).

    Returns
    -------
    (summary, mps) : tuple
        ``summary``: one row per (row, target) with ``n_mut``, ``n_wt``,
        ``mps_mut`` / ``mps_wt`` (mean MPS of carriers / WT), ``frac_mut_phenotype`` (share of
        carriers with the MPS-defined mutant phenotype), ``frac_mut_phenotype_wt`` (same among WT
        samples) and, with ``associate``, ``nes_G``, ``p_G``, ``nes_RPT``, ``p_RPT``
        (``nes_expr``, ``p_expr``), ``p_min`` and ``trait_min`` (G and RPT only). NES > 0 =
        carriers have higher activity. ``mps``: MPS per (row, target) x sample (MultiIndex).
    """

    traits = {"G": activity}
    if rpt is not None:
        traits["RPT"] = rpt
    integrate = list(traits)
    if expression is not None:
        traits["expr"] = expression

    def _targets(rid):
        return list(targets[rid]) if isinstance(targets, Mapping) else list(targets)

    rows, mps_rows = [], {}
    samples = _common_samples(mutations, activity, *([rpt] if rpt is not None else []))
    thr = lr_to_rl(lr)
    for rid in mutations.index:
        row = mutations.loc[rid, samples].to_numpy(dtype=float)
        carriers, wt = row == 1, row == 0
        for tgt in _targets(rid):
            rec = {"row": rid, "target": tgt, "n_mut": int(carriers.sum()), "n_wt": int(wt.sum())}
            if tgt not in activity.index:
                warnings.warn(f"Target {tgt!r} has no activity (not in the regulon); skipped")
                rows.append(rec)
                continue
            one = mutations.loc[[rid]]
            mps = mutant_phenotype_score(one, activity, rpt, min_mut=min_mut, min_wt=min_wt, key={rid: tgt})
            if len(mps):
                v = mps.iloc[0].to_numpy()
                mps_rows[(rid, tgt)] = mps.iloc[0]
                rec.update(mps_mut=np.nanmean(v[carriers]), mps_wt=np.nanmean(v[wt]),
                           frac_mut_phenotype=np.nanmean(v[carriers] > thr),
                           frac_mut_phenotype_wt=np.nanmean(v[wt] > thr))
            if associate:
                a = mutation_association(one, traits, integrate=integrate, min_mut=min_mut, key={rid: tgt})
                if len(a):
                    rec.update(a.iloc[0].drop("n_mut").to_dict())
            rows.append(rec)

    summary = pd.DataFrame(rows).set_index(["row", "target"])
    mps_df = pd.DataFrame(mps_rows).T if mps_rows else pd.DataFrame(columns=samples)
    mps_df.index = pd.MultiIndex.from_tuples(mps_df.index, names=["row", "target"]) if len(mps_df) else mps_df.index
    return summary, mps_df
