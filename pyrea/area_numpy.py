"""aREA and metaVIPER in numpy (no R): regulator activity of signatures, identical to ``viper::viper(method="none")``.

:func:`area_nes` computes the normalised enrichment score (NES) of regulons in every column of a signature matrix with the analytic rank-based
enrichment of aREA (Alvarez et al., Nat Genetics 2016): the genes of a signature are ranked, transformed to normal quantiles (two-tailed for the
mode of regulation, one-tailed for the part of the regulon without a sign), weighted with the interaction likelihoods and summed over the
targets. :func:`metaviper` integrates the NES of several networks like ``viper`` does when it is given a list of regulons (metaVIPER, Ding et al., Nat
Commun 2018): the weighted mean with weights ``|NES| ** mvws`` of the networks that contain the regulator.

The computation is vectorised over signatures and meant for matrices that are too large for the R round trip (LINCS: hundreds of thousands of
signatures). The result is tested against ``viper`` (tests/test_area_numpy.py). Signatures must not contain missing values.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

__all__ = ["compact_regulons", "rank_transform", "area_nes", "metaviper"]


def compact_regulons(regulon: pd.DataFrame, genes: Sequence[str], minsize: int = 25, regulators: Optional[Sequence[str]] = None) -> dict:
    """Regulon in long format (``source``, ``target``, ``mor``, optional ``likelihood``) restricted to ``genes`` (the rows of the signatures)
    as ``{regulator: (gene positions, mor, likelihood)}``; regulators with fewer than ``minsize`` targets among ``genes`` are dropped (as
    ``viper``'s ``minsize``). ``regulators`` limits the result to these regulators."""
    need = {"source", "target", "mor"}
    if not need <= set(regulon.columns):
        raise ValueError(f"regulon needs the columns {sorted(need)}")
    df = regulon
    if regulators is not None:
        df = df[df["source"].isin(set(map(str, regulators)))]
    pos = pd.Series(np.arange(len(genes)), index=pd.Index(genes))
    if not pos.index.is_unique:
        raise ValueError("genes must be unique")
    idx = pos.reindex(df["target"].astype(str).to_numpy())
    keep = idx.notna().to_numpy()
    df, idx = df[keep], idx[keep].to_numpy(dtype=np.int64)
    lik = df["likelihood"].to_numpy(dtype=np.float64) if "likelihood" in df else np.ones(len(df))
    mor = df["mor"].to_numpy(dtype=np.float64)
    out = {}
    src = df["source"].astype(str).to_numpy()
    order = np.argsort(src, kind="stable")
    src, idx, mor, lik = src[order], idx[order], mor[order], lik[order]
    bounds = np.flatnonzero(np.r_[True, src[1:] != src[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= minsize:
            out[src[a]] = (idx[a:b].astype(np.int32), mor[a:b], lik[a:b])
    return out


def rank_transform(X: np.ndarray) -> tuple:
    """The two normal-quantile transforms of aREA for every column of ``X`` (genes x signatures): ``t1`` (two-tailed: the distance of the rank from
    the median) and ``t2`` (signed)."""
    X = np.asarray(X)
    if not np.isfinite(X).all():
        raise ValueError("The signatures must not contain missing values")
    n = X.shape[0]
    t2 = rankdata(X, axis=0) / (n + 1)
    t1 = np.abs(t2 - 0.5) * 2
    t1 = t1 + (1 - t1.max(axis=0)) / 2
    return norm.ppf(t1), norm.ppf(t2)


def _nes_of(t1: np.ndarray, t2: np.ndarray, idx: np.ndarray, mor: np.ndarray, lik: np.ndarray) -> np.ndarray:
    sum1 = (mor * lik) @ t2[idx]
    sum2 = ((1 - np.abs(mor)) * lik) @ t1[idx]
    ss = np.where(sum1 == 0, 1.0, np.sign(sum1))
    es = (np.abs(sum1) + np.where(sum2 > 0, sum2, 0.0)) / lik.sum() * ss
    return es * np.sqrt(np.sum((lik / lik.max()) ** 2))


def area_nes(signatures: "pd.DataFrame | np.ndarray", regulons: "Mapping | pd.DataFrame", genes: Optional[Sequence[str]] = None, minsize: int = 25,
             regulators: Optional[Sequence[str]] = None, transformed: Optional[tuple] = None) -> "pd.DataFrame | np.ndarray":
    """NES of regulons in every signature.

    Parameters
    ----------
    signatures : DataFrame (genes x signatures) or ndarray
        Differential signatures (e.g. z-scores). With an ndarray give ``genes``.
    regulons : DataFrame or dict
        A long-format regulon (``source``, ``target``, ``mor``, optional ``likelihood``) or the output of :func:`compact_regulons` for these genes.
    genes : sequence of str, optional
        Row names of ``signatures`` (if an ndarray).
    minsize : int
        Minimum number of targets among the genes (long-format regulons only).
    regulators : sequence of str, optional
        Only these regulators (long-format regulons only).
    transformed : tuple, optional
        ``rank_transform(signatures)`` if already computed (to score several networks on the same signatures).

    Returns
    -------
    DataFrame (regulators x signatures) or ndarray with the order of the regulons (if ``signatures`` is an ndarray, the regulator names are
    not returned: use a DataFrame).
    """
    is_df = isinstance(signatures, pd.DataFrame)
    X = signatures.to_numpy() if is_df else np.asarray(signatures)
    names = list(signatures.index.astype(str)) if is_df else (None if genes is None else list(map(str, genes)))
    if isinstance(regulons, pd.DataFrame):
        if names is None:
            raise ValueError("Give genes for an ndarray")
        regs = compact_regulons(regulons, names, minsize, regulators)
    else:
        regs = dict(regulons)
    if not regs:
        raise ValueError("There are no regulons with enough targets among the genes of the signatures")
    t1, t2 = transformed if transformed is not None else rank_transform(X)
    out = np.vstack([_nes_of(t1, t2, *r) for r in regs.values()])
    if is_df:
        return pd.DataFrame(out, index=list(regs), columns=signatures.columns)
    return out


def metaviper(activities: "Mapping[str, pd.DataFrame] | np.ndarray", mvws: float = 1.0, allowed: Optional[np.ndarray] = None):
    """Integrate the NES of several networks like ``viper`` (metaVIPER): for every regulator and signature
    ``sum(NES * w) / sum(w)`` with ``w = |NES| ** mvws`` over the networks that contain the regulator (missing: weight 0).

    ``activities``: dict network -> DataFrame (regulators x signatures; regulators may differ between the networks; the same columns), or a
    3-d array (networks x regulators x signatures, NaN for a missing regulator). ``allowed`` (networks x signatures, bool) switches networks off for
    single signatures (e.g. only the networks of the lineage of a cell line). Returns a DataFrame (union of the regulators) or an array.
    NaN where no network has a nonzero NES.
    """
    if isinstance(activities, Mapping):
        nets = list(activities.values())
        if not nets:
            raise ValueError("No networks")
        regs = list(dict.fromkeys(r for a in nets for r in a.index))
        cols = nets[0].columns
        arr = np.full((len(nets), len(regs), len(cols)), np.nan)
        pos = {r: i for i, r in enumerate(regs)}
        for k, a in enumerate(nets):
            if not a.columns.equals(cols):
                raise ValueError("All networks need the same signatures (columns)")
            arr[k, [pos[r] for r in a.index], :] = a.to_numpy()
    else:
        arr, regs, cols = np.asarray(activities, dtype=float), None, None
    x = np.nan_to_num(arr, nan=0.0)
    w = np.abs(x) ** mvws
    if allowed is not None:
        w = w * np.asarray(allowed, dtype=float)[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (x * w).sum(axis=0) / w.sum(axis=0)
    return pd.DataFrame(out, index=regs, columns=cols) if regs is not None else out
