"""Is a set of regulators (a pair, a triplet, ...) more closely linked than other sets of the same kind in a network?

:func:`pyrea.regulon_overlap` asks whether two regulators share more targets than random gene sets would. That does not say whether
they are *special*: regulators of related functions share hub targets, and large regulons overlap by chance more. The reference here is
therefore the other **regulators of the same network**:

- **Size-matched random sets** (``p_random``): the overlap of the target tails of the set is compared with that of ``n_random`` sets of
  other regulators whose tails have about the same sizes (each member is replaced by a random regulator with a tail of similar
  size; the fold = overlap / expected is compared, which removes the remaining size differences).
- **Rank among all partners** (``extension``): for every member the other members are taken as a base (their common targets) and *all*
  regulators of the network are ranked by how strongly they overlap with it (hypergeometric p). For a pair this is the rank of B among the
  partners of A and of A among the partners of B (rank 1 for both: *mutual best partner*); for a triplet, how good C is as the third
  member given A and B.

The tail of a regulator is its positive (``mor > mor_min``) or negative (``mor < -mor_min``) targets; ``tail="both"`` evaluates every
combination of tails of the members (for a pair ``+/+``, ``-/-``, ``+/-``, ``-/+``), which is where anti-correlated regulators show up.
Only regulators with a tail of at least ``min_size`` targets are in the reference (as ``minsize`` in VIPER).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

__all__ = ["regulon_specificity", "RegulonSpecificity", "top_partners"]

_TAILS = ("positive", "negative", "both")
_MAX_BOTH = 5


def _patterns(k: int, tail: str) -> list:
    """Tail of each member: all positive, all negative, then (``tail="both"``) the mixed combinations."""
    pos, neg = ("+",) * k, ("-",) * k
    if tail == "positive":
        return [pos]
    if tail == "negative":
        return [neg]
    return [pos, neg] + [p for p in product("+-", repeat=k) if p not in (pos, neg)]


class _Network:
    """Tail membership of all regulators of a network: sparse boolean matrices regulators x genes, one per sign."""

    def __init__(self, regulon: pd.DataFrame, mor_min: float, min_likelihood: float, universe: Optional[Iterable[str]], min_size: int):
        need = {"source", "target", "mor"}
        if not need <= set(regulon.columns):
            raise ValueError(f"regulon needs the columns {sorted(need)} (long format, see read_regulon_rds)")
        cols = [c for c in ("source", "target", "mor", "likelihood") if c in regulon.columns]
        df = regulon[cols].copy()
        df["source"], df["target"] = df["source"].astype(str), df["target"].astype(str)
        if df.duplicated(["source", "target"]).any():
            raise ValueError("The regulon has duplicated source-target pairs")
        if "likelihood" in df:
            df = df[df["likelihood"] >= min_likelihood]
        if universe is not None:
            df = df[df["target"].isin(set(map(str, universe)))]
        src, self.regulators = pd.factorize(df["source"], sort=True)
        tgt, self.genes = pd.factorize(df["target"], sort=True)
        self.row = pd.Series(np.arange(len(self.regulators)), index=self.regulators)
        self.n_genes = len(self.genes)
        mor = df["mor"].to_numpy()
        self.M, self.size, self.elig = {}, {}, {}
        for sign, sel in (("+", mor > mor_min), ("-", mor < -mor_min)):
            m = sparse.csr_matrix((np.ones(int(sel.sum()), dtype=np.float32), (src[sel], tgt[sel])),
                                  shape=(len(self.regulators), self.n_genes))
            self.M[sign] = m
            self.size[sign] = np.asarray(m.sum(axis=1)).ravel().astype(int)
            self.elig[sign] = self.size[sign] >= min_size

    def vector(self, i: int, sign: str) -> np.ndarray:
        return self.M[sign].getrow(i).toarray().ravel() > 0

    def core(self, rows: Sequence[int], signs: Sequence[str]) -> np.ndarray:
        v = self.vector(rows[0], signs[0])
        for i, s in zip(rows[1:], signs[1:]):
            v = v & self.vector(i, s)
        return v

    def scores(self, base: np.ndarray, sign: str) -> tuple:
        """Overlap of every regulator's ``sign`` tail with the gene set ``base``: counts, sizes, expected, fold, hypergeometric p."""
        counts = np.asarray(self.M[sign] @ base.astype(np.float32)).ravel().round().astype(int)
        size = self.size[sign]
        n_b = int(base.sum())
        expected = n_b * size / self.n_genes
        with np.errstate(divide="ignore", invalid="ignore"):
            fold = np.where(expected > 0, counts / expected, np.nan)
        p = hypergeom.sf(counts - 1, self.n_genes, n_b, size)
        return counts, size, expected, fold, p


def _check_sets(sets, tail: str) -> list:
    if tail not in _TAILS:
        raise ValueError(f"tail must be one of {_TAILS}")
    if isinstance(sets, (str, bytes)) or not len(sets):
        raise ValueError("sets must be a non-empty sequence of regulator sequences, e.g. [('A', 'B'), ('A', 'B', 'C')]")
    out = []
    for s in sets:
        if isinstance(s, str):
            raise ValueError(f"Every set is a sequence of regulators, got the string {s!r}")
        s = tuple(map(str, s))
        if len(s) < 2:
            raise ValueError(f"A set needs at least two regulators, got {s}")
        if len(set(s)) != len(s):
            raise ValueError(f"Regulators of a set must be unique, got {s}")
        if tail == "both" and len(s) > _MAX_BOTH:
            raise ValueError(f"tail='both' evaluates 2^k tail combinations; at most {_MAX_BOTH} regulators per set")
        out.append(s)
    return out


def _random_folds(net: _Network, rows: Sequence[int], signs: Sequence[str], n_random: int, tol: float, min_pool: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Folds of ``n_random`` random sets of other regulators whose tails (sign by sign) have about the sizes of the observed ones."""
    k = len(rows)
    taken = np.zeros(len(net.regulators), dtype=bool)
    taken[list(rows)] = True
    pools = []
    for i, s in zip(rows, signs):
        n = net.size[s][i]
        cand = np.flatnonzero(net.elig[s] & ~taken)
        sz = net.size[s][cand]
        near = np.abs(sz - n) <= tol * n
        if near.sum() < min_pool:
            near = np.zeros(len(cand), dtype=bool)
            near[np.argsort(np.abs(np.log(sz / max(n, 1))))[:min_pool]] = True
        pools.append(cand[near])
    if any(len(p) == 0 for p in pools):
        return np.array([])
    idx = np.column_stack([rng.choice(p, n_random) for p in pools])
    for _ in range(50):                                                   # a regulator may not be drawn twice in one set
        dup = np.zeros(n_random, dtype=bool)
        for a in range(k):
            for b in range(a + 1, k):
                dup |= idx[:, a] == idx[:, b]
        if not dup.any():
            break
        idx[dup] = np.column_stack([rng.choice(p, int(dup.sum())) for p in pools])
    else:
        idx = idx[~dup]
    if not len(idx):
        return np.array([])
    prod = net.M[signs[0]][idx[:, 0]]
    sizes = [net.size[signs[0]][idx[:, 0]].astype(float)]
    for j in range(1, k):
        prod = prod.multiply(net.M[signs[j]][idx[:, j]])
        sizes.append(net.size[signs[j]][idx[:, j]].astype(float))
    counts = np.asarray(prod.sum(axis=1)).ravel()
    expected = net.n_genes * np.prod([s / net.n_genes for s in sizes], axis=0)
    return counts / expected


@dataclass
class RegulonSpecificity:
    """Result of :func:`regulon_specificity`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per set and tail combination: ``set`` ("A | B"), ``signs`` ("+/-": tail of each member), ``sizes`` (targets in the
        tails, "545/351"), ``overlap`` (targets in all tails), ``expected`` (independent tails), ``fold``, ``null_median_fold`` and
        ``null_p95_fold`` (size-matched random sets of other regulators), ``percentile`` (share of random sets with a smaller fold),
        ``p_random`` (one-sided, ``(1 + n at least as large) / (1 + n_null)``, smallest ``1 / (n_null + 1)``), ``n_null``,
        ``ext_rank_max`` / ``ext_percentile_max`` (the worst rank / percentile of a member in ``extension``: the set is only as
        specific as its weakest link) and ``mutual_best`` (every member is the best partner of the others).
    extension : pd.DataFrame
        One row per set, tail combination and member: the member is ranked among **all** regulators of the network as a partner of
        the other members (``base``; their common targets): ``rank``, ``n_candidates``, ``percentile`` (rank / n_candidates, small =
        good), ``overlap`` with the base, ``expected``, ``fold``, ``p_hyper`` and ``top`` (the best candidates with their overlap).
    settings : dict
        ``tail``, ``mor_min``, ``min_likelihood``, ``min_size``, ``n_random``, ``tol``, ``seed``, ``n_regulators`` (in the reference, per
        tail) and ``universe_size``.
    """

    table: pd.DataFrame
    extension: pd.DataFrame
    settings: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"RegulonSpecificity({self.table['set'].nunique()} set(s), {len(self.table)} tail combination(s), "
                f"{self.settings.get('n_regulators', {}).get('+', '?')} regulators in the reference)")


def regulon_specificity(regulon: pd.DataFrame, sets: Sequence[Sequence[str]], tail: str = "positive", mor_min: float = 0.0,
                        min_likelihood: float = 0.0, universe: Optional[Iterable[str]] = None, min_size: int = 25,
                        n_random: int = 5000, tol: float = 0.2, min_pool: int = 50, seed: int = 0, top: int = 5) -> RegulonSpecificity:
    """How specific is the overlap of a set of regulators compared with other regulators of the network?

    Parameters
    ----------
    regulon : pd.DataFrame
        Long format as returned by :func:`pyrea.read_regulon_rds` (``source``, ``target``, ``mor``, optionally ``likelihood``). The
        reference is **all regulators of this network** (with a tail of at least ``min_size`` targets).
    sets : sequence of sequences of str
        The sets to evaluate, e.g. ``[("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "BNC2")]`` (two or more regulators each).
    tail : {"positive", "negative", "both"}
        Tails of the members that are intersected: all positive (default), all negative, or ``"both"``: every combination of tails
        (``+/+``, ``-/-`` and the mixed ones such as ``+/-``: the first member's activated targets with the second's repressed ones;
        for a pair 4, for a triplet 8 combinations).
    mor_min, min_likelihood, universe :
        As in :func:`pyrea.regulon_overlap`.
    min_size : int
        Smallest tail for a regulator to be part of the reference (random sets and partner ranks).
    n_random : int
        Size-matched random sets per set and tail combination (the smallest possible ``p_random`` is ``1 / (n_random + 1)``).
    tol, min_pool : float, int
        A member is replaced by regulators whose tail is within ``tol`` (relative) of its size; if fewer than ``min_pool`` qualify, the
        ``min_pool`` regulators with the closest size are used.
    seed : int
    top : int
        Number of best candidates listed in ``extension["top"]``.

    Returns
    -------
    RegulonSpecificity

    Notes
    -----
    This compares regulators with each other. It complements :func:`pyrea.regulon_overlap`, whose null is a random draw of *genes*;
    here the null is a random draw of *regulators* with tails of the same size, which keeps the hub structure of the network
    (hub targets make every regulator overlap with others). A small ``p_random`` says: this set overlaps more than regulators of
    that size usually do, not that its members are related by function. If no other regulator has a tail (of that sign) of at least
    ``min_size`` targets there is no reference and ``p_random`` is NaN (typical for the negative tails when ``min_size`` is large).

    Examples
    --------
    ::

        sp = pr.regulon_specificity(reg, [("ZCCHC24", "ZEB1"), ("ZCCHC24", "ZEB1", "ZEB2")], n_random=5000)
        sp.table[["set", "signs", "overlap", "fold", "null_median_fold", "p_random", "mutual_best"]]
        sp.extension[["set", "member", "rank", "n_candidates", "top"]]
    """
    sets = _check_sets(sets, tail)
    if n_random < 1:
        raise ValueError("n_random must be at least 1")
    net = _Network(regulon, mor_min, min_likelihood, universe, min_size)
    missing = sorted({t for s in sets for t in s if t not in net.row.index})
    if missing:
        raise ValueError(f"Regulators not in the regulon: {missing}")
    rng = np.random.default_rng(seed)
    N = net.n_genes
    rows, ext_rows = [], []
    for tfs in sets:
        k = len(tfs)
        ridx = [int(net.row[t]) for t in tfs]
        label = " | ".join(tfs)
        for pat in _patterns(k, tail):
            sizes = [int(net.size[s][i]) for i, s in zip(ridx, pat)]
            rec = {"set": label, "signs": "/".join(pat), "sizes": "/".join(map(str, sizes))}
            core = net.core(ridx, pat)
            overlap = int(core.sum())
            expected = N * float(np.prod([n / N for n in sizes]))
            fold = overlap / expected if expected > 0 else np.nan
            rec.update(overlap=overlap, expected=expected, fold=fold)
            null = _random_folds(net, ridx, pat, n_random, tol, min_pool, rng) if min(sizes) > 0 else np.array([])
            if len(null) and np.isfinite(fold):
                rec.update(null_median_fold=float(np.median(null)), null_p95_fold=float(np.quantile(null, 0.95)),
                           percentile=float(np.mean(null < fold - 1e-12)), p_random=float((1 + np.sum(null >= fold - 1e-12)) / (1 + len(null))),
                           n_null=len(null))
            else:
                rec.update(null_median_fold=np.nan, null_p95_fold=np.nan, percentile=np.nan, p_random=np.nan, n_null=len(null))
            # rank of every member among all regulators as a partner of the others
            ranks, pcts = [], []
            for m in range(k):
                others = [j for j in range(k) if j != m]
                base = net.core([ridx[j] for j in others], [pat[j] for j in others])
                n_b = int(base.sum())
                counts, size, exp_, fold_, p = net.scores(base, pat[m])
                cand = net.elig[pat[m]].copy()
                cand[[ridx[j] for j in others]] = False
                cand[ridx[m]] = True
                ci = np.flatnonzero(cand)
                e = {"set": label, "signs": rec["signs"], "member": tfs[m], "base": " & ".join(tfs[j] for j in others), "n_base": n_b}
                if n_b == 0 or size[ridx[m]] == 0:
                    e.update(rank=np.nan, n_candidates=len(ci), percentile=np.nan, overlap=int(counts[ridx[m]]), expected=float(exp_[ridx[m]]),
                             fold=np.nan, p_hyper=np.nan, top="")
                    ranks.append(np.nan), pcts.append(np.nan)
                else:
                    pm, fm = p[ridx[m]], fold_[ridx[m]]
                    pc, fc = p[ci], np.nan_to_num(fold_[ci])
                    rank = 1 + int(np.sum(pc < pm)) + int(np.sum((pc == pm) & (fc > fm)))
                    order = np.lexsort((-fc, pc))[:top]
                    tops = ";".join(f"{net.regulators[ci[o]]}({counts[ci[o]]})" for o in order)
                    e.update(rank=rank, n_candidates=len(ci), percentile=rank / len(ci), overlap=int(counts[ridx[m]]),
                             expected=float(exp_[ridx[m]]), fold=float(fold_[ridx[m]]), p_hyper=float(pm), top=tops)
                    ranks.append(rank), pcts.append(rank / len(ci))
                ext_rows.append(e)
            r = np.array(ranks, dtype=float)
            rec["ext_rank_max"] = float(np.nanmax(r)) if np.isfinite(r).any() else np.nan
            rec["ext_percentile_max"] = float(np.nanmax(pcts)) if np.isfinite(pcts).any() else np.nan
            rec["mutual_best"] = bool(np.isfinite(r).all() and (r == 1).all())
            rows.append(rec)
    table = pd.DataFrame(rows, columns=["set", "signs", "sizes", "overlap", "expected", "fold", "null_median_fold", "null_p95_fold",
                                        "percentile", "p_random", "n_null", "ext_rank_max", "ext_percentile_max", "mutual_best"])
    ext = pd.DataFrame(ext_rows, columns=["set", "signs", "member", "base", "n_base", "rank", "n_candidates", "percentile", "overlap",
                                          "expected", "fold", "p_hyper", "top"])
    settings = {"tail": tail, "mor_min": float(mor_min), "min_likelihood": float(min_likelihood), "min_size": int(min_size),
                "n_random": int(n_random), "tol": float(tol), "min_pool": int(min_pool), "seed": int(seed),
                "n_regulators": {s: int(net.elig[s].sum()) for s in "+-"}, "universe_size": int(N)}
    return RegulonSpecificity(table=table, extension=ext, settings=settings)


def top_partners(regulon: pd.DataFrame, base: "str | Sequence[str]", sign: str = "+", partner_sign: Optional[str] = None,
                 mor_min: float = 0.0, min_likelihood: float = 0.0, universe: Optional[Iterable[str]] = None, min_size: int = 25,
                 top: Optional[int] = 20) -> pd.DataFrame:
    """The regulators that overlap most with ``base`` (one regulator, or the common targets of several): every regulator of the
    network ranked by the hypergeometric p of the overlap of its tail (``partner_sign``, default ``sign``) with the ``sign`` tails of the
    ``base`` regulators. Columns ``regulator``, ``n_targets`` (in the tail), ``overlap``, ``expected``, ``fold``, ``p_hyper``, ``rank``;
    ``top`` rows (``None``: all). The base members themselves are left out."""
    if sign not in ("+", "-") or (partner_sign not in (None, "+", "-")):
        raise ValueError("sign and partner_sign must be '+' or '-'")
    partner_sign = partner_sign or sign
    base = [base] if isinstance(base, str) else [str(b) for b in base]
    net = _Network(regulon, mor_min, min_likelihood, universe, min_size)
    missing = [b for b in base if b not in net.row.index]
    if missing:
        raise ValueError(f"Regulators not in the regulon: {missing}")
    rows = [int(net.row[b]) for b in base]
    core = net.core(rows, [sign] * len(rows))
    counts, size, expected, fold, p = net.scores(core, partner_sign)
    keep = net.elig[partner_sign].copy()
    keep[rows] = False
    ci = np.flatnonzero(keep)
    order = ci[np.lexsort((-np.nan_to_num(fold[ci]), p[ci]))]
    out = pd.DataFrame({"regulator": net.regulators[order], "n_targets": size[order], "overlap": counts[order],
                        "expected": expected[order], "fold": fold[order], "p_hyper": p[order]})
    out.insert(len(out.columns), "rank", np.arange(1, len(out) + 1))
    return out if top is None else out.head(top)
