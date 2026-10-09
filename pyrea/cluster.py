"""Unsupervised clustering of samples by gene expression or VIPER protein activity.

After the "Unsupervised data analysis" of Alvarez et al., Nat Genet 2018 (GEP-NET cohort):

1. expression signatures: gene-wise z-score across samples (:func:`zscore_signature`)
2. similarity between samples: Pearson correlation (expression) or ``viperSimilarity`` of the
   protein activity matrix (:func:`viper_similarity`)
3. partitioning into k clusters by PAM (partitioning around medoids, :func:`pam`)
4. number of clusters from the *cluster reliability*: for every sample the enrichment (aREA NES) of
   the members of its cluster among its nearest samples; per cluster and globally summarised as the
   area over the cumulative curve (AOC) of the scores scaled to [0, 1]. k is the first local
   maximum of the global reliability (:func:`cluster_reliability`, :func:`cluster_samples`)
5. t-SNE of the data, PAM on the embedding and comparison of two partitions by the adjusted Rand
   index with a permutation p-value (:func:`tsne_embedding`, :func:`adjusted_rand_index`)

Pure numpy / scipy; t-SNE needs scikit-learn (``pip install pyrea[cluster]``). ``viper_similarity``
and ``pam`` reproduce ``viper::viperSimilarity`` and ``cluster::pam`` (see the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

__all__ = [
    "zscore_signature",
    "viper_similarity",
    "similarity_to_distance",
    "pam",
    "cluster_reliability",
    "cluster_samples",
    "tsne_embedding",
    "adjusted_rand_index",
    "ari_permutation_test",
    "ClusterResult",
]


# --------------------------------------------------------------------------- signatures, similarity
def zscore_signature(expression: pd.DataFrame) -> pd.DataFrame:
    """Gene-wise z-score across samples: (x - mean) / sd (sd with n-1), genes x samples.

    Genes without variance are dropped.
    """
    x = expression.astype(float)
    sd = x.std(axis=1, ddof=1)
    z = x.sub(x.mean(axis=1), axis=0).div(sd, axis=0)
    return z.loc[sd > 0]


def _sigmoid_weight(x: np.ndarray, slope: float, inflection: float) -> np.ndarray:
    return 1.0 - 1.0 / (1.0 + np.exp(slope * (x - inflection)))


def viper_similarity(activity: pd.DataFrame,
                     nn: Optional[int] = None,
                     ws: Sequence[float] = (4, 2),
                     method: str = "two.sided") -> pd.DataFrame:
    """Similarity between samples from their protein activity, as ``viper::viperSimilarity``.

    The most active / inactive proteins of each sample (weighted by a sigmoid of their NES, or the
    ``nn`` top proteins) serve as a signature whose enrichment is tested in every other sample;
    the two directions are combined into a symmetric similarity.

    Parameters
    ----------
    activity : pd.DataFrame
        Regulators x samples (e.g. VIPER NES). Non-negative input is rank-transformed first.
    nn : int, optional
        Use the ``nn`` most active (and, for ``method="two.sided"``, inactive) proteins with equal
        weight instead of the sigmoid weights.
    ws : sequence of float
        Length 2 (default ``(4, 2)``): sigmoid weight of the absolute NES, 0.5 at ``ws[0]`` and
        0.1 at ``ws[1]`` (so proteins above NES 4 count almost fully, below 2 hardly); length 1:
        the NES scaled by its maximum and raised to the power ``ws[0]``.
    method : {"two.sided", "greater", "less"}
        Use active and inactive proteins, only active or only inactive ones.

    Returns
    -------
    pd.DataFrame
        Samples x samples, symmetric. Not scaled: see :func:`similarity_to_distance`.
        (With ``nn`` the tie-breaking of ranks is deterministic, viper breaks ties at random.)
    """

    if method not in ("two.sided", "greater", "less"):
        raise ValueError("method must be 'two.sided', 'greater' or 'less'")
    names = activity.columns
    x = activity.to_numpy(float).copy()
    nrow, ncol = x.shape

    if np.nanmin(x) >= 0:
        r = np.apply_along_axis(lambda v: rankdata(v, nan_policy="omit"), 0, x)
        x = norm.ppf(r / (np.sum(~np.isnan(x), axis=0) + 1))
    x[np.isnan(x)] = 0.0
    xw = x.copy()

    if nn is None:
        if len(ws) == 1:
            if method == "greater":
                xw[xw < 0] = 0
                xw = xw / x.max(axis=0)
            elif method == "less":
                xw[xw > 0] = 0
                xw = xw / np.abs(x).max(axis=0)
            else:
                xw = xw / np.abs(x).max(axis=0)
            xw = np.sign(xw) * np.abs(xw) ** ws[0]
        else:
            slope = 1.0 / (ws[1] - ws[0]) * np.log(1 / 0.9 - 1)
            if method == "greater":
                xw[xw < 0] = 0
            elif method == "less":
                xw[xw > 0] = 0
            xw = np.sign(xw) * _sigmoid_weight(np.abs(xw), slope, ws[0])
    else:
        keep = np.zeros_like(x, dtype=bool)
        for j in range(ncol):
            col = x[:, j]
            if method == "greater":
                pos = rankdata(-col, method="ordinal")
                keep[:, j] = pos <= nn
            elif method == "less":
                pos = rankdata(col, method="ordinal")
                keep[:, j] = pos <= nn
            else:
                half = int(round(nn / 2))
                pos = rankdata(col, method="ordinal")
                keep[:, j] = ~((pos > half) & (pos < nrow - half + 1))
        xw = np.where(keep, np.sign(x), 0.0)

    nes = np.sqrt((xw ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        xw = xw / np.abs(xw).sum(axis=0)
    t2 = norm.ppf(np.apply_along_axis(rankdata, 0, x) / (nrow + 1))
    vp = (xw.T @ t2) * nes[:, None]

    a, b = vp, vp.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sym = (a ** 3 + b ** 3) / (a ** 2 + b ** 2)
    out = np.tril(sym, -1) + np.tril(sym, -1).T + np.diag(np.diag(vp))
    return pd.DataFrame(out, index=names, columns=names)


def similarity_to_distance(similarity: pd.DataFrame, kind: str = "viper") -> pd.DataFrame:
    """Distance matrix from a similarity matrix.

    ``kind="viper"``: ``1 - S_ij / ((S_ii + S_jj) / 2)`` (``as.dist`` of a ``viperSimilarity``);
    ``kind="correlation"``: ``1 - r``.
    """
    s = similarity.to_numpy(float)
    if kind == "correlation":
        d = 1.0 - s
    elif kind == "viper":
        diag = np.diag(s)
        d = 1.0 - s / ((diag[:, None] + diag[None, :]) / 2.0)
    else:
        raise ValueError("kind must be 'viper' or 'correlation'")
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=similarity.index, columns=similarity.columns)


# --------------------------------------------------------------------------- PAM
def pam(dist: Union[pd.DataFrame, np.ndarray], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Partitioning around medoids (BUILD + SWAP) on a distance matrix, as ``cluster::pam``.

    Returns
    -------
    (labels, medoids) : arrays
        ``labels`` 0..k-1 per object (cluster j belongs to ``medoids[j]``), ``medoids`` the
        positions of the k medoids in the order they were chosen.
    """
    D = np.asarray(dist, dtype=float)
    n = D.shape[0]
    if D.shape != (n, n) or not 1 <= k <= n:
        raise ValueError("dist must be square and 1 <= k <= n")

    # BUILD: most central object, then the object that decreases the cost most
    medoids = [int(np.argmin(D.sum(axis=1)))]
    d1 = D[medoids[0]].copy()
    while len(medoids) < k:
        gain = np.maximum(d1[None, :] - D, 0.0).sum(axis=1)
        gain[medoids] = -np.inf
        m = int(np.argmax(gain))
        medoids.append(m)
        d1 = np.minimum(d1, D[m])

    # SWAP: steepest descent over (medoid, non-medoid) pairs
    med = np.array(medoids)
    cost = D[med].min(axis=0).sum()
    while True:
        sub = D[med]                                    # k x n
        order = np.argsort(sub, axis=0, kind="stable")
        nearest = order[0]
        d_near = sub[nearest, np.arange(n)]
        d_second = sub[order[1], np.arange(n)] if k > 1 else np.full(n, np.inf)
        best = (cost, None, None)
        for r in range(k):
            base = np.where(nearest == r, d_second, d_near)
            total = np.minimum(base[None, :], D).sum(axis=1)   # candidate h for medoid slot r
            total[med] = np.inf
            h = int(np.argmin(total))
            if total[h] < best[0] - 1e-10:
                best = (total[h], r, h)
        if best[1] is None:
            break
        cost = best[0]
        med[best[1]] = best[2]

    labels = D[med].argmin(axis=0)
    labels[med] = np.arange(k)
    return labels, med


# --------------------------------------------------------------------------- reliability
def cluster_reliability(dist: Union[pd.DataFrame, np.ndarray], labels: Sequence, scale: str = "bounds") -> pd.DataFrame:
    """Reliability of a partition: per sample, per cluster and global (after Alvarez et al. 2018).

    Sample reliability: the members of the sample's own cluster (without the sample) are a gene set,
    the signature is the vector of negative distances of the sample to all other samples; the
    enrichment is the one-tailed aREA NES ``sqrt(m) * mean(qnorm(rank / (N + 1)))`` with ``m``
    members among ``N`` other samples. High = the cluster mates are the nearest samples.

    The scores are scaled to [0, 1]; the reliability of a cluster is the area over the cumulative
    curve (AOC) of its scaled scores, which equals their mean; the global reliability is the AOC
    over all samples.

    Parameters
    ----------
    dist : square distance matrix
    labels : cluster per sample
    scale : {"bounds", "minmax"}
        Neither the manuscript nor its supplement says how the scores are "scaled between 0 and 1".
        ``"bounds"`` (default) scales each NES
        between the smallest and the largest value possible for a cluster of that size (0 = the
        members are the farthest samples, 1 = the nearest): the scores are then comparable between
        partitions, a random partition scores about 0.5. ``"minmax"`` scales between the lowest
        and highest score of the partition; this removes the absolute level, so the global value
        then says little about the quality of the partition.

    Returns
    -------
    pd.DataFrame
        One row per sample: ``cluster``, ``nes``, ``scaled``; ``attrs["cluster_aoc"]`` (Series) and
        ``attrs["global"]`` (float). Singletons have no score (NaN) and are left out of the AOC.
    """
    if scale not in ("bounds", "minmax"):
        raise ValueError("scale must be 'bounds' or 'minmax'")
    D = np.asarray(dist, dtype=float)
    n = D.shape[0]
    lab = np.asarray(labels)
    if len(lab) != n:
        raise ValueError("labels must have one entry per sample")
    names = dist.index if isinstance(dist, pd.DataFrame) else pd.RangeIndex(n)

    N = n - 1
    t_all = norm.ppf(np.arange(1, N + 1) / (N + 1))
    csum = np.concatenate([[0.0], np.cumsum(t_all)])
    nes = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    for i in range(n):
        others = np.arange(n) != i
        member = (lab[others] == lab[i])
        m = int(member.sum())
        if m == 0:
            continue
        ranks = rankdata(-D[i, others])                 # nearest = highest rank
        nes[i] = np.sqrt(m) * norm.ppf(ranks / (N + 1))[member].mean()
        lo[i] = np.sqrt(m) * csum[m] / m                  # members are the m farthest samples
        hi[i] = np.sqrt(m) * (csum[N] - csum[N - m]) / m  # members are the m nearest samples

    if scale == "bounds":
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = (nes - lo) / (hi - lo)
    else:
        a, b = np.nanmin(nes), np.nanmax(nes)
        scaled = (nes - a) / (b - a) if b > a else np.zeros(n)
        scaled[np.isnan(nes)] = np.nan
    out = pd.DataFrame({"cluster": lab, "nes": nes, "scaled": scaled}, index=names)
    out.attrs["cluster_aoc"] = out.groupby("cluster")["scaled"].mean()
    out.attrs["global"] = float(np.nanmean(scaled))
    return out


def _first_local_maximum(values: Sequence[float]) -> int:
    """Index of the first local maximum; the maximum if the curve has none; last point at the end."""
    v = np.asarray(values, dtype=float)
    for i in range(len(v)):
        left = v[i - 1] if i > 0 else -np.inf
        right = v[i + 1] if i < len(v) - 1 else -np.inf
        if v[i] > left and v[i] >= right and i < len(v) - 1:
            return i
    return int(np.nanargmax(v))


@dataclass
class ClusterResult:
    """Result of :func:`cluster_samples`.

    Attributes
    ----------
    labels : pd.Series
        Cluster per sample, 1..k, numbered by decreasing size.
    k : int
        Number of clusters (given or the first local maximum of the global reliability).
    medoids : list
        Sample names of the medoids, in cluster order.
    reliability : pd.DataFrame
        Per sample: ``cluster``, ``nes``, ``scaled`` (see :func:`cluster_reliability`).
    scores : pd.DataFrame
        Index k: global reliability for every k tried, with the chosen scaling (``global``) and with
        both scalings (``bounds``, ``minmax``), so that the choice of k can be checked; empty if k
        was fixed.
    similarity, distance : pd.DataFrame
        Samples x samples.
    """

    labels: pd.Series
    k: int
    medoids: list
    reliability: pd.DataFrame
    scores: pd.DataFrame
    similarity: pd.DataFrame
    distance: pd.DataFrame


def _relabel(labels: np.ndarray, medoids: np.ndarray):
    """Number clusters 1..k by decreasing size; medoids follow."""
    sizes = pd.Series(labels).value_counts()
    order = list(sizes.index)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels]), [int(medoids[o]) for o in order]


def cluster_samples(data: pd.DataFrame,
                    k: Optional[int] = None,
                    k_range: Sequence[int] = range(2, 11),
                    similarity: str = "viper",
                    scale: str = "bounds") -> ClusterResult:
    """Cluster the samples (columns) of a protein-activity or expression matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Features x samples. ``similarity="viper"``: VIPER activity (NES); ``"correlation"``:
        z-scored expression signature (see :func:`zscore_signature`), Pearson correlation.
    k : int, optional
        Fixed number of clusters; otherwise the first local maximum of the global reliability over
        ``k_range`` (the maximum if there is no local maximum).
    k_range : sequence of int
        Candidates for k.
    similarity : {"viper", "correlation"}
    scale : {"bounds", "minmax"}
        Scaling of the sample reliability that decides k, see :func:`cluster_reliability`. Neither
        the manuscript nor its supplement defines it; compare both columns of ``scores``.

    Returns
    -------
    ClusterResult

    Examples
    --------
    ::

        res = pr.cluster_samples(vpres)                     # VIPER similarity, PAM, k by reliability
        res.k, res.scores                                   # chosen k, reliability per k
        res.labels.value_counts()
        pd.crosstab(res.labels, subtype)
    """

    if similarity == "viper":
        sim = viper_similarity(data)
        dist = similarity_to_distance(sim, "viper")
    elif similarity == "correlation":
        sim = pd.DataFrame(np.corrcoef(data.to_numpy(float), rowvar=False), index=data.columns, columns=data.columns)
        dist = similarity_to_distance(sim, "correlation")
    else:
        raise ValueError("similarity must be 'viper' or 'correlation'")

    if k is None:
        ks = [int(x) for x in k_range]
        both = {"bounds": [], "minmax": []}
        for kk in ks:
            lab, _ = pam(dist, kk)
            for sc in both:
                both[sc].append(cluster_reliability(dist, lab, sc).attrs["global"])
        scores = pd.DataFrame({"global": both[scale], **both}, index=pd.Index(ks, name="k"))
        k = ks[_first_local_maximum(both[scale])]
    else:
        scores = pd.DataFrame({"global": [], "bounds": [], "minmax": []}, index=pd.Index([], name="k"))

    lab, med = pam(dist, k)
    lab, med = _relabel(lab, med)
    rel = cluster_reliability(dist, lab, scale)
    return ClusterResult(pd.Series(lab, index=data.columns, name="cluster"), int(k),
                         [data.columns[m] for m in med], rel, scores, sim, dist)


# --------------------------------------------------------------------------- t-SNE, Rand index
def tsne_embedding(data: pd.DataFrame, perplexity: float = 40, n_iter: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Two-dimensional t-SNE of the samples (columns), initialised with the first two principal components.

    Defaults as in Alvarez et al. 2018 (perplexity 40, 5,000 iterations, PCA initialisation).
    Needs scikit-learn (``pip install pyrea[cluster]``). The implementation differs from the R
    package ``tsne``; the embedding is therefore similar, not identical.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover
        raise ImportError("tsne_embedding needs scikit-learn: pip install 'pyrea[cluster]'") from exc
    X = data.to_numpy(float).T
    emb = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, init="pca", random_state=seed).fit_transform(X)
    return pd.DataFrame(emb, index=data.columns, columns=["tsne1", "tsne2"])


def adjusted_rand_index(a: Sequence, b: Sequence) -> float:
    """Adjusted Rand index between two partitions of the same samples (1 = identical, ~0 = random)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b):
        raise ValueError("partitions must have the same length")
    ca, ia = np.unique(a, return_inverse=True)
    cb, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ca), len(cb)))
    np.add.at(table, (ia, ib), 1)

    def c2(x):
        return x * (x - 1) / 2.0

    sum_ij = c2(table).sum()
    sum_a, sum_b = c2(table.sum(axis=1)).sum(), c2(table.sum(axis=0)).sum()
    total = c2(len(a))
    expected = sum_a * sum_b / total
    max_index = (sum_a + sum_b) / 2.0
    if max_index == expected:
        return 1.0
    return float((sum_ij - expected) / (max_index - expected))


def ari_permutation_test(a: Sequence, b: Sequence, n_perm: int = 10000, seed: int = 0) -> Tuple[float, float]:
    """Adjusted Rand index and its permutation p-value (labels of ``b`` shuffled ``n_perm`` times).

    Returns
    -------
    (ari, p) : p = (1 + number of permutations with ARI >= observed) / (1 + n_perm)
    """
    rng = np.random.default_rng(seed)
    b = np.asarray(b)
    obs = adjusted_rand_index(a, b)
    ge = sum(adjusted_rand_index(a, rng.permutation(b)) >= obs for _ in range(n_perm))
    return obs, (1 + ge) / (1 + n_perm)
