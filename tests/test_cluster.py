"""pyrea.cluster -- clustering by VIPER similarity, PAM and cluster reliability."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, rankdata

import pyrea as pr


def _r_ready(*pkgs):
    try:
        from pyrea.rbackend import r_package

        for p in pkgs:
            r_package(p)
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _r_ready("viper"), reason="needs R + viper")
rcluster = pytest.mark.skipif(not _r_ready("cluster"), reason="needs R package cluster")


@pytest.fixture
def blobs(rng):
    """Three groups of samples with distinct protein activity patterns."""
    genes = [f"p{i}" for i in range(150)]
    truth = np.repeat([0, 1, 2], [30, 25, 20])
    centers = rng.normal(0, 2.5, size=(3, 150))
    x = centers[truth].T + rng.normal(0, 1.5, size=(150, len(truth)))
    return pd.DataFrame(x, index=genes, columns=[f"s{i}" for i in range(len(truth))]), truth


def test_zscore_signature(rng):
    x = pd.DataFrame(rng.normal(5, 3, size=(4, 30)), index=list("abcd"))
    x.loc["e"] = 1.0                                           # no variance -> dropped
    z = pr.zscore_signature(x)
    assert list(z.index) == list("abcd")
    np.testing.assert_allclose(z.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(z.std(axis=1, ddof=1), 1)


def test_viper_similarity_properties(blobs):
    x, truth = blobs
    s = pr.viper_similarity(x)
    assert s.shape == (len(truth), len(truth)) and (s.index == x.columns).all()
    np.testing.assert_allclose(s.to_numpy(), s.to_numpy().T)
    same = truth[:, None] == truth[None, :]
    off = ~np.eye(len(truth), dtype=bool)
    assert s.to_numpy()[same & off].mean() > s.to_numpy()[~same].mean() + 1
    d = pr.similarity_to_distance(s)
    np.testing.assert_allclose(np.diag(d), 0)
    assert d.to_numpy()[same & off].mean() < d.to_numpy()[~same].mean()


@viper
@pytest.mark.parametrize("kwargs", [{}, {"method": "greater"}, {"method": "less"}, {"ws": (3,)}, {"nn": 40},
                                    {"nn": 40, "method": "greater"}])
def test_viper_similarity_matches_viper(blobs, kwargs):
    import rpy2.robjects as ro
    from pyrea.rbackend import r_package
    from pyrea.rbackend.regulon import frame_to_r_matrix, matrix_from_r

    x, _ = blobs
    x = x.iloc[:, ::3]
    r_package("viper")
    args = ", ".join(f"{k} = {repr(list(v)) if k == 'ws' else repr(v)}".replace("[", "c(").replace("]", ")")
                     .replace("'", '"') for k, v in kwargs.items())
    f = ro.r(f"function(x) unclass(viperSimilarity(x{', ' + args if args else ''}))")
    ref = matrix_from_r(f(frame_to_r_matrix(x)))
    mine = pr.viper_similarity(x, **kwargs)
    np.testing.assert_allclose(mine.to_numpy(), ref.to_numpy(), rtol=1e-8, atol=1e-8)


@viper
def test_distance_matches_as_dist(blobs):
    import rpy2.robjects as ro
    from pyrea.rbackend import r_package
    from pyrea.rbackend.regulon import frame_to_r_matrix, matrix_from_r

    x, _ = blobs
    x = x.iloc[:, ::3]
    r_package("viper")
    ref = matrix_from_r(ro.r("function(x) as.matrix(as.dist(viperSimilarity(x)))")(frame_to_r_matrix(x)))
    mine = pr.similarity_to_distance(pr.viper_similarity(x))
    np.testing.assert_allclose(mine.to_numpy(), ref.to_numpy(), atol=1e-8)


def test_pam_recovers_blobs_and_is_a_local_optimum(blobs):
    x, truth = blobs
    d = pr.similarity_to_distance(pr.viper_similarity(x))
    labels, med = pr.pam(d, 3)
    assert pr.adjusted_rand_index(labels, truth) > 0.95
    assert len(set(med)) == 3 and all(labels[m] == j for j, m in enumerate(med))
    D = d.to_numpy()
    cost = D[med].min(axis=0).sum()
    for r in range(3):                                # no single swap improves the cost
        for h in range(len(D)):
            if h in med:
                continue
            m2 = med.copy(); m2[r] = h
            assert D[m2].min(axis=0).sum() >= cost - 1e-9
    with pytest.raises(ValueError):
        pr.pam(d, 0)


@rcluster
@pytest.mark.parametrize("k", [2, 3, 5])
def test_pam_matches_cluster_pam(rng, k):
    import rpy2.robjects as ro
    from pyrea.rbackend import r_package
    from pyrea.rbackend.regulon import frame_to_r_matrix

    r_package("cluster")
    pts = np.vstack([rng.normal(c, 1.0, size=(25, 2)) for c in (0, 4, 8, 12, 16)])
    D = pd.DataFrame(np.linalg.norm(pts[:, None] - pts[None], axis=2))
    res = ro.r("function(m, k) {p <- cluster::pam(as.dist(m), k, diss = TRUE); list(p$clustering, p$id.med)}")(
        frame_to_r_matrix(D), k)
    r_labels, r_med = np.asarray(res[0]), np.asarray(res[1]) - 1
    labels, med = pr.pam(D, k)
    assert pr.adjusted_rand_index(labels, r_labels) == pytest.approx(1.0)
    assert set(med) == set(r_med)


def test_reliability_formula_and_scaling(blobs):
    x, truth = blobs
    d = pr.similarity_to_distance(pr.viper_similarity(x))
    rel = pr.cluster_reliability(d, truth)
    i = 0
    others = np.arange(len(truth)) != i
    ranks = rankdata(-d.to_numpy()[i, others])
    member = truth[others] == truth[i]
    expect = np.sqrt(member.sum()) * norm.ppf(ranks / (others.sum() + 1))[member].mean()
    assert rel["nes"].iloc[0] == pytest.approx(expect)
    assert rel["scaled"].between(-1e-9, 1 + 1e-9).all()
    assert rel.attrs["global"] == pytest.approx(rel["scaled"].mean()) and rel.attrs["global"] > 0.95
    mm = pr.cluster_reliability(d, truth, scale="minmax")
    assert mm["scaled"].min() == pytest.approx(0) and mm["scaled"].max() == pytest.approx(1)
    with pytest.raises(ValueError):
        pr.cluster_reliability(d, truth, scale="z")
    assert set(rel.attrs["cluster_aoc"].index) == {0, 1, 2}
    wrong = pr.cluster_reliability(d, np.random.default_rng(0).permutation(truth))
    assert 0.4 < wrong.attrs["global"] < 0.6                    # a shuffled partition is at chance level


@viper
def test_reliability_nes_matches_area(blobs):
    from pyrea.utils import gene_sets_to_regulon

    x, truth = blobs
    d = pr.similarity_to_distance(pr.viper_similarity(x))
    rel = pr.cluster_reliability(d, truth)
    i = 4
    names = np.array(x.columns)
    others = np.arange(len(truth)) != i
    sig = pd.Series(-d.to_numpy()[i, others], index=names[others], name="sig")
    members = list(names[others][truth[others] == truth[i]])
    reg = gene_sets_to_regulon({"cluster": members}, minsize=5)
    nes = pr.aREA(sig, reg, minsize=5).iloc[0, 0]
    assert rel["nes"].iloc[i] == pytest.approx(nes, rel=1e-6)


def test_cluster_samples_chooses_k_and_labels(blobs):
    x, truth = blobs
    res = pr.cluster_samples(x, k_range=range(2, 8))
    assert res.k == 3
    assert list(res.scores.index) == list(range(2, 8))
    assert list(res.scores.columns) == ["global", "bounds", "minmax"]
    assert (res.scores["global"] == res.scores["bounds"]).all()
    assert pr.adjusted_rand_index(res.labels, truth) > 0.95
    assert res.labels.value_counts().is_monotonic_decreasing and set(res.labels) == {1, 2, 3}
    assert len(res.medoids) == 3 and all(m in x.columns for m in res.medoids)
    assert res.reliability.attrs["global"] > 0.5
    # the scaling matters: the global value over k peaks at the true k only with 'bounds'
    assert res.scores["global"].idxmax() == 3 and res.scores.loc[3, "global"] > 0.95
    mm = pr.cluster_samples(x, k_range=range(2, 8), scale="minmax")
    assert (mm.scores["global"] == mm.scores["minmax"]).all() and mm.scores["bounds"].idxmax() == 3
    fixed = pr.cluster_samples(x, k=2)
    assert fixed.k == 2 and fixed.scores.empty


def test_cluster_samples_correlation_on_zscored_expression(blobs):
    x, truth = blobs
    res = pr.cluster_samples(pr.zscore_signature(x), similarity="correlation", k=3)
    assert pr.adjusted_rand_index(res.labels, truth) > 0.9
    with pytest.raises(ValueError):
        pr.cluster_samples(x, similarity="euclid")


def test_adjusted_rand_index_and_permutation(rng):
    sk = pytest.importorskip("sklearn.metrics")
    a = rng.integers(0, 4, 80)
    b = np.where(rng.random(80) < 0.7, a, rng.integers(0, 4, 80))
    assert pr.adjusted_rand_index(a, b) == pytest.approx(sk.adjusted_rand_score(a, b))
    assert pr.adjusted_rand_index(a, a + 10) == pytest.approx(1.0)
    ari, p = pr.ari_permutation_test(a, b, n_perm=200)
    assert ari > 0.3 and p == pytest.approx(1 / 201)
    _, p_null = pr.ari_permutation_test(a, rng.integers(0, 4, 80), n_perm=200)
    assert p_null > 0.05


def test_tsne_embedding_runs(blobs):
    pytest.importorskip("sklearn")
    x, truth = blobs
    emb = pr.tsne_embedding(x, perplexity=10, n_iter=300)
    assert emb.shape == (len(truth), 2) and (emb.index == x.columns).all()
    labels, _ = pr.pam(pr.similarity_to_distance(pd.DataFrame(np.linalg.norm(
        emb.to_numpy()[:, None] - emb.to_numpy()[None], axis=2)), "correlation") * 0 + np.linalg.norm(
        emb.to_numpy()[:, None] - emb.to_numpy()[None], axis=2), 3)
    assert pr.adjusted_rand_index(labels, truth) > 0.8
