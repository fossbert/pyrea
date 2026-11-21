"""pyrea.area_nes / metaviper: numpy aREA and metaVIPER, identical to viper."""

import numpy as np
import pandas as pd
import pytest

import pyrea as pr


def _viper_ready():
    try:
        from pyrea.rbackend import r_package

        r_package("viper")
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _viper_ready(), reason="needs R + Bioconductor package viper")


def _signatures(rng, n_genes=1500, m=12):
    genes = [f"g{i}" for i in range(n_genes)]
    return pd.DataFrame(rng.normal(size=(n_genes, m)), index=genes, columns=[f"s{j}" for j in range(m)])


def _regulon(rng, genes, n_reg=15, size=(30, 120), signs=True, lik=True):
    rows = []
    for k in range(n_reg):
        n = int(rng.integers(*size))
        t = rng.choice(genes, n, replace=False)
        mor = np.where(rng.random(n) < 0.25, -1.0, 1.0) if signs else np.ones(n)
        mor[: n // 10] *= rng.uniform(0.1, 0.9)                                  # some modes of regulation inside (-1, 1)
        rows += list(zip([f"TF{k}"] * n, t, mor, rng.uniform(0.1, 1.0, n) if lik else np.ones(n)))
    return pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"])


@viper
def test_area_nes_matches_viper(rng):
    X = _signatures(rng)
    reg = _regulon(rng, X.index)
    mine = pr.area_nes(X, reg, minsize=25)
    ref = pr.viper_activity(X, reg, method="none", minsize=25, eset_filter=False)
    assert set(mine.index) == set(ref.index) and list(mine.columns) == list(ref.columns)
    np.testing.assert_allclose(mine.loc[ref.index].to_numpy(), ref.to_numpy(), atol=1e-8, rtol=1e-8)


@viper
def test_area_nes_matches_viper_for_large_regulons_and_unit_weights(rng):
    X = _signatures(rng, n_genes=3000, m=6)
    big = _regulon(rng, X.index, n_reg=4, size=(1200, 1500))                           # more than 1000 targets: viper's loop path
    np.testing.assert_allclose(pr.area_nes(X, big).loc[:].to_numpy(),
                               pr.viper_activity(X, big, method="none", minsize=25, eset_filter=False).loc[pr.area_nes(X, big).index].to_numpy(), atol=1e-8)
    plain = _regulon(rng, X.index, signs=False, lik=False)
    np.testing.assert_allclose(pr.area_nes(X, plain).to_numpy(),
                               pr.viper_activity(X, plain, method="none", minsize=25, eset_filter=False).loc[pr.area_nes(X, plain).index].to_numpy(), atol=1e-8)


@viper
def test_regulons_are_restricted_to_the_genes_and_minsize(rng):
    X = _signatures(rng, n_genes=800).iloc[:600]                                      # 200 genes of the regulon are not in the signatures
    reg = _regulon(rng, [f"g{i}" for i in range(800)], size=(20, 80))
    mine = pr.area_nes(X, reg, minsize=30)
    ref = pr.viper_activity(X, reg, method="none", minsize=30, eset_filter=False)
    assert set(mine.index) == set(ref.index) and 0 < len(mine) < reg["source"].nunique()
    np.testing.assert_allclose(mine.loc[ref.index].to_numpy(), ref.to_numpy(), atol=1e-8)
    assert set(pr.area_nes(X, reg, minsize=30, regulators=["TF1", "TF2"]).index) <= {"TF1", "TF2"}


@viper
def test_metaviper_matches_viper_with_a_list_of_regulons(rng):
    X = _signatures(rng)
    nets = {"a": _regulon(rng, X.index, n_reg=10), "b": _regulon(rng, X.index, n_reg=14), "c": _regulon(rng, X.index, n_reg=8)}
    nets["c"]["source"] = nets["c"]["source"].str.replace("TF", "TZ").where(nets["c"]["source"].isin(["TF0", "TF1", "TF2", "TF3"]) == False, nets["c"]["source"])
    per = {k: pr.area_nes(X, v) for k, v in nets.items()}
    ref = pr.viper_activity(X, nets, method="none", minsize=25, eset_filter=False)                # metaVIPER in R
    mine = pr.metaviper(per)
    assert set(mine.index) == set(ref.index)
    np.testing.assert_allclose(mine.loc[ref.index].to_numpy(), ref.to_numpy(), atol=1e-8)
    for mvws in (0.0, 2.0):
        ref2 = pr.viper_activity(X, nets, method="none", minsize=25, eset_filter=False, mvws=mvws)
        np.testing.assert_allclose(pr.metaviper(per, mvws=mvws).loc[ref2.index].to_numpy(), ref2.to_numpy(), atol=1e-8)


def test_metaviper_integration_rules():
    a = pd.DataFrame({"s1": [2.0, -1.0], "s2": [0.5, 1.0]}, index=["X", "Y"])
    b = pd.DataFrame({"s1": [4.0], "s2": [-1.5]}, index=["X"])
    out = pr.metaviper({"a": a, "b": b})
    assert out.loc["X", "s1"] == pytest.approx((2 * 2 + 4 * 4) / (2 + 4))                # weighted by |NES|
    assert out.loc["X", "s2"] == pytest.approx((0.5 * 0.5 + -1.5 * 1.5) / (0.5 + 1.5))
    assert out.loc["Y", "s1"] == -1.0                                                     # only in one network: its NES
    assert out.loc["X", "s1"] != a.loc["X", "s1"]
    arr = np.stack([a.reindex(["X", "Y"]).to_numpy(), b.reindex(["X", "Y"]).to_numpy()])
    allowed = np.array([[True, True], [False, False]])                                    # network b off for both signatures
    off = pr.metaviper(arr, allowed=allowed)
    np.testing.assert_allclose(off[0], [2.0, 0.5])
    zero = pr.metaviper({"a": pd.DataFrame({"s": [0.0]}, index=["Z"])})
    assert np.isnan(zero.loc["Z", "s"])                                                   # 0 / 0
    with pytest.raises(ValueError, match="same signatures"):
        pr.metaviper({"a": a, "b": b.rename(columns={"s1": "q"})})
    with pytest.raises(ValueError, match="No networks"):
        pr.metaviper({})


def test_numpy_input_and_errors(rng):
    X = _signatures(rng, m=4)
    reg = _regulon(rng, X.index, n_reg=5)
    compact = pr.compact_regulons(reg, list(X.index), minsize=25)
    arr = pr.area_nes(X.to_numpy(), compact)
    pd.testing.assert_frame_equal(pr.area_nes(X, compact), pd.DataFrame(arr, index=list(compact), columns=X.columns))
    t = pr.rank_transform(X.to_numpy())
    np.testing.assert_allclose(pr.area_nes(X, compact, transformed=t).to_numpy(), arr)
    bad = X.copy()
    bad.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        pr.area_nes(bad, reg)
    with pytest.raises(ValueError, match="Give genes"):
        pr.area_nes(X.to_numpy(), reg)
    with pytest.raises(ValueError, match="no regulons"):
        pr.area_nes(X, reg, minsize=100000)
    with pytest.raises(ValueError, match="columns"):
        pr.compact_regulons(reg.drop(columns="mor"), list(X.index))
    sign = pd.DataFrame({"source": "T", "target": X.index[:40], "mor": 1.0})            # no likelihood column: all 1
    assert np.isfinite(pr.area_nes(X, sign)).all().all()
    shifted = X.copy()
    shifted.loc[X.index[:40]] += 3.0                                                       # the targets go up: positive NES
    assert (pr.area_nes(shifted, sign).iloc[0] > 3).all()
