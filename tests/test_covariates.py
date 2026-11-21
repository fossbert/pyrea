"""pyrea.covariates: gene list scores (aREA) and the registry of background variables."""

import numpy as np
import pandas as pd
import pytest

import pyrea as pr
from test_activity_specificity_multi import fake_viper


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PYREA_DATA", str(tmp_path / "pyrea_data"))
    return tmp_path / "pyrea_data" / "covariates"


def _expression(rng, n=60, n_genes=300):
    genes = [f"g{i}" for i in range(n_genes)]
    f = rng.normal(size=n)
    X = pd.DataFrame(rng.normal(size=(n_genes, n)), index=genes, columns=[f"s{i}" for i in range(n)])
    X.iloc[:40] += 1.0 * f                                                             # g0-g39 follow f
    X.iloc[40:80] -= 1.0 * f
    return X, f


# ---------------------------------------------------------------------- the registry
def test_bundled_absolute_table():
    av = pr.available_covariates().set_index("variable")
    assert {"absolute_purity", "absolute_ploidy", "absolute_called"} <= set(av.index)
    assert av.loc["absolute_purity", "location"] == "bundled" and av.loc["absolute_purity", "n_samples"] > 10000
    assert "ABSOLUTE" in av.loc["absolute_purity", "source"]
    full = pr.load_covariates("absolute_purity")
    assert full.index[0].startswith("TCGA-") and full["absolute_purity"].dropna().between(0, 1).all()
    assert pr.load_covariates().shape[1] >= 3
    got = pr.load_covariates(["absolute_purity", "absolute_called"], samples=["TCGA.OR.A5J1.01A.11R.A29S.07", "unknown"], sample_id=pr.tcga_sample_id)
    assert list(got.index) == ["TCGA.OR.A5J1.01A.11R.A29S.07", "unknown"] and got.iloc[0, 0] == pytest.approx(0.9) and got.iloc[1].isna().all()
    direct = pr.load_covariates("absolute_purity", samples=["TCGA-OR-A5J1-01"])
    assert direct.iloc[0, 0] == pytest.approx(0.9)
    with pytest.raises(ValueError, match="Unknown covariate"):
        pr.load_covariates("nope")


def test_add_and_remove_tables(user_dir):
    tab = pd.DataFrame({"my_stroma": [1.0, 2.0, 3.0], "my_immune": [0.1, np.nan, 0.3]}, index=["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0003-01"])
    f = pr.add_covariates("mine", tab, description={"my_stroma": "stroma score"}, source="test 2026")
    assert f.exists() and f.parent == user_dir
    av = pr.available_covariates().set_index("variable")
    assert av.loc["my_stroma", "location"] == "user" and av.loc["my_stroma", "description"] == "stroma score" and av.loc["my_immune", "n_samples"] == 2
    assert av.loc["my_stroma", "source"] == "test 2026" and "absolute_purity" in av.index                # bundled and user tables together
    np.testing.assert_allclose(pr.load_covariates("my_stroma")["my_stroma"].to_numpy(), [1, 2, 3])
    with pytest.raises(ValueError, match="exists"):
        pr.add_covariates("mine", tab)
    tab2 = tab.assign(my_stroma=[9.0, 9.0, 9.0])
    pr.add_covariates("mine", tab2, overwrite=True)
    assert pr.load_covariates("my_stroma")["my_stroma"].tolist() == [9, 9, 9]
    with pytest.raises(ValueError, match="already in the registry"):
        pr.add_covariates("other", pd.DataFrame({"my_stroma": [1.0]}, index=["x"]))
    with pytest.raises(ValueError, match="already in the registry"):
        pr.add_covariates("other", pd.DataFrame({"absolute_purity": [1.0]}, index=["x"]))
    with pytest.raises(ValueError, match="bundled"):
        pr.add_covariates("absolute", tab, overwrite=True)
    with pytest.raises(ValueError, match="unique"):
        pr.add_covariates("dup", pd.DataFrame({"v": [1.0, 2.0]}, index=["a", "a"]))
    with pytest.raises(ValueError, match="numeric"):
        pr.add_covariates("txt", pd.DataFrame({"v": ["a", "b"]}, index=["a", "b"]))
    with pytest.raises(ValueError, match="non-empty"):
        pr.add_covariates("empty", pd.DataFrame())
    with pytest.raises(ValueError, match="name may contain"):
        pr.add_covariates("bad name!", tab)
    pr.remove_covariates("mine")
    assert "my_stroma" not in set(pr.available_covariates()["variable"]) and not f.exists()
    with pytest.raises(ValueError, match="bundled tables cannot"):
        pr.remove_covariates("absolute")
    with pytest.raises(ValueError, match="No table"):
        pr.remove_covariates("never")


# ---------------------------------------------------------------------- resolving the ways to give covariates
def test_resolve_forms(rng, user_dir, monkeypatch):
    monkeypatch.setattr("pyrea.rbackend.viper.viper_activity", fake_viper)
    X, f = _expression(rng)
    X.columns = [f"TCGA-AA-{i:04d}-01" for i in range(X.shape[1])]
    pr.add_covariates("mine", pd.DataFrame({"my_stroma": np.arange(X.shape[1], dtype=float)}, index=X.columns))
    assert pr.resolve_covariates(None, X) is None
    one = pr.resolve_covariates("my_stroma", X)
    assert list(one.columns) == ["my_stroma"] and list(one.index) == list(X.columns) and one["my_stroma"].iloc[5] == 5
    ser = pr.resolve_covariates(pd.Series(rng.normal(size=60), index=X.columns, name="own"), X)
    assert list(ser.columns) == ["own"]
    half = pr.resolve_covariates(pd.DataFrame({"part": [1.0] * 30}, index=X.columns[:30]), X)
    assert half["part"].isna().sum() == 30                                                 # samples without a value are NaN
    sc = pr.resolve_covariates({"up": [f"g{i}" for i in range(40)]}, X, method="mad", min_genes=10)
    assert list(sc.columns) == ["up"] and np.corrcoef(sc["up"], f)[0, 1] > 0.9
    mixed = pr.resolve_covariates(["my_stroma", {"up": [f"g{i}" for i in range(40)]}, pd.Series(1.0, index=X.columns, name="own")], X, min_genes=10)
    assert list(mixed.columns) == ["my_stroma", "up", "own"]
    assert pr.resolve_covariates(("my_stroma",), X).shape == (60, 1)
    with pytest.raises(ValueError, match="repeat"):
        pr.resolve_covariates(["my_stroma", "my_stroma"], X)
    with pytest.raises(ValueError, match="duplicated"):
        pr.resolve_covariates(pd.Series(1.0, index=["a", "a"], name="x"), X)
    with pytest.raises(ValueError, match="registry name"):
        pr.resolve_covariates(3.14, X)
    X2 = X.copy()
    X2.columns = [f"TCGA.AA.{i:04d}.01A.11R" for i in range(60)]
    with pytest.raises(ValueError, match="sample_id=pr.tcga_sample_id"):
        pr.resolve_covariates("absolute_purity", X2)


# ---------------------------------------------------------------------- gene list scores
def test_score_genesets_input_forms(rng, monkeypatch):
    monkeypatch.setattr("pyrea.rbackend.viper.viper_activity", fake_viper)
    X, f = _expression(rng)
    up, down = [f"g{i}" for i in range(40)], [f"g{i}" for i in range(40, 80)]
    d = pr.score_genesets(X, {"up": up, "down": down}, min_genes=10)
    assert list(d.columns) == ["up", "down"] and list(d.index) == list(X.columns)
    assert np.corrcoef(d["up"], f)[0, 1] > 0.9 and np.corrcoef(d["down"], f)[0, 1] < -0.9
    assert d.attrs["n_genes"].to_dict() == {"up": 40, "down": 40}
    one = pr.score_genesets(X, up, min_genes=10)
    assert list(one.columns) == ["score"]
    pd.testing.assert_series_equal(one["score"], d["up"], check_names=False)
    partial = pr.score_genesets(X, {"mixed": up[:20] + ["not_a_gene"] * 3 + ["also_not"]}, min_genes=10)
    assert partial.attrs["n_genes"]["mixed"] == 20
    with pytest.warns(UserWarning, match="left out"):
        kept = pr.score_genesets(X, {"ok": up, "tiny": up[:3]}, min_genes=10)
    assert list(kept.columns) == ["ok"]
    with pytest.raises(ValueError, match="None of the gene lists"):
        pr.score_genesets(X, {"tiny": up[:3]}, min_genes=10)
    with pytest.raises(ValueError, match="unique"):
        pr.score_genesets(pd.concat([X, X.iloc[:2]]), up)
    with pytest.raises(ValueError, match="genesets must be"):
        pr.score_genesets(X, 3.14)


def test_score_genesets_bundled_collection_names(rng, monkeypatch):
    monkeypatch.setattr("pyrea.rbackend.viper.viper_activity", fake_viper)
    hallmark = pr.load_genesets("h", verbose=False)
    genes = sorted({g for v in hallmark.values() for g in v})[:2500]
    X = pd.DataFrame(rng.normal(size=(len(genes), 40)), index=genes, columns=[f"s{i}" for i in range(40)])
    h = pr.score_genesets(X, "h", min_genes=10)
    assert set(h.columns) <= set(hallmark) and h.shape[0] == 40 and h.shape[1] > 10
    h2 = pr.score_genesets(X, ["hallmark"], min_genes=10)
    assert list(h2.columns) == list(h.columns)


def _viper_ready():
    try:
        from pyrea.rbackend import r_package

        r_package("viper")
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _viper_ready(), reason="needs R + Bioconductor package viper")


@viper
def test_score_genesets_with_aREA(rng):
    X, f = _expression(rng, n=80)
    up, down = [f"g{i}" for i in range(40)], [f"g{i}" for i in range(40, 80)]
    d = pr.score_genesets(X, {"up": up, "down": down}, min_genes=10)
    assert np.corrcoef(d["up"], f)[0, 1] > 0.9 and np.corrcoef(d["down"], f)[0, 1] < -0.9
    other = pr.score_genesets(X, {"up": up}, method="rank", min_genes=10)
    assert np.corrcoef(other["up"], d["up"])[0, 1] > 0.8                                  # another signature, the same direction
    raw = pr.score_genesets(X, {"up": up}, nes=False, min_genes=10)
    assert np.corrcoef(raw["up"], d["up"])[0, 1] > 0.95
    null = pr.score_genesets(X, {"random": [f"g{i}" for i in range(200, 240)]}, min_genes=10)
    assert abs(np.corrcoef(null["random"], f)[0, 1]) < 0.4                                # a list without signal does not follow f
    pd.testing.assert_frame_equal(d[["up"]], pr.score_genesets(X, {"up": up}, min_genes=10), check_exact=False, atol=1e-8)


@viper
def test_score_genesets_method_none_scores_every_sample_on_its_own_ranks(rng):
    """With method="none" aREA ranks the genes within each sample: a sample's score does not depend on the other samples."""
    X, f = _expression(rng, n=60)
    up = [f"g{i}" for i in range(40)]
    full = pr.score_genesets(X, {"up": up}, method="none", min_genes=10)
    few = pr.score_genesets(X.iloc[:, :20], {"up": up}, method="none", min_genes=10)
    np.testing.assert_allclose(few["up"].to_numpy(), full["up"].iloc[:20].to_numpy(), atol=1e-8)
    assert np.corrcoef(full["up"], f)[0, 1] > 0.9
    rel = pr.score_genesets(X, {"up": up}, method="mad", min_genes=10)
    rel_few = pr.score_genesets(X.iloc[:, :20], {"up": up}, method="mad", min_genes=10)
    assert not np.allclose(rel_few["up"].to_numpy(), rel["up"].iloc[:20].to_numpy())           # "mad" is relative to the cohort
