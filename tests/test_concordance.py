"""pyrea.concordance -- activity vs expression effects."""

import numpy as np
import pandas as pd
import pytest

import pyrea as pr
from pyrea.concordance import _cliff


@pytest.fixture
def data(rng):
    n = 160
    samples = [f"s{i}" for i in range(n)]
    genes = ["CONC", "ACT", "EXP", "OPP", "NONE"]
    mut = pd.DataFrame(0, index=genes, columns=samples)
    for g in genes:
        mut.loc[g, rng.choice(samples, 50, replace=False)] = 1
    E = pd.DataFrame(rng.normal(size=(5, n)), index=genes, columns=samples)
    G = pd.DataFrame(rng.normal(size=(5, n)), index=genes, columns=samples)
    R = pd.DataFrame(rng.normal(size=(5, n)), index=genes, columns=samples)
    m = lambda g: (mut.loc[g] == 1).to_numpy()
    E.loc["CONC", m("CONC")] += 2; G.loc["CONC", m("CONC")] += 2
    G.loc["ACT", m("ACT")] += 2; R.loc["ACT", m("ACT")] += 2
    E.loc["EXP", m("EXP")] += 2
    E.loc["OPP", m("OPP")] += 2; G.loc["OPP", m("OPP")] -= 2
    return mut, G, E, R


def test_cliff_matches_auc(rng):
    from scipy.stats import mannwhitneyu
    a, b = rng.normal(1, 1, 30), rng.normal(0, 1, 50)
    auc = mannwhitneyu(a, b).statistic / (30 * 50)
    assert _cliff(a, b) == pytest.approx(2 * auc - 1)
    assert _cliff(a, a + 100) == -1.0


def test_categories(data):
    mut, G, E, R = data
    res = pr.activity_vs_expression(mut, G, E, rpt=R, n_boot=300)
    assert res.loc["CONC", "category"] == "concordant"
    assert res.loc["ACT", "category"] == "activity_only"
    assert res.loc["EXP", "category"] == "expression_only"
    assert res.loc["OPP", "category"] == "opposite"
    assert res.loc["NONE", "category"] == "none"
    assert set(res["category"]) <= set(pr.CONCORDANCE_CATEGORIES)
    assert set(pr.CONCORDANCE_COLORS) == set(pr.CONCORDANCE_CATEGORIES)


def test_effects_signs_and_intervals(data):
    mut, G, E, R = data
    res = pr.activity_vs_expression(mut, G, E, rpt=R, n_boot=400)
    assert res.loc["ACT", "delta_G"] > 0.6 and abs(res.loc["ACT", "delta_expr"]) < 0.3
    assert res.loc["OPP", "delta_expr"] > 0.6 > 0 > res.loc["OPP", "delta_G"]
    # the interval of a real difference excludes 0, that of equal effects contains it
    assert res.loc["ACT", "abs_low"] > 0 and res.loc["ACT", "p_abs"] < 0.05
    assert res.loc["EXP", "abs_high"] < 0
    assert res.loc["CONC", "abs_low"] < 0 < res.loc["CONC", "abs_high"]
    assert res.loc["OPP", "diff_high"] < 0
    # interval brackets the estimate
    assert (res["diff_low"] <= res["delta_diff"] + 1e-9).all() and (res["delta_diff"] <= res["diff_high"] + 1e-9).all()


def test_beyond_expression_needs_rpt_in_direction_of_g(data):
    mut, G, E, R = data
    res = pr.activity_vs_expression(mut, G, E, rpt=R, n_boot=100)
    assert res.loc["ACT", "beyond_expression"]
    assert not res.loc["EXP", "beyond_expression"]
    assert not res.loc["NONE", "beyond_expression"]
    assert "beyond_expression" not in pr.activity_vs_expression(mut, G, E, n_boot=100).columns
    assert "delta_RPT" not in pr.activity_vs_expression(mut, G, E, n_boot=100).columns


def test_reproducible_and_seeded(data):
    mut, G, E, _ = data
    a = pr.activity_vs_expression(mut, G, E, n_boot=200, seed=1)
    b = pr.activity_vs_expression(mut, G, E, n_boot=200, seed=1)
    c = pr.activity_vs_expression(mut, G, E, n_boot=200, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["diff_low"], c["diff_low"])
    np.testing.assert_allclose(a["delta_G"], c["delta_G"])   # point estimates do not depend on the seed


def test_nan_groups_and_thresholds_and_missing_genes(data):
    mut, G, E, _ = data
    m2 = mut.astype(float)
    m2.loc["ACT", m2.columns[:100]] = np.nan                 # not profiled: excluded
    res = pr.activity_vs_expression(m2, G, E, n_boot=50, min_mut=5, min_wt=5)
    assert res.loc["ACT", "n_mut"] + res.loc["ACT", "n_wt"] <= 60
    assert len(pr.activity_vs_expression(mut, G, E, n_boot=50, min_mut=1000)) == 0
    G2 = G.drop(index="EXP")
    assert "EXP" not in pr.activity_vs_expression(mut, G2, E, n_boot=50).index


def test_nan_in_traits_excludes_sample(data):
    mut, G, E, _ = data
    E2 = E.copy()
    E2.loc["CONC", E2.columns[:20]] = np.nan
    r = pr.activity_vs_expression(mut, G, E2, n_boot=50)
    assert r.loc["CONC", "n_mut"] + r.loc["CONC", "n_wt"] <= 140
    assert np.isfinite(r.loc["CONC", ["delta_expr", "delta_G", "p_diff"]].astype(float)).all()


def test_key_maps_fusion_rows(data):
    mut, G, E, _ = data
    one = mut.loc[["ACT"]].rename(index={"ACT": "FUSION_X"})
    res = pr.activity_vs_expression(one, G, E, key={"FUSION_X": "ACT"}, n_boot=100)
    assert list(res.index) == ["FUSION_X"] and res.loc["FUSION_X", "category"] == "activity_only"


def test_single_row_and_empty(data):
    mut, G, E, _ = data
    r = pr.activity_vs_expression(mut.loc[["CONC"]], G, E, n_boot=50)
    assert r.loc["CONC", "q_G"] == r.loc["CONC", "p_G"]
    empty = pr.activity_vs_expression(mut.iloc[:0], G, E)
    assert len(empty) == 0 and "category" in empty.columns
