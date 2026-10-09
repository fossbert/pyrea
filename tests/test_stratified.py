"""stratified_comparison and differential_activity."""

import numpy as np
import pandas as pd
import pytest

import pyrea as pr


@pytest.fixture
def cohort(rng):
    """Two subtypes with different baselines. The mutation lowers TGT_A in subtype A only,
    TGT_B in both, NONE in neither."""
    n_a, n_b = 120, 80
    samples = [f"s{i}" for i in range(n_a + n_b)]
    strata = pd.Series(["A"] * n_a + ["B"] * n_b, index=samples)
    mut = pd.Series(0.0, index=samples)
    mut.iloc[:25] = 1                       # 25 mutants in A
    mut.iloc[n_a:n_a + 20] = 1              # 20 in B
    act = pd.DataFrame(rng.normal(size=(3, len(samples))), index=["TGT_A", "TGT_B", "NONE"], columns=samples)
    act.loc[:, strata == "B"] += 5          # baseline difference between the strata
    act.loc["TGT_A", (mut == 1) & (strata == "A")] -= 2
    act.loc["TGT_B", mut == 1] -= 2
    return mut, strata, act


def test_effects_within_strata_ignore_baseline(cohort):
    mut, strata, act = cohort
    res = pr.stratified_comparison(mut, strata, "TGT_A", act, n_boot=300)
    e = res.effects
    assert e.loc[("A", "G"), "auc"] < 0.15 and e.loc[("A", "G"), "p"] < 1e-4
    assert 0.3 < e.loc[("B", "G"), "auc"] < 0.7 and e.loc[("B", "G"), "p"] > 0.05
    assert e.loc[("A", "G"), "ci_high"] < 0.3 and e.loc[("A", "G"), "ci_low"] <= e.loc[("A", "G"), "auc"] <= e.loc[("A", "G"), "ci_high"]
    assert (e.loc[("A", "G"), "n_mut"], e.loc[("A", "G"), "n_wt"]) == (25, 95)


def test_interaction_detects_different_effects_only(cohort):
    mut, strata, act = cohort
    assert pr.stratified_comparison(mut, strata, "TGT_A", act, n_boot=300).interaction.loc["G", "p_interaction"] < 0.01
    same = pr.stratified_comparison(mut, strata, "TGT_B", act, n_boot=300)
    assert same.interaction.loc["G", "p_interaction"] > 0.05           # same effect in both strata
    pw = pr.stratified_comparison(mut, strata, "TGT_A", act, n_boot=500).pairwise.loc[("G", "A", "B")]
    assert pw["auc_diff"] < -0.2 and pw["ci_high"] < 0 and pw["p"] < 0.01


def test_traits_nan_and_min_group(cohort):
    mut, strata, act = cohort
    rpt = act.copy()
    res = pr.stratified_comparison(mut, strata, "NONE", act, rpt=rpt, expression=act, n_boot=100)
    assert set(res.effects.index.get_level_values("trait")) == {"G", "RPT", "expression"}
    m2 = mut.copy()
    m2[(strata == "B") & (m2 == 1)] = np.nan                          # B has no mutants left
    one = pr.stratified_comparison(m2, strata, "TGT_A", act, n_boot=100)
    assert set(one.effects.index.get_level_values("stratum")) == {"A"}
    assert one.interaction.empty or len(one.pairwise) == 0
    with pytest.raises(ValueError, match="enough"):
        pr.stratified_comparison(m2, strata, "TGT_A", act, min_group=200)
    with pytest.raises(ValueError, match="not in"):
        pr.stratified_comparison(mut, strata, "NOPE", act)


def test_differential_activity_all_regulators(cohort):
    mut, strata, act = cohort
    res = pr.differential_activity(mut, act, strata)
    assert res.index[0] == "TGT_A"                                     # strongest interaction first
    assert res.loc["TGT_A", "A:auc"] < 0.15 and res.loc["TGT_A", "interaction:p"] < 1e-3
    assert res.loc["TGT_B", "interaction:p"] > 0.05
    assert res.loc["TGT_B", "A:auc"] < 0.3 and res.loc["TGT_B", "B:auc"] < 0.3
    assert res.loc["NONE", "A:q"] > 0.1
    assert (res.filter(like=":q").to_numpy() >= res.filter(like=":p").to_numpy() - 1e-12).all()
    single = pr.differential_activity(mut, act)                        # no strata: pooled "all"
    assert list(single.columns) == ["all:auc", "all:p", "all:q"]


def test_differential_matches_stratified(cohort):
    mut, strata, act = cohort
    full = pr.differential_activity(mut, act, strata)
    one = pr.stratified_comparison(mut, strata, "TGT_B", act, n_boot=50)
    for st in "AB":
        assert full.loc["TGT_B", f"{st}:auc"] == pytest.approx(one.effects.loc[(st, "G"), "auc"])
        assert full.loc["TGT_B", f"{st}:p"] == pytest.approx(one.effects.loc[(st, "G"), "p"], rel=1e-6)
    assert full.loc["TGT_B", "interaction:p"] == pytest.approx(one.interaction.loc["G", "p_interaction"])
