"""compare_phenotypes."""

import numpy as np
import pandas as pd
import pytest

import pyrea as pr


@pytest.fixture
def setup(rng):
    """A and B both lower the activity of TGT (phenocopy), C does not; D overlaps A."""
    n = 150
    s = [f"s{i}" for i in range(n)]
    g = pd.Series(0.0, index=s)
    A, B, C = g.copy(), g.copy(), g.copy()
    A.iloc[:15] = 1
    B.iloc[15:30] = 1
    C.iloc[30:45] = 1
    B.iloc[10:12] = 1                 # two samples in A and B -> overlap
    C.iloc[145:] = np.nan             # not profiled for C
    act = pd.DataFrame(rng.normal(size=(2, n)), index=["TGT", "OTHER"], columns=s)
    rpt = pd.DataFrame(rng.normal(size=(2, n)), index=["TGT", "OTHER"], columns=s)
    act.loc["TGT", A[A == 1].index] -= 3
    act.loc["TGT", B[B == 1].index] -= 3
    return {"A": A, "B": B, "C": C}, act, rpt


def test_groups_overlap_and_reference(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt)
    sc = cmp.scores
    assert (sc["group"] == "overlap").sum() == 2
    assert (sc["group"] == "A").sum() == 13 and (sc["group"] == "B").sum() == 15
    assert sc["group"].isna().sum() == 5                      # not profiled for C -> no reference
    assert (sc["group"] == "neither").sum() == 150 - 13 - 15 - 2 - 15 - 5
    ref = sc.index[sc["group"] == "neither"]
    assert not set(ref) & set(groups["A"][groups["A"] == 1].index)
    assert not set(ref) & set(groups["C"][groups["C"].isna()].index)


def test_summary_and_phenocopy(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt)
    s = cmp.summary
    assert s.loc[("A", "G"), "auc"] < 0.1 and s.loc[("A", "G"), "p"] < 1e-4   # lower than reference
    assert 0.2 < s.loc[("C", "G"), "auc"] < 0.8 and s.loc[("C", "G"), "p"] > 0.05
    c = cmp.cross
    assert c.loc[("A", "B"), "frac_mutant_phenotype"] > 0.5 and c.loc[("A", "B"), "p"] < 1e-3   # B scores like A
    assert c.loc[("B", "A"), "frac_mutant_phenotype"] > 0.5
    assert c.loc[("A", "C"), "frac_mutant_phenotype"] < 0.2                                     # C does not
    assert c.loc[("A", "A"), "in_sample"] and not c.loc[("A", "B"), "in_sample"]
    pw = cmp.pairwise.loc[("G", "A", "B")]
    assert pw["p"] > 0.05                                      # A and B do not differ from each other
    assert cmp.pairwise.loc[("G", "A", "C"), "p"] < 1e-3


def test_without_rpt_and_small_group(setup):
    groups, act, _ = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, min_group=14)
    assert "RPT" not in cmp.scores.columns
    assert cmp.scores["MPS[A]"].isna().all()                   # 13 exclusive A samples < 14


def test_unknown_target(setup):
    groups, act, rpt = setup
    with pytest.raises(ValueError, match="no activity"):
        pr.compare_phenotypes(groups, "NOPE", act, rpt)


def test_group_without_carriers_warns_and_stays_quiet_otherwise(setup):
    import warnings

    groups, act, rpt = setup
    ref = pr.compare_phenotypes(groups, "TGT", act, rpt)
    empty = pd.Series(0.0, index=groups["A"].index)                   # nobody carries it
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cmp = pr.compare_phenotypes({**groups, "none": empty}, "TGT", act, rpt)
    assert not [x for x in w if issubclass(x.category, RuntimeWarning)]            # no numpy noise
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert len(msgs) == 1 and "'none' (n=0)" in msgs[0] and "nobody carries" in msgs[0]
    assert cmp.scores["MPS[none]"].isna().all()
    assert cmp.summary.loc[("none", "MPS[none]"), "n"] == 0 and np.isnan(cmp.summary.loc[("none", "G"), "mean"])
    assert np.isnan(cmp.cross.loc[("none", "A"), "mean_rl"])
    # the other groups are unaffected by the empty one
    pd.testing.assert_frame_equal(cmp.summary.loc[["A", "B", "C"]].reindex(ref.summary.loc[["A", "B", "C"]].index), ref.summary.loc[["A", "B", "C"]])


def test_no_reference_is_an_error(setup):
    groups, act, rpt = setup
    bad = {k: v.where(v == 1) for k, v in groups.items()}               # 1 or NaN, never 0
    with pytest.raises(ValueError, match="No reference samples"):
        pr.compare_phenotypes(bad, "TGT", act, rpt)


def test_auc_intervals(setup):
    groups, act, rpt = setup
    cmp = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=300)
    s, c, p = cmp.summary, cmp.cross, cmp.pairwise
    for tab in (s.drop(index="neither", level=0), c, p):
        assert {"auc_low", "auc_high"} <= set(tab.columns)
        ok = tab.dropna(subset=["auc"])
        assert (ok["auc_low"] <= ok["auc_high"]).all()
        assert ((ok["auc_low"] >= 0) & (ok["auc_high"] <= 1)).all()
    # strong effect: interval far from 0.5 and brackets the estimate; null group: interval contains 0.5
    assert s.loc[("A", "G"), "auc_high"] < 0.2 and s.loc[("A", "G"), "auc_low"] <= s.loc[("A", "G"), "auc"] <= s.loc[("A", "G"), "auc_high"]
    assert s.loc[("C", "G"), "auc_low"] < 0.5 < s.loc[("C", "G"), "auc_high"]
    # phenocopy: B on A's scale is clearly above the reference, C is not
    assert c.loc[("A", "B"), "auc_low"] > 0.5
    assert c.loc[("A", "C"), "auc_low"] < 0.5 < c.loc[("A", "C"), "auc_high"] + 0.2
    # the summary row (group, MPS[scale]) and the cross row are the same quantity
    assert s.loc[("B", "MPS[A]"), "auc_low"] == c.loc[("A", "B"), "auc_low"]
    assert s.loc[("B", "MPS[A]"), "auc_high"] == c.loc[("A", "B"), "auc_high"]
    assert p.loc[("G", "A", "B"), "auc_low"] < 0.5 < p.loc[("G", "A", "B"), "auc_high"]   # A and B do not differ


def test_auc_intervals_off_and_reproducible(setup):
    groups, act, rpt = setup
    off = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=0)
    assert off.summary["auc_low"].isna().all() and off.cross["auc_high"].isna().all()
    a = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=100, seed=3)
    b = pr.compare_phenotypes(groups, "TGT", act, rpt, n_boot=100, seed=3)
    pd.testing.assert_frame_equal(a.cross, b.cross)
    pd.testing.assert_series_equal(a.summary["auc"], off.summary["auc"])     # point estimates unchanged


def test_auc_interval_narrower_with_more_samples(rng):
    def run(n_group):
        n = 60 + 4 * n_group
        s = [f"s{i}" for i in range(n)]
        A = pd.Series(0.0, index=s); A.iloc[:n_group] = 1
        B = pd.Series(0.0, index=s); B.iloc[n_group:2 * n_group] = 1
        act = pd.DataFrame(rng.normal(size=(1, n)), index=["T"], columns=s)
        act.loc["T", A[A == 1].index] -= 1.5
        act.loc["T", B[B == 1].index] -= 1.0
        sm = pr.compare_phenotypes({"A": A, "B": B}, "T", act, n_boot=200).cross
        return float(sm.loc[("A", "B"), "auc_high"] - sm.loc[("A", "B"), "auc_low"])
    assert run(12) > run(60)


def test_auc_intervals_small_or_missing_groups_are_nan(setup):
    groups, act, rpt = setup
    with pytest.warns(UserWarning):
        cmp = pr.compare_phenotypes(groups, "TGT", act, min_group=14, n_boot=100)
    assert np.isnan(cmp.cross.loc[("A", "B"), "auc_low"])          # A has 13 samples: no scale


def test_vectorised_kde_matches_relative_likelihood(rng):
    from pyrea.phenotypes import _rl_rows
    m = rng.normal(1, 1, (4, 12))
    w = rng.normal(0, 1, (4, 40))
    x = rng.normal(0, 2, (4, 30))
    for bw in (1.0, 1.7):
        got = _rl_rows(x, m, w, bw)
        ref = np.vstack([pr.relative_likelihood(x[i], m[i], w[i], bw_adjust=bw) for i in range(4)])
        np.testing.assert_allclose(got, ref, atol=1e-12)
    m[2] = 5.0                                            # one distinct value: no KDE
    assert np.isnan(_rl_rows(x, m, w, 1.0)[2]).all()


def test_auc_rows_matches_scalar_and_handles_nan(rng):
    from pyrea.phenotypes import _auc, _auc_rows
    x, y = rng.normal(size=(5, 9)), rng.normal(size=(5, 14))
    x[1, 2] = np.nan
    got = _auc_rows(x, y)
    np.testing.assert_allclose(got, [_auc(x[i], y[i]) for i in range(5)])
