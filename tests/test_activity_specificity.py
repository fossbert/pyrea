"""pyrea.activity_specificity: how closely are the activities of a set of regulators linked, compared with other regulators?"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr

import pyrea as pr


def _activity(seed=0, n=240, m=300):
    """Regulators x samples. A stroma factor S (observed as `purity` with noise) drives a module (A, B, C, D and 40 others);
    A, B, C also share a factor U, so after removing the purity only they belong together. E and F are anti-correlated via V."""
    rng = np.random.default_rng(seed)
    S, U, V = rng.normal(size=(3, n))
    rows = {}
    for i in range(m):
        load = 1.0 if i < 40 else 0.0
        rows[f"bg{i}"] = load * S + rng.normal(size=n) * (0.8 if load else 1.0)
    rows["A"] = 1.0 * S + 1.4 * U + 0.6 * rng.normal(size=n)
    rows["B"] = 1.0 * S + 1.4 * U + 0.6 * rng.normal(size=n)
    rows["C"] = 1.0 * S + 1.0 * U + 0.8 * rng.normal(size=n)
    rows["D"] = 1.4 * S + 0.6 * rng.normal(size=n)                                     # shares the stroma factor only
    rows["E"] = 1.5 * V + 0.6 * rng.normal(size=n)
    rows["F"] = -1.5 * V + 0.6 * rng.normal(size=n)
    samples = [f"s{i}" for i in range(n)]
    act = pd.DataFrame(rows, index=samples).T
    purity = pd.Series(S + 0.3 * rng.normal(size=n), index=samples, name="purity")
    return act, purity


@pytest.fixture(scope="module")
def data():
    return _activity(0)


def test_pair_correlations_match_scipy_and_the_partial_by_hand(data):
    act, purity = data
    sp = pr.activity_specificity(act, [("A", "B")], covariates=purity)
    r = sp.pairs.iloc[0]
    rho, p = spearmanr(act.loc["A"], act.loc["B"])
    assert r["variant"] == "full" and r["n"] == 240 and r["rho"] == pytest.approx(rho) and r["p"] == pytest.approx(p, rel=1e-6)
    ra, rb, rp = (rankdata(act.loc["A"]), rankdata(act.loc["B"]), rankdata(purity))
    D = np.column_stack([np.ones(240), rp])
    res = lambda y: y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    manual = np.corrcoef(res(ra), res(rb))[0, 1]
    assert r["partial_rho"] == pytest.approx(manual)
    assert r["partial_p"] < 1e-10 and r["partial_rho"] < r["rho"]
    no_cov = pr.activity_specificity(act, [("A", "B")])
    assert "partial_rho" not in no_cov.pairs.columns and "padj_ext_rank_max" not in no_cov.table.columns
    assert set(no_cov.extension["adjusted"]) == {False}


def test_ranks_raw_and_adjusted(data):
    act, purity = data
    sp = pr.activity_specificity(act, [("A", "B"), ("A", "D")], covariates=purity)
    t = sp.table.set_index("set")
    ab, ad = t.loc["A | B"], t.loc["A | D"]
    assert ab["rho_mean"] > 0.8 and ab["padj_mutual_best"] and ab["padj_ext_rank_max"] == 1 and ab["padj_ext_percentile_max"] < 0.01
    assert ad["rho_mean"] > 0.4                                                              # raw: the stroma factor links them
    assert ad["partial_abs_rho_min"] < 0.3 and not ad["padj_mutual_best"] and ad["padj_ext_percentile_max"] > 0.2
    e = sp.extension[(sp.extension["set"] == "A | B") & sp.extension["adjusted"]].set_index("member")
    assert e.loc["A", "rank"] == 1 and e.loc["B", "rank"] == 1 and e.loc["A", "base"] == "B"
    assert e.loc["A", "n_candidates"] == len(act) - 1
    assert e.loc["A", "top"].startswith("A(") and "C(" in e.loc["A", "top"]                  # the member itself leads its list (base left out)
    assert e.loc["A", "stat"] > 0.5


def test_triplet_ranks_by_r2(data):
    act, purity = data
    sp = pr.activity_specificity(act, [("A", "B", "C"), ("A", "B", "D")], covariates=purity)
    ext = sp.extension[sp.extension["adjusted"]]
    c = ext[(ext["set"] == "A | B | C")].set_index("member")
    assert list(c["base"]) == ["B & C", "A & C", "A & B"] and (c["rank"] <= 3).all()
    d = ext[(ext["set"] == "A | B | D")].set_index("member")
    assert d.loc["D", "rank"] > 20                                                           # D adds nothing once the stroma is removed
    t = sp.table.set_index("set")
    assert t.loc["A | B | C", "padj_ext_percentile_max"] < t.loc["A | B | D", "padj_ext_percentile_max"]
    assert len(sp.pairs) == 5 * 1 and set(sp.pairs["variant"]) == {"full"}                   # 5 distinct pairs (A|B once)


def test_direction_for_anti_correlated_pairs(data):
    act, _ = data
    ab = pr.activity_specificity(act, [("E", "F")]).table.iloc[0]
    assert ab["rho_mean"] < -0.8 and ab["mutual_best"]                                       # abs: anti-correlation counts
    pos = pr.activity_specificity(act, [("E", "F")], direction="positive").table.iloc[0]
    assert not pos["mutual_best"] and pos["ext_rank_max"] > 50
    neg = pr.activity_specificity(act, [("E", "F")], direction="negative").table.iloc[0]
    assert neg["mutual_best"]
    ext = pr.activity_specificity(act, [("E", "F")], direction="negative").extension
    assert (ext["stat"] < -0.8).all()                                                        # the signed correlation is shown


def test_missing_covariates_variants_and_validation(data):
    act, purity = data
    cov = purity.copy()
    cov.iloc[:30] = np.nan
    sp = pr.activity_specificity(act, [("A", "B")], covariates=cov)
    assert sp.settings["n_samples"] == 210 and sp.settings["n_dropped"] == 30 and sp.pairs["n"].eq(210).all()
    assert sp.settings["covariates"] == ["purity"]
    rho = spearmanr(act.loc["A", act.columns[30:]], act.loc["B", act.columns[30:]])[0]
    assert sp.pairs["rho"].iloc[0] == pytest.approx(rho)                                     # the raw statistic uses the same samples

    var = {"candidate": act.loc[["A", "B"]], "pleiotropy": act.loc[["A", "B"]] * 1.0 + np.random.default_rng(1).normal(size=(2, 240))}
    sv = pr.activity_specificity(act, [("A", "B")], covariates=purity, variants=var)
    assert list(sv.pairs["variant"]) == ["full", "candidate", "pleiotropy"] and sv.settings["variants"] == ["full", "candidate", "pleiotropy"]
    assert sv.pairs.set_index("variant").loc["candidate", "rho"] == pytest.approx(sv.pairs.set_index("variant").loc["full", "rho"])
    assert sv.pairs.set_index("variant").loc["pleiotropy", "rho"] < sv.pairs.set_index("variant").loc["full", "rho"]

    with pytest.raises(ValueError, match="direction"):
        pr.activity_specificity(act, [("A", "B")], direction="x")
    with pytest.raises(ValueError, match="at least two"):
        pr.activity_specificity(act, [("A",)])
    with pytest.raises(ValueError, match="unique"):
        pr.activity_specificity(act, [("A", "A")])
    with pytest.raises(ValueError, match="string"):
        pr.activity_specificity(act, ["AB"])
    with pytest.raises(ValueError, match="non-empty"):
        pr.activity_specificity(act, [])
    with pytest.raises(ValueError, match="not in activity"):
        pr.activity_specificity(act, [("A", "NOPE")])
    with pytest.raises(ValueError, match="not in candidate"):
        pr.activity_specificity(act, [("A", "C")], variants={"candidate": act.loc[["A"]]})
    with pytest.raises(ValueError, match="'full'"):
        pr.activity_specificity(act, [("A", "B")], variants={"full": act})
    with pytest.raises(ValueError, match="duplicated"):
        pr.activity_specificity(act, [("A", "B")], covariates=pd.concat([purity, purity.iloc[:2]]))
    with pytest.raises(ValueError, match="usable samples"):
        pr.activity_specificity(act, [("A", "B")], min_samples=500)
    bad = act.copy()
    bad.loc["A", "s0"] = np.nan
    with pytest.raises(ValueError, match="missing or constant"):
        pr.activity_specificity(bad, [("A", "B")])
    assert "ActivitySpecificity(1 set(s), 240 samples" in repr(pr.activity_specificity(act, [("A", "B")]))


def test_two_covariates_and_activity_partners(data):
    act, purity = data
    cov = pd.DataFrame({"purity": purity, "noise": np.random.default_rng(3).normal(size=240)})
    sp = pr.activity_specificity(act, [("A", "B")], covariates=cov)
    assert sp.settings["covariates"] == ["purity", "noise"]
    assert sp.pairs["partial_p"].iloc[0] == pytest.approx(
        spearman_partial_p(act, cov, "A", "B"), rel=1e-6)
    tp = pr.activity_partners(act, "A", covariates=purity, top=3)
    assert list(tp.columns) == ["regulator", "stat", "rank"] and tp["regulator"].iloc[0] == "B" and list(tp["rank"]) == [1, 2, 3]
    raw = pr.activity_partners(act, "A", top=3)
    assert raw["regulator"].iloc[0] == "B" and set(raw["regulator"]) & {"C", "D"}              # raw: the module shows up
    both = pr.activity_partners(act, ["A", "B"], covariates=purity, top=2)
    assert both["regulator"].iloc[0] == "C" and not {"A", "B"} & set(pr.activity_partners(act, ["A", "B"], top=None)["regulator"])
    neg = pr.activity_partners(act, "E", direction="negative", top=1)
    assert neg["regulator"].iloc[0] == "F" and neg["stat"].iloc[0] < -0.8
    with pytest.raises(ValueError, match="not in the activity"):
        pr.activity_partners(act, "NOPE")
    with pytest.raises(ValueError, match="direction"):
        pr.activity_partners(act, "A", direction="x")


def spearman_partial_p(act, cov, a, b):
    from scipy.stats import t
    D = np.column_stack([np.ones(len(cov)), rankdata(cov["purity"]), rankdata(cov["noise"])])
    res = lambda y: y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    r = np.corrcoef(res(rankdata(act.loc[a])), res(rankdata(act.loc[b])))[0, 1]
    df = len(cov) - 2 - 2
    return 2 * t.sf(abs(r * np.sqrt(df / (1 - r ** 2))), df)
