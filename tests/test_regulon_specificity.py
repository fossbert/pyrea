"""pyrea.regulon_specificity / specificity_across_networks: is a set of regulators more closely linked than other regulators?"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

import pyrea as pr


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _network(seed=0, n_genes=3000, n_bg=250, planted=40, anti=False):
    """Hub network (targets drawn with heavy-tailed weights, so every regulator overlaps with others). A and B share ``planted``
    targets, A, B and C share the first 25 of them; D is an ordinary regulator. ``anti``: E's positive targets are F's negative ones."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"g{i}" for i in range(n_genes)])
    w = rng.pareto(1.5, n_genes) + 0.05
    w /= w.sum()
    rows = []

    def add(tf, idx, signs=None):
        idx = np.unique(idx)
        for i, s in zip(idx, np.ones(len(idx)) if signs is None else signs):
            rows.append((tf, genes[i], s, rng.uniform(0.2, 1.0)))

    shared = rng.choice(n_genes, planted, replace=False)
    add("A", np.concatenate([shared, rng.choice(n_genes, 150, p=w)]))
    add("B", np.concatenate([shared, rng.choice(n_genes, 130, p=w)]))
    add("C", np.concatenate([shared[:25], rng.choice(n_genes, 120, p=w)]))
    add("D", rng.choice(n_genes, 140, p=w))
    for k in range(n_bg):
        idx = np.unique(rng.choice(n_genes, int(rng.integers(30, 250)), p=w))
        add(f"bg{k}", idx, np.where(rng.random(len(idx)) < 0.3, -1.0, 1.0) if anti else None)     # anti: background has negative tails too
    if anti:
        both = rng.choice(n_genes, 45, replace=False)
        add("E", np.concatenate([both, rng.choice(n_genes, 100, p=w)]))
        f = np.unique(np.concatenate([both, rng.choice(n_genes, 100, p=w)]))
        add("F", f, np.where(np.isin(f, both), -1.0, 1.0))
    reg = pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"])
    return reg.drop_duplicates(["source", "target"])


@pytest.fixture(scope="module")
def reg():
    return _network(0)


# ---------------------------------------------------------------------- one network
def test_planted_pair_is_specific_and_ordinary_pair_is_not(reg):
    sp = pr.regulon_specificity(reg, [("A", "B"), ("A", "D")], n_random=2000, seed=1)
    t = sp.table.set_index("set")
    ab, ad = t.loc["A | B"], t.loc["A | D"]
    assert ab["signs"] == "+/+" and ab["overlap"] >= 40 and ab["fold"] > ab["null_p95_fold"]
    assert ab["p_random"] < 0.01 and ab["percentile"] > 0.99 and bool(ab["mutual_best"]) and ab["ext_rank_max"] == 1
    assert ad["p_random"] > 0.1 and not bool(ad["mutual_best"]) and ad["ext_rank_max"] > 10
    assert ab["p_random"] >= 1 / (ab["n_null"] + 1)
    ext = sp.extension.set_index(["set", "member"])
    assert ext.loc[("A | B", "B"), "rank"] == 1 and ext.loc[("A | B", "A"), "rank"] == 1
    assert ext.loc[("A | B", "B"), "n_candidates"] == ext.loc[("A | B", "A"), "n_candidates"]
    assert ext.loc[("A | B", "B"), "top"].startswith("B(")                              # the member itself leads its own top list
    assert "C(" in ext.loc[("A | B", "A"), "top"]                                       # C is the next best partner of B


def test_hub_overlap_is_significant_against_genes_but_not_against_regulators(reg):
    ov = pr.regulon_overlap(reg, ["A", "D"])
    assert ov.table["p_hyper"].iloc[0] < 1e-3                                           # gene-level null: "significant"
    sp = pr.regulon_specificity(reg, [("A", "D")], n_random=2000, seed=1)
    assert sp.table["p_random"].iloc[0] > 0.1                                           # other regulators overlap as much
    assert sp.table["fold"].iloc[0] > 2


def test_triplet(reg):
    sp = pr.regulon_specificity(reg, [("A", "B", "C"), ("A", "B", "D")], n_random=2000, seed=1)
    t = sp.table.set_index("set")
    assert t.loc["A | B | C", "signs"] == "+/+/+" and t.loc["A | B | C", "p_random"] < 0.01 and bool(t.loc["A | B | C", "mutual_best"])
    assert t.loc["A | B | D", "p_random"] > 0.05 and not bool(t.loc["A | B | D", "mutual_best"])
    e = sp.extension[sp.extension["set"] == "A | B | C"].set_index("member")
    assert list(e["base"]) == ["B & C", "A & C", "A & B"] and (e["rank"] == 1).all()
    assert e.loc["C", "overlap"] == sp.table.set_index("set").loc["A | B | C", "overlap"]


def test_tail_both_finds_the_anti_correlated_pair():
    reg = _network(2, anti=True)
    sp = pr.regulon_specificity(reg, [("E", "F")], tail="both", n_random=1500, seed=1, min_size=10)
    t = sp.table.set_index("signs")
    assert list(sp.table["signs"]) == ["+/+", "-/-", "+/-", "-/+"]
    assert t.loc["+/-", "overlap"] >= 45 and t.loc["+/-", "p_random"] < 0.01 and bool(t.loc["+/-", "mutual_best"])
    assert t.loc["+/+", "p_random"] > 0.05
    three = pr.regulon_specificity(reg, [("E", "F", "A")], tail="both", n_random=200, seed=1, min_size=10)
    assert len(three.table) == 8 and len(three.extension) == 24


def test_negative_tail_and_zero_sized_tails(reg):
    sp = pr.regulon_specificity(reg, [("A", "B")], tail="negative", n_random=200, seed=1)    # A and B have no negative targets
    r = sp.table.iloc[0]
    assert r["signs"] == "-/-" and r["sizes"] == "0/0" and np.isnan(r["p_random"]) and not bool(r["mutual_best"])
    assert np.isnan(sp.extension["rank"]).all()


def test_seed_and_reproducibility(reg):
    a = pr.regulon_specificity(reg, [("A", "B"), ("A", "D")], n_random=500, seed=3)
    b = pr.regulon_specificity(reg, [("A", "B"), ("A", "D")], n_random=500, seed=3)
    pd.testing.assert_frame_equal(a.table, b.table)
    pd.testing.assert_frame_equal(a.extension, b.extension)
    c = pr.regulon_specificity(reg, [("A", "D")], n_random=500, seed=4)
    assert c.table["null_median_fold"].iloc[0] != a.table["null_median_fold"].iloc[1]


def test_min_size_defines_the_reference(reg):
    tiny = pd.DataFrame({"source": "tiny", "target": [f"g{i}" for i in range(8)], "mor": 1.0, "likelihood": 1.0})
    a_targets = reg[reg["source"] == "A"]["target"].iloc[:8]
    tiny["target"] = a_targets.to_numpy()                                                # 8 targets, all shared with A
    reg2 = pd.concat([reg, tiny], ignore_index=True)
    big = pr.regulon_specificity(reg2, [("A", "B")], n_random=100, min_size=25)
    small = pr.regulon_specificity(reg2, [("A", "B")], n_random=100, min_size=5)
    assert small.settings["n_regulators"]["+"] == big.settings["n_regulators"]["+"] + 1
    assert "tiny" not in top_names(reg2, "A", 1000, min_size=25) and "tiny" in top_names(reg2, "A", 1000, min_size=5)


def top_names(regulon, base, top, **kw):
    return set(pr.top_partners(regulon, base, top=top, **kw)["regulator"])


def test_top_partners(reg):
    tp = pr.top_partners(reg, "A", top=5)
    assert list(tp.columns) == ["regulator", "n_targets", "overlap", "expected", "fold", "p_hyper", "rank"]
    assert tp["regulator"].iloc[0] == "B" and list(tp["rank"]) == [1, 2, 3, 4, 5] and "A" not in set(tp["regulator"])
    assert tp["p_hyper"].is_monotonic_increasing
    both = pr.top_partners(reg, ["A", "B"], top=3)                                       # common targets of A and B as the base
    assert both["regulator"].iloc[0] == "C" and not {"A", "B"} & set(pr.top_partners(reg, ["A", "B"], top=None)["regulator"])
    assert len(pr.top_partners(reg, "A", top=None)) > 100
    assert pr.top_partners(reg, "A", sign="+", partner_sign="-").empty                    # nobody has a negative tail of 25 targets
    with pytest.raises(ValueError, match="sign"):
        pr.top_partners(reg, "A", sign="x")
    with pytest.raises(ValueError, match="not in the regulon"):
        pr.top_partners(reg, "NOPE")


def test_validation(reg):
    with pytest.raises(ValueError, match="tail"):
        pr.regulon_specificity(reg, [("A", "B")], tail="x")
    with pytest.raises(ValueError, match="non-empty"):
        pr.regulon_specificity(reg, [])
    with pytest.raises(ValueError, match="at least two"):
        pr.regulon_specificity(reg, [("A",)])
    with pytest.raises(ValueError, match="unique"):
        pr.regulon_specificity(reg, [("A", "A")])
    with pytest.raises(ValueError, match="string"):
        pr.regulon_specificity(reg, ["AB"])
    with pytest.raises(ValueError, match="at most"):
        pr.regulon_specificity(reg, [("A", "B", "C", "D", "bg1", "bg2")], tail="both")
    with pytest.raises(ValueError, match="not in the regulon"):
        pr.regulon_specificity(reg, [("A", "NOPE")])
    with pytest.raises(ValueError, match="n_random"):
        pr.regulon_specificity(reg, [("A", "B")], n_random=0)
    with pytest.raises(ValueError, match="duplicated"):
        pr.regulon_specificity(pd.concat([reg, reg.iloc[:3]]), [("A", "B")])
    with pytest.raises(ValueError, match="columns"):
        pr.regulon_specificity(reg.drop(columns="mor"), [("A", "B")])
    assert "RegulonSpecificity(2 set(s)" in repr(pr.regulon_specificity(reg, [("A", "B"), ("A", "D")], n_random=50))


# ---------------------------------------------------------------------- across networks
@pytest.fixture(scope="module")
def nets():
    return {"n1": _network(10), "n2": _network(11), "n3": _network(12)}


SETS = [("A", "B"), ("A", "D")]


def test_across_networks_summary_matrix_and_figure(nets):
    ms = pr.specificity_across_networks(nets, SETS, n_random=500, seed=1, verbose=False)
    assert isinstance(ms, pr.MultiSpecificity) and "MultiSpecificity(3 network(s)" in repr(ms)
    assert list(ms.results) == ["n1", "n2", "n3"] and ms.table.columns[0] == "network" and len(ms.table) == 6 and len(ms.extension) == 12
    s = ms.summary().set_index("set")
    assert s.loc["A | B", "n_hit"] == 3 and s.loc["A | B", "n_mutual_best"] == 3 and s.index[0] == "A | B"
    assert s.loc["A | D", "n_hit"] == 0 and s.loc["A | B", "median_percentile"] > 0.99
    assert s.loc["A | B", "n_top"] == 3 and s.loc["A | D", "n_top"] == 0 and s.loc["A | B", "median_ext_percentile"] < 0.01
    assert ms.summary(top_frac=1.0).set_index("set").loc["A | D", "n_top"] == 3
    assert np.allclose(ms.table["q_random_all"], false_discovery_control(ms.table["p_random"].to_numpy(), method="bh"))
    mat = ms.matrix("percentile")
    assert mat.shape == (3, 2) and list(mat.index) == ["n1", "n2", "n3"] and (mat["A | B"] > 0.99).all()
    assert ms.matrix("fold", signs="+/+").equals(ms.matrix("fold"))
    with pytest.raises(ValueError, match="Unknown value"):
        ms.matrix("nope")
    with pytest.raises(ValueError, match="not in the table"):
        ms.matrix("fold", signs="+/-")
    fig, ax = plt.subplots()
    assert ms.plot_matrix("percentile", ax=ax) is ax
    n0 = len(ax.texts)
    ms.add_hits(ax=ax)
    assert len(ax.texts) - n0 == int(ms.hits().sum()) == 3
    with pytest.raises(ValueError, match="Draw the matrix first"):
        ms.add_hits(ax=plt.subplots()[1])


def test_missing_regulators_skipped_networks_and_paths(nets, tmp_path):
    n = dict(nets)
    n["n2"] = n["n2"][n["n2"]["source"] != "D"]
    n["x"] = n["n3"][n["n3"]["source"].str.startswith("bg")]
    with pytest.warns(UserWarning, match="skipped"):
        ms = pr.specificity_across_networks(n, SETS, n_random=100, verbose=False)
    cov = ms.coverage.set_index("network")
    assert cov.loc["n2", "n_sets"] == 1 and cov.loc["n2", "missing"] == "D" and cov.loc["n1", "n_sets"] == 2
    assert set(ms.skipped) == {"x"} and list(ms.results) == ["n1", "n2", "n3"]
    s = ms.summary().set_index("set")
    assert s.loc["A | B", "n_networks"] == 3 and s.loc["A | D", "n_networks"] == 2
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="None of the networks"):
        pr.specificity_across_networks({"x": n["x"]}, SETS, n_random=50, verbose=False)
    f = tmp_path / "net.csv"
    nets["n1"].to_csv(f, index=False)
    ms2 = pr.specificity_across_networks({"BRCA": f}, SETS, n_random=100, reader=pd.read_csv, verbose=False)
    assert list(ms2.results) == ["BRCA"] and ms2.coverage["path"].iloc[0] == str(f)


def test_cache(nets, tmp_path):
    files = []
    for name, reg in nets.items():
        f = tmp_path / f"{name}.csv"
        reg.to_csv(f, index=False)
        files.append(f)
    calls = []

    def reader(p):
        calls.append(p.name)
        return pd.read_csv(p)

    kw = dict(n_random=100, reader=reader, cache_dir=tmp_path / "cache", verbose=False)
    a = pr.specificity_across_networks(files, SETS, **kw)
    assert len(calls) == 3 and list(a.results) == ["n1", "n2", "n3"]
    b = pr.specificity_across_networks(files, SETS, **kw)
    assert len(calls) == 3
    pd.testing.assert_frame_equal(a.table, b.table)
    pr.specificity_across_networks(files, SETS, **{**kw, "n_random": 50})                # other settings recompute
    assert len(calls) == 6
    pr.specificity_across_networks(files, [("A", "B")], **kw)                            # other sets recompute
    assert len(calls) == 9


def test_parallel_equals_sequential(nets):
    seq = pr.specificity_across_networks(nets, SETS, n_random=100, seed=2, n_jobs=1, verbose=False)
    par = pr.specificity_across_networks(nets, SETS, n_random=100, seed=2, n_jobs=2, verbose=False)
    pd.testing.assert_frame_equal(seq.table, par.table)
    pd.testing.assert_frame_equal(seq.extension, par.extension)


def test_merge_by_name(nets):
    kw = dict(n_random=100, seed=3, verbose=False)
    joint = pr.specificity_across_networks(nets, SETS, **kw)
    a = pr.specificity_across_networks({"n1": nets["n1"]}, SETS, **kw)
    b = pr.specificity_across_networks({"n2": nets["n2"], "n3": nets["n3"]}, SETS, **kw)
    mg = a.merge(b)
    pd.testing.assert_frame_equal(mg.table, joint.table)
    pd.testing.assert_frame_equal(mg.extension, joint.extension)
    pd.testing.assert_frame_equal(mg.summary(), joint.summary())
    assert list(mg.results) == ["n1", "n2", "n3"] and mg.settings["n_networks"] == 3
    with pytest.raises(ValueError, match="more than one analysis"):
        a.merge(pr.specificity_across_networks({"n1": nets["n2"]}, SETS, **kw))
    with pytest.raises(ValueError, match="differs in"):
        a.merge(pr.specificity_across_networks({"n2": nets["n2"]}, SETS, **{**kw, "n_random": 120}))
    with pytest.raises(ValueError, match="Nothing"):
        pr.MultiSpecificity.concat([])


def test_tail_both_across_networks_and_export(tmp_path):
    n = {"a": _network(20, anti=True), "b": _network(21, anti=True)}
    ms = pr.specificity_across_networks(n, [("E", "F")], tail="both", n_random=300, seed=1, min_size=10, verbose=False)
    s = ms.summary().set_index("signs")
    assert set(s.index) == {"+/+", "-/-", "+/-", "-/+"} and s.loc["+/-", "n_hit"] == 2 and s.loc["+/+", "n_hit"] == 0
    assert ms.matrix("fold", signs="+/-").notna().all().all()
    out = ms.export(tmp_path / "spec.xlsx")
    sheets = pd.ExcelFile(out).sheet_names
    assert {"settings", "coverage", "summary", "table", "extension"} <= set(sheets)
    assert len(pd.read_excel(out, sheet_name="table")) == len(ms.table)
    assert (ms.export(tmp_path / "csv") / "extension.csv").exists()
