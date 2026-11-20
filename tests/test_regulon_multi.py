"""pyrea.overlap_across_networks and the pair selection (focus, pairs) of regulon_overlap."""

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import false_discovery_control

import pyrea as pr
from test_regulon_overlap import _regulon


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _variant(base, seed):
    """Same A, B, C regulons as ``base`` (so their shared targets recur) with different background regulators."""
    other = _regulon(np.random.default_rng(seed))
    return pd.concat([base[base["source"].isin(list("ABC"))], other[~other["source"].isin(list("ABCD"))]], ignore_index=True)


@pytest.fixture
def nets():
    base = _regulon(np.random.default_rng(0))
    return {"n1": base, "n2": _variant(base, 11), "n3": _variant(base, 12)}


# ---------------------------------------------------------------------- pair selection in regulon_overlap
def test_focus_and_pairs_select_the_same_rows(rng):
    reg = _regulon(rng, discordant=20)
    full = pr.regulon_overlap(reg, ["A", "B", "C", "D"], n_perm=0)
    foc = pr.regulon_overlap(reg, ["A", "B", "C", "D"], focus="A", n_perm=0)
    sel = pr.regulon_overlap(reg, pairs=[("A", "B"), ("C", "D")], n_perm=0)
    assert [(a, b) for a, b in zip(foc.table.tf1, foc.table.tf2)] == [("A", "B"), ("A", "C"), ("A", "D")]
    assert [(a, b) for a, b in zip(sel.table.tf1, sel.table.tf2)] == [("A", "B"), ("C", "D")]
    cols = ["overlap", "expected", "fold", "jaccard", "p_hyper"]
    for part in (foc, sel):
        for _, r in part.table.iterrows():
            f = full.table[(full.table.tf1 == r.tf1) & (full.table.tf2 == r.tf2)].iloc[0]
            assert np.allclose(r[cols].astype(float), f[cols].astype(float))
    assert set(sel.shared_targets["tf1"] + sel.shared_targets["tf2"]) == {"AB", "CD"}
    assert len(sel.pairs) == 2
    assert set(sel.tfs) == {"A", "B", "C", "D"}


def test_pair_selection_changes_the_multiple_testing_only(rng):
    reg = _regulon(rng)
    full = pr.regulon_overlap(reg, ["A", "B", "C"])
    ab = pr.regulon_overlap(reg, pairs=[("A", "B")])
    assert ab.table["q_hyper"].iloc[0] == ab.table["p_hyper"].iloc[0]                    # one test: q = p
    assert full.table["q_hyper"].iloc[0] >= full.table["p_hyper"].iloc[0]


def test_pair_selection_validation(rng):
    reg = _regulon(rng)
    with pytest.raises(ValueError, match="focus"):
        pr.regulon_overlap(reg, ["A", "B"], focus="C")
    with pytest.raises(ValueError, match="either focus or pairs"):
        pr.regulon_overlap(reg, ["A", "B"], focus="A", pairs=[("A", "B")])
    with pytest.raises(ValueError, match="two different"):
        pr.regulon_overlap(reg, pairs=[("A", "A")])
    with pytest.raises(ValueError, match="tfs or pairs"):
        pr.regulon_overlap(reg)
    with pytest.raises(ValueError, match="not in tfs"):
        pr.regulon_overlap(reg, ["A", "B"], pairs=[("A", "C")])
    with pytest.raises(ValueError, match="not in the regulon"):
        pr.regulon_overlap(reg, pairs=[("A", "NOPE")])
    both = pr.regulon_overlap(reg, pairs=[("A", "B"), ("B", "A")])                      # a pair and its reverse are one pair
    assert len(both.pairs) == 1


# ---------------------------------------------------------------------- across networks
def test_across_networks_table_and_summary(nets):
    mo = pr.overlap_across_networks(nets, ["A", "B", "C"], n_perm=300, verbose=False)
    assert isinstance(mo, pr.MultiOverlap)
    assert list(mo.results) == ["n1", "n2", "n3"]
    assert mo.table.columns[0] == "network" and set(mo.table["network"]) == {"n1", "n2", "n3"}
    assert len(mo.table) == 9 and len(mo.pairs) == 9 and len(mo.coverage) == 3
    s = mo.summary().set_index(["tf1", "tf2"])
    assert s.loc[("A", "B"), "n_hit"] == 3 and s.loc[("A", "B"), "frac_hit"] == 1.0
    assert s.loc[("A", "B"), "median_fold"] > 3
    assert s.loc[("B", "C"), "n_hit"] == 0 and s.loc[("B", "C"), "networks_hit"] == ""
    assert s.index[0] == ("A", "B")                                                      # most consistent pair first
    # BH over all rows of all networks
    expect = false_discovery_control(mo.table["p_perm"].to_numpy(), method="bh")
    assert np.allclose(mo.table["q_perm_all"], expect)
    assert (mo.table["q_perm_all"] >= mo.table["q_perm"] - 1e-12).all()
    mat = mo.matrix("fold")
    assert mat.shape == (3, 3) and list(mat.index) == ["n1", "n2", "n3"] and "A | B" in mat.columns
    assert mo.matrix("overlap", signs=("+", "+")).loc["n1", "A | B"] == mo.results["n1"].table.iloc[0]["overlap"]
    assert "MultiOverlap(3 network(s)" in repr(mo)


def test_hit_definition_uses_hypergeometric_without_permutations(nets):
    mo = pr.overlap_across_networks(nets, ["A", "B"], n_perm=0, verbose=False)
    assert "q_perm_all" not in mo.table.columns and "q_hyper_all" in mo.table.columns
    assert mo.hits().all()
    with pytest.raises(ValueError, match="without permutations"):
        mo.hits(p="perm")
    assert not mo.hits(fold_min=1e6).any() and not mo.hits(q=0).any()


def test_focus_across_networks(nets):
    mo = pr.overlap_across_networks(nets, ["A", "B", "C"], focus="A", n_perm=0, verbose=False)
    assert set(zip(mo.table.tf1, mo.table.tf2)) == {("A", "B"), ("A", "C")}
    assert mo.settings["pairs"] == ["A|B", "A|C"]
    mo2 = pr.overlap_across_networks(nets, pairs=[("B", "C")], n_perm=0, verbose=False)
    assert set(zip(mo2.table.tf1, mo2.table.tf2)) == {("B", "C")}


def test_missing_regulators_and_skipped_networks(nets, rng):
    nets = dict(nets)
    nets["n2"] = nets["n2"][nets["n2"]["source"] != "C"]                                 # C missing in n2
    nets["empty"] = nets["n3"][nets["n3"]["source"].isin(["B", "C"]) | nets["n3"]["source"].str.startswith("bg")]
    nets["empty"] = nets["empty"][nets["empty"]["source"] != "A"]
    mo = pr.overlap_across_networks(nets, ["A", "B", "C"], n_perm=0, verbose=False)
    cov = mo.coverage.set_index("network")
    assert cov.loc["n2", "n_pairs"] == 1 and cov.loc["n2", "missing"] == "C"
    assert cov.loc["empty", "n_pairs"] == 1                                              # only B | C is covered
    assert set(mo.table.loc[mo.table.network == "n2", "tf2"]) == {"B"}
    s = mo.summary().set_index(["tf1", "tf2"])
    assert s.loc[("A", "B"), "n_networks"] == 3          # n1, n2, n3 (not "empty": no A)
    assert s.loc[("A", "C"), "n_networks"] == 2          # n1, n3 (C missing in n2, A in "empty")
    assert s.loc[("B", "C"), "n_networks"] == 3          # n1, n3, "empty" (C missing in n2)
    only_a = {k: v[v["source"] != "A"] for k, v in nets.items() if k in ("n1", "n2")}
    only_a["n3"] = nets["n3"]
    with pytest.warns(UserWarning, match="skipped"):
        mo2 = pr.overlap_across_networks(only_a, pairs=[("A", "B")], n_perm=0, verbose=False)
    assert list(mo2.results) == ["n3"] and set(mo2.skipped) == {"n1", "n2"}
    with pytest.warns(UserWarning, match="skipped"), pytest.raises(ValueError, match="None of the networks"):
        pr.overlap_across_networks({"x": nets["n1"]}, pairs=[("A", "NOPE")], n_perm=0, verbose=False)


def test_paths_reader_names_and_cache(nets, tmp_path):
    paths = []
    for name, reg in nets.items():
        f = tmp_path / f"{name}.csv"
        reg.to_csv(f, index=False)
        paths.append(f)
    calls = []

    def reader(p):
        calls.append(p.name)
        return pd.read_csv(p)

    cache = tmp_path / "cache"
    kw = dict(n_perm=200, reader=reader, cache_dir=cache, verbose=False)
    mo = pr.overlap_across_networks(paths, ["A", "B", "C"], **kw)
    assert list(mo.results) == ["n1", "n2", "n3"] and len(calls) == 3                    # names = file stems
    assert len(list(cache.glob("*.pkl"))) == 3
    again = pr.overlap_across_networks(paths, ["A", "B", "C"], **kw)
    assert len(calls) == 3                                                               # served from the cache
    pd.testing.assert_frame_equal(mo.table, again.table)
    pr.overlap_across_networks(paths, ["A", "B", "C"], **{**kw, "n_perm": 100})           # new settings: recomputed
    assert len(calls) == 6 and len(list(cache.glob("*.pkl"))) == 6
    pr.overlap_across_networks(paths, pairs=[("A", "B")], **kw)                           # new pairs: recomputed
    assert len(calls) == 9
    paths[0].write_text(paths[0].read_text() + "\n")                                      # changed file: recomputed
    pr.overlap_across_networks(paths, ["A", "B", "C"], **kw)
    assert len(calls) == 10 and calls[-1] == "n1.csv"
    named = pr.overlap_across_networks(paths, ["A", "B"], names=["x", "y", "z"], **{**kw, "cache_dir": None})
    assert list(named.results) == ["x", "y", "z"]


def test_names_from_folders_when_file_names_repeat(nets, tmp_path):
    paths = []
    for name, sub in zip(nets, ["BRCA/ARACNe", "GBM", "STAD_TCGA/ARACNe"]):
        d = tmp_path / "networks" / sub
        d.mkdir(parents=True)
        nets[name].to_csv(d / "FullReg.csv", index=False)
        paths.append(d / "FullReg.csv")
    mo = pr.overlap_across_networks(paths, ["A", "B"], n_perm=0, reader=pd.read_csv, verbose=False)
    assert list(mo.results) == ["BRCA/ARACNe", "GBM", "STAD_TCGA/ARACNe"]
    same = pr.overlap_across_networks(paths[:1], ["A", "B"], n_perm=0, reader=pd.read_csv, verbose=False)
    assert list(same.results) == ["FullReg"]


def test_names_validation(nets):
    with pytest.raises(ValueError, match="need names"):
        pr.overlap_across_networks(list(nets.values()), ["A", "B"], n_perm=0)
    with pytest.raises(ValueError, match="unique"):
        pr.overlap_across_networks(list(nets.values()), ["A", "B"], names=["a", "a", "b"], n_perm=0)
    with pytest.raises(ValueError, match="differ in length"):
        pr.overlap_across_networks(list(nets.values()), ["A", "B"], names=["a"], n_perm=0)
    with pytest.raises(ValueError, match="only for a sequence"):
        pr.overlap_across_networks(nets, ["A", "B"], names=["a", "b", "c"], n_perm=0)
    with pytest.raises(ValueError, match="No networks"):
        pr.overlap_across_networks([], ["A", "B"], n_perm=0)


def test_parallel_equals_sequential(nets):
    seq = pr.overlap_across_networks(nets, ["A", "B", "C"], n_perm=100, n_jobs=1, verbose=False)
    par = pr.overlap_across_networks(nets, ["A", "B", "C"], n_perm=100, n_jobs=2, verbose=False)
    pd.testing.assert_frame_equal(seq.table, par.table)
    pd.testing.assert_frame_equal(seq.shared_targets, par.shared_targets)


def test_recurrent_targets_collect_enrich_export(nets, tmp_path):
    mo = pr.overlap_across_networks(nets, ["A", "B", "C"], n_perm=0, verbose=False)
    sh = [set(mo.results[n].shared("A", "B", signs=("+", "+")).index) for n in nets]
    in3 = set.intersection(*sh)
    in2 = {g for g in set.union(*sh) if sum(g in s for s in sh) >= 2}
    rec = mo.recurrent_targets(min_networks=3)
    ab = rec[(rec.tf1 == "A") & (rec.tf2 == "B")]
    assert set(ab["gene"]) == in3 and len(in3) >= 25 and (ab["n_networks"] == 3).all() and (ab["frac"] == 1.0).all()
    assert set(mo.recurrent_targets(min_networks=2).query("tf1 == 'A' and tf2 == 'B'")["gene"]) == in2
    assert mo.recurrent_targets(min_networks=4).empty
    assert {"networks", "mean_likelihood1", "n_tested"} <= set(rec.columns)

    col = mo.collect(min_networks=3)
    name = "A&B (+/+) in >= 3 networks"
    assert set(col[name]) == in3 and col.meta(name)["n_tested"] == 3
    assert len(mo.collect(min_networks=3, per_network=True)) > len(col)

    sets = {"planted": sorted(in3)[:25], "other": [f"g{i}" for i in range(1500, 1560)]}
    e = mo.enrich(sets, min_networks=3, min_size=5, min_overlap=2)
    top = e[e["term"] == "planted"].iloc[0]
    assert top["target_set"] == name and top["overlap"] == 25 and top["q"] < 1e-6

    out = mo.export(tmp_path / "multi.xlsx", min_networks=3, enrichment=e)
    sheets = pd.ExcelFile(out).sheet_names
    for s in ("settings", "coverage", "summary", "table", "shared_targets", "recurrent", "lists", "genes", "enrichment"):
        assert s in sheets
    assert len(pd.read_excel(out, sheet_name="table")) == len(mo.table)
    gmt = mo.export(tmp_path / "multi.gmt", min_networks=3)
    assert name.replace(" ", "_") in gmt.read_text() or name in gmt.read_text()
    csv_dir = mo.export(tmp_path / "csv", min_networks=3)
    assert (csv_dir / "table.csv").exists()


def test_tail_both_summary_and_matrix(rng):
    reg = _regulon(rng, discordant=20)
    mo = pr.overlap_across_networks({"a": reg, "b": reg}, ["A", "B", "D"], tail="both", n_perm=0, verbose=False)
    s = mo.summary()
    assert set(zip(s.sign1, s.sign2)) == {("+", "+"), ("-", "-"), ("+", "-"), ("-", "+")}
    disc = mo.matrix("overlap", signs=("+", "-"))
    assert disc["A | D"].gt(0).all()
    with pytest.raises(ValueError, match="Unknown value"):
        mo.matrix("nope")
    pos = pr.overlap_across_networks({"a": reg}, ["A", "B"], n_perm=0, verbose=False)
    with pytest.raises(ValueError, match="not in the table"):
        pos.matrix("fold", signs=("+", "-"))


def test_figure_and_hits(nets):
    mo = pr.overlap_across_networks(nets, ["A", "B", "C"], n_perm=200, verbose=False)
    fig, ax = plt.subplots()
    assert mo.plot_matrix("fold", ax=ax) is ax
    assert [t.get_text() for t in ax.get_yticklabels()] == ["n1", "n2", "n3"]
    n0 = len(ax.texts)
    mo.add_hits(ax=ax)
    assert len(ax.texts) - n0 == int(mo.hits().sum()) == 3                               # A | B in every network
    with pytest.raises(ValueError, match="Draw the matrix first"):
        mo.add_hits(ax=plt.subplots()[1])


# ---------------------------------------------------------------------- path in coverage, merging by name
def test_coverage_has_the_paths(nets, tmp_path):
    f = tmp_path / "net.csv"
    nets["n1"].to_csv(f, index=False)
    mo = pr.overlap_across_networks({"BRCA": f, "mem": nets["n2"]}, ["A", "B"], n_perm=0, reader=pd.read_csv, verbose=False)
    cov = mo.coverage.set_index("network")
    assert cov.loc["BRCA", "path"] == str(f) and cov.loc["mem", "path"] == ""
    assert list(mo.results) == ["BRCA", "mem"]                                           # dict keys are the names


def test_merge_equals_a_joint_run(nets):
    kw = dict(n_perm=100, seed=3, verbose=False)
    joint = pr.overlap_across_networks(nets, ["A", "B", "C"], **kw)
    a = pr.overlap_across_networks({"n1": nets["n1"]}, ["A", "B", "C"], **kw)
    b = pr.overlap_across_networks({"n2": nets["n2"], "n3": nets["n3"]}, ["A", "B", "C"], **kw)
    assert not np.allclose(a.table["q_perm_all"].to_numpy(), joint.table.loc[joint.table.network == "n1", "q_perm_all"].to_numpy())
    mg = a.merge(b)
    for attr in ("table", "pairs", "sizes", "shared_targets", "coverage"):
        pd.testing.assert_frame_equal(getattr(mg, attr), getattr(joint, attr))
    assert list(mg.results) == ["n1", "n2", "n3"] and mg.settings["n_networks"] == 3
    pd.testing.assert_frame_equal(mg.summary(), joint.summary())
    same = pr.MultiOverlap.concat([a, b])
    pd.testing.assert_frame_equal(same.table, mg.table)
    c = pr.overlap_across_networks({"n4": nets["n1"]}, ["A", "B", "C"], **kw)
    assert list(a.merge(b, c).results) == ["n1", "n2", "n3", "n4"]


def test_merge_with_different_pairs_and_skipped(nets):
    kw = dict(n_perm=0, verbose=False)
    a = pr.overlap_across_networks({"n1": nets["n1"]}, pairs=[("A", "B")], **kw)
    b = pr.overlap_across_networks({"n2": nets["n2"]}, pairs=[("B", "C")], **kw)
    with pytest.warns(UserWarning):
        c = pr.overlap_across_networks({"n3": nets["n3"], "x": nets["n3"][nets["n3"].source == "bg1"]}, pairs=[("A", "B")], **kw)
    mg = pr.MultiOverlap.concat([a, b, c])
    assert mg.settings["pairs"] == ["A|B", "B|C"] and set(mg.skipped) == {"x"}
    s = mg.summary().set_index(["tf1", "tf2"])
    assert s.loc[("A", "B"), "n_networks"] == 2 and s.loc[("B", "C"), "n_networks"] == 1
    assert len(mg.coverage) == 4 and len(mg.table) == 3


def test_merge_refuses_duplicates_and_different_settings(nets):
    a = pr.overlap_across_networks({"n1": nets["n1"]}, ["A", "B"], n_perm=50, verbose=False)
    with pytest.raises(ValueError, match="more than one analysis"):
        a.merge(pr.overlap_across_networks({"n1": nets["n2"]}, ["A", "B"], n_perm=50, verbose=False))
    b = pr.overlap_across_networks({"n2": nets["n2"]}, ["A", "B"], n_perm=50, tail="both", verbose=False)
    with pytest.raises(ValueError, match="differs in"):
        a.merge(b)
    c = pr.overlap_across_networks({"n2": nets["n2"]}, ["A", "B"], n_perm=60, verbose=False)
    with pytest.raises(ValueError, match="n_perm"):
        a.merge(c)
    with pytest.raises(ValueError, match="Nothing"):
        pr.MultiOverlap.concat([])
