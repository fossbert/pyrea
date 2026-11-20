"""Gene list collection, enrichment, export and activity split of regulon overlaps."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

import pyrea as pr
from test_regulon_overlap import _regulon


def _viper_ready():
    try:
        from pyrea.rbackend import r_package

        r_package("viper")
        return True
    except Exception:
        return False


viper = pytest.mark.skipif(not _viper_ready(), reason="needs R + Bioconductor package viper")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# ------------------------------------------------------------------ GeneListCollection
def test_collection_basics_and_set_operations():
    col = pr.GeneListCollection()
    assert col.add("a", ["g2", "g1", "g1"], source="x", n=3) is col
    col.add("b", ["g2", "g3"])
    assert col["a"] == ["g1", "g2"] and len(col) == 2 and "b" in col and col.names == ["a", "b"] and "2 lists" in repr(col)
    assert col.union() == ["g1", "g2", "g3"] and col.intersection(["a", "b"]) == ["g2"] and col.intersection([]) == []
    s = col.summary()
    assert list(s["name"]) == ["a", "b"] and list(s["n_genes"]) == [2, 2] and {"source", "n"} <= set(s.columns)
    assert col.meta("a") == {"source": "x", "n": 3} and list(col.to_frame().columns) == ["set", "gene"] and len(col.to_frame()) == 4
    with pytest.raises(ValueError, match="exists"):
        col.add("a", ["g9"])
    col.add("a", ["g9"], overwrite=True)
    assert col["a"] == ["g9"] and col.meta("a") == {}                                       # overwrite replaces list and metadata
    m = col.membership()
    assert m.loc["g2", "b"] and not m.loc["g2", "a"] and list(m.index) == ["g2", "g3", "g9"] and m.loc["g9", "a"]
    col.remove("b")
    assert col.names == ["a"]


def test_collection_export_formats(tmp_path):
    col = pr.GeneListCollection().add("pair (+/+)", ["g1", "g2"], source="net").add("other", ["g3"])
    extra = {"enrichment": pd.DataFrame({"term": ["T"], "p": [0.01]})}
    x = col.export(tmp_path / "out.xlsx", extra=extra)
    sheets = pd.read_excel(x, sheet_name=None)
    assert set(sheets) == {"lists", "genes", "enrichment"} and len(sheets["genes"]) == 3 and sheets["enrichment"]["term"][0] == "T"
    d = col.export(tmp_path / "csvs", extra=extra)
    assert {f.name for f in d.iterdir()} == {"lists.csv", "genes.csv", "enrichment.csv"}
    g = col.export(tmp_path / "sets.gmt")
    lines = g.read_text().splitlines()
    assert lines[0].split("\t") == ["pair (+/+)", "2 genes", "g1", "g2"] and len(lines) == 2


# ------------------------------------------------------------------ gene_set_enrichment
def test_enrichment_matches_hypergeometric_by_hand():
    universe = [f"g{i}" for i in range(100)]
    sets = {"hit": [f"g{i}" for i in range(10)], "miss": [f"g{i}" for i in range(50, 60)], "tiny": ["g1", "g2"],
            "outside": ["g1", "g2", "g3", "g4", "g5", "zzz", "yyy"]}
    genes = [f"g{i}" for i in range(8)] + ["g50"]
    res = pr.gene_set_enrichment(genes, sets, universe=universe, min_size=5, min_overlap=1)
    r = res.set_index("term")
    assert "tiny" not in r.index                                                           # below min_size
    assert r.loc["hit", "overlap"] == 8 and r.loc["hit", "n_term"] == 10 and r.loc["hit", "n_genes"] == 9
    assert r.loc["hit", "p"] == pytest.approx(hypergeom.sf(7, 100, 10, 9)) and r.loc["hit", "expected"] == pytest.approx(0.9)
    assert r.loc["outside", "n_term"] == 5                                                 # restricted to the universe
    assert res.iloc[0]["term"] == "hit" and res["p"].is_monotonic_increasing and (res["q"] >= res["p"] - 1e-12).all()
    assert r.loc["hit", "genes"] == ";".join(sorted(f"g{i}" for i in range(8)))
    assert set(pr.gene_set_enrichment(genes, sets, universe=universe, min_overlap=2)["term"]) == {"hit", "outside"} - {"outside"} | {"outside"}
    none = pr.gene_set_enrichment(["zzz"], sets, universe=universe)                        # list outside the universe
    assert len(none) == 0 and list(none.columns)[:3] == ["term", "n_genes", "n_term"]


# ------------------------------------------------------------------ RegulonOverlap: collecting, enrichment, export
def test_shared_targets_gene_sets_and_core(rng):
    reg = _regulon(rng, discordant=25)
    ov = pr.regulon_overlap(reg, ["A", "B", "C", "D"], tail="both")
    st = ov.shared_targets
    assert list(st.columns) == ["tf1", "tf2", "gene", "mor1", "mor2", "likelihood1", "likelihood2", "sign1", "sign2", "concordant", "in_tail"]
    assert len(st) == ov.pairs["shared"].sum() and st["in_tail"].all()                    # tail="both": every sign combination
    pos = pr.regulon_overlap(reg, ["A", "B", "D"])
    assert pos.shared_targets["in_tail"].sum() == pos.table["overlap"].sum() and (~pos.shared_targets["in_tail"]).any()
    gs = ov.gene_sets()
    for name, genes in gs.items():
        tf1, rest = name.split("&")
        tf2, signs = rest.split(" (")
        row = ov.table[(ov.table.tf1 == tf1) & (ov.table.tf2 == tf2) & (ov.table.sign1 == signs[0]) & (ov.table.sign2 == signs[2])].iloc[0]
        assert len(genes) == row["overlap"]
    assert set(ov.gene_sets(signs=("+", "+"))) <= set(gs) and all(k.endswith("(+/+)") for k in ov.gene_sets(signs=("+", "+")))
    core = ov.core("+", ["A", "B"])
    assert set(core) == set(ov.gene_sets(signs=("+", "+"))["A&B (+/+)"])
    assert set(ov.core("+")) <= set(core)
    with pytest.raises(ValueError, match="sign"):
        ov.core("x")
    with pytest.raises(ValueError, match="Unknown regulator"):
        ov.core("+", ["A", "Z"])


def test_collect_label_and_enrich_planted_set(rng):
    reg = _regulon(rng)
    ov = pr.regulon_overlap(reg, ["A", "B", "C"], n_perm=50, seed=2)
    col = ov.collect(label="net1")
    assert col is not None and "net1: A&B (+/+)" in col.names and not any("core" in n for n in col.names)       # C shares no target with A and B
    meta = col.meta("net1: A&B (+/+)")
    assert meta["source"] == "net1" and meta["tfs"] == "A,B" and meta["signs"] == "+/+" and "p_perm" in meta and meta["overlap"] == len(col["net1: A&B (+/+)"])
    again = ov.collect(into=col, label="net2")                                              # a second analysis into the same collection
    assert again is col and "net2: A&B (+/+)" in col.names
    shared_pp = ov.gene_sets(("+", "+"))["A&B (+/+)"]
    sets = {"PLANTED": shared_pp[:25], "OTHER": [g for g in ov.universe_genes if g not in shared_pp][:60]}
    e = ov.enrich(sets, signs=("+", "+"), min_size=5, min_overlap=3)
    assert list(e.columns) == ["target_set", "term", "n_genes", "n_term", "overlap", "expected", "fold", "p", "q", "genes"]
    top = e[e["target_set"] == "A&B (+/+)"].iloc[0]
    assert top["term"] == "PLANTED" and top["overlap"] == 25 and top["q"] < 1e-6
    full = ov.enrich(sets, min_overlap=3)
    assert {"A&B (+/+)"} <= set(full["target_set"])


def test_core_of_three_regulators_is_collected():
    reg = pd.DataFrame({"source": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["bg"] * 3,
                        "target": ["g1", "g2", "g3", "g4", "g2", "g3", "g5", "g6", "g3", "g7", "g8", "g9", "g1", "g2", "g10"],
                        "mor": [1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1.0], "likelihood": 1.0})
    ov = pr.regulon_overlap(reg, ["A", "B", "C"], tail="both")
    assert ov.core("+") == ["g3"] and ov.core("+", ["A", "B"]) == ["g2", "g3"] and ov.core("-") == []
    col = ov.collect(label="t")
    assert col["t: A&B&C core (+)"] == ["g3"] and col.meta("t: A&B&C core (+)")["signs"] == "+" and "t: A&B&C core (-)" not in col
    assert col["t: A&B (+/+)"] == ["g2", "g3"]
    e = ov.enrich({"T": ["g3", "g2", "g11", "g12", "g13", "g14"]}, min_size=2, min_overlap=1, include_core=True)
    assert "A&B&C core (+)" in set(e["target_set"])


def test_export_workbook_contains_everything(rng, tmp_path):
    reg = _regulon(rng, discordant=20)
    ov = pr.regulon_overlap(reg, ["A", "B", "D"], tail="both", n_perm=30)
    enr = ov.enrich({"X": list(ov.universe_genes[:200])}, min_size=5, min_overlap=1)
    path = ov.export(tmp_path / "ov.xlsx", enrichment=enr)
    sheets = pd.read_excel(path, sheet_name=None)
    assert {"settings", "table", "pairs", "sizes", "shared_targets", "lists", "genes", "enrichment"} <= set(sheets)
    assert len(sheets["shared_targets"]) == len(ov.shared_targets) and sheets["settings"]["tail"][0] == "both"
    assert len(sheets["genes"]) == sum(len(v) for v in ov.collect().to_dict().values())
    pr.regulon_overlap(reg, ["A", "B"]).export(tmp_path / "sets.gmt")
    assert (tmp_path / "sets.gmt").read_text().startswith("A&B (+/+)")
    col = pr.GeneListCollection().add("mine", ["g1"])
    custom = pd.read_excel(ov.export(tmp_path / "c.xlsx", collection=col), sheet_name="lists")
    assert list(custom["name"]) == ["mine"]


# ------------------------------------------------------------------ split_activity (needs R)
def _expression(rng, drivers, n_samples=120, n_other=3000, noise=1.0):
    """Genes in three blocks plus noise genes; drivers: dict block -> latent vector (samples)."""
    blocks = {"shared": 60, "a_unique": 60, "b_unique": 60}
    rows, names = [], []
    for b, k in blocks.items():
        for i in range(k):
            rows.append(drivers.get(b, np.zeros(n_samples)) + noise * rng.normal(size=n_samples))
            names.append(f"{b}{i}")
    for i in range(n_other):
        rows.append(rng.normal(size=n_samples))
        names.append(f"x{i}")
    X = pd.DataFrame(rows, index=names, columns=[f"s{j}" for j in range(n_samples)])
    X.loc["A"], X.loc["B"] = rng.normal(size=n_samples), rng.normal(size=n_samples)
    return (X.sub(X.mean(axis=1), axis=0)).div(X.std(axis=1), axis=0)


def _split_regulon(**_):
    rows = [("A", f"shared{i}", 1.0, 1.0) for i in range(60)] + [("A", f"a_unique{i}", 1.0, 1.0) for i in range(60)]
    rows += [("B", f"shared{i}", 1.0, 1.0) for i in range(60)] + [("B", f"b_unique{i}", 1.0, 1.0) for i in range(60)]
    return pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"])


@viper
def test_split_isolates_the_shared_targets(rng):
    f = rng.normal(size=120)
    X = _expression(rng, {"shared": 1.0 * f})                                               # only the shared targets move together
    ov = pr.regulon_overlap(_split_regulon(), ["A", "B"])
    sp = ov.split_activity(X, "A", "B", method="none", minsize=20, n_random=40, seed=1)
    assert isinstance(sp, pr.ActivitySplit) and "ActivitySplit(A x B" in repr(sp)
    c = sp.correlation.set_index(["part1", "part2"])
    assert c.loc[("shared", "shared"), "target_overlap"] == 60 and c.loc[("shared", "shared"), "circular"] and c.loc[("shared", "shared"), "rho"] > 0.95
    assert c.loc[("unique", "unique"), "target_overlap"] == 0 and not c.loc[("unique", "unique"), "circular"]
    assert c.loc[("full", "full"), "target_overlap"] == 60 and c.loc[("full", "full"), "rho"] > 0.3                 # via the shared targets
    assert abs(c.loc[("unique", "unique"), "rho"]) < 0.3 and abs(c.loc[("shared", "unique"), "rho"]) < 0.3         # nothing beyond them
    assert c.loc[("full", "unique"), "target_overlap"] == 0 and c.loc[("unique", "full"), "target_overlap"] == 0
    assert list(sp.parts["n_regulon"]) == [120, 120, 60, 60, 60, 60] and sp.parts["in_activity"].all()
    assert set(sp.activity.index) == {f"{t}|{p}" for t in "AB" for p in ("full", "shared", "unique")}               # random subsets not kept
    ns = sp.null_summary()
    assert (ns["p"] > 0.05).all() and ns["n_random"].min() == 40
    m = sp.matrix()
    assert list(m.index) == ["full", "shared", "unique"] and m.loc["shared", "shared"] == c.loc[("shared", "shared"), "rho"]
    assert sp.matrix("target_overlap").loc["unique", "unique"] == 0 and np.isfinite(sp.expression_corr)
    with pytest.raises(ValueError, match="value must"):
        sp.matrix("nope")


@viper
def test_split_full_part_matches_the_standard_activity(rng):
    """full x full must equal the correlation of the ordinary VIPER activities, with the signature ranked over all genes
    (the default eset_filter would rank only the few regulon targets and make the parts depend on each other)."""
    from scipy.stats import spearmanr
    from pyrea.rbackend.viper import viper_activity
    X = _expression(rng, {"shared": rng.normal(size=120)})
    reg = _split_regulon()
    sp = pr.regulon_overlap(reg, ["A", "B"]).split_activity(X, "A", "B", method="none", minsize=20)
    act = viper_activity(X, reg, method="none", minsize=20, eset_filter=False)
    assert sp.correlation.set_index(["part1", "part2"]).loc[("full", "full"), "rho"] == pytest.approx(spearmanr(act.loc["A"], act.loc["B"])[0])
    np.testing.assert_allclose(sp.activity.loc["A|full"].to_numpy(), act.loc["A"].to_numpy(), atol=1e-8)


@viper
def test_split_detects_correlation_beyond_the_shared_targets_and_special_shared_part(rng):
    c = rng.normal(size=120)
    both = _expression(rng, {"a_unique": 1.0 * c, "b_unique": 1.0 * c})                     # the own targets of A and B move together
    sp = pr.regulon_overlap(_split_regulon(), ["A", "B"]).split_activity(both, "A", "B", method="none", minsize=20)
    assert sp.correlation.set_index(["part1", "part2"]).loc[("unique", "unique"), "rho"] > 0.7
    b = rng.normal(size=120)
    X = _expression(rng, {"shared": 1.0 * b, "b_unique": 1.0 * b})                          # A's shared targets follow B's own targets
    sp2 = pr.regulon_overlap(_split_regulon(), ["A", "B"]).split_activity(X, "A", "B", method="none", minsize=20, n_random=60, seed=2)
    ns = sp2.null_summary()
    assert ns.loc["A|shared vs B|unique", "observed"] > 0.7 and ns.loc["A|shared vs B|unique", "p"] < 0.05
    assert ns.loc["A|shared vs B|unique", "observed"] > ns.loc["A|shared vs B|unique", "null_mean"] + 3 * ns.loc["A|shared vs B|unique", "null_sd"]


@viper
def test_split_validation_small_parts_and_figures(rng):
    X = _expression(rng, {})
    ov = pr.regulon_overlap(_split_regulon(), ["A", "B"])
    with pytest.raises(ValueError, match="Unknown regulator"):
        ov.split_activity(X, "A", "Z")
    with pytest.raises(ValueError, match="different"):
        ov.split_activity(X, "A", "A")
    with pytest.warns(UserWarning, match="minsize"):
        small = ov.split_activity(X, "A", "B", method="none", minsize=70)                   # shared / unique parts have 60 targets only
    assert not small.parts["in_activity"].all() and small.correlation["rho"].isna().sum() > 0
    with pytest.raises(ValueError, match="No null"):
        small.null_summary()
    sp = ov.split_activity(X, "A", "B", method="none", minsize=20)
    ax = sp.plot_correlations()
    assert len(ax.texts) == 0 and len(ax.patches) == 4 and sp.add_correlation_values(ax=ax) is ax and len(ax.texts) == 9    # 4 hatched (circular) cells
    with pytest.raises(ValueError, match="plot_correlations"):
        sp.add_correlation_values(ax=plt.figure().add_subplot())
    sc = sp.plot_scatter("unique", "unique")
    assert "A, unique" in sc.get_xlabel() and "circular" not in sp.add_scatter_stats("unique", "unique", ax=sc).texts[0].get_text()
    assert "circular" in sp.add_scatter_stats("full", "full", ax=sp.plot_scatter("full", "full")).texts[0].get_text()
    with pytest.raises(ValueError, match="part must"):
        sp.plot_scatter("nope", "unique")


# ------------------------------------------------------------------ bundled gene set collections by name
def test_available_genesets_and_quiet_loading(capsys):
    av = pr.available_genesets()
    assert len(av) == 8 and set(av["collection"]) == {"h", "pid", "wikipathways", "reactome"} and set(av["species"]) == {"human", "mouse"}
    assert list(av.columns) == ["collection", "description", "species", "n_sets", "size_min", "size_median", "size_max"]
    row = av.query("collection == 'h' and species == 'human'").iloc[0]
    assert row["n_sets"] == 50 and row["size_max"] <= 200 and av.query("collection == 'reactome' and species == 'human'")["n_sets"].iloc[0] > 1000
    assert capsys.readouterr().out == ""                                                   # no "Found 1 gene set collection" noise
    assert len(pr.load_genesets("h", "human", verbose=False)) == 50 and capsys.readouterr().out == ""
    assert len(pr.load_genesets("H", "Human", verbose=False)) == 50 and pr.load_genesets("hallmark", verbose=False).keys() == pr.load_genesets("h", verbose=False).keys()
    pr.load_genesets("h")
    assert "Loaded" in capsys.readouterr().out                                             # default stays verbose
    with pytest.raises(ValueError, match="suitable"):
        pr.load_genesets("kegg")


def test_resolve_genesets():
    from pyrea.genelists import resolve_genesets
    d = {"T": ["g1", "g2"]}
    out, coll = resolve_genesets(d)
    assert out == d and out is not d and coll is None
    h, ch = resolve_genesets("h")
    assert len(h) == 50 and set(ch.values()) == {"h"} and "HALLMARK_HYPOXIA" in h
    both, cb = resolve_genesets(["h", "PID"])
    assert len(both) == 50 + 196 and set(cb.values()) == {"h", "pid"} and cb["HALLMARK_HYPOXIA"] == "h"
    assert resolve_genesets("hallmark")[0].keys() == h.keys()
    m, _ = resolve_genesets("h", species="mouse")
    assert len(m) == 50 and m["HALLMARK_HYPOXIA"] != h["HALLMARK_HYPOXIA"]
    for bad in ("kegg", [], [1, 2]):
        with pytest.raises(ValueError):
            resolve_genesets(bad)


def test_enrichment_by_collection_name():
    h = pr.load_genesets("h", "human", verbose=False)
    emt = h["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"]
    uni = sorted(set().union(*h.values()))
    res = pr.gene_set_enrichment(emt[:60], "h", universe=uni, min_overlap=3)
    assert res.iloc[0]["term"] == "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION" and res.iloc[0]["overlap"] == 60 and res.iloc[0]["q"] < 1e-20
    assert list(res.columns)[:3] == ["term", "collection", "n_genes"] and set(res["collection"]) == {"h"}
    named = pr.gene_set_enrichment(emt[:60], h, universe=uni, min_overlap=3)                  # the same with the dict
    assert "collection" not in named.columns
    pd.testing.assert_frame_equal(res.drop(columns="collection"), named)
    multi = pr.gene_set_enrichment(emt[:60], ["h", "reactome"], universe=uni, min_overlap=3)
    assert {"h", "reactome"} <= set(multi["collection"])
    q_alone = res.set_index("term")["q"]
    q_multi = multi.set_index("term")["q"].reindex(q_alone.index)
    assert (q_multi >= q_alone - 1e-12).all()                                                # more tests, more correction


def test_overlap_enrich_by_collection_name(rng):
    h = pr.load_genesets("h", "human", verbose=False)
    genes = sorted(set().union(*h.values()))
    emt = h["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"]
    rows = []
    for tf in ("A", "B"):
        own = list(rng.choice([g for g in genes if g not in emt], 100, replace=False))
        rows += [(tf, g, 1.0, 1.0) for g in emt[:25] + own]
    for i in range(30):
        rows += [(f"bg{i}", g, 1.0, 1.0) for g in rng.choice(genes, 150, replace=False)]
    ov = pr.regulon_overlap(pd.DataFrame(rows, columns=["source", "target", "mor", "likelihood"]).drop_duplicates(["source", "target"]), ["A", "B"])
    e = ov.enrich("h", signs=("+", "+"), min_size=10, min_overlap=3)
    assert list(e.columns) == ["target_set", "term", "collection", "n_genes", "n_term", "overlap", "expected", "fold", "p", "q", "genes"]
    assert e.iloc[0]["term"] == "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION" and e.iloc[0]["collection"] == "h" and e.iloc[0]["q"] < 1e-10
    by_dict = ov.enrich(h, signs=("+", "+"), min_size=10, min_overlap=3)
    pd.testing.assert_frame_equal(e.drop(columns="collection"), by_dict)
    assert set(ov.enrich(["h", "pid"], signs=("+", "+"), min_size=10, min_overlap=3)["collection"]) <= {"h", "pid"}
    with pytest.raises(ValueError):
        ov.enrich("kegg")
