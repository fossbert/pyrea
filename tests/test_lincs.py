"""pyrea.lincs: regulator activity in LINCS-like signatures (a small synthetic GCTX file)."""

import h5py
import numpy as np
import pandas as pd
import pytest

import pyrea as pr

N_GENES = 600
GENES = ["T0", "T1", "X", "U"] + [f"G{i}" for i in range(4, N_GENES)]
CELLS = ["c1", "c2", "c3", "c4"]


def _regulons(rng):
    """Two networks. T0, T1, X, U regulate 40 targets each (positive and negative); network B knows T0 and X only."""
    rows = {}
    for k, name in enumerate(["T0", "T1", "X", "U"]):
        t = rng.choice(np.arange(40, N_GENES), 40, replace=False)
        rows[name] = (t, np.where(rng.random(40) < 0.3, -1.0, 1.0))
    def long(names, seed):
        r = np.random.default_rng(seed)
        out = []
        for n in names:
            t, m = rows[n]
            for g, mor in zip(t, m):
                out.append((n, GENES[g], mor, r.uniform(0.3, 1.0)))
        # a few background regulators
        for b in range(8):
            t = r.choice(N_GENES, 50, replace=False)
            out += [(f"R{b}", GENES[g], 1.0, 1.0) for g in t]
        return pd.DataFrame(out, columns=["source", "target", "mor", "likelihood"])
    return rows, {"netA": long(["T0", "T1", "X", "U"], 1), "netB": long(["T0", "X"], 2)}


def _build(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    rows, nets = _regulons(rng)
    sigs, ids = [], []
    meta = []

    def add(kind, pert, cell, effects=None, own=None, time=24, dose=10.0, pert_id=None):
        z = rng.normal(size=N_GENES)
        for reg, (strength) in (effects or {}).items():
            t, m = rows[reg]
            z[t] += strength * m
        if own is not None:
            z[GENES.index(own[0])] += own[1]
        sig_id = f"{kind}{len(ids)}_{cell}_{time}H:{pert}:{dose}"
        ids.append(sig_id)
        sigs.append(z.astype(np.float32))
        meta.append({"sig_id": sig_id, "pert_iname": pert, "cell_iname": cell, "pert_type": f"trt_{kind}", "pert_itime": f"{time} h",
                     "pert_idose": f"{dose} uM", "pert_id": pert_id or pert})

    # XPR: knockout of T0 lowers T0's targets in every cell line; of U lowers X (so X is perturbed by U); other genes: noise
    for cell in CELLS:
        add("xpr", "T0", cell, effects={"T0": -1.2}, own=("T0", -3.0))
        add("xpr", "T0", cell, effects={"T0": -1.0}, own=("T0", -2.5))
        add("xpr", "U", cell, effects={"X": -1.2}, own=("U", -3.0))
        add("xpr", "T1", cell, own=("T1", -3.0))                                     # T1: knockout without an effect on its regulon
        for g in range(80, 150):
            add("xpr", f"G{g}", cell)
    for cell in CELLS:
        add("oe", "T0", cell, effects={"T0": 1.2}, own=("T0", 3.0))
        add("oe", "U", cell, effects={"X": 1.2}, own=("U", 3.0))
        for g in range(200, 240):
            add("oe", f"G{g}", cell)
    # compounds: C0-C4 (moa HDAC) raise T1, C5-C7 (moa KINASE) lower T1; the others do nothing; two doses, two times
    for cell in CELLS:
        for c in range(25):
            eff = {"T1": 1.0} if c < 5 else ({"T1": -1.0} if c < 8 else None)
            for dose in (1.0, 10.0):
                add("cp", f"drug{c}", cell, effects=eff, time=6 if c % 2 else 24, dose=dose, pert_id=f"BRD-{c:03d}")
    data = np.vstack(sigs)
    gene_ids = [str(i + 1) for i in range(N_GENES)]
    files = {}
    for kind in ("oe", "xpr", "cp"):
        sel = [i for i, m in enumerate(meta) if m["pert_type"] == f"trt_{kind}"]
        f = tmp_path / f"level5_{kind}.gctx"
        with h5py.File(f, "w") as h:
            h.create_dataset("/0/DATA/0/matrix", data=data[sel])
            h.create_dataset("/0/META/COL/id", data=np.array([ids[i] for i in sel], dtype="S58"))
            h.create_dataset("/0/META/ROW/id", data=np.array(gene_ids, dtype="S12"))
        files[kind] = f
    siginfo = pd.DataFrame(meta)
    geneinfo = pd.DataFrame({"gene_id": gene_ids, "gene_symbol": GENES, "feature_space": ["landmark"] * 100 + ["inferred"] * (N_GENES - 100)})
    cellinfo = pd.DataFrame({"cell_iname": CELLS, "cell_lineage": ["breast", "breast", "colon", "colon"], "primary_disease": ["BRCA", "BRCA", "COAD", "COAD"]})
    comp = pd.DataFrame({"pert_id": [f"BRD-{c:03d}" for c in range(25)], "cmap_name": [f"drug{c}" for c in range(25)],
                         "moa": ["HDAC inhibitor" if c < 5 else ("kinase inhibitor|MEK inhibitor" if c < 8 else "other") for c in range(25)],
                         "target": ["HDAC1|HDAC2" if c < 5 else "" for c in range(25)]})
    return files, siginfo, geneinfo, cellinfo, comp, nets, rows


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lincs")
    files, siginfo, geneinfo, cellinfo, comp, nets, rows = _build(tmp)
    lib = pr.LincsLibrary(siginfo, geneinfo, cellinfo=cellinfo, compoundinfo=comp, **files)
    net = pr.LincsNetworks(lib, nets, minsize=25)
    return lib, net, nets, rows, tmp, (files, siginfo, geneinfo, cellinfo, comp)


# ---------------------------------------------------------------------- library and networks
def test_library_reads_and_selects(world):
    lib, net, nets, rows, tmp, _ = world
    assert "LincsLibrary(" in repr(lib) and len(lib.genes) == N_GENES and set(lib.sig["kind"]) == {"oe", "xpr", "cp"}
    assert lib.sig.loc[lib.sig["kind"] == "cp", "dose"].isin([1.0, 10.0]).all() and set(lib.sig["time"].dropna()) == {6.0, 24.0}
    s = lib.signatures("xpr", gene="t0")                                              # case-insensitive
    assert len(s) == 8 and set(s["cell"]) == set(CELLS) and set(s["lineage"]) == {"breast", "colon"}
    assert len(lib.signatures("xpr", gene="T0", cell="c1")) == 2 and len(lib.signatures("xpr", lineage="colon", gene="T0")) == 4
    assert "T0" not in set(lib.signatures("xpr", exclude_gene="T0")["pert"])
    assert len(lib.signatures("cp", compound="drug3")) == 8 and len(lib.signatures("cp", compound="BRD-003")) == 8
    assert set(lib.signatures("cp", times=[6])["time"]) == {6.0} and set(lib.signatures("cp", min_dose=5)["dose"]) == {10.0}
    ids = list(lib.signatures("xpr", gene="T0").index)
    X = lib.read(ids)
    assert X.shape == (N_GENES, 8) and X.dtypes.eq(np.float32).all() and list(X.columns) == ids
    assert X.loc["T0"].mean() < -2                                                     # the knocked-out gene is down
    mixed = ids[:2] + list(lib.signatures("oe", gene="T0").index)[:2] + ids[2:4]
    Y = lib.read(mixed)
    np.testing.assert_array_equal(Y[mixed[:2]].to_numpy(), X[ids[:2]].to_numpy())      # kinds and orders are mixed correctly
    assert Y.loc["T0", mixed[2]] > 2 and Y.loc["T0", mixed[0]] < -2
    scattered = list(lib.signatures("xpr").index[::7])
    Z = lib.read(scattered[::-1])
    np.testing.assert_array_equal(Z.to_numpy()[:, ::-1], lib.read(scattered).to_numpy())
    blocks = list(lib.blocks(list(lib.signatures("xpr").index), size=100))
    assert sum(len(b[0]) for b in blocks) == len(lib.signatures("xpr")) and all(b[1].shape[1] == len(b[0]) for b in blocks)
    info = lib.compound_info(["BRD-001", "BRD-999"])
    assert info.loc["BRD-001", "moa"] == "HDAC inhibitor" and info.loc["BRD-999", "moa"] == ""
    with pytest.raises(ValueError, match="Unknown signature"):
        lib.read(["nope"])
    with pytest.raises(ValueError, match="No signatures given"):
        lib.read([])


def test_library_validation(world, tmp_path):
    _, _, _, _, tmp, (files, siginfo, geneinfo, cellinfo, comp) = world
    with pytest.raises(ValueError, match="at least one"):
        pr.LincsLibrary(siginfo, geneinfo)
    with pytest.raises(ValueError, match="lacks the columns"):
        pr.LincsLibrary(siginfo.drop(columns="cell_iname"), geneinfo, oe=files["oe"])
    with pytest.raises(ValueError, match="gene_id"):
        pr.LincsLibrary(siginfo, geneinfo.drop(columns="gene_symbol"), oe=files["oe"])
    with pytest.raises(ValueError, match="No signature of siginfo"):
        pr.LincsLibrary(siginfo.assign(sig_id="x" + siginfo["sig_id"]), geneinfo, oe=files["oe"])
    with pytest.raises(ValueError, match="not in geneinfo"):
        pr.LincsLibrary(siginfo, geneinfo.iloc[:-5], oe=files["oe"])
    lib2 = pr.LincsLibrary(siginfo, geneinfo.assign(gene_symbol=lambda d: d["gene_symbol"].where(d.index != 3, "T0")), oe=files["oe"])
    assert len(lib2.genes) == N_GENES - 1                                              # a duplicated symbol: the first is kept
    only = pr.LincsLibrary(siginfo, geneinfo, xpr=files["xpr"])
    assert set(only.sig["kind"]) == {"xpr"} and "cp" not in repr(only)


def test_networks_coverage_and_cache(world, tmp_path):
    lib, net, nets, rows, tmp, _ = world
    assert net.names == ["netA", "netB"] and net.coverage.set_index("network").loc["netA", "n_usable"] >= 4
    assert "T0" in net.regulators() and "U" in net.regulators() and "LincsNetworks(2 network(s)" in repr(net)
    cov = net.tf_coverage("U").set_index("network")["n_targets"]
    assert cov["netA"] == 40 and np.isnan(cov["netB"])                                 # U is not in network B
    small = pr.LincsNetworks(lib, nets, minsize=45)
    assert "T0" not in small.regulators() and "R0" in small.regulators()               # minsize drops the 40-target regulons
    keep = pr.LincsNetworks(lib, nets, regulators=["T0"])
    assert keep.regulators() == ["T0"]
    import pickle
    p = tmp_path / "net.pkl"
    pickle.dump(nets["netA"], open(p, "wb"))
    reader = lambda path: pickle.load(open(path, "rb"))
    a = pr.LincsNetworks(lib, {"A": p}, cache_dir=tmp_path / "cache", regulon_reader=reader)
    calls = []
    b = pr.LincsNetworks(lib, {"A": p}, cache_dir=tmp_path / "cache", regulon_reader=lambda q: calls.append(q) or reader(q))
    assert calls == [] and a.regs["A"].keys() == b.regs["A"].keys()                    # the second run comes from the cache
    with pytest.raises(ValueError, match="unique names"):
        pr.LincsNetworks(lib, [p, p])


# ---------------------------------------------------------------------- activity
def test_activity_matches_the_numpy_area_and_integrates_networks(world):
    lib, net, nets, rows, tmp, _ = world
    ids = list(lib.signatures("xpr", gene="T0").index) + list(lib.signatures("oe", gene="T0").index)
    X = lib.read(ids).astype(float)
    a = pr.area_nes(X, nets["netA"], minsize=25).loc[["T0", "X"]]
    b = pr.area_nes(X, nets["netB"], minsize=25).loc[["T0", "X"]]
    meta = pr.metaviper({"A": a, "B": b})
    act = lib.activity(ids, net, regulators=["T0", "X"])
    np.testing.assert_allclose(act.loc[["T0", "X"]].to_numpy(), meta.loc[["T0", "X"]].to_numpy(), atol=1e-4)
    per = lib.activity(ids, net, regulators=["T0"], per_network=True)
    np.testing.assert_allclose(per["netA"].loc["T0"].to_numpy(), a.loc["T0"].to_numpy(), atol=1e-4)
    only_a = lib.activity(ids, net, regulators=["T0"], use=["netA"])
    np.testing.assert_allclose(only_a.loc["T0"].to_numpy(), a.loc["T0"].to_numpy(), atol=1e-4)
    assert list(act.columns) == ids
    u = lib.activity(ids, net, regulators=["U"])                                       # only in network A
    np.testing.assert_allclose(u.loc["U"].to_numpy(), pr.area_nes(X, nets["netA"], minsize=25).loc["U"].to_numpy(), atol=1e-4)
    xs = lib.signatures("xpr", gene="T0")
    assert lib.activity(list(xs.index), net, regulators=["T0"]).loc["T0"].median() < -3        # T0 targets fall in the T0 knockout
    with pytest.raises(ValueError, match="Unknown network"):
        lib.activity(ids, net, use=["nope"])
    with pytest.raises(ValueError, match="None of the regulators"):
        lib.activity(ids, net, regulators=["NOPE"])
    with pytest.warns(UserWarning, match="without a usable regulon"):
        lib.activity(ids, net, regulators=["T0", "NOPE"])


def test_cell_specific_networks_with_a_network_map(world):
    lib, net, nets, rows, tmp, _ = world
    ids = list(lib.signatures("xpr", gene="T0").index)                                # breast: c1, c2; colon: c3, c4
    allnet = lib.activity(ids, net, regulators=["T0"]).loc["T0"]
    mapped = lib.activity(ids, net, regulators=["T0"], network_map={"breast": ["netA"], "c3": ["netB"]}).loc["T0"]
    per = lib.activity(ids, net, regulators=["T0"], per_network=True)
    meta = lib.sig.loc[ids]
    for sid, cell, lin in zip(ids, meta["cell"], meta["lineage"]):
        if lin == "breast":
            assert mapped[sid] == pytest.approx(per["netA"].loc["T0", sid], abs=1e-6)  # the lineage map
        elif cell == "c3":
            assert mapped[sid] == pytest.approx(per["netB"].loc["T0", sid], abs=1e-6)  # the cell line map
        else:
            assert mapped[sid] == pytest.approx(allnet[sid], abs=1e-6)                  # not in the map: all networks


# ---------------------------------------------------------------------- the four questions
def test_tf_response(world):
    lib, net, *_ = world
    r = pr.tf_response(lib, "T0", net, background=200, seed=1)
    s = r.summary.set_index("kind")
    assert s.loc["xpr", "median_nes"] < -3 and s.loc["oe", "median_nes"] > 3
    assert s.loc["xpr", "frac_expected"] == 1.0 and s.loc["oe", "frac_expected"] == 1.0 and s.loc["xpr", "auc"] > 0.99 and s.loc["oe", "auc"] > 0.99
    assert s.loc["xpr", "p_cells"] < 0.01 and s.loc["oe", "p_cells"] < 0.01 and s.loc["xpr", "n_cells"] == 4
    assert r.table["bg_percentile"].lt(0.05).sum() >= 6 and set(r.table.columns) >= {"sig_id", "nes", "expression_z", "bg_percentile", "expected"}
    assert (r.table.query("kind == 'xpr'")["expression_z"] < -0.5).all()
    assert len(r.cells) == 8 and r.missing == []
    flat = pr.tf_response(lib, "T1", net, kinds=["xpr"], background=200, seed=1)       # T1: the knockout does not move its regulon
    f = flat.summary.iloc[0]
    assert abs(f["median_nes"]) < 3 and f["p_cells"] > 0.05 and f["auc"] < 0.9
    one = pr.tf_response(lib, "T0", net, kinds=["xpr"], cells=["c1", "c2"], background=100, seed=1, use=["netA"])
    assert one.summary["n_cells"].iloc[0] == 2
    miss = pr.tf_response(lib, "T0", net, kinds=["xpr", "oe"], cells="c1", background=100)
    assert set(miss.table["kind"]) == {"xpr", "oe"}
    with pytest.raises(ValueError, match="No signatures of X"):
        pr.tf_response(lib, "X", net, background=100)                                  # no XPR/OE of X
    again = pr.tf_response(lib, "T0", net, background=200, seed=1)
    pd.testing.assert_frame_equal(r.summary, again.summary)


def test_perturbation_targets(world):
    lib, net, *_ = world
    e = pr.perturbation_targets(lib, "U", net, kind="xpr", background=300, seed=1)
    assert isinstance(e, pr.RankedEffects) and "RankedEffects(" in repr(e)
    top = e.down(3)
    assert top["regulator"].iloc[0] == "X" and top["z"].iloc[0] < -3 and top["n_cells"].iloc[0] == 4 and top["frac_same_sign"].iloc[0] == 1.0
    assert e.table.set_index("regulator").loc["U", "self"] and not e.table.set_index("regulator").loc["X", "self"]
    assert e.up(3)["z"].max() < e.down(1)["z"].abs().iloc[0]                           # the strongest effect is the fall of X
    assert (e.down(5)["z"].diff().dropna() >= 0).all()                                 # sorted
    oe = pr.perturbation_targets(lib, "U", net, kind="oe", background=300, seed=1)
    assert oe.up(1)["regulator"].iloc[0] == "X" and oe.up(1)["z"].iloc[0] > 3
    sub = pr.perturbation_targets(lib, "U", net, kind="xpr", cells=["c1", "c2"], regulators=["X", "T1"], background=300)
    assert set(sub.table["regulator"]) == {"X", "T1"} and sub.meta["cells"] == ["c1", "c2"]
    with pytest.raises(ValueError, match="No xpr signatures of NOPE"):
        pr.perturbation_targets(lib, "NOPE", net)


def test_perturbed_by(world, tmp_path):
    lib, net, *_ = world
    res = pr.perturbed_by(lib, "X", net, min_cells=2, cache_dir=tmp_path / "cache")
    t = res.table
    xpr = t[t["kind"] == "xpr"].set_index("gene")
    assert xpr.loc["U", "z"] < -3 and res.down(1, kind="xpr")["gene"].iloc[0] == "U" and xpr.loc["U", "n_cells"] == 4
    assert res.up(1, kind="oe")["gene"].iloc[0] == "U"                                  # overexpression raises X
    assert xpr.loc["U", "q"] < 0.01 and (xpr.drop("U")["q"] > 0.0001).mean() > 0.9
    assert (t["z"].abs() < 3).mean() > 0.9                                              # the other genes do nothing
    assert len(list((tmp_path / "cache").glob("scan-X-*.pkl"))) == 2                    # one scan per kind, cached
    again = pr.perturbed_by(lib, "X", net, min_cells=2, cache_dir=tmp_path / "cache")
    pd.testing.assert_frame_equal(res.table, again.table)
    only = pr.perturbed_by(lib, "X", net, kinds=["xpr"], cells=["c1", "c3"], min_cells=2)
    assert set(only.table["kind"]) == {"xpr"} and only.table["n_cells"].max() == 2
    assert pr.perturbed_by(lib, "X", net, kinds=["xpr"], min_cells=5).table.empty
    mapped = pr.perturbed_by(lib, "X", net, kinds=["xpr"], network_map={"breast": ["netA"], "colon": ["netB"]})
    assert mapped.down(1)["gene"].iloc[0] == "U"
    with pytest.raises(ValueError, match="No signatures of the kinds"):
        pr.perturbed_by(lib, "X", net, kinds=["xpr"], cells=["nope"])


def test_tf_in_compounds_and_moa_enrichment(world):
    lib, net, *_ = world
    res = pr.tf_in_compounds(lib, "T1", net, min_cells=3)
    t = res.table
    assert {"pert_id", "compound", "moa", "target", "z", "p", "q"} <= set(t.columns)
    top_up, top_down = set(res.up(5)["pert_id"]), set(res.down(3)["pert_id"])
    assert top_up == {f"BRD-{c:03d}" for c in range(5)} and top_down == {f"BRD-{c:03d}" for c in range(5, 8)}
    assert res.up(1)["compound"].iloc[0].startswith("drug") and res.up(1)["moa"].iloc[0] == "HDAC inhibitor"
    assert t.set_index("pert_id").loc["BRD-001", "n_signatures"] == 8
    enr_up = pr.compound_set_enrichment(res, top=5, direction="up")
    assert enr_up.iloc[0]["term"] == "HDAC inhibitor" and enr_up.iloc[0]["n_top"] == 5 and enr_up.iloc[0]["q"] < 0.01
    enr_down = pr.compound_set_enrichment(res, top=3, direction="down")
    assert set(enr_down["term"][:2]) == {"kinase inhibitor", "MEK inhibitor"} and enr_down["q"].iloc[0] < 0.05      # terms are split at "|"
    tgt = pr.compound_set_enrichment(res, top=5, direction="up", column="target")
    assert set(tgt["term"]) == {"HDAC1", "HDAC2"}
    sel = pr.tf_in_compounds(lib, "T1", net, times=[24], min_cells=2)
    assert set(sel.table["pert_id"]) <= {f"BRD-{c:03d}" for c in range(25) if c % 2 == 0}
    br = pr.tf_in_compounds(lib, "T1", net, lineage="breast", min_cells=2)
    assert br.table["n_cells"].max() == 2
    with pytest.raises(ValueError, match="No compound signatures"):
        pr.tf_in_compounds(lib, "T1", net, cells=["nope"])
    with pytest.raises(ValueError, match="direction"):
        pr.compound_set_enrichment(res, direction="x")
    with pytest.raises(ValueError, match="no column"):
        pr.compound_set_enrichment(res, column="nope")


def test_siginfo_variants_quality_and_file_reading(world, tmp_path):
    _, _, _, _, tmp, (files, siginfo, geneinfo, cellinfo, comp) = world
    real_like = siginfo.rename(columns={"pert_iname": "cmap_name"})                 # the beta file calls the perturbation cmap_name
    real_like["qc_pass"] = np.where(real_like.index % 5 == 0, 0, 1)
    real_like["junk_column"] = "x"
    real_like.loc[real_like.index[:3], "pert_type"] = "ctl_vector"                  # signatures in a file that are no perturbation of the kind
    path = tmp_path / "siginfo.txt"
    real_like.to_csv(path, sep="\t", index=False)
    lib = pr.LincsLibrary(path, geneinfo, cellinfo=cellinfo, **files)                # read from a file, only the needed columns
    assert "junk_column" not in lib.sig.columns and "qc_pass" in lib.sig.columns and (lib.sig["pert"] != "nan").all()
    full = pr.LincsLibrary(siginfo, geneinfo, **files)
    assert len(lib.sig) == len(full.sig) - 3                                         # the control rows are dropped
    allx = lib.signatures("xpr")
    good = lib.signatures("xpr", qc_pass=True)
    assert 0 < len(good) < len(allx) and (good["qc_pass"] == 1).all()
    assert len(full.signatures("xpr", qc_pass=True)) == len(full.signatures("xpr"))  # no qc column: no filter
    no_name = real_like.drop(columns="cmap_name")
    with pytest.raises(ValueError, match="pert_iname or cmap_name"):
        pr.LincsLibrary(no_name, geneinfo, **files)
    net = pr.LincsNetworks(lib, _regulons(np.random.default_rng(0))[1], minsize=25)
    r = pr.tf_response(lib, "T0", net, kinds=["xpr"], background=100, seed=1)         # qc_pass=True by default
    assert r.table["sig_id"].isin(good.index).all()
    r2 = pr.tf_response(lib, "T0", net, kinds=["xpr"], background=100, seed=1, qc_pass=False)
    assert len(r2.table) > len(r.table)


def test_files_with_the_genes_in_another_order(world, tmp_path):
    _, _, _, _, tmp, (files, siginfo, geneinfo, cellinfo, comp) = world
    perm = np.random.default_rng(5).permutation(N_GENES)
    shuffled = tmp_path / "xpr_shuffled.gctx"
    with h5py.File(files["xpr"], "r") as src, h5py.File(shuffled, "w") as h:
        h.create_dataset("/0/DATA/0/matrix", data=src["/0/DATA/0/matrix"][:, :][:, perm])
        h.create_dataset("/0/META/COL/id", data=src["/0/META/COL/id"][:])
        h.create_dataset("/0/META/ROW/id", data=src["/0/META/ROW/id"][:][perm])
    a = pr.LincsLibrary(siginfo, geneinfo, oe=files["oe"], xpr=files["xpr"])
    b = pr.LincsLibrary(siginfo, geneinfo, oe=files["oe"], xpr=shuffled)             # the second file lists the genes in another order
    ids = list(a.signatures("xpr", gene="T0").index) + list(a.signatures("oe", gene="T0").index)
    np.testing.assert_array_equal(a.read(ids).to_numpy(), b.read(ids).to_numpy())
    assert list(a.genes) == list(b.genes)
    other = tmp_path / "other.gctx"
    with h5py.File(files["xpr"], "r") as src, h5py.File(other, "w") as h:
        h.create_dataset("/0/DATA/0/matrix", data=src["/0/DATA/0/matrix"][:, :][:, :-3])
        h.create_dataset("/0/META/COL/id", data=src["/0/META/COL/id"][:])
        h.create_dataset("/0/META/ROW/id", data=src["/0/META/ROW/id"][:][:-3])
    with pytest.raises(ValueError, match="differ from those"):
        pr.LincsLibrary(siginfo, geneinfo, oe=files["oe"], xpr=other)


def test_perturbation_specificity(world):
    lib, net, *_ = world
    out = pr.perturbation_specificity(lib, "X", ["U", "G90", "NOPE"], net, kind="xpr", background=300, seed=1)
    assert list(out["gene"]) == ["U", "G90", "NOPE"] and list(out.columns)[:3] == ["gene", "n_signatures", "n_cells"]
    u = out.set_index("gene")
    assert u.loc["U", "z"] < -3 and u.loc["U", "rank"] == 1 and u.loc["U", "percentile"] < 0.2 and u.loc["U", "n_changed"] <= 3   # specific: X is the regulator that changes
    assert u.loc["U", "n_down"] >= 1 and u.loc["U", "n_regulators"] >= 4 and u.loc["U", "n_cells"] == 4
    assert abs(u.loc["G90", "z"]) < 3 and u.loc["G90", "rank"] > 1                           # an unrelated knockout: X is not special
    assert u.loc["NOPE", "n_signatures"] == 0 and np.isnan(u.loc["NOPE", "z"])
    assert pr.perturbation_specificity(lib, "NOPE_TF", ["U"], net, background=200).loc[0, "n_signatures"] == 4     # the regulator is not in the networks


def test_scan_cache_does_not_depend_on_files_of_other_kinds(world, tmp_path):
    lib, net, nets, rows, tmp, (files, siginfo, geneinfo, cellinfo, comp) = world
    xpr_only = pr.LincsLibrary(siginfo, geneinfo, xpr=files["xpr"])
    both = pr.LincsLibrary(siginfo, geneinfo, xpr=files["xpr"], cp=files["cp"])                 # another file added later (as with the CP download)
    cache = tmp_path / "cache"
    n1 = pr.LincsNetworks(xpr_only, nets, minsize=25)
    pr.perturbed_by(xpr_only, "X", n1, kinds=["xpr"], min_cells=2, cache_dir=cache)
    assert len(list(cache.glob("scan-*.pkl"))) == 1
    n2 = pr.LincsNetworks(both, nets, minsize=25)
    pr.perturbed_by(both, "X", n2, kinds=["xpr"], min_cells=2, cache_dir=cache)
    assert len(list(cache.glob("scan-*.pkl"))) == 1                                              # the xpr scan was reused


def test_lineage_map_completes_unknown_lineages(world):
    _, _, _, _, tmp, (files, siginfo, geneinfo, cellinfo, comp) = world
    ci = cellinfo.copy()
    ci["cell_lineage"] = ["unknown", "breast", "unknown", "colon"]
    lib = pr.LincsLibrary(siginfo, geneinfo, cellinfo=ci, **files)                   # default map names none of the test cells
    assert set(lib.sig.loc[lib.sig["cell"] == "c1", "lineage"]) == {"unknown"}
    lib = pr.LincsLibrary(siginfo, geneinfo, cellinfo=ci, lineage_map={"c1": "stomach", "c2": "ovary"}, **files)
    lin = lib.sig.groupby("cell")["lineage"].first()
    assert lin["c1"] == "stomach" and lin["c2"] == "ovary" and lin["c3"] == "unknown" and lin["c4"] == "colon"    # the map also overrides known ones
    assert "HEK293T" in pr.DEFAULT_LINEAGE_MAP


def test_effect_concordance_runs_and_compares_with_random_pairs(world):
    lib, net, *_ = world
    ran = 0
    for kind in ("xpr", "oe", "cp"):
        try:
            r = pr.effect_concordance(lib, "T0", "T1", net, kind=kind, n_random=1, n_matched=1, min_cells=1)
        except ValueError as e:                                                  # too few perturbations of that kind in the synthetic library
            assert "perturbations" in str(e) or "signatures" in str(e)
            continue
        assert -1 <= r.rho <= 1 and r.n >= 10 and np.isfinite(r.activity_rho)
        assert len(r.random) == 1 and not ({"T0", "T1"} & (set(r.random["a"]) | set(r.random["b"])))
        assert 0 < r.p_random <= 1 and 0 < r.p_matched <= 1 and list(r.effects.columns) == ["T0", "T1"]
        assert r.meta["kind"] == kind
        ran += 1
    assert ran >= 1
    with pytest.raises(ValueError, match="No usable regulon"):
        pr.effect_concordance(lib, "T0", "NOPE", net)
