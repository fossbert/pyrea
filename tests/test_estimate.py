"""pyrea.read_estimate / import_estimate: loader for ESTIMATE scores downloaded per disease."""

import numpy as np
import pandas as pd
import pytest

import pyrea as pr


def _write(path, ids, seed=0):
    rng = np.random.default_rng(seed)
    st, im = rng.normal(size=len(ids)) * 800, rng.normal(size=len(ids)) * 800 + 1500
    pd.DataFrame({"ID": ids, "Stromal_score": st.round(2), "Immune_score": im.round(2), "ESTIMATE_score": (st + im).round(2)}).to_csv(
        path, sep="\t", index=False)


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "estimate"
    d.mkdir()
    _write(d / "stomach_adenocarcinoma_RNAseqV2.txt", [f"TCGA-AA-{i:04d}-01" for i in range(30)], 0)
    _write(d / "acute_myeloid_leukemia_RNAseqV2.txt", [f"TCGA-AB-{i:04d}-03" for i in range(20)], 1)
    _write(d / "colorectal_adenocarcinoma_RNAseqV2.txt", [f"TCGA-AC-{i:04d}-01" for i in range(25)], 2)
    return d


def test_read_directory_file_list_and_filter(folder):
    df = pr.read_estimate(folder)
    assert list(df.columns) == ["estimate_stromal", "estimate_immune", "estimate_score", "cohort"] and df.shape == (75, 4)
    assert df.index.name == "sample" and df.index.is_unique and df["cohort"].value_counts().to_dict() == {"STAD": 30, "COADREAD": 25, "LAML": 20}
    assert (df["estimate_stromal"] + df["estimate_immune"] - df["estimate_score"]).abs().max() < 0.02
    assert df.loc["TCGA-AB-0003-03", "cohort"] == "LAML"
    one = pr.read_estimate(folder / "stomach_adenocarcinoma_RNAseqV2.txt")
    assert len(one) == 30 and set(one["cohort"]) == {"STAD"}
    both = pr.read_estimate([folder / "stomach_adenocarcinoma_RNAseqV2.txt", folder / "acute_myeloid_leukemia_RNAseqV2.txt"])
    assert len(both) == 50
    assert set(pr.read_estimate(folder, cohorts=["STAD", "LAML"])["cohort"]) == {"STAD", "LAML"}
    assert set(pr.read_estimate(folder, cohorts=["colorectal_adenocarcinoma"])["cohort"]) == {"COADREAD"}
    with pytest.raises(ValueError, match="No ESTIMATE file selected"):
        pr.read_estimate(folder, cohorts=["XYZ"])
    assert pr.ESTIMATE_COHORTS["stomach_adenocarcinoma"] == "STAD" and len(pr.ESTIMATE_COHORTS) >= 24


def test_unknown_disease_duplicates_and_format_errors(folder, tmp_path):
    _write(folder / "some_new_tumour_RNAseqV2.txt", [f"TCGA-AD-{i:04d}-01" for i in range(5)])
    assert "some_new_tumour" in set(pr.read_estimate(folder)["cohort"])                       # unknown file name: kept as it is
    _write(folder / "zz_stomach_copy.txt", [f"TCGA-AA-{i:04d}-01" for i in range(3)], 5)      # samples of the STAD file again
    with pytest.warns(UserWarning, match="several files"):
        df = pr.read_estimate(folder)
    assert df.index.is_unique and df.loc["TCGA-AA-0001-01", "cohort"] == "STAD"              # the first file is kept
    bad = tmp_path / "bad"
    bad.mkdir()
    pd.DataFrame({"ID": ["a"], "Stromal_score": [1.0]}).to_csv(bad / "x.txt", sep="\t", index=False)
    with pytest.raises(ValueError, match="columns .* missing"):
        pr.read_estimate(bad)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No .txt files"):
        pr.read_estimate(empty)


def test_import_into_the_registry(folder, tmp_path, monkeypatch):
    monkeypatch.setenv("PYREA_DATA", str(tmp_path / "data"))
    f = pr.import_estimate(folder)
    assert f.exists() and str(f).startswith(str(tmp_path / "data"))                          # in the user's registry, not in the package
    av = pr.available_covariates().set_index("variable")
    assert {"estimate_stromal", "estimate_immune", "estimate_score"} <= set(av.index)
    assert av.loc["estimate_stromal", "location"] == "user" and "Yoshihara" in av.loc["estimate_stromal", "source"]
    got = pr.load_covariates(["estimate_stromal", "estimate_immune"], samples=["TCGA.AA.0002.01A.11R", "TCGA.AB.0001.03A.11R", "zzz"],
                             sample_id=pr.tcga_sample_id)
    ref = pr.read_estimate(folder)
    assert got.iloc[0, 0] == pytest.approx(ref.loc["TCGA-AA-0002-01", "estimate_stromal"])
    assert got.iloc[1, 1] == pytest.approx(ref.loc["TCGA-AB-0001-03", "estimate_immune"]) and got.iloc[2].isna().all()
    with pytest.raises(ValueError, match="exists"):
        pr.import_estimate(folder)
    pr.import_estimate(folder, overwrite=True)
    assert "cohort" not in pr.load_covariates("estimate_score").columns
    from pathlib import Path
    assert not list((Path(pr.__file__).parent / "data" / "covariates").glob("estimate*"))     # and nothing in the package
