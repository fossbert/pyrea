"""Loader for ESTIMATE scores (stromal, immune and ESTIMATE score; Yoshihara et al., Nat Commun 2013) downloaded per disease.

The scores are **not** part of the package (the terms of use of the source allow no redistribution that we know of): read the files you downloaded
with :func:`read_estimate`, or put them into your own registry with :func:`import_estimate` so that ``covariates=["estimate_stromal"]`` works
(see :mod:`pyrea.covariates`).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

__all__ = ["read_estimate", "import_estimate", "ESTIMATE_COHORTS"]

#: file name (without ``_RNAseqV2.txt``) of the per-disease downloads of the ESTIMATE site -> TCGA study code
ESTIMATE_COHORTS = {
    "acute_myeloid_leukemia": "LAML", "bladder_urothelial_carcinoma": "BLCA", "breast_cancer": "BRCA", "cervical_carcinoma": "CESC",
    "colorectal_adenocarcinoma": "COADREAD", "esophageal_carcinoma": "ESCA", "glioblastoma_multiforme": "GBM",
    "head_and_neck_squamous_cell_carcinoma": "HNSC", "kidney_chromophobe_renal_cell_carcinoma": "KICH",
    "kidney_renal_clear_cell_carcinoma": "KIRC", "kidney_renal_papillary_cell_carcinoma": "KIRP",
    "liver_hepatocellular_carcinoma": "LIHC", "low_grade_glioma": "LGG", "lung_adenocarcinoma": "LUAD",
    "lung_squamous_cell_carcinoma": "LUSC", "ovarian_serous_cystadenocarcinoma": "OV", "pancreatic_ductal_adenocarcinoma": "PAAD",
    "paraganglioma_and_pheochromocytoma": "PCPG", "prostate_adenocarcinoma": "PRAD", "skin_cutaneous_melanoma": "SKCM",
    "stomach_adenocarcinoma": "STAD", "thyroid_papillary_carcinoma": "THCA", "uterine_carcinosarcoma": "UCS",
    "uterine_corpus_endometrial_carcinoma": "UCEC", "adrenocortical_carcinoma": "ACC",
}
_COLUMNS = {"Stromal_score": "estimate_stromal", "Immune_score": "estimate_immune", "ESTIMATE_score": "estimate_score"}
_SUFFIXES = ("_RNAseqV2", "_RNAseq", "_Affymetrix", "_Agilent")


def _files(source) -> list:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_dir():
            files = sorted(p.glob("*.txt"))
            if not files:
                raise ValueError(f"No .txt files in {p}")
            return files
        return [p]
    return [Path(s) for s in source]


def _disease(path: Path) -> str:
    stem = path.stem
    for s in _SUFFIXES:
        if stem.endswith(s):
            stem = stem[: -len(s)]
    return stem


def read_estimate(source: "str | Path | Sequence[str | Path]", cohorts: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read ESTIMATE scores downloaded per disease from the ESTIMATE site (tab-separated: ``ID``, ``Stromal_score``, ``Immune_score``,
    ``ESTIMATE_score``).

    Parameters
    ----------
    source : path or list of paths
        A directory with the downloaded ``.txt`` files, one file, or a list of files. The disease is read from the file name
        (``stomach_adenocarcinoma_RNAseqV2.txt``; see :data:`ESTIMATE_COHORTS`).
    cohorts : iterable of str, optional
        Keep these TCGA study codes (``"STAD"``) or disease file names only. Colorectal cancer is one file: ``"COADREAD"``.

    Returns
    -------
    pd.DataFrame
        Index: the sample key of the file (``TCGA-XX-XXXX-01``; the last two digits are the sample type, e.g. 03 for blood-derived LAML).
        Columns ``estimate_stromal``, ``estimate_immune``, ``estimate_score`` (unit-free scores, comparable between samples; the ESTIMATE
        score is their sum) and ``cohort`` (the study code, or the file name if unknown). A sample that occurs in several files is kept once
        (the first) with a warning.
    """
    files = _files(source)
    want = None if cohorts is None else {str(c) for c in cohorts}
    frames = []
    for f in files:
        disease = _disease(f)
        code = ESTIMATE_COHORTS.get(disease, disease)
        if want is not None and not ({disease, code} & want):
            continue
        df = pd.read_csv(f, sep="\t", dtype={"ID": str})
        missing = {"ID", *_COLUMNS} - set(df.columns)
        if missing:
            raise ValueError(f"{f.name}: columns {sorted(missing)} missing; expected ID, Stromal_score, Immune_score, ESTIMATE_score")
        df = df.rename(columns=_COLUMNS).set_index("ID")[list(_COLUMNS.values())].astype(float)
        df["cohort"] = code
        frames.append(df)
    if not frames:
        raise ValueError("No ESTIMATE file selected" + ("" if want is None else f" for cohorts {sorted(want)}"))
    out = pd.concat(frames)
    dup = out.index.duplicated()
    if dup.any():
        warnings.warn(f"{int(dup.sum())} sample(s) occur in several files; the first is kept")
        out = out[~dup]
    out.index.name = "sample"
    return out


def import_estimate(source: "str | Path | Sequence[str | Path]", name: str = "estimate", overwrite: bool = False) -> Path:
    """Put downloaded ESTIMATE scores into **your** registry of background variables (``$PYREA_DATA/covariates``, see
    :func:`pyrea.add_covariates`): afterwards ``covariates=["estimate_stromal", "estimate_immune"]`` works in
    :func:`pyrea.activity_specificity_across_cohorts` and :func:`pyrea.load_covariates` finds the variables. ``source`` as in
    :func:`read_estimate`; returns the path of the stored table. The scores stay in your directory, not in the package."""
    df = read_estimate(source).drop(columns="cohort")
    return __import__("pyrea.covariates", fromlist=["add_covariates"]).add_covariates(
        name, df,
        description={"estimate_stromal": "ESTIMATE stromal score (ssGSEA of the stromal signature)",
                     "estimate_immune": "ESTIMATE immune score (ssGSEA of the immune signature)",
                     "estimate_score": "ESTIMATE score = stromal + immune (low: high tumour purity)"},
        source="ESTIMATE scores (Yoshihara et al., Nat Commun 2013, 4:2612) for TCGA RNA-seq V2 from bioinformatics.mdanderson.org/estimate; "
               "downloaded per disease. Sample key TCGA-XX-XXXX-NN.", overwrite=overwrite)
