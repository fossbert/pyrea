"""Regulon conversion between pandas and Bioconductor ``viper`` (needs R)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import require_rpy2

__all__ = ["regulon_to_r", "frame_to_r_matrix", "matrix_from_r", "read_regulon_rds"]

_COLUMNS = ["source", "target", "mor", "likelihood"]


def regulon_to_r(regulon: pd.DataFrame):
    """Long-format regulon (``source, target, mor, likelihood``) -> R ``regulon`` list."""
    ro, *_ = require_rpy2()
    if not all(regulon.columns.values == np.array(_COLUMNS)):
        regulon = regulon.set_axis(_COLUMNS, axis=1)

    entries = {}
    for source, grp in regulon.groupby("source", observed=True):
        tfmode = ro.FloatVector(grp["mor"].to_numpy(dtype=float))
        tfmode.names = ro.StrVector(grp["target"].astype(str))
        likelihood = ro.FloatVector(grp["likelihood"].to_numpy(dtype=float))
        entries[str(source)] = ro.ListVector({"tfmode": tfmode, "likelihood": likelihood})
    regulon_r = ro.ListVector(entries)
    regulon_r.rclass = ro.StrVector(["regulon"])
    return regulon_r


def frame_to_r_matrix(frame: pd.DataFrame):
    """DataFrame -> numeric R matrix keeping row and column names."""
    ro, pandas2ri, localconverter, _ = require_rpy2()
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(frame.astype(float))
    return ro.r["as.matrix"](df_r)


def matrix_from_r(mat) -> pd.DataFrame:
    """R matrix with dimnames -> DataFrame."""
    return pd.DataFrame(np.asarray(mat), index=list(mat.rownames), columns=list(mat.colnames))


def read_regulon_rds(path) -> pd.DataFrame:
    """Read a ``viper`` regulon saved with ``saveRDS`` into the long format.

    Returns columns ``source, target, mor, likelihood``. Regulons without a
    ``likelihood`` element get likelihood 1 (as ``viper`` itself assumes).
    """
    ro, *_ = require_rpy2()
    reg = ro.r["readRDS"](str(path))
    frames = []
    for name, entry in zip(reg.names, reg):
        entry = dict(zip(entry.names, entry))
        tfmode = entry["tfmode"]
        targets = list(tfmode.names)
        lik = np.asarray(entry["likelihood"]) if "likelihood" in entry else np.ones(len(targets))
        frames.append(pd.DataFrame({"source": name, "target": targets,
                                    "mor": np.asarray(tfmode), "likelihood": lik}))
    return pd.concat(frames, ignore_index=True)
