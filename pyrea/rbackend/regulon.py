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
    """DataFrame -> numeric R matrix keeping row and column names.

    The values are copied into R memory explicitly. (Converting a temporary ``frame.astype(float)``
    with pandas2ri and calling ``as.matrix`` afterwards let R read memory of an already freed numpy
    array: results were intermittently NaN or wrong.)
    """
    ro, *_ = require_rpy2()
    values = np.ascontiguousarray(frame.to_numpy(dtype=float))
    flat = values.ravel(order="F")                         # kept alive until R has made its own matrix
    mat = ro.r["matrix"](ro.FloatVector(flat), nrow=values.shape[0], ncol=values.shape[1])
    set_dimnames = ro.r("function(m, r, c) { dimnames(m) <- list(r, c); m }")
    mat = set_dimnames(mat, ro.StrVector([str(i) for i in frame.index]), ro.StrVector([str(c) for c in frame.columns]))
    del flat, values
    return mat


def matrix_from_r(mat) -> pd.DataFrame:
    """R matrix with dimnames -> DataFrame."""
    # np.array copies: np.asarray would be a view of R memory, which R may reuse after the object is collected
    return pd.DataFrame(np.array(mat, dtype=float), index=list(mat.rownames), columns=list(mat.colnames))


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
        lik = np.array(entry["likelihood"], dtype=float) if "likelihood" in entry else np.ones(len(targets))
        frames.append(pd.DataFrame({"source": name, "target": targets,
                                    "mor": np.array(tfmode, dtype=float), "likelihood": lik}))
    return pd.concat(frames, ignore_index=True)
