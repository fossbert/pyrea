"""Analytic rank-based enrichment (aREA), delegated to Bioconductor ``viper``.

Needs the ``r`` extra (``pip install pyrea[r]``) plus the R package ``viper``::

    BiocManager::install("viper")
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import r_package, require_rpy2
from .regulon import regulon_to_r

__all__ = ["aREA"]


def aREA(dset, regulon, minsize=20, dset_filter=False):
    """Analytic rank-based enrichment (Alvarez et al., Nat Genetics 2016).

    Thin wrapper around ``viper::aREA`` -- computes normalized enrichment
    scores (NES) for each regulator/pathway (``regulon["source"]``) across
    each sample in ``dset``.

    Parameters
    ----------
    dset : pd.Series or pd.DataFrame
        Gene expression signature(s), genes on the index. A Series is a
        single signature; its ``name`` becomes the one output column.
    regulon : pd.DataFrame
        Long-format regulon with columns ``source``, ``target``, ``mor``,
        ``likelihood`` (as produced by :func:`pyrea.utils.gene_sets_to_regulon`
        / :func:`pyrea.utils.sig_to_reg`).
    minsize : int
        Minimum number of regulon targets present in ``dset`` for a
        regulator to be scored.
    dset_filter : bool
        Restrict ``dset`` to genes that appear as a regulon target before
        ranking, instead of ranking over the full signature.

    Returns
    -------
    pd.DataFrame
        NES, regulators (``regulon["source"]``) x samples.
    """
    ro, pandas2ri, localconverter, importr = require_rpy2()
    viper = r_package("viper")

    if dset_filter:
        dset = dset.loc[dset.index.isin(regulon["target"])]

    is_series = isinstance(dset, pd.Series)
    frame = dset.to_frame(name=dset.name or "ges") if is_series else dset

    with localconverter(ro.default_converter + pandas2ri.converter):
        eset_r = ro.conversion.py2rpy(frame)

    regulon_r = regulon_to_r(regulon)

    res = viper.aREA(eset_r, regulon_r, minsize=minsize)
    nes = res.rx2("nes")

    nes_df = pd.DataFrame(
        np.asarray(nes),
        index=list(nes.rownames),
        columns=list(nes.colnames),
    )
    return nes_df
