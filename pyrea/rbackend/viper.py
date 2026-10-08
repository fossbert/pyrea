"""Protein activity inference with ``viper::viper`` (needs R)."""

from __future__ import annotations

import pandas as pd

from . import r_package
from .regulon import frame_to_r_matrix, matrix_from_r, regulon_to_r

__all__ = ["viper_activity"]


def viper_activity(expression: pd.DataFrame,
                   regulon: pd.DataFrame,
                   method: str = "mad",
                   minsize: int = 25,
                   eset_filter: bool = True,
                   pleiotropy: bool = False,
                   nes: bool = True,
                   cores: int = 1) -> pd.DataFrame:
    """VIPER protein activity (Alvarez et al., Nat Genet 2016) via ``viper::viper``.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression, genes x samples. Single-sample signatures are made
        by ``method``.
    regulon : pd.DataFrame
        Long-format regulon, columns ``source, target, mor, likelihood``
        (see :func:`pyrea.read_regulon_rds`, :func:`pyrea.sig_to_reg`).
    method : {"none", "scale", "rank", "mad", "ttest"}
        Signature method of ``viper::viper``; "mad" centers on the sample
        median and scales by the MAD.
    minsize : int
        Minimum number of regulon targets present in the data.
    eset_filter : bool
        Restrict the expression matrix to regulon targets before ranking.
    pleiotropy : bool
        Apply the pleiotropy correction (default arguments of ``viper``).
    nes : bool
        Return normalized enrichment scores (default); otherwise raw scores.
    cores : int
        Parallelism of ``viper``.

    Returns
    -------
    pd.DataFrame
        Activity, regulators x samples (the ``vpres`` of the VIPER workflow).
    """
    viper = r_package("viper")
    res = viper.viper(frame_to_r_matrix(expression), regulon_to_r(regulon),
                      method=method, minsize=minsize, eset_filter=eset_filter,
                      pleiotropy=pleiotropy, nes=nes, cores=cores, verbose=False)
    return matrix_from_r(res)
