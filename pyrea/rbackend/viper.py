"""Protein activity inference with ``viper::viper`` (needs R)."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from . import r_package, require_rpy2
from .regulon import frame_to_r_matrix, matrix_from_r, regulon_to_r

__all__ = ["viper_activity"]


def viper_activity(expression: pd.DataFrame,
                   regulon: "pd.DataFrame | Mapping[str, pd.DataFrame]",
                   method: str = "mad",
                   minsize: int = 25,
                   eset_filter: bool = True,
                   pleiotropy: bool = False,
                   nes: bool = True,
                   cores: int = 1,
                   pleiotropy_args: Optional[dict] = None,
                   mvws: float = 1.0) -> pd.DataFrame:
    """VIPER protein activity (Alvarez et al., Nat Genet 2016) via ``viper::viper``.

    Parameters
    ----------
    expression : pd.DataFrame
        Gene expression, genes x samples. Single-sample signatures are made
        by ``method``.
    regulon : pd.DataFrame or dict of DataFrames
        Long-format regulon, columns ``source, target, mor, likelihood``
        (see :func:`pyrea.read_regulon_rds`, :func:`pyrea.sig_to_reg`). A **dict name -> regulon** (several networks) runs metaVIPER: the NES of
        the networks are integrated per regulator as the mean weighted by ``|NES| ** mvws`` (see :func:`pyrea.metaviper`).
    method : {"none", "scale", "rank", "mad", "ttest"}
        Signature method of ``viper::viper``; "mad" centers on the sample
        median and scales by the MAD.
    minsize : int
        Minimum number of regulon targets present in the data.
    eset_filter : bool
        Restrict the expression matrix to regulon targets before ranking.
    pleiotropy : bool
        Apply the pleiotropy ("shadow") correction of ``viper``: the targets that a regulator shares with a regulon of a more
        significant regulator are down-weighted. It acts on the regulons *given*: with a regulon restricted to a few candidate
        regulators only these are corrected against each other.
    pleiotropy_args : dict, optional
        Arguments of the correction (``pleiotropyArgs`` of ``viper``); defaults ``regulators=0.05`` (p-value threshold of the regulators
        considered), ``shadow=0.05`` (p-value of the shadow effect), ``targets=10`` (minimum shared targets), ``penalty=20`` and
        ``method="adaptive"`` (or ``"absolute"``). Only the keys given are changed. Ignored without ``pleiotropy=True``.
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
    extra = {}
    if pleiotropy and pleiotropy_args is not None:
        ro, *_ = require_rpy2()
        defaults = {"regulators": 0.05, "shadow": 0.05, "targets": 10, "penalty": 20, "method": "adaptive"}
        unknown = sorted(set(pleiotropy_args) - set(defaults))
        if unknown:
            raise ValueError(f"Unknown pleiotropy_args {unknown}; allowed: {sorted(defaults)}")
        merged = {**defaults, **pleiotropy_args}
        if merged["method"] not in ("adaptive", "absolute"):
            raise ValueError("pleiotropy_args['method'] must be 'adaptive' or 'absolute'")
        extra["pleiotropyArgs"] = ro.ListVector({k: (ro.StrVector([v]) if isinstance(v, str) else ro.FloatVector([float(v)]))
                                                  for k, v in merged.items()})
    try:
        if isinstance(regulon, Mapping):
            ro, *_ = require_rpy2()
            regulon_r = ro.ListVector({str(k): regulon_to_r(v) for k, v in regulon.items()})
            extra["mvws"] = ro.FloatVector([float(mvws)])
        else:
            regulon_r = regulon_to_r(regulon)
        res = viper.viper(frame_to_r_matrix(expression), regulon_r,
                          method=method, minsize=minsize, eset_filter=eset_filter,
                          pleiotropy=pleiotropy, nes=nes, cores=cores, verbose=False, **extra)
    except Exception as exc:
        msg = str(exc)
        if pleiotropy and "regind" in msg:
            raise RuntimeError("viper's pleiotropy correction failed (known problem of its R code when, in a sample, only one pair of the "
                               "significant regulons shares enough targets): give it more regulators with overlapping regulons, or "
                               "use pleiotropy=False. R: " + msg.strip().splitlines()[0]) from exc
        raise
    return matrix_from_r(res)
