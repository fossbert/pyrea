"""
The pyrea package provides classes and functions for gene set enrichment analysis.
"""

from .Gsea1T import Gsea1T, Gsea1TMultSets, Gsea1TMultSigs
from .Gsea2T import Gsea2T, GseaReg, GseaRegMultSigs, GseaMultReg
from .Viper import Viper
from .rbackend.area import aREA
from .rbackend.viper import viper_activity
from .rbackend.regulon import read_regulon_rds
from .rpt import viper_rpt
from .mps import (relative_likelihood, mutant_phenotype_score, classify_phenotype,
                  mutation_association, locus_specific_mps, mps_targets, lr_to_rl)
from .mps_plot import plot_mps_rank, plot_rl_diagnostic
from .utils import gene_sets_to_regulon, sig_to_reg, load_genesets, load_species_converter
