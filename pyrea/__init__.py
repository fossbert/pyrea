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
from .stratified import stratified_comparison, differential_activity, StratifiedComparison
from .cluster import (zscore_signature, viper_similarity, similarity_to_distance, pam, cluster_reliability,
                      cluster_samples, tsne_embedding, adjusted_rand_index, ari_permutation_test, ClusterResult)
from .phenotypes import compare_phenotypes, PhenotypeComparison
from .phenotypes_plot import plot_phenocopy
from .comutation import comutation_effect, CoMutationResult
from .regulon_overlap import regulon_overlap, RegulonOverlap
from .regulon_overlap_multi import overlap_across_networks, MultiOverlap
from .regulon_specificity import regulon_specificity, RegulonSpecificity, top_partners
from .regulon_specificity_multi import specificity_across_networks, MultiSpecificity
from .activity_specificity import activity_specificity, ActivitySpecificity, activity_partners
from .covariates import score_genesets, available_covariates, load_covariates, add_covariates, remove_covariates, resolve_covariates
from .activity_specificity_multi import activity_specificity_across_cohorts, MultiActivity, tcga_sample_id
from .genelists import GeneListCollection, gene_set_enrichment, available_genesets
from .regulon_activity import split_activity, ActivitySplit
from .concordance import activity_vs_expression, CONCORDANCE_CATEGORIES, CONCORDANCE_COLORS
from .mps_plot import plot_mps_rank, plot_mps_traits, plot_rl_diagnostic
from .utils import gene_sets_to_regulon, sig_to_reg, load_genesets, load_species_converter
