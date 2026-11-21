# pyrea

Gene set enrichment and VIPER regulon activity in Python, and the analyses built on them: mutant phenotype scores, phenocopy and co-mutation
tests, overlap and specificity of regulons and of activities across networks and cohorts, gene list scores and background variables.

Enrichment scoring (`aREA`, and therefore `Viper`, `Gsea1T`, `Gsea2T`, `viper_activity` and friends) is computed by the Bioconductor package
[`viper`](https://bioconductor.org/packages/viper/) via `rpy2` -- not a Python reimplementation -- so scores are identical to running `viper::aREA`
in R. Many analyses on top of it (`regulon_overlap`, `regulon_specificity`, `activity_specificity`, gene list collections, the figures) are pure
Python and need no R.

## Install

```bash
pip install -e .            # core: gene set / regulon utilities, no R needed
pip install -e '.[r]'       # aREA / Viper / Gsea1T / Gsea2T (needs R + Bioconductor viper)
pip install -e '.[r,test]'
```

```r
BiocManager::install("viper")
```

## Documentation

The documentation is split by topic; every page starts with the question it answers and shows the functions with short examples.

| Page | Topic | Main functions |
| --- | --- | --- |
| [docs/mps.md](docs/mps.md) | Mutant phenotype score, RPT methods, the relative likelihood (`density` / `cdf`), rank plots, locus-specific variants | `mutant_phenotype_score`, `viper_rpt`, `locus_specific_mps`, `plot_mps_rank`, `plot_mps_traits` |
| [docs/phenocopy.md](docs/phenocopy.md) | Do two alterations act alike? Cross-scoring, AUC with bootstrap intervals, figures, stratified comparison | `compare_phenotypes`, `PhenotypeComparison`, `stratified_comparison` |
| [docs/concordance.md](docs/concordance.md) | Is an effect seen in the activity but not (or oppositely) in the expression? | `activity_vs_expression` |
| [docs/comutation.md](docs/comutation.md) | Does a co-mutation modify the effect of a driver? (2 x 2 design) | `comutation_effect` |
| [docs/regulon-overlap.md](docs/regulon-overlap.md) | Overlap of regulons by tail, gene list collection and enrichment, activity split, many networks | `regulon_overlap`, `split_activity`, `overlap_across_networks`, `GeneListCollection` |
| [docs/specificity.md](docs/specificity.md) | Is a set of regulators more closely linked than other regulators? Targets (`regulon_specificity`) and activities (`activity_specificity`), across networks and cohorts | `regulon_specificity`, `specificity_across_networks`, `activity_specificity`, `activity_specificity_across_cohorts` |
| [docs/covariates.md](docs/covariates.md) | Gene list scores per sample, registry of background variables (ABSOLUTE purity, ESTIMATE import) | `score_genesets`, `load_covariates`, `add_covariates`, `read_estimate` |
| [docs/clustering.md](docs/clustering.md) | Clustering samples by protein activity | `pyrea.cluster` |

## Layout

| Module | Contents |
| --- | --- |
| `pyrea.rbackend.area` | `aREA` -- analytic rank-based enrichment (Alvarez et al., Nat Genetics 2016), via `viper::aREA` (needs the `r` extra) |
| `pyrea.Gsea1T` | `Gsea1T`, `Gsea1TMultSets`, `Gsea1TMultSigs` -- single-tailed GSEA |
| `pyrea.Gsea2T` | `Gsea2T`, `GseaReg`, `GseaRegMultSigs`, `GseaMultReg` -- two-tailed / regulon GSEA |
| `pyrea.Viper` | `Viper` -- VIPER regulon activity inference |
| `pyrea.utils` | `gene_sets_to_regulon`, `sig_to_reg`, `load_genesets`, `load_species_converter` |
| `pyrea.plotting` | internal plotting helpers used by the classes above |
| `pyrea.mps` | Mutant phenotype score (Alvarez et al. 2016): `relative_likelihood`, `mutant_phenotype_score`, `classify_phenotype`, `mutation_association`, `locus_specific_mps`. Only the association tests need R |
| `pyrea.rbackend.viper` | `viper_activity` -- `viper::viper` (mad/rank/... signatures) for protein activity, as DataFrame; `pleiotropy=True` with `pleiotropy_args` (needs R) |
| `pyrea.rbackend.regulon` | `read_regulon_rds` -- ARACNe/viper regulon from `.rds` to long format (needs R) |
| `pyrea.rpt` | `viper_rpt` -- residual post-translational activity (`rank`, `lineal`, `spline`), pure Python, matches `viper::viperRPT` |
| `pyrea.mps_plot` | `plot_mps_rank`, `plot_mps_traits`, `plot_rl_diagnostic` |
| `pyrea.phenotypes_plot` | figures of `compare_phenotypes` results: `cmp.plot()`, `plot_scores`, `plot_scale_scatter`, `plot_cross`, `plot_auc` |
| `pyrea.comutation` | `comutation_effect` -- effect of a co-alteration with and without a driver (2 x 2 design); `CoMutationResult` with figures (`pyrea.comutation_plot`) |
| `pyrea.regulon_overlap` | `regulon_overlap` -- overlap of the positive / negative target tails of several regulators, with hypergeometric and degree-weighted permutation null; `RegulonOverlap` with figures (`pyrea.regulon_overlap_plot`), gene list collection and export, enrichment, activity split (`pyrea.regulon_activity`); `pairs=` / `focus=` select the pairs |
| `pyrea.regulon_overlap_multi` | `overlap_across_networks` -- the overlap in a list of networks (paths of `.rds` files, one in memory at a time); `MultiOverlap` with per-pair summary across networks, recurrent shared targets, enrichment, export and a networks x pairs figure |
| `pyrea.regulon_specificity` | `regulon_specificity`, `top_partners` -- is a set of regulators (pair, triplet, ...) more closely linked than other regulators of the network? Size-matched random regulator sets and partner ranks; `pyrea.regulon_specificity_multi`: `specificity_across_networks`, `MultiSpecificity` (summary, matrix, merge, export, figure via `pyrea.regulon_specificity_plot`) |
| `pyrea.activity_specificity` | `activity_specificity`, `activity_partners` -- the activity counterpart (Spearman and partial correlations, partner ranks among all regulators, activity variants); `pyrea.activity_specificity_multi`: `activity_specificity_across_cohorts` (needs R), `MultiActivity` (summary, random-effects meta-analysis, pleiotropy effect, merge, export; figures in `pyrea.activity_specificity_plot`), `tcga_sample_id` |
| `pyrea.covariates` | `score_genesets` (gene lists scored per sample with aREA, needs R), registry of background variables (`available_covariates`, `load_covariates`, `add_covariates`, `remove_covariates`; bundled: ABSOLUTE purity / ploidy), `resolve_covariates` |
| `pyrea.estimate` | `read_estimate`, `import_estimate` -- loader for ESTIMATE stromal / immune scores downloaded per disease (not bundled; into your own registry) |
| `pyrea.genelists` | `GeneListCollection` (named gene lists with provenance, export to xlsx / csv / gmt), `gene_set_enrichment` (over-representation; bundled collections by name), `available_genesets` |
| `pyrea.concordance` | `activity_vs_expression` -- is an effect seen in the activity but not (or oppositely) in the expression? |

```python
reg = pr.read_regulon_rds("FullReg.rds")
vpres = pr.viper_activity(emat, reg)                         # method='mad'
rpt = pr.viper_rpt(vpres, emat)                              # method='rank'
mps = pr.mutant_phenotype_score(mutations, vpres, rpt)       # genes x samples, in [-1, 1]
pr.classify_phenotype(mps.loc["ERBB2"])                       # mutant / wt / intermediate (LR > 3)
assoc, var_mps = pr.locus_specific_mps(variants, vpres, rpt, expression=emat)  # rows "GENE:variant"
```

The public API is re-exported at the top level:

```python
import pyrea as pr

regulon = pr.gene_sets_to_regulon(pr.load_genesets("h", "human"))
res = pr.Gsea1T(signature, regulon)
```

## Bundled gene sets

`pyrea/data/` ships MSigDb v7.4 collections (Hallmark, PID, Reactome,
WikiPathways; human + mouse symbols) plus a human/mouse homology map, loaded
via `load_genesets(collection, species)` (`verbose=False` for no messages) / `load_species_converter(keys)`.
`pr.available_genesets()` lists the collections with their sizes; the enrichment functions (`pr.gene_set_enrichment`, `RegulonOverlap.enrich`)
accept the collection names directly.

## Tests

```bash
pytest -q                  # tests that need R and viper are skipped if they are not installed
```
