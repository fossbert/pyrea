# pyrea

Gene set enrichment tools: single- and two-tailed GSEA, VIPER/aREA regulon
activity, and the plotting helpers that go with them. Extracted from a
working analysis script.

Enrichment scoring (`aREA`, and therefore `Viper`, `Gsea1T`, `Gsea2T` and
friends) is computed by the Bioconductor package
[`viper`](https://bioconductor.org/packages/viper/) via `rpy2` -- not a
Python reimplementation -- so scores are identical to running `viper::aREA`
in R.

## Install

```bash
pip install -e .            # core: gene set / regulon utilities, no R needed
pip install -e '.[r]'       # aREA / Viper / Gsea1T / Gsea2T (needs R + Bioconductor viper)
pip install -e '.[r,test]'
```

```r
BiocManager::install("viper")
```

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
| `pyrea.mps_plot` | `plot_mps_rank`, `plot_rl_diagnostic` |

```python
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
via `load_genesets(collection, species)` / `load_species_converter(keys)`.
