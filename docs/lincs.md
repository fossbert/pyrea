[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Regulator activity in LINCS perturbation signatures (`pyrea.lincs`)

The level 5 signatures of the CLUE **beta** release (LINCS L1000; moderated z-scores of 12 328 genes per signature) contain overexpression (**OE**), CRISPR
knockout (**XPR**) and compound (**CP**) perturbations. `pyrea.lincs` determines the activity of regulators in them and answers, for a candidate such as ZEB1
or ZCCHC24:

1. Does the activity of the regulator fall when its own gene is knocked out and rise when it is overexpressed? (`tf_response`)
2. Which other regulators change when the gene is perturbed (top X)? (`perturbation_targets`)
3. For genes **without** a perturbation of their own: the perturbation of which genes changes the activity most? (`perturbed_by`)
4. Which compounds raise or lower the activity, and which mechanisms of action are enriched? (`tf_in_compounds`, `compound_set_enrichment`)

## Files and opening the library

Needed from CLUE: the `level5_beta_trt_{oe,xpr,cp}_n..x12328.gctx` matrices and `siginfo_beta.txt`, `geneinfo_beta.txt`; optional `cellinfo_beta.txt` (lineage of
the cell lines) and `compoundinfo_beta.txt` (mechanism of action, targets).

```python
from pathlib import Path
d = Path("~/Downloads").expanduser()
lib = pr.LincsLibrary(d / "siginfo_beta.txt", d / "geneinfo_beta.txt",
                      oe=d / "level5_beta_trt_oe_n34171x12328.gctx", xpr=d / "level5_beta_trt_xpr_n142901x12328.gctx",
                      cp=d / "level5_beta_trt_cp_...gctx",                                    # when the download is complete
                      cellinfo=d / "cellinfo_beta.txt", compoundinfo=d / "compoundinfo_beta.txt")
lib.signatures("xpr", gene="ZEB1", qc_pass=True)         # metadata of the signatures (cell line, time, dose, quality, lineage, ...)
lib.read(ids)                                             # genes x signatures (float32), read from disk on demand
```

The matrices are read from disk in blocks with `h5py` (nothing is loaded as a whole; the XPR matrix has 6.5 GB). Only the needed columns of `siginfo` are read
(the beta file has 1.2 million rows and 40 columns). The name of the perturbation is `pert_iname` or `cmap_name` (the gene symbol for OE and XPR); signatures in a
file whose `pert_type` does not fit (controls) are left out; files that list the 12 328 genes in a different order are aligned. `lib.sig` is the table of all
signatures (`kind`, `pert`, `cell`, `lineage`, `disease`, `time`, `dose`, quality columns).

**Quality.** `qc_pass=True` keeps the signatures that passed the CLUE quality control (`qc_pass == 1`); it is the **default** of all analyses
(`qc_pass=False` uses everything; no effect without the column).

## Networks and the activity (aREA, metaVIPER)

The activity of a regulator in a signature is the aREA enrichment of its regulon in the signature: the genes of the signature are ranked, the targets enter with
mode of regulation and likelihood, the result is the normalised enrichment score (NES). The signatures are already differential (z-scores against the plate
controls), so no signature step is applied (`method="none"`). The calculation is `pr.area_nes`, a numpy implementation that is tested to equal `viper::viper` (also
for regulons above 1000 targets); it is needed because the XPR matrix alone has 140 000 signatures.

```python
net = pr.LincsNetworks(lib, net_paths, minsize=25, cache_dir="results/lincs_networks")     # dict name -> .rds path or regulon DataFrame
net.coverage; net.tf_coverage("ZEB1")        # regulators usable in LINCS (>= minsize targets among its 12 328 genes), targets of a regulator per network
act = lib.activity(sig_ids, net, regulators=["ZEB1", "ZCCHC24"])                          # regulators x signatures
```

- **All networks are the default** and are integrated like metaVIPER, as `viper` does with a list of regulons: for every regulator and signature the weighted mean of the
  NES of the networks that contain it, with weights `|NES| ** mvws` (`pr.metaviper`, tested against `viper` with a list of regulons; `viper_activity` accepts a dict of
  regulons too). `per_network=True` returns the NES of every network.
- **Tumour-line-specific analyses.** `use=["BRCA", "STAD"]` restricts the networks, and `network_map={"breast": ["BRCA"], "MCF7": ["BRCA"], "colon": ["COAD"]}` integrates
  only the networks of the cell line (looked up first) or of its lineage (`cellinfo` `cell_lineage`) for each signature; cell lines that are not in the map use all networks.
  The cell lines of a tumour type are selected with `cells=` / `lineage=`.
  81 of the 245 cell lines have an unknown lineage in `cellinfo_beta.txt` (patient-derived pools, normal and stem cells, hardly any of them in CCLE/DepMap). `LincsLibrary(..., lineage_map=)` completes
  them: by default `pr.DEFAULT_LINEAGE_MAP` (HEK293T, HPTEC, AALE, 1HAE, HUH751, LHSAR, C42, HME, HMELZ, HAP1, CD34); a mapping of your own overrides it, `{}` switches it off.
- The NES of **one regulator** is comparable between signatures; NES of different regulators are only roughly comparable (the size of the regulon enters). The analyses
  therefore standardise every regulator against its own background (below).

## 1. Activity in the own perturbation: `tf_response`

```python
r = pr.tf_response(lib, "ZEB1", net, kinds=["xpr", "oe"], background=2000, seed=1)
r.summary     # per kind: n_signatures, n_cells, median_nes, frac_expected, auc, z_cells, p_cells, expression_rho
r.table       # per signature: nes, expression_z (the gene's own level in the signature), bg_percentile, expected
r.cells       # per kind and cell line: median activity and its percentile among the signatures of other genes in this cell line
```

Expected direction: XPR negative, OE positive. The activity is compared with its activity in `background` signatures of **other genes in the same cell lines** (same kind), so the
question is whether the change is larger than the one of an arbitrary perturbation (baselines of cell lines and plates cancel). `auc` is the probability that an own signature
is more extreme in the expected direction than a background signature; `z_cells` / `p_cells` are Stouffer's z over the cell lines of the percentile of the cell line's median
among the background of that cell line (one-sided); `expression_rho` correlates the gene's own level in the signature with the activity (the gene itself is not in its regulon,
so this is an independent check of the network).

## 2. Which regulators change when a gene is perturbed: `perturbation_targets`

```python
e = pr.perturbation_targets(lib, "ZEB1", net, kind="xpr", background=1500)
e.down(25); e.up(25)          # regulator, z, z_median, n_cells, n_signatures, frac_same_sign, self
```

The activity of all regulators in the signatures of the gene is standardised against the activity of the same regulator in the background signatures of other genes in the same
cell line (`z` per signature); the signatures of a cell line are summarised by their median and the cell lines by Stouffer's z (`min_cells` cell lines). `self` marks the regulator of
the perturbed gene.

## 3. Genes without a perturbation: which perturbations change the regulator? `perturbed_by`

```python
res = pr.perturbed_by(lib, "ZCCHC24", net, kinds=["xpr", "oe"], min_cells=3, cache_dir="results/lincs_scans")
res.down(25, kind="xpr"); res.up(25, kind="oe")        # kind, gene, z, z_median, n_cells, n_signatures, frac_same_sign, p, q, self
```

The activity of the regulator is determined in **all** signatures of the kind (chunked; one scan per regulator and kind is cached), standardised within the cell line and kind (a
cell line with a high baseline does not dominate), summarised per perturbed gene (median per cell line, Stouffer's z over cell lines, at least `min_cells`). `p` is a normal
approximation and `q` Benjamini-Hochberg within a kind; replicate and guide signatures are correlated, so read `q` as an aid for the ranking.

### Specific or a broad response? `perturbation_specificity`

The knockout of an essential gene (transcription, replication, translation) shifts the state of the cell and with it the activity of hundreds of regulators; such a hit of
`perturbed_by` says little about the regulator. `perturbation_specificity` places the regulator among **all** regulators for the signatures of each hit:

```python
hits = list(res.down(10, kind="xpr")["gene"]) + list(res.up(10, kind="xpr")["gene"])
pr.perturbation_specificity(lib, "ZCCHC24", hits, net, kind="xpr")
# gene, n_signatures, n_cells, z, rank (by |z| among all regulators), n_regulators, percentile (small = specific), n_changed (|z| >= 3), n_up, n_down
```

A hit is specific if the regulator is among the few that change (small `percentile`, small `n_changed`); a broad response (hundreds of regulators with `|z| >= 3`) points to a
generic cell-state change. Cost: one `perturbation_targets` per gene (all regulators, all networks).

## 4. Compounds: `tf_in_compounds`

```python
c = pr.tf_in_compounds(lib, "ZEB1", net, lineage="breast", times=[24], min_dose=1.0, min_cells=3)
c.up(25); c.down(25)                                   # pert_id, compound, moa, target, z, n_cells, ...
pr.compound_set_enrichment(c, top=100, direction="down", column="moa")     # mechanisms of action among the strongest decreases
```

Selection by `cells`, `lineage`, `times` (hours), `min_dose`. The activity is standardised within cell line and time (dose and plate effects are not modelled), summarised per compound
(median per cell line, Stouffer's z over cell lines). The enrichment splits `moa` / `target` at `|` and tests the top compounds against all compounds of the table (one-sided Fisher test,
Benjamini-Hochberg).

## What the results do and do not say

- The activity is inferred from the signature, not measured. A fall of the activity after the knockout of its own gene (question 1) is a check of the network and of the knockdown in that cell
  line; a missing fall does not show that the regulator is inactive (the regulon may not hold in the cell line).
- Only 12 328 genes are in the signatures (978 measured, the rest inferred); regulons are cut to them (`net.tf_coverage`). Regulators with few targets left are not scored.
- Perturbation signatures of different genes in one cell line share plate and batch effects; the standardisation within the cell line removes the baseline, not every structure.
- The z values are descriptive. They rank candidates; replicates (guides, constructs, doses) are not independent.

## Example (CLUE beta, pan-cancer TF networks of 23 TCGA cohorts, `qc_pass`)

Which perturbations exist: ZEB1 has 20 XPR and no OE signatures (10 cell lines), SNAI2 20 XPR and 18 OE, **ZCCHC24, GRHL2 and HNF1A have none** (ZEB2: 20 XPR, 1 OE). The
LINCS genes cover 69 to 440 targets of ZEB1 and 73 to 466 of ZCCHC24 per network.

- **Own perturbation (`tf_response`).** The activity responds only weakly in the expected direction: ZEB1 XPR median NES +0.39 (frac. expected 0.5, AUC 0.51, `p_cells` 0.38), SNAI2 XPR
  -0.81 (`p_cells` 0.045) and OE +0.43 (0.076), TP53 OE +1.1 (69 % expected, AUC 0.65, `p_cells` 0.35). Networks from tumours, a single-signature z-score and cell lines in which the regulator
  is little active limit what a knockout can show; a missing fall is not evidence of an inactive regulator.
- **Regulators changed by ZEB1 XPR (`perturbation_targets`, 20 signatures, 10 cell lines).** The epithelial regulators rise: OVOL1 (z 3.8), GRHL1, IRX4, ELF3, OVOL2, GRHL3, GRHL2 (z 3.1), SOX9 -
  the known antagonism of ZEB1 with OVOL and GRHL2. (z values are not corrected for the 1 700 regulators tested.)
- **Genes that change ZCCHC24 (`perturbed_by`, 5 157 XPR and 2 507 OE genes, about 9 minutes with 23 networks, then cached).** The activity rises most after XPR of TRRAP, CCNA2, RBBP6, COPA,
  RRM2, XPO1 and BRAF, and falls after XPR of NRG1, LAG3, DUSP7, RASAL2 and TP53. **`perturbation_specificity` shows that most are broad responses**: 200 to 970 of 1 688 regulators change
  with the same perturbation (essential genes). More specific: DUSP7 (rank 10 of 1 688, only 39 regulators change), LAG3 (rank 10, 236), BRAF (rank 6, 202), RBBP6 (rank 14, 344). BRAF and
  DUSP7 point in opposite MAPK directions (BRAF XPR up, DUSP7 XPR down); other MAPK genes (KRAS, RAF1, MAP2K2: up; MAPK3: down) lean the same way but none passes `q < 0.1`, so this is a lead, not
  a result.
- **Compounds (`tf_in_compounds`, ZEB1, MCF7 / A549 / HT29 at 24 h: 75 000 signatures, 8 600 compounds, 3 minutes).** No single compound passes the correction (three cell lines), but the
  mechanisms of action are informative: among the compounds that raise ZEB1 activity **MTOR inhibitors** (5 of 15, `q` 0.0005), **PI3K inhibitors** (3 of 16, 0.05) and IGF-1 inhibitors (2 of 6,
  0.05) are enriched.


## Do two regulators respond alike? `effect_concordance`

```python
r = pr.effect_concordance(lib, "ZCCHC24", "ZEB1", net_stad, kind="xpr", cells=["AGS"], n_random=200)
r                      # rho over the perturbations, activity rho, p_random, p_matched
r.random               # the random regulator pairs (rho, activity_rho)
```

Spearman correlation of the effects of both regulators over all perturbed genes (`kind="xpr"`/`"oe"`) or compounds (`"cp"`, with `lineage=` / `times=`), compared with `n_random` random pairs of other
regulators of the same networks. Regulators of one network respond alike for trivial reasons (shared targets, shared baseline activity), so `p_random` is lenient; `p_matched` compares only with the random pairs
whose activity correlation over the signatures is closest to that of the pair (`n_matched`). A concordance that beats these pairs is evidence for shared upstream control.
