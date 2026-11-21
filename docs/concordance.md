[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Does activity show something that expression does not? (`activity_vs_expression`)

A recurring question: does the VIPER activity of a gene show an effect of an alteration that its own expression
does not, or one pointing the other way? `activity_vs_expression` makes the same two-group comparison on expression,
G and (optionally) RPT and compares the effects directly:

```python
res = pr.activity_vs_expression(mutations, vpres, emat, rpt=rpt)     # rows like mutant_phenotype_score
res.sort_values("abs_diff")[["delta_expr", "delta_G", "delta_RPT", "abs_diff", "abs_low", "abs_high", "category"]]
```

- The effect on each scale is **Cliff's delta** (`2 * AUC - 1`, in [-1, 1], > 0 = higher in the altered samples). It is rank-based
  and therefore comparable between expression (log counts) and activity (NES-like).
- "Significant on one scale, not on the other" does not show a difference. The **difference** of the effects is
  estimated directly with a paired bootstrap over the samples (the scales are measured on the same samples and are
  correlated, since activity is computed from expression): `delta_diff = delta_G - delta_expr` and
  `abs_diff = |delta_G| - |delta_expr|` (is the activity effect larger or smaller than the expression effect?), each with a
  `1 - alpha` interval.
- `category`: `concordant`, `opposite`, `activity_only`, `expression_only`, `unclear` (only one scale shows an effect, the
  difference is not established), `none`. An effect means BH `q < alpha` (per scale, over all rows) and
  `|delta| >= min_effect` (default 0.2). The categories sort results; the effects and intervals are the result.
- With `rpt`, `beyond_expression` marks rows where RPT shows an effect in the same direction as G.

```python
# scatter of the effects with cbrviz (colour = category, dotted diagonal = same effect on both scales)
import matplotlib.pyplot as plt
from cbrviz import XYview

cats = [c for c in pr.CONCORDANCE_CATEGORIES if c in set(res["category"])]
xv = XYview(res[["delta_expr", "delta_G"]], s1=res["category"], s1_order=cats,
            s1_colors=[pr.CONCORDANCE_COLORS[c] for c in cats], highlight=list(res.index), s=30, alpha=0.9)
fig, ax = plt.subplots(figsize=(4, 3.6))
ax.scatter(**xv.scatter_call)
ax.set(xlim=(-1, 1), ylim=(-1, 1), xlabel="Effect on expression", ylabel="Effect on activity (G)")
xv.add_xy_line(ax=ax)
xv.label_dots(ax=ax, adjust=True, fontsize=7)
ax.legend(handles=xv.handles, frameon=False, fontsize=6)
```

pyrea itself does not depend on cbrviz; the table is all a plot needs. (`XYview.label_dots` needs column names that are valid
identifiers, so keep `delta_expr` / `delta_G` and set the axis labels afterwards.)

Reading the result (examples from TCGA STAD/EAC amplifications and GBM drivers):

- An amplified gene usually shows a **larger effect on expression than on activity** (`abs_diff < 0`, mostly
  `expression_only`): the copy number raises the transcript directly, the regulon follows less. ERBB2 amplification is
  concordant (delta_expr 0.95, delta_G 0.64). Activity larger than expression is rare (VEGFA: 0.89 vs 0.82, RPT +0.49).
- In GBM, NF1 alterations lower the expression strongly (-0.85) but not the activity (-0.16): `expression_only`.
- **RPT is not independent of this difference**: it is the residual of the activity after regressing on expression, so where the
  activity does not follow a large expression effect, RPT takes the opposite sign (NF1: RPT +0.63) by construction. Read
  `beyond_expression` together with the effect on G, not on its own.
- A signal only in the activity needs checks before it is believed: copy-number effects on neighbouring regulon genes (compare
  G and the amplification itself), subtype composition (`stratified_comparison`), and replication in another cohort.
