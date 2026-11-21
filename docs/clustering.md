[pyrea](../README.md) · [documentation index](../README.md#documentation)

# Clustering samples by protein activity

`pyrea.cluster` follows the *Unsupervised data analysis* of
[Alvarez et al., Nat Genet 2018](https://doi.org/10.1038/s41588-018-0138-4):

```python
res = pr.cluster_samples(vpres)                       # VIPER similarity, PAM, k from the cluster reliability
res.k, res.scores                                     # chosen k; global reliability for every k tried
res.labels.value_counts()                             # cluster 1..k, largest first
res.reliability                                       # per patient: nes and scaled reliability
pd.crosstab(res.labels, subtype)

pr.cluster_samples(pr.zscore_signature(expr), similarity="correlation")   # expression instead (Pearson)

emb = pr.tsne_embedding(vpres, perplexity=40, n_iter=5000)                # needs scikit-learn (pyrea[cluster])
pr.ari_permutation_test(res.labels, other_labels, n_perm=10000)           # adjusted Rand index and p
```

Building blocks: `zscore_signature`, `viper_similarity` (= `viper::viperSimilarity`),
`similarity_to_distance`, `pam` (= `cluster::pam`: BUILD + SWAP), `cluster_reliability`,
`adjusted_rand_index`. `viper_similarity`, the distance and `pam` are checked against R in the tests.

- **Cluster reliability.** For every sample the members of its cluster are a gene set and the
  negative distances to all other samples the signature; the aREA NES says how close the cluster
  mates are. The scores are scaled to [0, 1] and averaged (= area over the cumulative curve) per
  cluster and overall; k is the first local maximum of the global value.
- **Neither the manuscript nor its supplement says how the scores are scaled** (Supplementary Fig. 2d-g only
  shows the result). Both options are kept in `res.scores`; compare them before trusting k. The default (`scale="bounds"`) uses the
  smallest and largest NES possible for a cluster of that size, so that 0.5 is chance level and
  values are comparable between k. `scale="minmax"` scales within the partition; then the global
  value hardly tells good from random partitions (a synthetic test shows it).
- With continuous data the global reliability keeps rising with k and the first local maximum can
  be fragile: look at `res.scores` before using the clusters.
