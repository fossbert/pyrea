import pandas as pd

import pyrea as pr


def test_import_exposes_public_api():
    for name in [
        "Gsea1T",
        "Gsea1TMultSets",
        "Gsea1TMultSigs",
        "Gsea2T",
        "GseaReg",
        "GseaRegMultSigs",
        "GseaMultReg",
        "Viper",
        "aREA",
        "gene_sets_to_regulon",
        "sig_to_reg",
        "load_genesets",
        "load_species_converter",
    ]:
        assert hasattr(pr, name)


def test_load_genesets_bundled_hallmark():
    genesets = pr.load_genesets("h", "human")
    assert isinstance(genesets, dict)
    assert len(genesets) > 0


def test_area_smoke(rng):
    genes = [f"gene{i}" for i in range(60)]
    genesets = {
        "set_a": genes[:25],
        "set_b": genes[20:45],
    }
    regulon = pr.gene_sets_to_regulon(genesets, minsize=20)

    dset = pd.Series(rng.normal(size=len(genes)), index=genes)
    nes = pr.aREA(dset, regulon)

    assert isinstance(nes, pd.DataFrame)
    assert set(nes.index) == {"set_a", "set_b"}
