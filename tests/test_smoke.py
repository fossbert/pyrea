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


def test_rbackend_submodule_imports_without_r():
    # importing pyrea and pyrea.rbackend must not require rpy2/R
    import pyrea.rbackend as rb

    assert hasattr(rb, "require_rpy2")
    assert hasattr(rb, "r_package")


def test_load_genesets_bundled_hallmark():
    genesets = pr.load_genesets("h", "human")
    assert isinstance(genesets, dict)
    assert len(genesets) > 0
