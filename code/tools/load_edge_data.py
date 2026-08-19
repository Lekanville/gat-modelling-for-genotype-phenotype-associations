import pandas as pd

def edge_data(INPUT):

    ######################################################################
    # Define the File Paths                                           #
    ######################################################################

    # The edges files
    ancestry_phenotype = f"{INPUT}/edges/ancestry_phenotype_edges.csv"
    gene_gene = f"{INPUT}/edges/df_ppi_final.csv"
    variant_gene = f"{INPUT}/edges/df_vg_master_clean.csv"
    drug_phenotype_treats = f"{INPUT}/edges/drug_treats_phenotype_edges.csv"
    drug_phenotype_causes = f"{INPUT}/edges/drug_causes_phenotype_edges.csv"
    drug_gene = f"{INPUT}/edges/drug_gene_edges_final.csv"
    gene_pathway = f"{INPUT}/edges/gene_pathway_edges.csv"
    gene_phenotype = f"{INPUT}/edges/gene_phenotype_edges.csv"
    gene_tissue = f"{INPUT}/edges/gene_tissue_edges_final.csv"
    phenotype_phenotype_lin = f"{INPUT}/edges/phenotype_phenotype_edges.csv"
    phenotypes_phenotypes_ldsc = f"{INPUT}/edges/phenotype_phenotype_ldsc.csv"
    tissue_phenotype = f"{INPUT}/edges/tissue_phenotype_edges.csv"
    variant_ancestry = f"{INPUT}/edges/va_final.csv"
    variant_phenotype = f"{INPUT}/edges/vp_edges_refined.csv"

    # The edges dataframes
    df_ancestry_phenotype = pd.read_csv(ancestry_phenotype)
    df_gene_gene = pd.read_csv(gene_gene)
    df_variant_gene = pd.read_csv(variant_gene)
    df_drug_phenotype_treats = pd.read_csv(drug_phenotype_treats)
    df_drug_phenotype_causes = pd.read_csv(drug_phenotype_causes)
    df_drug_gene = pd.read_csv(drug_gene)
    df_gene_pathway = pd.read_csv(gene_pathway)
    df_gene_phenotype = pd.read_csv(gene_phenotype)
    df_gene_tissue = pd.read_csv(gene_tissue)
    df_phenotype_phenotype_lin = pd.read_csv(phenotype_phenotype_lin)
    df_phenotypes_phenotypes_ldsc = pd.read_csv(phenotypes_phenotypes_ldsc)
    df_tissue_phenotype = pd.read_csv(tissue_phenotype)
    df_variant_ancestry = pd.read_csv(variant_ancestry)
    df_variant_phenotype = pd.read_csv(variant_phenotype)

    return (df_ancestry_phenotype, df_gene_gene, df_variant_gene, df_drug_phenotype_treats, df_drug_phenotype_causes, df_drug_gene, df_gene_pathway, 
            df_gene_phenotype, df_gene_tissue, df_phenotype_phenotype_lin, df_phenotypes_phenotypes_ldsc, df_tissue_phenotype,
            df_variant_ancestry, df_variant_phenotype)