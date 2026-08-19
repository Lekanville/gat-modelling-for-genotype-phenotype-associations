import pandas as pd

def node_data(INPUT):

    ######################################################################
    # Define the File Paths                                           #
    ######################################################################

    # The node files
    variants = f"{INPUT}/nodes/variant_nodes.csv"
    tissues = f"{INPUT}/nodes/tissue_nodes.csv"
    genes = f"{INPUT}/nodes/genes_node_dedup_mapped.csv"
    ancestry = f"{INPUT}/nodes/ancestry_nodes.csv"
    phenotype = f"{INPUT}/nodes/phenotype_features.csv"
    side_effects = f"{INPUT}/nodes/side_effects_features.csv"
    pathway = f"{INPUT}/nodes/pathway_nodes.csv"
    drugs = f"{INPUT}/nodes/all_drug_features.csv"

    # The nodes dataframes
    df_variants = pd.read_csv(variants)
    df_tissues = pd.read_csv(tissues)
    df_genes = pd.read_csv(genes)
    df_ancestry = pd.read_csv(ancestry)
    df_phenotype = pd.read_csv(phenotype)
    df_meddra = pd.read_csv(side_effects)
    df_pathway = pd.read_csv(pathway)
    df_drugs = pd.read_csv(drugs)

    return (df_variants, df_tissues, df_genes, df_ancestry, df_phenotype, df_meddra, df_pathway, df_drugs)