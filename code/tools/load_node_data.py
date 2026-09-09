import pandas as pd

def node_data(INPUT):

    ######################################################################
    # Define the File Paths                                           #
    ######################################################################

    # The node files
    drugs = f"{INPUT}/nodes/all_drug_features.csv"
    ancestry = f"{INPUT}/nodes/ancestry_nodes.csv"
    clinical_outcomes = f"{INPUT}/nodes/clinical_outcomes_final_cui.csv"
    genes = f"{INPUT}/nodes/genes_node_dedup_mapped.csv"
    pathway = f"{INPUT}/nodes/pathway_nodes.csv"
    tissues = f"{INPUT}/nodes/tissue_nodes.csv"
    variants = f"{INPUT}/nodes/variant_nodes.csv"

    # The nodes dataframes
    df_variants = pd.read_csv(variants)
    df_tissues = pd.read_csv(tissues)
    df_genes = pd.read_csv(genes)
    df_ancestry = pd.read_csv(ancestry)
    df_clinical_outcomes = pd.read_csv(clinical_outcomes)
    df_pathway = pd.read_csv(pathway)
    df_drugs = pd.read_csv(drugs)

    return (df_variants, df_tissues, df_genes, df_ancestry, df_clinical_outcomes, df_pathway, df_drugs)