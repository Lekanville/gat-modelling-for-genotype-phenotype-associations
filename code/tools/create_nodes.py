import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tools.smiles_features import smiles_to_feature_vector


# def nodes_init(data, df_variants, df_tissues, df_genes, df_ancestry, df_phenotype, df_meddra, df_pathway, df_drugs):
#     scaler = MinMaxScaler()

#     # --- VARIANT NODES ---
#     df_variants['CADD_Score'] = scaler.fit_transform(df_variants[['CADD_Score']])
#     variant_rsids = df_variants['rsid'].tolist()
#     variant_to_idx = {rsid: i for i, rsid in enumerate(variant_rsids)}
#     variant_features = df_variants.drop(columns=['rsid']).apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
#     data['variant'].x = torch.from_numpy(variant_features)

#     # --- TISSUE NODES ---
#     tissue_ids = df_tissues['Tissue_ID'].tolist() 
#     tissue_to_idx = {tid: i for i, tid in enumerate(tissue_ids)}
#     tissue_features = df_tissues.drop(columns=['Tissue_ID']).apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
#     data['tissue'].x = torch.from_numpy(tissue_features)

#     # --- GENE NODES ---
#     gene_ids = df_genes['Gene_Symbol'].tolist()
#     for col in ['Gene_Length', 'Molecular_Weight', 'PPI_Count', "Expression_Specificity_Score"]:
#         df_genes[col] = np.log10(df_genes[col] + 1)
#         df_genes[col] = scaler.fit_transform(df_genes[[col]])

#     for col in ['pLI_Score']:
#         df_genes[col] = scaler.fit_transform(df_genes[[col]])

#     gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
#     gene_features_df = df_genes.drop(columns=["Gene_Symbol", "Entrez_ID", "Gencode_ID", "Ensembl_ID", "flag_sum"])
#     gene_features = gene_features_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
#     data['gene'].x = torch.from_numpy(gene_features)

#     # --- ANCESTRY NODES ---
#     ancestry_ids = df_ancestry['node_id'].tolist()
#     ancestry_to_idx = {aid: i for i, aid in enumerate(ancestry_ids)}
#     data['ancestry'].x = torch.eye(len(ancestry_ids), dtype=torch.float32)

#     # --- PHENOTYPE NODES ---
#     pheno_ids = df_phenotype['HPO_ID'].tolist()
#     pheno_to_idx = {pid: i for i, pid in enumerate(pheno_ids)}
#     pheno_features = df_phenotype.drop(columns=['HPO_ID', 'phenotype']).apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
#     data['phenotype'].x = torch.from_numpy(pheno_features)

#     # --- SIDE_EFFECT NODES ---
#     meddra_codes = df_meddra['meddraCode'].tolist()
#     meddra_to_idx = {code: i for i, code in enumerate(meddra_codes)}
#     meddra_features = df_meddra.drop(columns=['count', 'llr']).apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
#     data['side_effect'].x = torch.from_numpy(meddra_features)

#     # --- PATHWAY NODES ---
#     pathway_ids = df_pathway['Pathway_ID'].tolist()
#     pathway_to_idx = {pwid: i for i, pwid in enumerate(pathway_ids)}
    
#     df_pathway['gene_count_log'] = np.log10(df_pathway['Gene_Count_In_Pathway'] + 1)
#     df_pathway['gene_count_norm'] = scaler.fit_transform(df_pathway[['gene_count_log']])

#     df_pathway['fdr_clipped'] = df_pathway['P_Value (FDR)'].clip(lower=1e-300)
#     df_pathway['log_fdr'] = -np.log10(df_pathway['fdr_clipped'])
#     df_pathway['log_fdr_norm'] = scaler.fit_transform(df_pathway[['log_fdr']])
    
#     cols_to_drop = [
#         'Pathway_ID', 'Pathway_Name', 'Gene_Count_In_Pathway', 
#         'gene_count_log', 'P_Value (FDR)', 'fdr_clipped', 'log_fdr', 'Gene_Count_Normalized'
#     ]
#     pathway_features = df_pathway.drop(columns=cols_to_drop, errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
#     data['pathway'].x = torch.from_numpy(pathway_features)

#     # --- DRUG NODES ---
#     drug_ids = df_drugs['chembl_id'].tolist()
#     drug_to_idx = {did: i for i, did in enumerate(drug_ids)}
#     drug_feature_matrix = np.vstack([smiles_to_feature_vector(s) for s in df_drugs['smiles']]).astype(np.float32)
#     data['drug'].x = torch.from_numpy(drug_feature_matrix)

#     return (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, pheno_to_idx, meddra_to_idx, pathway_to_idx, drug_to_idx, data)

def nodes_init(data, df_variants, df_tissues, df_genes, df_ancestry, df_clinical_outcomes, df_pathway, df_drugs):
    scaler = MinMaxScaler()
    
    # Initialize Features & ID Mappings                  
    
    # --- VARIANT NODES ---
    # will need to edit this in the preprocessing
    df_variants['CADD_Score'] = scaler.fit_transform(df_variants[['CADD_Score']])
    variant_rsids = df_variants['rsid'].tolist()
    variant_to_idx = {rsid: i for i, rsid in enumerate(variant_rsids)}
    # other_variant_cols = [col for col in df_variants.columns if col != 'rsid']
    # all_cols = ['rsid'] + other_variant_cols
    variant_features_df = df_variants.drop(columns=['rsid'])
    # variant_features_df = df_variants.drop(columns=['rsid'])
    data['variant'].x = torch.tensor(variant_features_df.values, dtype=torch.float32)

    # --- TISSUE NODES ---
    tissue_ids = df_tissues['Tissue_ID'].tolist() 
    tissue_to_idx = {tid: i for i, tid in enumerate(tissue_ids)}
    tissue_features_df = df_tissues.drop(columns=['Tissue_ID'])
    tissue_features_df = tissue_features_df.astype(int)
    data['tissue'].x = torch.tensor(tissue_features_df.values, dtype=torch.float32)

    # --- GENE NODES ---
    # will need to edit this in the preprocessing
    gene_ids = df_genes['Gene_Symbol'].tolist()
    # Skewed Continuous Normalization (Log + MinMax)
    for col in ['Gene_Length', 'Molecular_Weight', 'PPI_Count', "Expression_Specificity_Score"]:
        df_genes[col] = np.log10(df_genes[col] + 1)
        df_genes[col] = scaler.fit_transform(df_genes[[col]])

    # Direct Bounded Normalization
    for col in ['pLI_Score']:
        df_genes[col] = scaler.fit_transform(df_genes[[col]])

    gene_to_idx = {gid: i for i, gid in enumerate(gene_ids)}
    gene_features_df = df_genes.drop(columns=["Gene_Symbol", "Entrez_ID", "Gencode_ID", "Ensembl_ID", "flag_sum"])
    data['gene'].x = torch.tensor(gene_features_df.values, dtype=torch.float32)

    # --- ANCESTRY NODES ---
    ancestry_ids = df_ancestry['node_id'].tolist()
    ancestry_to_idx = {aid: i for i, aid in enumerate(ancestry_ids)}
    num_ancestry_nodes = len(ancestry_ids)
    data['ancestry'].x = torch.eye(num_ancestry_nodes, dtype=torch.float32)

    # # --- PHENOTYPE NODES ---
    # pheno_ids = df_phenotype['HPO_ID'].tolist()
    # pheno_to_idx = {pid: i for i, pid in enumerate(pheno_ids)}
    # pheno_features_df = df_phenotype.drop(columns=['HPO_ID', 'phenotype'])
    # data['phenotype'].x = torch.tensor(pheno_features_df.values, dtype=torch.float32)

    # # --- SIDE_EFFECTS NODES ---
    # meddra_codes = df_meddra['meddraCode'].tolist()
    # meddra_to_idx = {code: i for i, code in enumerate(meddra_codes)}
    # meddra_features_df = df_meddra.drop(columns=['count', 'llr'])
    # data['side_effect'].x = torch.tensor(meddra_features_df.values, dtype=torch.float32)

    # --- CLINICAL OUTCOMES ---
    cui_codes = df_clinical_outcomes['cui'].tolist()
    clinical_outcomes_to_idx = {code: i for i, code in enumerate(cui_codes)}
    cui_features_df = df_clinical_outcomes.drop(columns=['HPO_ID', 'cui', 'meddraCode', 'outcome', 'count', 'llr'])
    data['clinical_outcome'].x = torch.tensor(cui_features_df.values, dtype=torch.float32)

    # --- PATHWAY NODES ---
    pathway_ids = df_pathway['Pathway_ID'].tolist()
    pathway_to_idx = {pwid: i for i, pwid in enumerate(pathway_ids)}
    
    # will need to edit this in the preprocessing
    df_pathway['gene_count_log'] = np.log10(df_pathway['Gene_Count_In_Pathway'] + 1)
    df_pathway['gene_count_norm'] = scaler.fit_transform(df_pathway[['gene_count_log']])

    df_pathway['fdr_clipped'] = df_pathway['P_Value (FDR)'].clip(lower=1e-300)
    df_pathway['log_fdr'] = -np.log10(df_pathway['fdr_clipped'])
    df_pathway['log_fdr_norm'] = scaler.fit_transform(df_pathway[['log_fdr']])
    cols_to_drop = [
        'Pathway_ID', 'Pathway_Name',
        'Gene_Count_In_Pathway', 'gene_count_log', 
        'P_Value (FDR)', 'fdr_clipped', 'log_fdr', 'Gene_Count_Normalized'
    ]

    pathway_features_df = df_pathway.drop(columns=cols_to_drop)
    data['pathway'].x = torch.tensor(pathway_features_df.values, dtype=torch.float32)

    # --- DRUG NODES ---
    drug_ids = df_drugs['chembl_id'].tolist()
    drug_to_idx = {did: i for i, did in enumerate(drug_ids)}
    drug_feature_list = [smiles_to_feature_vector(s) for s in df_drugs['smiles']]
    data['drug'].x = torch.tensor(np.array(drug_feature_list), dtype=torch.float32)

    return (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, clinical_outcomes_to_idx, pathway_to_idx, drug_to_idx, data)