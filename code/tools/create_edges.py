import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_edge_data(df, src_col, dst_col, src_mapping, dst_mapping, weight_col=None):
    """
    Translates raw string edge tables into PyTorch Geometric edge tensors.
    """
    # 1. Keep only valid edges where both source and target exist in our node sets
    valid_mask = df[src_col].isin(src_mapping) & df[dst_col].isin(dst_mapping)
    clean_df = df[valid_mask]
    
    # 2. Map string identifiers to mapped integer indices
    src_indices = clean_df[src_col].map(src_mapping).values
    dst_indices = clean_df[dst_col].map(dst_mapping).values
    
    # 3. Create PyG Edge Index Tensor with shape [2, Num_Edges]
    edge_index = torch.tensor(np.array([src_indices, dst_indices]), dtype=torch.long)
    
    # 4. Extract edge weights if column exists
    edge_weight = None
    if weight_col and weight_col in clean_df.columns:
        edge_weight = torch.tensor(clean_df[weight_col].values, dtype=torch.float32)
        
    return edge_index, edge_weight



def edges_init(data, edge_dfs, mappings):
    scaler = MinMaxScaler()

    # Unpack edge dataframes
    (df_ancestry_phenotype, df_gene_gene, df_variant_gene, df_drug_phenotype_treats, df_drug_phenotype_causes,
     df_drug_gene, df_gene_pathway, df_gene_phenotype, df_gene_tissue, 
     df_phenotype_phenotype_lin, df_phenotypes_phenotypes_ldsc, df_tissue_phenotype,
     df_variant_ancestry, df_variant_phenotype) = edge_dfs
    
    # Unpack dictionaries for readability
    (variant_to_idx, tissue_to_idx, gene_to_idx, ancestry_to_idx, 
         clinical_outcomes_to_idx, pathway_to_idx, drug_to_idx) = mappings

    # --- 1. Ancestry -> Phenotype ---
    idx, w = process_edge_data(df_ancestry_phenotype, 'ancestry', 'cui', ancestry_to_idx, clinical_outcomes_to_idx, weight_col='prevalence')
    data['ancestry', 'prevalent_in', 'phenotype'].edge_index = idx
    if w is not None: data['ancestry', 'prevalent_in', 'phenotype'].edge_weight = w

    # --- 2. Gene -> Gene (PPI) ---
    # will need to edit this in the preprocessing
    df_gene_gene['combined_score'] = df_gene_gene['combined_score'] / 1000.0
    idx, w = process_edge_data(df_gene_gene, 'source', 'target', gene_to_idx, gene_to_idx, weight_col='combined_score')
    data['gene', 'interacts_with', 'gene'].edge_index = idx
    if w is not None: data['gene', 'interacts_with', 'gene'].edge_weight = w

    # --- 3. Variant -> Gene ---
    idx, w = process_edge_data(df_variant_gene, 'source', 'target', variant_to_idx, gene_to_idx, weight_col='Unified_Weight')
    data['variant', 'maps_to', 'gene'].edge_index = idx
    if w is not None: data['variant', 'maps_to', 'gene'].edge_weight = w

    # --- 4a. Drug -> Phenotype (Treats) ---
    idx, w = process_edge_data(df_drug_phenotype_treats, 'source', 'cui', drug_to_idx, clinical_outcomes_to_idx)
    data['drug', 'treats', 'phenotype'].edge_index = idx

    # --- 4b. Drug -> Phenotype (Causes) ---
    idx, w = process_edge_data(df_drug_phenotype_causes, 'chembl_id', 'cui', drug_to_idx, clinical_outcomes_to_idx, weight_col='llr_norm')
    data['drug', 'causes', 'side_effect'].edge_index = idx
    if w is not None: data['drug', 'causes', 'side_effect'].edge_weight = w

    # --- 5. Drug -> Gene ---
    idx, w = process_edge_data(df_drug_gene, 'source', 'target', drug_to_idx, gene_to_idx, weight_col='weight_norm')
    type_cols = [c for c in df_drug_gene.columns if c.startswith('type_')]
    edge_attr = torch.tensor(df_drug_gene[type_cols].values, dtype=torch.float32)
    data['drug', 'targets', 'gene'].edge_index = idx
    data['drug', 'targets', 'gene'].edge_weight = w
    data['drug', 'targets', 'gene'].edge_attr = edge_attr

    # --- 6. Gene -> Pathway ---
    idx, w = process_edge_data(df_gene_pathway, 'Gene_Symbol', 'Pathway_ID', gene_to_idx, pathway_to_idx)
    data['gene', 'belongs_to', 'pathway'].edge_index = idx

    # --- 7. Gene -> Phenotype ---
    idx, w = process_edge_data(df_gene_phenotype, 'source', 'target', gene_to_idx, clinical_outcomes_to_idx, weight_col='weight')
    data['gene', 'associated_with', 'phenotype'].edge_index = idx
    if w is not None: data['gene', 'associated_with', 'phenotype'].edge_weight = w

    # --- 8. Gene -> Tissue ---
    idx, w = process_edge_data(df_gene_tissue, 'Gene_Symbol', 'Tissue_ID', gene_to_idx, tissue_to_idx, weight_col='Weight_Norm')
    data['gene', 'expressed_in', 'tissue'].edge_index = idx
    if w is not None: data['gene', 'expressed_in', 'tissue'].edge_weight = w

    # --- 9. Phenotype -> Phenotype (Lin Similarity) ---
    idx, w = process_edge_data(df_phenotype_phenotype_lin, 'source_cui', 'target_cui', clinical_outcomes_to_idx, clinical_outcomes_to_idx, weight_col='lin_similarity')
    data['phenotype', 'lin_similarity_with', 'phenotype'].edge_index = idx
    if w is not None: data['phenotype', 'lin_similarity_with', 'phenotype'].edge_weight = w

    # --- 10. Phenotype -> Phenotype (LDSC) ---
    idx, w = process_edge_data(df_phenotypes_phenotypes_ldsc, 'source_cui', 'target_cui', clinical_outcomes_to_idx, clinical_outcomes_to_idx, weight_col='weight')
    data['phenotype', 'genetically_correlated', 'phenotype'].edge_index = idx
    if w is not None: data['phenotype', 'genetically_correlated', 'phenotype'].edge_weight = w

    # --- 11. Tissue -> Phenotype ---
    idx, w = process_edge_data(df_tissue_phenotype, 'source_tissue', 'cui', tissue_to_idx, clinical_outcomes_to_idx, weight_col='weight_norm')
    data['tissue', 'enriched_for', 'phenotype'].edge_index = idx
    if w is not None: data['tissue', 'enriched_for', 'phenotype'].edge_weight = w

    # --- 12. Variant -> Ancestry ---
    idx, w = process_edge_data(df_variant_ancestry, 'rsid', 'target_ancestry', variant_to_idx, ancestry_to_idx)
    data['variant', 'observed_in', 'ancestry'].edge_index = idx
    # if w is not None: data['variant', 'observed_in', 'ancestry'].edge_weight = w

    # --- 13. Variant -> Phenotype ---
    idx, w = process_edge_data(df_variant_phenotype, 'rsid', 'cui', variant_to_idx, clinical_outcomes_to_idx, weight_col='magnitude')
    data['variant', 'associated_with', 'phenotype'].edge_index = idx
    if w is not None: data['variant', 'associated_with', 'phenotype'].edge_weight = w

    return data