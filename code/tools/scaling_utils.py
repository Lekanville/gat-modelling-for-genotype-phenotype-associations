import numpy as np
import pandas as pd


def _safe_abs_max(series):
    """Return the max absolute value, with a safe fall-back for empty/all-zero data."""
    vals = pd.to_numeric(series, errors="coerce").abs().replace(0, np.nan)
    max_val = vals.max()
    return 1.0 if pd.isna(max_val) or max_val <= 0 else float(max_val)


def _train_only_minmax(series, split_col=None, train_label="train"):
    """Apply min-max scaling using the train subset only when split labels are available."""
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if split_col is None:
        train_vals = vals
    else:
        train_vals = vals[series.index.map(lambda i: split_col in series.index.names)]

    if train_vals.empty:
        train_vals = vals

    min_val = train_vals.min()
    max_val = train_vals.max()
    if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
        return pd.Series(1.0, index=vals.index, dtype=float)
    return (vals - min_val) / (max_val - min_val)


def compute_variant_phenotype_magnitude(df_vp_edges, split_col=None, train_label="train"):
    """Compute a split-aware magnitude for variant-phenotype edges.

    If split_col is provided, the scaler is fit on the training split only and then
    applied to train/val/test. This avoids leakage from the full dataset.
    """
    df = df_vp_edges.copy()
    # df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    # df["beta_unified"] = df["beta"].fillna(np.log(pd.to_numeric(df["or_value"], errors="coerce")))

    if split_col is not None and split_col in df.columns:
        train_mask = df[split_col].astype(str).str.lower().eq(str(train_label).lower())
        train_max = _safe_abs_max(df.loc[train_mask, "beta_unified"])
        df["magnitude"] = df["beta_unified"].abs() / train_max
    else:
        max_val = _safe_abs_max(df["beta_unified"])
        df["magnitude"] = df["beta_unified"].abs() / max_val

    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce").fillna(0.0)
    return df


def scale_tissue_phenotype_edges(df_tissue_phenotype, split_col=None, train_label="train"):
    """Normalize tissue-to-phenotype edge weights per phenotype, using training-only bounds."""
    df = df_tissue_phenotype.copy()
    if 'weight' not in df.columns:
        return df

    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0.0)
    df['weight_norm'] = 1.0

    for cui, group in df.groupby('cui', dropna=False):
        group_idx = group.index
        if split_col is not None and split_col in df.columns:
            train_mask = df.loc[group_idx, split_col].astype(str).str.lower().eq(str(train_label).lower())
            train_vals = pd.to_numeric(group.loc[train_mask, 'weight'], errors='coerce').fillna(0.0)
        else:
            train_vals = pd.to_numeric(group['weight'], errors='coerce').fillna(0.0)

        if train_vals.empty:
            df.loc[group_idx, 'weight_norm'] = 1.0
            continue

        min_val = train_vals.min()
        max_val = train_vals.max()
        if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
            df.loc[group_idx, 'weight_norm'] = 1.0
            continue

        group_weights = pd.to_numeric(group['weight'], errors='coerce').fillna(0.0)
        df.loc[group_idx, 'weight_norm'] = (group_weights - min_val) / (max_val - min_val)

    return df


def scale_drug_causes_llr(df_drug_causes, split_col=None, train_label="train"):
    """Apply train-only normalization to drug-causes LLR values after log-stabilization."""
    df = df_drug_causes.copy()
    if 'llr' not in df.columns:
        return df

    df['llr'] = pd.to_numeric(df['llr'], errors='coerce').fillna(0.0)
    df['llr_log'] = np.log1p(df['llr'].clip(lower=0))
    df['llr_norm'] = 1.0

    if split_col is not None and split_col in df.columns:
        train_mask = df[split_col].astype(str).str.lower().eq(str(train_label).lower())
        train_vals = pd.to_numeric(df.loc[train_mask, 'llr_log'], errors='coerce').fillna(0.0)
    else:
        train_vals = pd.to_numeric(df['llr_log'], errors='coerce').fillna(0.0)

    if train_vals.empty:
        return df

    min_val = train_vals.min()
    max_val = train_vals.max()
    if pd.isna(min_val) or pd.isna(max_val) or np.isclose(max_val, min_val):
        df['llr_norm'] = 1.0
        return df

    df['llr_norm'] = (df['llr_log'] - min_val) / (max_val - min_val)
    return df