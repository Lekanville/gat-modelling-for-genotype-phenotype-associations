import torch

############################################
# Inference utilities                   #
############################################

@torch.no_grad()
def _topk_for_relation_candidates(model, data, device, source_idx, rel, k=10):
    """Rank only the actual candidate destination nodes connected to a source under rel."""
    model.eval()
    z = model(data)

    src_type, _, dst_type = rel
    edge_index = data[rel].edge_index
    if edge_index.numel() == 0:
        return [], []

    mask = edge_index[0] == source_idx
    candidates = edge_index[1, mask].unique()
    if candidates.numel() == 0:
        return [], []

    k_eff = min(int(k), int(candidates.numel()))
    if k_eff <= 0:
        return [], []

    src = torch.full((candidates.numel(),), source_idx, dtype=torch.long, device=device)
    dst = candidates.to(device)
    ei = torch.stack([src, dst], dim=0)
    logits = model.link_logits(z, ei, rel)
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k_eff)
    return candidates[topk.indices].tolist(), topk.values.tolist()

@torch.no_grad()
def topk_pheno_for_variant(model, data, device, variant_idx, k=10):
    return _topk_for_relation_candidates(
        model,
        data,
        device,
        source_idx=variant_idx,
        rel=('variant', 'associated_with', 'clinical_outcome'),
        k=k,
    )

@torch.no_grad()
def topk_pheno_for_drug(model, data, device, drug_idx, k=10):
    return _topk_for_relation_candidates(
        model,
        data,
        device,
        source_idx=drug_idx,
        rel=('drug', 'treats', 'clinical_outcome'),
        k=k,
    )

@torch.no_grad()
def topk_side_effect_for_drug(model, data, device, drug_idx, k=10):
    return _topk_for_relation_candidates(
        model,
        data,
        device,
        source_idx=drug_idx,
        rel=('drug', 'causes', 'clinical_outcome'),
        k=k,
    )

@torch.no_grad()
def polypharmacy_risk(model, data, drug_a, drug_b, k=10, combine='add'):
    """Compose two drug embeddings, then score clinical outcomes."""
    model.eval()
    z = model(data)
    za = z['drug'][drug_a]
    zb = z['drug'][drug_b]

    if combine == 'add':
        zpair = za + zb
    elif combine == 'mean':
        zpair = (za + zb) / 2.0
    else:
        zpair = za + zb

    n_outcome = data['clinical_outcome'].x.size(0)
    k_eff = min(int(k), n_outcome)
    if k_eff <= 0:
        return [], []

    src_embed = zpair.unsqueeze(0).expand(n_outcome, -1)
    dst_embed = z['clinical_outcome']

    rel_key = 'drug__causes__clinical_outcome'
    r = model.scorer.rel_params[rel_key]

    logits = (src_embed * r * dst_embed).sum(-1)
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k_eff)
    return topk.indices.tolist(), topk.values.tolist()