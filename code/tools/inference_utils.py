import torch

############################################
# Inference utilities                   #
############################################

@torch.no_grad()
def topk_pheno_for_variant(model, data, device, variant_idx, k=10):
    model.eval()
    z = model(data)
    n_outcome = data['clinical_outcome'].x.size(0)
    k_eff = min(int(k), n_outcome)
    if k_eff <= 0:
        return [], []

    src = torch.full((n_outcome,), variant_idx, dtype=torch.long, device=device)
    dst = torch.arange(n_outcome, device=device)
    ei = torch.stack([src, dst], 0)
    logits = model.link_logits(z, ei, ('variant', 'associated_with', 'clinical_outcome'))
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k_eff)
    return topk.indices.tolist(), topk.values.tolist()

@torch.no_grad()
def topk_pheno_for_drug(model, data, device, drug_idx, k=10):
    model.eval()
    z = model(data)
    n_outcome = data['clinical_outcome'].x.size(0)
    k_eff = min(int(k), n_outcome)
    if k_eff <= 0:
        return [], []

    src = torch.full((n_outcome,), drug_idx, dtype=torch.long, device=device)
    dst = torch.arange(n_outcome, device=device)
    ei = torch.stack([src, dst], 0)
    logits = model.link_logits(z, ei, ('drug', 'treats', 'clinical_outcome'))
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k_eff)
    return topk.indices.tolist(), topk.values.tolist()

@torch.no_grad()
def topk_side_effect_for_drug(model, data, device, drug_idx, k=10):
    model.eval()
    z = model(data)
    n_outcome = data['clinical_outcome'].x.size(0)
    k_eff = min(int(k), n_outcome)
    if k_eff <= 0:
        return [], []

    src = torch.full((n_outcome,), drug_idx, dtype=torch.long, device=device)
    dst = torch.arange(n_outcome, device=device)
    ei = torch.stack([src, dst], 0)
    logits = model.link_logits(z, ei, ('drug', 'causes', 'clinical_outcome'))
    probs = torch.sigmoid(logits)
    topk = torch.topk(probs, k=k_eff)
    return topk.indices.tolist(), topk.values.tolist()

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