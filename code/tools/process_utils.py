import torch

def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def parse_relation_string(value):
    if value is None or value == '':
        return set()
    raw = str(value).strip()
    if not raw or raw.lower() in {'none', 'all'}:
        return set()

    # Treat each relation tuple as a separate group and only split within each tuple.
    # Accept either semicolon or pipe separators between tuples, and commas within a tuple.
    raw = raw.replace(';', '|').replace('-', ',')
    groups = [group.strip() for group in raw.split('|') if group.strip()]

    out = set()
    for group in groups:
        tokens = [token.strip().strip("()[]'\"") for token in group.split(',') if token.strip()]
        if len(tokens) == 3:
            out.add(tuple(tokens))
    return out


def drop_relation_types(data, relations_to_drop):
    relations_to_drop = {tuple(rel) for rel in relations_to_drop}
    for rel in list(data.edge_types):
        if tuple(rel) in relations_to_drop:
            del data[rel]
    return data


def apply_relation_filters(data, ignore_relations=''):
    ignored = parse_relation_string(ignore_relations)
    if ignored:
        data = drop_relation_types(data, ignored)
    return data


def restrict_one_edge_per_source(edge_index, mask):
    """Keep exactly one edge per source node in the selected mask.

    This is used only for validation/test edges after the usual source-aware split.
    Training masks are left untouched, so the graph keeps the original split semantics.
    """
    if edge_index is None or edge_index.numel() == 0:
        return mask.clone()

    selected_idx = torch.nonzero(mask, as_tuple=False).flatten()
    if selected_idx.numel() == 0:
        return mask.clone()

    source_ids = edge_index[0, selected_idx]
    kept = torch.zeros_like(mask, dtype=torch.bool)
    seen = set()
    for idx, src in zip(selected_idx.tolist(), source_ids.tolist()):
        src_id = int(src)
        if src_id in seen:
            continue
        seen.add(src_id)
        kept[idx] = True
    return kept