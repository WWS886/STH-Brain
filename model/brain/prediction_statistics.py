"""Subject-balanced pretraining prediction and normalization statistics."""
import torch


def grouped_prediction(error, indices, batch_size, target_type=None, weights=()):
    counts = torch.bincount(indices[:, 0], minlength=batch_size).to(error)
    if torch.any(counts == 0):
        raise ValueError('Every scan must have prediction targets')
    plain = error.new_zeros(batch_size).index_add(0, indices[:, 0], error) / counts
    diagnostics = {}
    if target_type is None:
        if weights:
            raise ValueError('Grouped loss requires target_type')
        return plain, diagnostics
    kinds = target_type[tuple(indices.T)]
    if torch.any((kinds < 1) | (kinds > 3)):
        raise ValueError('Target types must partition targets into 1,2,3')
    grouped, denominator = torch.zeros_like(plain), torch.zeros_like(plain)
    for kind in (1, 2, 3):
        selected = kinds == kind
        n = torch.bincount(indices[selected, 0], minlength=batch_size).to(error)
        means = error.new_zeros(batch_size).index_add(0, indices[selected, 0], error[selected]) / n.clamp_min(1)
        diagnostics[str(kind)] = {'token_count': int(n.sum().detach()),
                                 'scan_count': int((n > 0).sum().detach()),
                                 'error_sum': float(error[selected].detach().sum())}
        if weights:
            grouped += means * float(weights[kind - 1]) * (n > 0)
            denominator += float(weights[kind - 1]) * (n > 0)
    if not weights:
        return plain, diagnostics
    if torch.any(denominator <= 0):
        raise ValueError('A scan has no positively weighted target group')
    return grouped / denominator, diagnostics


@torch.no_grad()
def prediction_group_report(error,indices,subjects,sources,target_type):
    """Subject-uniform diagnostic group errors, explicitly local to this rank."""
    if target_type is None:return {}
    kinds=target_type[tuple(indices.T)];result={}
    for source in ['all',*sorted(set(sources))]:
        by_type={}
        for kind in (1,2,3):
            rows={};targets=0
            for scan,subject in enumerate(subjects):
                if source!='all' and sources[scan]!=source:continue
                keep=(indices[:,0]==scan)&(kinds==kind)
                n=int(keep.sum());targets+=n
                if n:rows.setdefault(subject,[]).append(float(error[keep].float().mean()))
            by_type[str(kind)]={'target_units':targets,'subjects':len(rows),
                'subject_mean_error':sum(sum(v)/len(v) for v in rows.values())/len(rows) if rows else None}
        result[source]=by_type
    return {'scope':'rank_local_subject_uniform_within_each_observed_group','by_source':result}


def source_subject_weights(names, counts, reference):
    if not counts or any(type(v) is not int or v <= 0 for v in counts.values()):
        raise ValueError('Source macro risk needs positive train-only subject counts')
    if any(n not in counts for n in names):
        raise ValueError('Unregistered source in macro risk')
    total = sum(counts.values())
    return reference.new_tensor([total / (len(counts) * counts[n]) for n in names])


def normalization_summary(stats, roi_valid, names=None):
    if not stats:
        return {}
    exposed = (stats['support_count'] > 0) & roi_valid.bool()
    fallback = stats['fallback_mask'] & exposed
    def summary(mask):
        count = int(mask.sum())
        return {'exposed_rois': count, 'fallback_rois': int((fallback & mask).sum()),
                'fallback_fraction': float((fallback & mask).sum()) / max(count, 1),
                'minimum_native_support': int(stats['support_count'][mask].min()) if count else 0}
    result = {'all': summary(exposed)}
    if names is not None:
        for name in sorted(set(names)):
            rows = torch.tensor([n == name for n in names], device=exposed.device)
            result[str(name)] = summary(exposed & rows[:, None])
    return result

