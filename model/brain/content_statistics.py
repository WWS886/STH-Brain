"""Observed content statistics, independent of missing-slot patterns."""
import torch


def observed_center(values, present):
    """Center each slot over its observers, preserving exact nonzero constants.

    Subtract an observed anchor before the FP64 mean. Missing observations and
    slots with fewer than two observers have zero residual, including gradients.
    """
    if values.ndim != 3 or present.shape != values.shape[:2] or present.dtype != torch.bool:
        raise ValueError('Expected [subjects, slots, features] and boolean presence')
    if not len(values) or not bool(torch.isfinite(values[present]).all()):
        raise ValueError('Observed content must be nonempty and finite')
    counts = present.sum(0)
    observed = present & (counts >= 2)[None]
    x = values.double()
    first = observed.long().argmax(0)
    anchor = x[first, torch.arange(x.shape[1], device=x.device)]
    delta = torch.where(observed[..., None], x - anchor, 0.)
    mean = delta.sum(0) / counts.clamp_min(1)[:, None]
    centered = torch.where(observed[..., None], delta - mean, 0.)
    return centered.float(), counts, observed


def pool_slots(values, scan, slot, batch, slots):
    """Mean tokens by fixed identity within a scan; never mix subjects."""
    key = scan * slots + slot
    count = torch.bincount(key, minlength=batch * slots)
    total = values.float().new_zeros((batch * slots, values.shape[-1]))
    total.index_add_(0, key, values.float())
    return (total / count.clamp_min(1)[:, None]).reshape(batch, slots, -1), count.reshape(batch, slots) > 0


def subject_slots(rows):
    """Visit means use only observed visits for each subject and slot."""
    values, masks = [], []
    for subject in sorted(rows):
        visits = rows[subject]
        x = torch.stack([v[0] for v in visits]).double()
        valid = torch.stack([v[1] for v in visits]).bool()
        count = valid.sum(0)
        # Anchored mean also preserves constants across different visit counts.
        first = valid.long().argmax(0)
        anchor = x[first, torch.arange(x.shape[1], device=x.device)]
        delta = torch.where(valid[..., None], x - anchor, 0.)
        mean = anchor + delta.sum(0) / count.clamp_min(1)[:, None]
        values.append(mean.masked_fill(~(count > 0)[..., None], 0.).float())
        masks.append(count > 0)
    return torch.stack(values), torch.stack(masks)
