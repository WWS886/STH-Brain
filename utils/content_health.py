"""Health of observed subject content during pretraining."""
import torch
import torch.nn.functional as F
from model.brain.content_statistics import observed_center

CONTENT_HEALTH_PROTOCOL = 'observed_subject_content'


@torch.no_grad()
def content_health(values, present, normalize=False):
    if normalize:
        values = F.layer_norm(values.float(), (values.shape[-1],))
    centered, counts, observed = observed_center(values, present)
    eligible = counts >= 2
    z = centered.double().flatten(1)
    width = int(eligible.sum()) * values.shape[-1]
    capacity = min(max(len(z) - 1, 0), width)
    gram = z @ z.T
    energy = gram.diagonal().sum()
    eig = torch.linalg.eigvalsh((gram + gram.T) * .5).clamp_min(0)
    # Filter eigensolver noise relative to the largest eigenvalue.
    eig = torch.where(eig > eig.max() * 1e-12, eig, 0.)
    p = eig / eig.sum().clamp_min(1e-30)
    rank = float((-(p * p.clamp_min(1e-30).log()).sum()).exp()) if energy > 1e-12 else 0.
    per_slot_std = (centered.double().square().sum(0) / counts.clamp_min(1)[:, None]).sqrt()
    std = per_slot_std[eligible]
    return dict(protocol=CONTENT_HEALTH_PROTOCOL, subjects=len(z), dimension=values.shape[1]*values.shape[2],
                supported_dimension=width, eligible_slots=int(eligible.sum()), capacity=capacity,
                normalized_effective_rank=rank/max(width,1), capacity_normalized_rank=rank/max(capacity,1),
                feature_std=float(std.mean()) if std.numel() else 0., centered_energy=float(energy),
                exact_constant=bool(eligible.any() and not centered.any()),
                top1_energy=float(eig[-1]/eig.sum()) if energy>1e-12 else 0.,
                observed_fraction=float(present.float().mean()),
                min_slot_subjects=int(counts.min()),max_slot_subjects=int(counts.max()))


def branch_health(subject_branches):
    roi, network = subject_branches
    result={name:content_health(*item) for name,item in zip(('roi','network'),subject_branches)}
    # Widths differ: one-feature slots retain exactly the same observation mask.
    x=torch.cat([v.flatten(1) for v,_ in subject_branches],1)[...,None]
    m=torch.cat([valid[...,None].expand_as(v).flatten(1) for v,valid in subject_branches],1)
    result['joint']=content_health(x,m)
    return result
