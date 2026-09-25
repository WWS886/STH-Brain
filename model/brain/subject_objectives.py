"""Full-backbone subject diversity on observed ROI and network features.

The trace-normalized log-det replaces per-slot spectral penalties. It measures
cross-slot redundancy on each entire ROI/network branch using an N x N Gram.
"""
import math

import torch
import torch.nn.functional as F


FOUR_TERM_PROTOCOL = 'roi_network_four_term'
LOGDET_OBJECTIVE = 'full_branch_logdet_isotropy'
LOGDET_RELATIVE_ENERGY_FLOOR = (32 * torch.finfo(torch.float32).eps) ** 2


def validate_four_term_objective(cfg, objective):
    """Enforce four atomic terms even when constructing the model without YAML."""
    if objective.loss_protocol != FOUR_TERM_PROTOCOL:
        raise ValueError('Unknown loss protocol')
    if cfg.representation_architecture != 'roi_temporal_hierarchy':
        raise ValueError('Four-term objective requires ROI identity fusion')
    eta=objective.context_moment_weight
    if isinstance(eta,bool) or not math.isfinite(eta) or not 0<=eta<=1:
        raise ValueError('context_moment_weight must lie in [0,1]')
    if objective.teacher_target!='signal_residual' or cfg.teacher_target_mode!='signal_residual':
        raise ValueError('Hierarchy targets require the signal residual teacher')
    weights=objective.logdet_branch_weights
    if (len(weights)!=2 or any(not math.isfinite(w) or w<=0 for w in weights)
            or not math.isclose(sum(weights),1.,abs_tol=1e-8)):
        raise ValueError('Logdet ROI/Network weights must be positive and sum to one')
    if (cfg.huber_weight != 1.0 or cfg.cosine_weight != 0.0
            or not math.isfinite(cfg.huber_beta) or cfg.huber_beta <= 0):
        raise ValueError('Four-term prediction must be a single SmoothL1 loss per branch')
    if objective.covariance_objective != LOGDET_OBJECTIVE:
        raise ValueError('Four-term objective replaces slot spectral loss with full-branch logdet')
    if cfg.signal_normalization not in {'visible_scan_roi_zscore', 'train_roi_frozen'}:
        raise ValueError('Four-term protocol requires normalization with audited native support')
    for name in ('prediction_weight', 'roi_prediction_weight', 'variance_weight',
                 'covariance_weight', 'logdet_alpha', 'variance_target', 'variance_epsilon'):
        if not math.isfinite(getattr(objective, name)) or getattr(objective, name) <= 0:
            raise ValueError(name + ' must be finite and positive')
    if objective.variance_epsilon >= objective.variance_target**2:
        raise ValueError('Variance numerical floor must be below the variance target')
    start, end = objective.covariance_start_epoch, objective.covariance_ramp_end_epoch
    if not all(math.isfinite(v) and v >= 0 for v in (start, end)):
        raise ValueError('Covariance schedule must be finite and nonnegative')
    if (end and not start < end <= 500) or (not end and start):
        raise ValueError('Covariance ramp requires 0 <= start < end <= 500')


def covariance_schedule(objective, update=0, steps_per_epoch=1):
    start, end = objective.covariance_start_epoch, objective.covariance_ramp_end_epoch
    if end == 0:
        return 1.0
    return min(1., max(0., (update / max(steps_per_epoch, 1) - start) / (end - start)))


def centered_logdet_penalty(centered, feature_count, alpha=1., reference_energy=None):
    """Normalized gap from the finite-sample isotropic log-det optimum.

    K = X X^T / tr(X X^T), c = min(N-1, supported feature count).
    L = log(1+alpha) - logdet(I + alpha*c*K)/c.
    Above a numerical near-zero region this is scale invariant. Constants and
    energy below 32 FP32 eps relative RMS receive the constant maximum gap,
    with zero spectral gradient; variance/monitoring remain separate mechanisms.
    """
    with torch.autocast(device_type=centered.device.type, enabled=False):
        x = centered.float()
        capacity = min(max(len(x)-1, 0), int(feature_count))
        if capacity == 0:
            return x.sum()*0
        if not math.isfinite(alpha) or alpha <= 0 or not bool(torch.isfinite(x).all()):
            raise ValueError('Logdet requires finite features and positive alpha')
        energy = x.double().square().sum()
        reference = float(x.numel()) if reference_energy is None else float(reference_energy)
        if not math.isfinite(reference) or reference < 0:
            raise ValueError('Logdet reference energy must be finite and nonnegative')
        floor = max(1e-12, reference * LOGDET_RELATIVE_ENERGY_FLOOR)
        if float(energy.detach()) <= floor:
            return x.sum() * 0 + math.log1p(alpha)
        scaled = x / energy.sqrt().to(x.dtype)
        gram = scaled @ scaled.T
        gram = (gram + gram.T) * .5
        matrix = torch.eye(len(x), device=x.device, dtype=x.dtype) + alpha*capacity*gram
        chol, info = torch.linalg.cholesky_ex(matrix)
        if bool((info != 0).any()) or not bool(torch.isfinite(chol).all()):
            raise FloatingPointError('Full-branch logdet Cholesky failed; objective not skipped')
        logdet = 2 * chol.diagonal().log().sum()
        return (math.log1p(alpha) - logdet / capacity).clamp_min(0)


def full_branch_moments(subject_slots, objective):
    """Observed-subject centering; missing slots cannot act as variance features.

    Inputs have already been differentiably gathered and reduced to one row
    per independent subject. ROI and network branches receive equal weight.
    """
    metrics, evidence = [], []
    for values, present in subject_slots:
        with torch.autocast(device_type=values.device.type, enabled=False):
            values = values.float()
            from model.brain.content_statistics import observed_center
            centered, counts, observed = observed_center(values, present)
            eligible = counts >= 2
            variance = centered.square().sum(0) / (counts-1).clamp_min(1)[:, None]
            std = (variance + objective.variance_epsilon).sqrt()
            denom = eligible.sum().clamp_min(1)
            var_loss = (F.relu(objective.variance_target-std).mean(-1)*eligible).sum()/denom
            mean_std = (std.mean(-1)*eligible).sum()/denom
            reference_energy = values.detach().double()[observed].square().sum()
            centered_energy = centered.detach().double().square().sum()
            energy_floor = max(1e-12, float(reference_energy) * LOGDET_RELATIVE_ENERGY_FLOOR)
            spectral = centered_logdet_penalty(centered.flatten(1),
                int(eligible.sum())*values.shape[-1], objective.logdet_alpha, reference_energy)
            metrics.append(torch.stack((var_loss, spectral, mean_std)))
            evidence.append(dict(eligible_slots=int(eligible.sum()), total_slots=values.shape[1],
                subject_counts=counts.cpu().tolist(), variance=float(var_loss.detach()),
                branch_logdet=float(spectral.detach()), feature_std=float(mean_std.detach()),
                regularizer=LOGDET_OBJECTIVE, subject_count=len(values),
                centered_energy=float(centered_energy), reference_energy=float(reference_energy),
                spectral_energy_floor=energy_floor, near_zero_energy=bool(centered_energy <= energy_floor),
                raw_min_feature_std=float(variance.detach()[eligible].sqrt().min()) if bool(eligible.any()) else 0.,
                fraction_below_variance_target=float((std[eligible]<objective.variance_target).float().mean()) if bool(eligible.any()) else 0.,
                exact_constant=bool(eligible.any() and not centered.any())))
    branches = torch.stack(metrics)
    average=branches.mean(0)
    spectral=(branches[:,1]*branches.new_tensor(objective.logdet_branch_weights)).sum()
    return torch.stack((average[0],spectral,average[2])), evidence, branches
