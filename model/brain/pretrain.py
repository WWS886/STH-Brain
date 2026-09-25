"""Pure ROI time-series JEPA. This pretrainer directly composes modules; no FC parent."""
import copy, math
from dataclasses import asdict
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather
from model.config import STHBrainConfig
from model.sth_brain_encoder import STHBrainEncoder
from model.sth_brain_predictor import STHBrainPredictor
from model.brain.core import PhysicalGeometry, PhysicalBackboneMethods, PhysicalTokenizer, _pack_flat
from settings.pretrain import ALGORITHM_VERSION, PretrainObjective
from model.brain.readout import encoded_readouts
from model.brain.objectives import prediction_error
from model.brain.normalization import visible_scan_roi_zscore
from utils.normalization import load_normalization_stats

def global_rows(x, enabled=True, sizes=None):
    """Differentiable gather, including unequal final-rank sizes; DDP averages once."""
    if not enabled or not dist.is_initialized():
        return x
    if sizes is None:
        size = torch.tensor([len(x)], device=x.device, dtype=torch.long)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        sizes = [int(v.item()) for v in sizes]
    padded = F.pad(x, (0, 0, 0, max(sizes) - len(x)))
    return torch.cat([v[:n] for (v, n) in zip(all_gather(padded), sizes)], 0)

class SubjectReduction:
    """One per forward; reuse metadata, never reuse features across updates."""

    def __init__(self, subjects, device, enabled=True):
        groups = [list(subjects)]
        if enabled and dist.is_initialized():
            groups = [None] * dist.get_world_size()
            dist.all_gather_object(groups, list(subjects))
        self.sizes = [len(group) for group in groups]
        self.subjects = [s for group in groups for s in group]
        unique = sorted(set(self.subjects))
        mapping = {s: i for (i, s) in enumerate(unique)}
        self.index = torch.tensor([mapping[s] for s in self.subjects], device=device, dtype=torch.long)
        self.count = torch.bincount(self.index, minlength=len(unique)).clamp_min(1)

def unique_subject_rows(values, subjects, enabled=True, reduction=None):
    plan = reduction if reduction is not None else SubjectReduction(subjects, values.device, enabled)
    values = global_rows(values, enabled, sizes=plan.sizes)
    sums = values.new_zeros((len(plan.count), values.shape[-1])).index_add(0, plan.index, values)
    return sums / plan.count.to(values)[:, None]


def representation_health(z):
    with torch.no_grad(), torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        z = z - z.mean(0)
        eig = torch.linalg.eigvalsh(z @ z.T / max(len(z) - 1, 1)).clamp_min(0)
        energy = eig.sum()
        p = eig / energy.clamp_min(1e-12)
        rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum()) if energy > 1e-12 else eig.new_zeros(())
        return {'normalized_effective_rank': float(rank / z.shape[1]), 'capacity_normalized_rank': float(rank / max(1, min(len(z) - 1, z.shape[1]))), 'feature_std': float(z.std(0, unbiased=False).mean())}


@torch.no_grad()
def teacher_target_diagnostics(target, clean):
    """Rank-local target signal BEFORE LayerNorm; no training graph retained."""
    rms = target.detach().float().square().mean(-1).sqrt()
    clean_rms = clean.detach().float().square().mean(-1).sqrt()
    return {'target_count': rms.numel(), 'pre_ln_rms_mean': float(rms.mean()), 'pre_ln_rms_min': float(rms.min()), 'pre_ln_rms_p05': float(torch.quantile(rms, 0.05)), 'pre_ln_rms_median': float(rms.median()), 'fraction_rms_below_1e_6': float((rms < 1e-06).float().mean()), 'clean_rms_mean': float(clean_rms.mean())}

class BrainTokenizer(PhysicalTokenizer):
    """Retain time/ROI rotations and bounded signal-position fusion, without auxiliary heads."""

    def __init__(self, geometry, cfg):
        if not isinstance(cfg, STHBrainConfig):
            cfg = STHBrainConfig(**asdict(cfg))
        super().__init__(geometry, cfg)
        self.normalization_epsilon = cfg.block_normalization_epsilon
        self.min_normalization_samples = cfg.min_normalization_samples
        if type(self.min_normalization_samples) is not int or self.min_normalization_samples < 1:
            raise ValueError('Minimum normalization samples must be a positive integer')
        self.signal_normalization = cfg.signal_normalization
        location = torch.zeros(geometry.roi_to_network.numel(), dtype=torch.float32)
        scale = torch.ones_like(location)
        if self.signal_normalization in {'train_roi_frozen', 'visible_scan_roi_zscore'}:
            if not cfg.normalization_stats_path or not cfg.normalization_stats_sha256:
                raise ValueError('ROI normalization fallback requires the train-only statistics path and SHA-256')
            stats = load_normalization_stats(cfg.normalization_stats_path, cfg.normalization_stats_sha256, expected_n_roi=geometry.roi_to_network.numel())
            self.normalization_provenance = {k: stats[k] for k in ('fit_split_col', 'fit_split', 'subject_count', 'scan_count', 'manifest_sha256')}
            location = torch.as_tensor(stats['location'], dtype=torch.float32)
            scale = torch.as_tensor(stats['scale'], dtype=torch.float32)
            expected = (geometry.roi_to_network.numel(),)
            if location.shape != expected or scale.shape != expected or (not torch.isfinite(location).all()) or (not torch.isfinite(scale).all()) or (not (scale > 0).all()):
                raise ValueError('ROI normalization fallback requires finite ROI-aligned location and strictly positive scale')
        elif self.signal_normalization != 'roi_block_zscore':
            raise ValueError('Unregistered ROI signal normalization')
        self.register_buffer('normalization_location', location, persistent=True)
        self.register_buffer('normalization_scale', scale, persistent=True)
        d = cfg.model_dim
        self.signal_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.identity_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.signal_projection = nn.Linear(d, d, bias=False)
        self.identity_projection = nn.Linear(d, d, bias=False)
        self.interaction_projection = nn.Linear(d, d, bias=False)
        self.gain_logit = nn.Parameter(torch.tensor(math.log(0.5)))
        for m in (self.signal_projection, self.identity_projection, self.interaction_projection):
            nn.init.orthogonal_(m.weight)

    def forward(self, block_values, block_sample_valid, *args, **kwargs):
        x = block_values.float()
        valid = block_sample_valid[..., None].to(x)
        if self.signal_normalization == 'train_roi_frozen':
            x = (x - self.normalization_location) / self.normalization_scale * valid
            normalization_debug = {}
        elif self.signal_normalization == 'visible_scan_roi_zscore':
            if len(args) < 3:
                raise ValueError('Visible scan normalization requires offsets, durations and token indices')
            roi_valid_mask = kwargs.get('roi_valid_mask')
            if roi_valid_mask is None:
                raise ValueError('Visible scan normalization requires roi_valid_mask')
            (x, normalization_debug) = visible_scan_roi_zscore(x, block_sample_valid, roi_valid_mask, args[2], self.geometry.roi_to_network, self.normalization_location, self.normalization_scale, network_count=self.geometry.num_networks, minimum_samples=self.min_normalization_samples, epsilon=self.normalization_epsilon)
        else:
            n = valid.sum(2, keepdim=True).clamp_min(1)
            mean = (x * valid).sum(2, keepdim=True) / n
            centered = (x - mean) * valid
            scale = (centered.square().sum(2, keepdim=True) / n).clamp_min(self.normalization_epsilon ** 2).sqrt()
            x = centered / scale
            normalization_debug = {}
        self._last_normalization_debug = normalization_debug
        c = super().forward(x, block_sample_valid, *args, **kwargs)
        if normalization_debug:
            c.update({f'normalization_{key}': value for (key, value) in normalization_debug.items()})
        return c

    def fuse_identity(self, signal, identity):
        interaction = self.interaction_projection(F.silu(self.signal_projection(self.signal_norm(signal.float()))) * F.silu(self.identity_projection(self.identity_norm(identity.float()))))
        return (signal + identity + (0.25 + 0.75 * self.gain_logit.sigmoid()) * interaction).to(signal.dtype)

    def summarize_networks(self, features, roi_indices, token_indices, stage_outputs=None):
        """Aggregate temporal ROI states within the same visible scan/block/network."""
        networks = self.geometry.num_networks
        blocks = int(token_indices[:, 1].max()) + 1
        scan_count = int(token_indices[:, 0].max()) + 1
        token_keys = (token_indices[:, 0] * blocks + token_indices[:, 1]) * networks + token_indices[:, 2]
        lookup = torch.full((scan_count * blocks * networks,), -1, device=features.device, dtype=torch.long)
        lookup[token_keys] = torch.arange(len(token_indices), device=features.device)
        keys = (roi_indices[:, 0] * blocks + roi_indices[:, 1]) * networks + self.geometry.roi_to_network[roi_indices[:, 2]]
        group = lookup[keys]
        if bool((group < 0).any()):
            raise ValueError('ROI content outside visible network support')
        counts = torch.bincount(group, minlength=len(token_indices))
        if bool((counts == 0).any()):
            raise ValueError('Network summary requires observed ROI content')
        x = features.float()
        count = counts[:, None].float()
        mean = x.new_zeros((len(counts), x.shape[-1])).index_add(0, group, x) / count
        centered = x - mean[group]
        variance = x.new_zeros(mean.shape).index_add(0, group, centered.square()) / count
        summaries = (mean, variance.clamp_min(1e-12).sqrt())
        attention = None
        if self.roi_pool_query is not None:
            scores = (x * self.roi_pool_query.float()).sum(-1) / math.sqrt(x.shape[-1])
            maxima = x.new_full((len(counts),), -torch.inf).scatter_reduce(0, group, scores, reduce='amax')
            exp = (scores - maxima[group]).exp()
            denominator = x.new_zeros(len(counts)).index_add(0, group, exp)
            attention = x.new_zeros(mean.shape).index_add(0, group, x * exp[:, None]) / denominator[:, None]
            summaries = (*summaries, attention)
        signal = self.local_to_model(torch.cat(summaries, -1))
        identity = self.network_embedding_scale * self.network_embedding(token_indices[:, 2])
        fused=self.fuse_identity(signal, identity)
        if stage_outputs is not None:
            from model.brain.content_statistics import pool_slots
            for name,value in (('roi_mean',mean),('roi_std',variance.clamp_min(1e-12).sqrt()),('roi_attention_pool',attention),('projection',signal),('identity_fusion',fused)):
                if value is None:continue
                stage_outputs['network_aggregation/'+name]=pool_slots(value.detach(),token_indices[:,0],token_indices[:,2],scan_count,networks)
        return fused

class BrainBackbone(nn.Module):
    _basis_for_indices = PhysicalBackboneMethods._basis_for_indices
    def _encode_rows(self, *args, prepared_tokens=None, **kwargs):
        if prepared_tokens is None:
            raise ValueError('Hierarchy Network encoding requires temporal ROI summaries')
        return PhysicalBackboneMethods._encode_rows(self, *args, prepared_tokens=prepared_tokens, **kwargs)
    _network_valid_mask = PhysicalBackboneMethods._network_valid_mask

    def __init__(self, cfg):
        super().__init__()
        if not isinstance(cfg, STHBrainConfig):
            cfg = STHBrainConfig(**asdict(cfg))
        if cfg.algorithm_version != ALGORITHM_VERSION:
            raise ValueError('STH-Brain TS backbone identity mismatch')
        if float(cfg.dropout) != 0.0:
            raise ValueError('STH-Brain pretraining backbone requires dropout=0; stochastic diversity is not a moment target')
        if cfg.representation_architecture != 'roi_temporal_hierarchy':
            raise ValueError('Unknown ROI-network architecture')
        cfg.validate_roi_temporal()
        self.cfg = cfg
        g = PhysicalGeometry(cfg.atlas_path, cfg.time_periods, maximum_time_frequency_octaves=cfg.maximum_time_frequency_octaves)
        self.tokenizer = BrainTokenizer(g, cfg)
        self.encoder = STHBrainEncoder(dim=cfg.model_dim, layers=cfg.encoder_layers, heads=cfg.encoder_heads, ffn_dim=cfg.encoder_ffn_dim, dropout=cfg.dropout, spatial_head_fraction=cfg.sope_head_fraction, geometry_additive_scale=cfg.geometry_additive_scale)
        from model.brain.cross_level import ROITemporalEncoder
        self.roi_temporal = ROITemporalEncoder(cfg)

    @property
    def geometry(self):
        return self.tokenizer.geometry

        
    def encode_full_tokens(self, **inputs):
        return self.encode_multilevel_tokens(**inputs)

    def encode_multilevel_tokens(self, *, block_values, block_sample_valid, block_sample_offsets, block_valid, block_center_seconds, block_duration_seconds, tr_seconds, roi_valid_mask, token_mask=None, stage_diagnostics=False):
        valid_grid = block_valid.bool()[..., None] & self._network_valid_mask(roi_valid_mask)[:, None]
        if token_mask is not None:
            if token_mask.shape != valid_grid.shape or torch.any(token_mask.bool() & ~valid_grid):
                raise ValueError('Invalid multilevel view mask')
            valid_grid = valid_grid & token_mask.bool()
        if torch.any(valid_grid.flatten(1).sum(1) == 0):
            raise ValueError('Each scan requires at least one visible supported network token')
        indices = valid_grid.nonzero()
        layer_slots={} if stage_diagnostics else None
        components = self.tokenizer(block_values, block_sample_valid, block_sample_offsets, block_duration_seconds, indices, roi_valid_mask=roi_valid_mask, stage_outputs=layer_slots)
        ridx = components['roi_indices']
        temporal = self.roi_temporal(components['roi_features'], ridx, block_center_seconds, self.geometry.bounded_time_frequencies(), self.geometry.roi_to_network.numel(),stage_outputs=layer_slots)
        prepared = self.tokenizer.summarize_networks(temporal, ridx, indices,stage_outputs=layer_slots)
        stages = {} if stage_diagnostics else None
        (_, network, basis, valid, positions) = self._encode_rows(indices, block_values, block_sample_valid, block_sample_offsets, block_center_seconds, block_duration_seconds, tr_seconds, prepared_tokens=prepared, roi_valid_mask=roi_valid_mask, stage_outputs=stages)
        block_ids = torch.full_like(valid, -1, dtype=torch.long)
        network_ids = torch.full_like(block_ids, -1)
        block_ids[indices[:, 0], positions] = indices[:, 1]
        network_ids[indices[:, 0], positions] = indices[:, 2]
        batch_size = block_values.shape[0]
        (roi, roi_valid, rp) = _pack_flat(temporal, ridx[:, 0], batch_size)
        (roi_local, _, _) = _pack_flat(components['roi_features'], ridx[:, 0], batch_size)
        (roi_temporal, _, _) = _pack_flat(temporal, ridx[:, 0], batch_size)
        roi_ids = torch.full_like(roi_valid, -1, dtype=torch.long)
        roi_blocks = torch.full_like(roi_ids, -1)
        roi_times = block_center_seconds.new_zeros(roi_valid.shape)
        roi_network_ids = torch.full_like(roi_ids, -1)
        target = (ridx[:, 0], rp)
        (roi_ids[target], roi_blocks[target]) = (ridx[:, 2], ridx[:, 1])
        roi_times[target] = block_center_seconds[ridx[:, 0], ridx[:, 1]]
        roi_network_ids[target] = self.geometry.roi_to_network[ridx[:, 2]]
        result = dict(tokens=network, token_valid=valid, token_basis=basis, token_block_ids=block_ids, token_network_ids=network_ids, roi_tokens=roi, roi_valid=roi_valid, roi_ids=roi_ids, roi_block_ids=roi_blocks, roi_time_seconds=roi_times, roi_network_ids=roi_network_ids, roi_local_tokens=roi_local, roi_temporal_tokens=roi_temporal, num_rois=self.geometry.roi_to_network.numel(), num_networks=self.geometry.num_networks, identity_preserving=self.cfg.representation_architecture == 'roi_temporal_hierarchy')
        if stages is not None:
            result['network_stage_tokens'] = stages
            from model.brain.readout import network_mean_readouts
            for name,value in {**stages,'network_final_norm':network}.items():
                pooled=network_mean_readouts(value.detach(),valid,network_ids,self.geometry.num_networks)
                layer_slots['network_encoder/'+name]=(pooled['raw'],pooled['valid'])
                means=pooled['raw'].gather(1,network_ids.clamp_min(0)[...,None].expand(-1,-1,value.shape[-1]))
                residual=(value.detach().float()-means).square()
                variance=network_mean_readouts(residual,valid,network_ids,self.geometry.num_networks)
                layer_slots['network_encoder/temporal_std/'+name]=(variance['raw'].clamp_min(0).sqrt(),variance['valid'])
            result['layer_slots']=layer_slots
        return result

class BrainPretrainer(nn.Module):

    def __init__(self, cfg, objective=None):
        if not isinstance(cfg, STHBrainConfig):
            cfg = STHBrainConfig(**asdict(cfg))
        super().__init__()
        self.cfg = cfg
        cfg.validate_predictors()
        self.objective = objective or PretrainObjective()
        from model.brain.subject_objectives import validate_four_term_objective
        validate_four_term_objective(cfg, self.objective)
        if self.objective.normalization != cfg.signal_normalization:
            raise ValueError('Model signal normalization disagrees with the objective contract')
        if self.objective.representation != 'layernorm_full_joint_student':
            raise ValueError('Hierarchy moments require the full joint student view')
        self.student = BrainBackbone(cfg)
        self.teacher = copy.deepcopy(self.student)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.predictor = STHBrainPredictor(model_dim=cfg.model_dim, predictor_dim=cfg.network_predictor_dim, layers=cfg.network_predictor_layers, heads=cfg.network_predictor_heads, ffn_dim=cfg.network_predictor_ffn_dim, dropout=cfg.dropout, spatial_head_fraction=cfg.sope_head_fraction, mode=cfg.predictor_mode, first_layer_query_residual=cfg.first_layer_query_residual, cross_attention_output_bias=cfg.cross_attention_output_bias)
        from model.brain.roi_identity import ROIPredictor
        if not math.isfinite(self.objective.roi_prediction_weight) or self.objective.roi_prediction_weight <= 0:
            raise ValueError('Identity fusion requires positive ROI prediction weight')
        self.predictor.roi_predictor = ROIPredictor(cfg, self.student.geometry)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def ema_update(self, momentum):
        for (t, s) in zip(self.teacher.parameters(), self.student.parameters()):
            t.mul_(momentum).add_(s, alpha=1 - momentum)
        for (t, s) in zip(self.teacher.buffers(), self.student.buffers()):
            t.copy_(s)

    def architecture_fingerprint(self):
        return {'algorithm_version': ALGORITHM_VERSION, 'model': asdict(self.cfg), 'objective': asdict(self.objective), 'normalization': self.cfg.signal_normalization, 'fc_path_present': False}

    def forward(self, *, block_values, block_sample_valid, block_sample_offsets, block_valid, block_center_seconds, block_duration_seconds, tr_seconds, roi_valid_mask, context_mask, target_mask, exclusion_mask, subject_uids, objective_distributed=True, return_debug=False, target_type=None, source_names=None, evaluation_only=False, diagnostics=True, stage_diagnostics=False, **unused):
        if evaluation_only and (self.training or torch.is_grad_enabled() or objective_distributed):
            raise ValueError('Lean validation requires eval(), no_grad and local objectives')
        diagnostics = diagnostics and (not evaluation_only)
        reduction = SubjectReduction(subject_uids, block_values.device, objective_distributed) if not evaluation_only else None
        b = len(block_values)
        valid = block_valid.bool()[..., None] & self.student._network_valid_mask(roi_valid_mask)[:, None]
        partition = context_mask.int() + target_mask.int() + exclusion_mask.int()
        if not torch.all(partition[valid] == 1) or torch.any(partition[~valid] != 0):
            raise ValueError('Invalid C/M/E partition')
        if torch.any(context_mask.flatten(1).sum(1) == 0) or torch.any(target_mask.flatten(1).sum(1) == 0):
            raise ValueError('Empty scan context/target')
        c = context_mask.nonzero()
        m = target_mask.nonzero()
        a = valid.nonzero()
        args = (block_values, block_sample_valid, block_sample_offsets, block_center_seconds, block_duration_seconds, tr_seconds)
        inputs = dict(block_values=block_values, block_sample_valid=block_sample_valid, block_sample_offsets=block_sample_offsets, block_valid=block_valid, block_center_seconds=block_center_seconds, block_duration_seconds=block_duration_seconds, tr_seconds=tr_seconds, roi_valid_mask=roi_valid_mask)
        identity_fusion = self.cfg.representation_architecture == 'roi_temporal_hierarchy'
        context = self.student.encode_multilevel_tokens(**inputs, token_mask=context_mask,stage_diagnostics=stage_diagnostics)
        (h, cb, cv) = (context['tokens'], context['token_basis'], context['token_valid'])
        context_normalization = dict(getattr(self.student.tokenizer, '_last_normalization_debug', {}))
        with torch.no_grad():
            teacher = self.teacher.encode_full_tokens(**inputs,stage_diagnostics=stage_diagnostics)
            (teacher_full_h, teacher_full_valid) = (teacher['tokens'], teacher['token_valid'])
            clean = teacher_full_h[teacher_full_valid]
            target = clean
            if self.objective.teacher_target == 'signal_residual':
                teacher_zero = self.teacher.encode_full_tokens(**{**inputs, 'block_values': torch.zeros_like(block_values)})
                zero_network = teacher_zero['tokens'][teacher_zero['token_valid']]
                target = clean - zero_network
            grid = target.new_zeros((*valid.shape, self.cfg.model_dim))
            grid[tuple(a.T)] = target
            clean_grid = clean.new_zeros(grid.shape)
            clean_grid[tuple(a.T)] = clean
            target_stats = {'network': teacher_target_diagnostics(grid[tuple(m.T)], clean_grid[tuple(m.T)])} if diagnostics else {}
            target = F.layer_norm(grid[tuple(m.T)].float(), (self.cfg.model_dim,)).detach()
        basis = self.student._basis_for_indices(m, block_center_seconds, block_duration_seconds, tr_seconds, torch.zeros(len(m), dtype=torch.long, device=m.device))
        (mb, mv, positions) = _pack_flat(basis, m[:, 0], b)
        network_predictor_stages={} if stage_diagnostics else None
        prediction = self.predictor(h, cb, cv, mb, mv,stage_outputs=network_predictor_stages)[m[:, 0], positions].float()
        error = prediction_error(prediction, target, self.cfg)
        from model.brain.prediction_statistics import grouped_prediction, source_subject_weights, normalization_summary
        (scan_error, group_diagnostics) = grouped_prediction(error, m, b, target_type, self.objective.target_group_weights)
        full_encoded = self.student.encode_full_tokens(**inputs, stage_diagnostics=stage_diagnostics or return_debug)
        full_normalization = dict(getattr(self.student.tokenizer, '_last_normalization_debug', {}))
        full_readouts = encoded_readouts(full_encoded)
        subject_z = unique_subject_rows(full_readouts['layernorm'], subject_uids, objective_distributed, reduction=reduction) if not evaluation_only else None
        risk_error = scan_error
        risk_weights = torch.ones_like(scan_error)
        if self.objective.source_risk == 'source_macro' and self.training:
            if source_names is None:
                raise ValueError('Source macro risk requires dataset identities')
            risk_weights = source_subject_weights(source_names, self.objective.source_subject_counts, scan_error)
            risk_error = scan_error * risk_weights
        roi_scan = scan_error * 0
        moment_evidence = []
        from model.brain.roi_identity import identity_moments
        with torch.no_grad():
            rt = teacher['roi_tokens']
            if self.objective.teacher_target == 'signal_residual':
                rt = rt - teacher_zero['roi_tokens']
            si = torch.arange(b, device=rt.device)[:, None]
            select = teacher['roi_valid'] & target_mask[si, teacher['roi_block_ids'].clamp_min(0), teacher['roi_network_ids'].clamp_min(0)]
            if diagnostics:
                target_stats['roi'] = teacher_target_diagnostics(rt[select], teacher['roi_tokens'][select])
            roi_residual = rt[select].float() if stage_diagnostics else None
            rt = F.layer_norm(rt.float(), (self.cfg.local_dim,))
        roi_predictor_stages={} if stage_diagnostics else None
        (rp, (rs, rpos)) = self.predictor.roi_predictor(context, teacher, select, self.student.geometry,stage_outputs=roi_predictor_stages)
        re = prediction_error(rp.float(), rt[rs, rpos].detach(), self.cfg)
        from model.brain.roi_identity import target_local_support
        local_support = target_local_support(context, teacher, select)
        routing = {'targets': len(re), 'local_targets': int(local_support.sum()), 'network_fallback_targets': int((~local_support).sum()), 'local_fraction': float(local_support.float().mean()), 'local_target_error': float(re.detach()[local_support].mean()) if bool(local_support.any()) else 0.0, 'fallback_target_error': float(re.detach()[~local_support].mean()) if bool((~local_support).any()) else 0.0}
        units = (rs * valid.shape[1] + teacher['roi_block_ids'][rs, rpos]) * valid.shape[2] + teacher['roi_network_ids'][rs, rpos]
        sums = re.new_zeros(valid.numel()).index_add(0, units, re)
        counts = re.new_zeros(valid.numel()).index_add(0, units, torch.ones_like(re))
        unit_errors = (sums / counts.clamp_min(1)).reshape(valid.shape)[tuple(m.T)]
        (roi_scan, _) = grouped_prediction(unit_errors, m, b, target_type, self.objective.target_group_weights)
        scan_error = scan_error + self.objective.roi_prediction_weight * roi_scan
        z = encoded_readouts(context)['layernorm']
        layer_slots={}
        if stage_diagnostics:
            from model.brain.content_statistics import pool_slots
            for prefix,encoded in (('student_context',context),('student_full',full_encoded),('teacher_full',teacher)):
                layer_slots.update({prefix+'/'+name:item for name,item in encoded['layer_slots'].items()})
            layer_slots.update({'roi_predictor/'+name:item for name,item in roi_predictor_stages.items()})
            for name,value in network_predictor_stages.items():
                layer_slots['network_predictor/'+name]=pool_slots(value[m[:,0],positions],m[:,0],m[:,2],b,self.student.geometry.num_networks)
            for name,value in (('roi_pre_ln',roi_residual),('roi_ln',rt[rs,rpos])):
                layer_slots['teacher_target/'+name]=pool_slots(value,rs,teacher['roi_ids'][rs,rpos],b,self.student.geometry.roi_to_network.numel())
            for name,value in (('network_pre_ln',grid[tuple(m.T)]),('network_ln',target)):
                layer_slots['teacher_target/'+name]=pool_slots(value,m[:,0],m[:,2],b,self.student.geometry.num_networks)
        if not evaluation_only:
            moments = identity_moments(full_encoded, subject_uids, self.objective, objective_distributed, slots=full_readouts.get('slots'), reduction=reduction, return_branches=True, return_subject_slots=True)
            ((var, cov, std), moment_evidence) = moments[:2]
            regularizer_branches = moments[2]
            regularizer_view_evidence = {'context_weight':self.objective.context_moment_weight,
                                        'full':moment_evidence,'context':[]}
            if self.objective.context_moment_weight > 0:
                context_moments = identity_moments(context, subject_uids, self.objective,
                    objective_distributed, reduction=reduction, return_branches=True)
                eta = self.objective.context_moment_weight
                var = (1-eta)*var + eta*context_moments[0][0]
                cov = (1-eta)*cov + eta*context_moments[0][1]
                regularizer_branches = (1-eta)*regularizer_branches + eta*context_moments[2]
                regularizer_view_evidence['context'] = context_moments[1]
        if evaluation_only:
            evaluation = {'scan_prediction_loss': scan_error.detach(), 'student_embeddings': z.detach(), 'student_full_embeddings': full_readouts['layernorm'].detach(), 'student_full_raw_embeddings': full_readouts['raw'].detach()}
            if stage_diagnostics and identity_fusion:
                from utils.stage_health import stage_vectors
                evaluation['stage_vectors'] = stage_vectors(full_encoded)
                evaluation['layer_slots'] = layer_slots
            return evaluation
        subject_losses = unique_subject_rows(torch.stack((risk_error, roi_scan * risk_weights), -1), subject_uids, objective_distributed, reduction=reduction).mean(0)
        (network_pred_loss, roi_pred_loss) = subject_losses.unbind()
        pred_loss = network_pred_loss + self.objective.roi_prediction_weight * roi_pred_loss
        from model.brain.subject_objectives import covariance_schedule
        covariance_weight = self.objective.covariance_weight * covariance_schedule(self.objective, getattr(self, 'optimizer_update', 0), getattr(self, 'steps_per_epoch', 1))
        objective_terms = {}
        objective_terms = {'network_prediction': self.objective.prediction_weight * network_pred_loss, 'roi_prediction': self.objective.prediction_weight * self.objective.roi_prediction_weight * roi_pred_loss, 'variance': self.objective.variance_weight * var, 'subject_logdet': covariance_weight * cov}
        loss = sum(objective_terms.values())
        result = {'loss': loss, 'prediction_loss': pred_loss, 'variance_loss': var, 'covariance_loss': cov, 'student_feature_std': std, 'scan_prediction_loss': scan_error.detach(), 'student_embeddings': z.detach(), 'student_full_embeddings': full_readouts['layernorm'].detach(), 'student_full_raw_embeddings': full_readouts['raw'].detach(), 'subject_count': len(subject_z)}
        if diagnostics:
            from utils.content_health import branch_health
            result['branch_health'] = branch_health(moments[3])
        result['network_prediction_loss'] = network_pred_loss
        result['loss_protocol'] = self.objective.loss_protocol
        result['objective_terms'] = objective_terms
        result['covariance_actual_weight'] = covariance_weight
        result['subject_logdet_loss'] = cov
        if (stage_diagnostics or return_debug) and identity_fusion:
            from utils.stage_health import stage_vectors
            result['stage_vectors'] = stage_vectors(full_encoded)
            if stage_diagnostics:result['layer_slots']=layer_slots
        for (i, branch) in enumerate(('roi', 'network')):
            result[branch + '_variance_loss'] = regularizer_branches[i, 0]
            spectral_name = '_logdet_loss'
            result[branch + spectral_name] = regularizer_branches[i, 1]
        result['roi_prediction_loss'] = roi_pred_loss
        result['roi_routing_diagnostics'] = routing
        result['identity_moment_evidence'] = moment_evidence
        result['regularizer_view_evidence'] = regularizer_view_evidence
        result['identity_moment_scope'] = 'full_view_global_unique_subjects'
        result['teacher_target_diagnostics'] = target_stats
        total_valid = int(valid.sum()) if diagnostics else 1
        result['mask_diagnostics'] = {name: {'token_count': int(mask.sum()), 'fraction': float(mask.sum()) / total_valid} for (name, mask) in [('context', context_mask), ('target', target_mask), ('exclusion', exclusion_mask)]} if diagnostics else {}
        result['native_support_diagnostics'] = {}
        if diagnostics:
            native_counts = block_sample_valid.sum(-1)
            for source in sorted(set(source_names or ['unknown'] * b)):
                scans = torch.tensor([s == source for s in source_names or ['unknown'] * b], device=block_valid.device)
                support_counts = native_counts[block_valid & scans[:, None]]
                result['native_support_diagnostics'][source] = {'blocks': len(support_counts), 'native_samples_min': int(support_counts.min()) if len(support_counts) else 0, 'blocks_below_normalization_minimum': int((support_counts < self.cfg.min_normalization_samples).sum()), 'normalization': self.cfg.signal_normalization, 'block_zscore_minimum_is_not_enforced': self.cfg.signal_normalization == 'roi_block_zscore'}
        if return_debug:
            parts = {'huber': F.smooth_l1_loss(prediction, target, reduction='none', beta=self.cfg.huber_beta).mean(-1), 'cosine': 1 - F.cosine_similarity(prediction, target, dim=-1)}
            for (name, per_target) in parts.items():
                (per_scan, _) = grouped_prediction(per_target, m, b, target_type, self.objective.target_group_weights)
                if self.objective.source_risk == 'source_macro' and self.training:
                    per_scan = per_scan * source_subject_weights(source_names, self.objective.source_subject_counts, per_scan)
                result['prediction_' + name + '_loss'] = unique_subject_rows(per_scan[:, None], subject_uids, objective_distributed, reduction=reduction).mean()
                roi_part = F.smooth_l1_loss(rp.float(), rt[rs, rpos], reduction='none', beta=self.cfg.huber_beta).mean(-1) if name == 'huber' else 1 - F.cosine_similarity(rp.float(), rt[rs, rpos], dim=-1)
                ps = roi_part.new_zeros(valid.numel()).index_add(0, units, roi_part)
                pe = (ps / counts.clamp_min(1)).reshape(valid.shape)[tuple(m.T)]
                (per_scan_roi, _) = grouped_prediction(pe, m, b, target_type, self.objective.target_group_weights)
                per_scan_roi = per_scan_roi * risk_weights
                result['prediction_' + name + '_loss'] = result['prediction_' + name + '_loss'] + self.objective.roi_prediction_weight * unique_subject_rows(per_scan_roi[:, None], subject_uids, objective_distributed, reduction=reduction).mean()
        result['target_group_diagnostics'] = group_diagnostics
        result['prediction_group_report'] = {}
        if diagnostics:
            from model.brain.prediction_statistics import prediction_group_report
            sources=source_names or ['unknown']*b
            result['prediction_group_report']={
                'network':prediction_group_report(error,m,subject_uids,sources,target_type),
                'roi':prediction_group_report(unit_errors,m,subject_uids,sources,target_type)}
        result['normalization_diagnostics'] = {name: normalization_summary(stats, roi_valid_mask, source_names) for (name, stats) in [('context', context_normalization), ('full', full_normalization)]}
        result['huber_loss'] = F.smooth_l1_loss(prediction, target, reduction='none', beta=self.cfg.huber_beta).mean().detach()
        result['cosine_loss'] = (1 - F.cosine_similarity(prediction, target, dim=-1)).mean().detach()
        result['moment_health'] = dict(subject_count=len(subject_z),
            raw_min_feature_std=min(item['raw_min_feature_std'] for item in moment_evidence),
            centered_energy=sum(item['centered_energy'] for item in moment_evidence),
            exact_constant=any(item['exact_constant'] for item in moment_evidence),
            spectral_energy_floor=max(item['spectral_energy_floor'] for item in moment_evidence),
            epsilon_std_floor=math.sqrt(self.objective.variance_epsilon),
            fraction_below_variance_target=sum(item['fraction_below_variance_target'] for item in moment_evidence)/2,
            scope='observed_subject_content; stop if either supported branch is constant')
        if return_debug:
            result.update(teacher_target=target, prediction=prediction, target_indices=m)
        return result
