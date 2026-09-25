"""Registered hierarchy configuration with exactly four atomic losses."""
from dataclasses import dataclass, field
from pathlib import Path
import math
import re
import yaml
from model.config import STHBrainConfig
from utils.paths import resolve_paths

ALGORITHM_VERSION = 'hierarchical_ts_jepa'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/pretrain.yaml'


def resolve_pretrain_runtime(cfg, *, seed=None, output_dir=None):
    """Resolve one seed and its output identically in the launcher and trainer."""
    runtime = cfg['runtime']
    if seed is not None:
        # Rank-local RNG initialization adds up to expected_world_size - 1.
        maximum = 2**32 - int(runtime.get('expected_world_size', 1))
        if type(seed) is not int or not 0 <= seed <= maximum:
            raise ValueError(f'Pretraining seed must be an integer in 0..{maximum}')
        if seed != runtime['seed'] and output_dir is None:
            previous = Path(runtime['output_dir'])
            if re.fullmatch(r'seed\d+_ep\d+', previous.name):
                output_dir = previous.with_name(f"seed{seed}_ep{cfg['optim']['epochs']}")
            else:
                output_dir = previous.with_name(previous.name + f'_seed{seed}')
        runtime['seed'] = seed
    if output_dir is not None:
        runtime['output_dir'] = str(Path(output_dir).expanduser().resolve())
    return cfg

@dataclass(frozen=True)
class PretrainObjective:
    loss_protocol: str = 'roi_network_four_term'
    logdet_alpha: float = 1.0
    covariance_start_epoch: float = 0.0
    covariance_ramp_end_epoch: float = 5.0
    logdet_branch_weights: tuple = (0.5, 0.5)
    context_moment_weight: float = 0.0
    target_group_weights: tuple = ()
    source_risk: str = "subject_mean"
    source_subject_counts: dict = field(default_factory=dict)
    roi_prediction_weight: float = 1.0
    prediction_weight: float = 1.0
    variance_weight: float = 0.1
    covariance_weight: float = 0.1
    covariance_objective: str = 'full_branch_logdet_isotropy'
    variance_target: float = 0.5
    variance_epsilon: float = 1e-4
    normalization: str = 'visible_scan_roi_zscore'
    teacher_target: str = 'signal_residual'
    # No projector: moments act directly on the student's tokens.
    representation: str = 'layernorm_full_joint_student'

def load_config(path=DEFAULT_CONFIG):
    cfg = resolve_paths(yaml.safe_load(Path(path).read_text()))
    expected = {'schema_version','name','model','objective','mask','data','optim','runtime'}
    if set(cfg) != expected or cfg['schema_version'] != 'sth_brain_hierarchical_ts_jepa':
        raise ValueError('Invalid STH-Brain TS-JEPA config sections')
    cfg['runtime'].setdefault('health_every', 10)
    cfg['runtime'].setdefault('stage_health_every_epochs', 1)
    cfg['runtime'].setdefault('stage_health_max_subjects', 128)
    cfg['runtime'].setdefault('compression_rank_ratio', .5)
    cfg['runtime'].setdefault('compression_minimum_reference_rank', .05)
    cfg['runtime'].setdefault('compression_patience', 3)
    for key in ('compression_rank_ratio','compression_minimum_reference_rank'):
        value=cfg['runtime'][key]
        if isinstance(value,bool) or not math.isfinite(value) or not 0<value<1:
            raise ValueError(key+' must lie in (0,1)')
    if type(cfg['runtime']['compression_patience']) is not int or cfg['runtime']['compression_patience']<1:
        raise ValueError('compression_patience must be a positive integer')
    if type(cfg['runtime']['stage_health_max_subjects']) is not int or cfg['runtime']['stage_health_max_subjects']<2:
        raise ValueError('stage_health_max_subjects must be an integer >= 2')
    cfg['optim'].setdefault('gradient_diagnostic_batches', 3)
    for section,key in (('runtime','stage_health_every_epochs'),('optim','gradient_diagnostic_batches')):
        if type(cfg[section][key]) is not int or cfg[section][key] < 1:
            raise ValueError(key+' must be a positive integer')
    for key in ('health_every', 'log_every'):
        if type(cfg['runtime'][key]) is not int or cfg['runtime'][key] < 1:
            raise ValueError(key+' must be a positive integer')
    if cfg['model']['algorithm_version'] != ALGORITHM_VERSION:
        raise ValueError('This trainer accepts only STH-Brain TS-JEPA; old checkpoints are not resumes')
    if any('fc_' in str(k).lower() or 'structured_fc' in str(k).lower() for section in cfg.values() if isinstance(section,dict) for k in section):
        raise ValueError('FC keys are forbidden in STH-Brain TS-JEPA')
    legacy = {'predictor_' + key for key in ('dim', 'layers', 'heads', 'ffn_dim')}
    if legacy & cfg['model'].keys():
        raise ValueError('Use explicit network_predictor_* and roi_predictor_* parameters')
    required_predictors = {branch + '_predictor_' + key
                           for branch in ('roi', 'network')
                           for key in ('dim', 'layers', 'heads', 'ffn_dim')}
    missing_predictors = required_predictors - cfg['model'].keys()
    if missing_predictors:
        raise ValueError('Missing predictor parameters: ' + ', '.join(sorted(missing_predictors)))
    model = STHBrainConfig(**cfg['model']); obj = PretrainObjective(**cfg['objective'])
    model.validate_predictors()
    model.validate_roi_temporal()
    model.validate_network_summary()
    if model.representation_architecture != 'roi_temporal_hierarchy':
        raise ValueError('Unknown ROI-network architecture')
    if float(cfg['model']['dropout']) != 0.0:
        raise ValueError('Hierarchical pretraining requires dropout=0 in tokenizer, encoder and predictor')
    if obj.normalization not in {'roi_block_zscore','train_roi_frozen','visible_scan_roi_zscore'} or obj.normalization!=model.signal_normalization or obj.representation != 'layernorm_full_joint_student' or obj.teacher_target not in {'signal_residual','clean'}:
        raise ValueError('Unregistered normalization/representation/teacher target')
    if model.signal_normalization in {'train_roi_frozen','visible_scan_roi_zscore'} and (not model.normalization_stats_path or not model.normalization_stats_sha256):
        raise ValueError('train_roi_frozen requires a pre-registered train-only normalization artifact')
    if model.signal_normalization=='roi_block_zscore' and (model.normalization_stats_path is not None or model.normalization_stats_sha256 is not None):
        raise ValueError('roi_block_zscore must not declare frozen normalization artifacts')
    from model.brain.subject_objectives import validate_four_term_objective
    validate_four_term_objective(model, obj)
    for key in ('prediction_weight','variance_weight','variance_target','variance_epsilon'):
        if not math.isfinite(getattr(obj,key)) or getattr(obj,key)<=0: raise ValueError(key)
    if type(model.min_normalization_samples) is not int or model.min_normalization_samples<3:
        raise ValueError('min_normalization_samples must be an integer >= 3')
    if cfg['model']['teacher_target_mode'] != obj.teacher_target or not cfg['model']['target_residual_normalize'] or cfg['model']['target_query_use_mask_type']:
        raise ValueError('Teacher/query declarations disagree with the registered objective')
    if any(k in cfg['model'] for k in ('variance_weight','covariance_weight','variance_target_std')):
        raise ValueError('Use objective section for student moment weights')
    if cfg['model']['block_normalization_epsilon']<=0 or cfg['model']['max_interpolation_gap_seconds']<=0:raise ValueError('Invalid normalization or interpolation support')
    if model.local_time_mode not in {'relative_block_cycles','physical_seconds'}:
        raise ValueError('Invalid local time encoding mode')
    local_frequencies=(model.local_time_frequency_min_hz,model.local_time_frequency_max_hz)
    if (not all(math.isfinite(value) and value>0 for value in local_frequencies)
            or model.local_time_frequency_max_hz<model.local_time_frequency_min_hz):
        raise ValueError('Invalid local time frequency range')
    if (model.local_time_mode=='physical_seconds'
            and model.local_time_frequency_max_hz>0.5/model.physical_grid_step_seconds):
        raise ValueError('Local physical time frequencies exceed the query-grid Nyquist frequency')
    if cfg['data']['normalize'] != 'none' or cfg['data']['split_col'] != 'split' or cfg['data'].get('interface') != 'registered_subject_tasks':
        raise ValueError('Use raw sequences and the frozen component-exclusive split')
    if cfg['data'].get('timestamp_receipt_path') or cfg['data'].get('timestamp_receipt_sha256'):
        raise ValueError('Registered TXT split uses audited regular TR; external timestamp overrides require a new audited interface')
    if cfg['data']['train_split']!='train' or cfg['data']['val_split']!='val':raise ValueError('Invalid roles')
    if cfg['optim']['epochs']!=500:raise ValueError('Formal schedule is fixed at 500; use --stop-after for short runs')
    if cfg['optim']['batch_size']<1 or cfg['optim']['accumulation_steps']<1:raise ValueError('Invalid batching')
    if cfg['optim']['clip_mode'] not in {'global','student_predictor'}:raise ValueError('Invalid clip_mode')
    for key in ('grad_clip','student_grad_clip','predictor_grad_clip'):
        if not math.isfinite(cfg['optim'][key]) or cfg['optim'][key]<=0:raise ValueError(key)
    for key in ('gradient_diagnostic_every','update_diagnostic_every'):
        if type(cfg['optim'][key]) is not int or cfg['optim'][key]<0:raise ValueError(key)
    if type(cfg['optim']['exact_collapse_patience']) is not int or cfg['optim']['exact_collapse_patience']<1:
        raise ValueError('exact_collapse_patience')
    if cfg['model']['block_seconds'] not in (8.,16.,32.):raise ValueError('Unvalidated physical block duration')
    if obj.target_group_weights and (len(obj.target_group_weights) != 3 or any(not math.isfinite(w) or w <= 0 for w in obj.target_group_weights)):
        raise ValueError('Three strictly positive target group weights required')
    if obj.source_risk not in {'subject_mean','source_macro'}:
        raise ValueError('Unknown source risk')
    if obj.source_risk == 'source_macro':
        import pandas as pd
        frame = pd.read_csv(cfg['data']['manifest'], low_memory=False)
        counts = frame.loc[frame[cfg['data']['split_col']].eq('train')].drop_duplicates('subject_uid').dataset.value_counts().to_dict()
        if obj.source_subject_counts != counts:
            raise ValueError('Macro source counts must exactly match train manifest subjects')
    if cfg['mask'].get('strategy','union') not in {'union','mixture'}:
        raise ValueError('Unknown structured mask strategy')
    candidates = cfg['runtime'].get('checkpoint_epochs', [50,150,250,350,450,500])
    if not candidates or len(set(candidates)) != len(candidates) or any(type(e) is not int or not 1 <= e <= 500 for e in candidates):
        raise ValueError('Invalid preregistered checkpoint candidates')
    return cfg
