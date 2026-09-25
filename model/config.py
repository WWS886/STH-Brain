"""Immutable physical-time backbone configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class STHBrainConfig:
    atlas_path: str
    algorithm_version: str = "hierarchical_ts_jepa"
    block_seconds: float = 8.0
    min_block_coverage: float = 0.75
    local_dim: int = 64
    local_layers: int = 1
    local_heads: int = 4
    local_ffn_dim: int = 256
    model_dim: int = 256
    encoder_layers: int = 8
    encoder_heads: int = 8
    encoder_ffn_dim: int = 1024
    network_predictor_dim: int = 128
    network_predictor_layers: int = 4
    network_predictor_heads: int = 4
    network_predictor_ffn_dim: int = 512
    roi_predictor_dim: int = 128
    roi_predictor_layers: int = 4
    roi_predictor_heads: int = 4
    roi_predictor_ffn_dim: int = 512
    dropout: float = 0.0
    sope_head_fraction: float = 0.5
    time_periods: tuple[float, ...] = (8, 16, 32, 64, 120, 240, 480, 960)
    huber_beta: float = 1.0
    huber_weight: float = 1.0
    cosine_weight: float = 0.0
    ema_base: float = 0.994
    ema_final: float = 0.9999
    predictor_mode: str = "context_anchored_cross_attention"
    first_layer_query_residual: bool = False
    cross_attention_output_bias: bool = False
    target_query_use_mask_type: bool = False
    teacher_target_mode: str = "signal_residual"
    target_residual_normalize: bool = True
    network_embedding_init_std: float | None = None
    network_embedding_scale: float = 1.0
    geometry_additive_scale: float = 0.05
    network_summary_mode: str = "mean_std_attention"
    roi_sope_heads: int = 4
    roi_sope_head_fraction: float = 1.0
    maximum_time_frequency_octaves: float = 1.0
    physical_grid_step_seconds: float = 1.0
    physical_kernel_seconds: tuple[float, ...] = (1.0, 3.0, 5.0)
    block_normalization_epsilon: float = 1.0e-6
    max_interpolation_gap_seconds: float = 6.5
    signal_normalization: str = "visible_scan_roi_zscore"
    normalization_stats_path: str | None = None
    normalization_stats_sha256: str | None = None
    min_normalization_samples: int = 8
    representation_architecture: str = "roi_temporal_hierarchy"
    roi_temporal_layers: int = 8
    roi_temporal_heads: int = 16
    roi_temporal_dim: int = 256
    roi_temporal_ffn_dim: int = 1024
    roi_temporal_checkpointing: bool = True
    local_time_mode: str = "physical_seconds"
    local_time_frequency_min_hz: float = 0.03125
    local_time_frequency_max_hz: float = 0.5




    def validate_predictors(self):
        """Each prediction branch owns its attention depth, width and heads."""
        for branch in ('roi', 'network'):
            prefix = branch + '_predictor_'
            for suffix in ('dim', 'layers', 'heads', 'ffn_dim'):
                name = prefix + suffix
                if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                    raise ValueError(name + ' must be a positive integer')
            if getattr(self, prefix + 'dim') % getattr(self, prefix + 'heads'):
                raise ValueError(prefix + 'dim must be divisible by ' + prefix + 'heads')

    def validate_network_summary(self):
        if self.network_summary_mode not in {'mean_std_attention','mean_std'}:
            raise ValueError('Unknown network_summary_mode')

    def validate_roi_temporal(self):
        """The temporal attention width is independent of the ROI slot width."""
        for name in ('local_dim', 'roi_temporal_dim', 'roi_temporal_layers',
                     'roi_temporal_heads', 'roi_temporal_ffn_dim'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if (self.roi_temporal_dim % self.roi_temporal_heads or
                self.roi_temporal_dim // self.roi_temporal_heads < 16):
            raise ValueError('ROI temporal heads require at least eight rotary channel pairs')
        if type(self.roi_temporal_checkpointing) is not bool:
            raise ValueError('roi_temporal_checkpointing must be a boolean')
