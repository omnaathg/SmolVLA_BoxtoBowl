from dataclasses import dataclass

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("contextvla")
@dataclass
class ContextVLAConfig(SmolVLAConfig):
    """SmolVLA extended with wide temporal sampling for non-Markovian tasks.

    Feeds n_obs_steps frames spaced temporal_stride frames apart into the VLM,
    giving the policy memory over a window of (n_obs_steps-1)*temporal_stride/fps seconds.

    Default: 11 frames × stride 45 @ 30fps = 15-second memory window.
    delta_timestamps: [-15.0, -13.5, ..., -1.5, 0.0] seconds.
    """

    n_obs_steps: int = 11        # 11 frames × stride 45 @ 30fps = 15-second memory window
    temporal_stride: int = 45   # frames between sampled steps (45 = 1.5 sec at 30 fps)
    compression_layer: int = 8  # Phase 2 only — layer at which past tokens are pooled
    use_compression: bool = False  # False = Phase 1 (full sequence), True = Phase 2

    @property
    def observation_delta_indices(self) -> list[int]:
        """Frame offsets for temporal sampling, e.g. [-210, -180, ..., -30, 0]."""
        start = -(self.n_obs_steps - 1) * self.temporal_stride
        return list(range(start, 1, self.temporal_stride))
