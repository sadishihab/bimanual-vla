#!/usr/bin/env python3
"""Shared pieces for taking an ACT checkpoint to OpenVINO: model, stats, calibration.

The exported graph is *self-contained*: normalization of the inputs and
unnormalization of the action are compiled into it, so the IR takes raw
observations -- images in [0, 1] and joint positions in radians -- and returns an
action in real units.  Two reasons for that.  A deployment target like the NPU
should not need a Python preprocessing pipeline alongside it, and this dataset's
action mixes units (ten arm channels in radians, two gripper channels in
newton-metres), so the per-channel statistics are exactly the thing you do not
want to reimplement by hand at the edge.  Baked in, they cannot drift.

Shapes are static and the batch is fixed, because the NPU plugin requires static
shapes and because latency is what this pipeline is measured on.

Nothing here needs a trained checkpoint: ``build_policy`` will construct the
architecture with randomly initialized weights, which is enough to convert,
quantize and benchmark against.  The numbers from random weights are meaningless
as behaviour and exactly as meaningful as trained ones for latency.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

STATE_KEY = OBS_STATE
ACTION_KEY = "action"
DEFAULT_CAMERAS = ("observation.images.overhead", "observation.images.front")
DEFAULT_RESOLUTION = 256
DEFAULT_STATE_DIM = 12
DEFAULT_CHUNK = 100
# Channels 5 and 11 of the action are gripper torques in N.m; the other ten are
# arm joint position targets in radians.  Recorded here so the conversion report
# can show that the two are normalized on their own scales.
GRIPPER_CHANNELS = (5, 11)


# ---------------------------------------------------------------------------
# Policy and statistics
# ---------------------------------------------------------------------------


def dataset_meta(root: Optional[pathlib.Path], repo_id: str):
    """Dataset metadata, or ``None``.  Only needed for real statistics."""
    if root is None:
        return None
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id, root=root)


def synthetic_features(cameras: Sequence[str], resolution: int, state_dim: int,
                       chunk: int) -> Tuple[dict, dict]:
    """Feature dicts matching the recorded dataset, for use without one on disk."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    inputs = {STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))}
    for cam in cameras:
        inputs[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, resolution, resolution))
    outputs = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(state_dim,))}
    return inputs, outputs


def identity_stats(inputs: dict, outputs: dict) -> Dict[str, Dict[str, torch.Tensor]]:
    """Zero-mean, unit-std statistics, so an untrained export is still well defined."""
    stats = {}
    for key, feat in {**inputs, **outputs}.items():
        shape = (3, 1, 1) if len(feat.shape) == 3 else feat.shape
        stats[key] = {"mean": torch.zeros(shape), "std": torch.ones(shape),
                      "min": -torch.ones(shape), "max": torch.ones(shape)}
    return stats


def build_policy(checkpoint: Optional[pathlib.Path], meta, *, chunk: int,
                 cameras: Sequence[str], resolution: int, state_dim: int
                 ) -> Tuple[ACTPolicy, ACTConfig, Dict[str, Dict[str, torch.Tensor]], str]:
    """The ACT policy, its config, the normalization statistics, and where they came from.

    Three ways in, in descending order of authority: a trained checkpoint, which
    carries the statistics it was trained with; a dataset on disk, whose statistics
    are what a checkpoint trained on it would have; or neither, in which case the
    architecture is built with random weights and identity statistics so the rest
    of the pipeline can still be exercised.
    """
    if checkpoint is not None:
        policy = ACTPolicy.from_pretrained(checkpoint)
        cfg = policy.config
        cfg.device = "cpu"
        stats = _stats_from_checkpoint(cfg, checkpoint)
        if stats is not None:
            source = f"checkpoint {checkpoint}"
        elif meta is not None:
            stats, source = meta.stats, f"dataset {meta.root}"
        else:
            stats = identity_stats(cfg.input_features, cfg.output_features)
            source = "identity (checkpoint carried no normalizer)"
        policy.eval()
        return policy, cfg, stats, source

    cfg = ACTConfig(chunk_size=chunk, n_action_steps=chunk, device="cpu")
    if meta is not None:
        policy = make_policy(cfg=cfg, ds_meta=meta)
        stats, source = meta.stats, f"dataset {meta.root}"
    else:
        cfg.input_features, cfg.output_features = synthetic_features(
            cameras, resolution, state_dim, chunk)
        cfg.validate_features()
        policy = ACTPolicy(cfg)
        stats = identity_stats(cfg.input_features, cfg.output_features)
        source = "identity (no checkpoint, no dataset: random weights)"
    policy.eval()
    return policy, cfg, stats, source


def _stats_from_checkpoint(cfg: ACTConfig, checkpoint: pathlib.Path):
    """Statistics the checkpoint's own preprocessor was saved with, or ``None``."""
    if not any(checkpoint.glob("*normalizer_processor*")):
        return None
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))
    for step in pre.steps:
        if isinstance(step, NormalizerProcessorStep):
            return step.stats
    return None


# ---------------------------------------------------------------------------
# The exported module
# ---------------------------------------------------------------------------


def _as_tensor(v) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.detach().float().cpu()
    return torch.as_tensor(np.asarray(v), dtype=torch.float32)


class ACTInference(nn.Module):
    """ACT with normalization and unnormalization folded in.

    Takes the state and one tensor per camera, in the order of ``cameras``, and
    returns the action chunk in real units.  The VAE encoder is not part of this
    graph: at inference ACT sets the latent to zeros, so only the decoder runs.
    """

    EPS = 1e-8      # matches NormalizerProcessorStep's guard

    def __init__(self, policy: ACTPolicy, stats: Dict[str, Dict[str, torch.Tensor]],
                 cameras: Sequence[str]):
        super().__init__()
        self.model = policy.model
        self.cameras = list(cameras)
        norm_map = policy.config.normalization_mapping
        from lerobot.configs.types import FeatureType

        self.state_mode = norm_map[FeatureType.STATE].value
        self.visual_mode = norm_map[FeatureType.VISUAL].value
        self.action_mode = norm_map[FeatureType.ACTION].value
        for mode in (self.state_mode, self.visual_mode, self.action_mode):
            if mode not in ("MEAN_STD", "MIN_MAX", "IDENTITY"):
                raise NotImplementedError(
                    f"{mode} normalization is not folded into the export; use "
                    "MEAN_STD, MIN_MAX or IDENTITY, or normalize outside the IR.")

        self._register(STATE_KEY, "state", stats)
        for i, cam in enumerate(self.cameras):
            self._register(cam, f"cam{i}", stats)
        self._register(ACTION_KEY, "action", stats)

    def _register(self, key: str, prefix: str, stats) -> None:
        s = stats.get(key)
        if s is None:
            raise KeyError(f"no statistics for {key}; cannot fold its normalization in")
        for name in ("mean", "std", "min", "max"):
            if name in s:
                self.register_buffer(f"{prefix}_{name}", _as_tensor(s[name]))

    def _norm(self, x: torch.Tensor, prefix: str, mode: str) -> torch.Tensor:
        if mode == "IDENTITY":
            return x
        if mode == "MEAN_STD":
            mean = getattr(self, f"{prefix}_mean")
            std = getattr(self, f"{prefix}_std")
            return (x - mean) / (std + self.EPS)
        lo, hi = getattr(self, f"{prefix}_min"), getattr(self, f"{prefix}_max")
        return (x - lo) / (hi - lo + self.EPS) * 2.0 - 1.0

    def _denorm(self, x: torch.Tensor, prefix: str, mode: str) -> torch.Tensor:
        if mode == "IDENTITY":
            return x
        if mode == "MEAN_STD":
            mean = getattr(self, f"{prefix}_mean")
            std = getattr(self, f"{prefix}_std")
            return x * (std + self.EPS) + mean
        lo, hi = getattr(self, f"{prefix}_min"), getattr(self, f"{prefix}_max")
        return (x + 1.0) / 2.0 * (hi - lo + self.EPS) + lo

    def forward(self, state: torch.Tensor, *images: torch.Tensor) -> torch.Tensor:
        batch = {
            STATE_KEY: self._norm(state, "state", self.state_mode),
            OBS_IMAGES: [self._norm(img, f"cam{i}", self.visual_mode)
                         for i, img in enumerate(images)],
        }
        actions, _ = self.model(batch)
        return self._denorm(actions, "action", self.action_mode)


def input_specs(cfg: ACTConfig, cameras: Sequence[str], batch: int
                ) -> List[Tuple[str, Tuple[int, ...]]]:
    """Ordered (name, static shape) for the exported graph's inputs."""
    state_dim = cfg.input_features[STATE_KEY].shape[0]
    specs = [(STATE_KEY, (batch, state_dim))]
    for cam in cameras:
        specs.append((cam, (batch, *cfg.input_features[cam].shape)))
    return specs


def example_inputs(cfg: ACTConfig, cameras: Sequence[str], batch: int
                   ) -> Tuple[torch.Tensor, ...]:
    """A plausible batch: images in [0, 1], state at a mid-range joint angle."""
    g = torch.Generator().manual_seed(0)
    out = []
    for name, shape in input_specs(cfg, cameras, batch):
        if name == STATE_KEY:
            out.append(torch.rand(shape, generator=g) * 2.0 - 1.0)
        else:
            out.append(torch.rand(shape, generator=g))
    return tuple(out)


def resolve_cameras(cfg: ACTConfig, fallback: Sequence[str]) -> List[str]:
    """Camera keys in the order the graph will take them."""
    from lerobot.configs.types import FeatureType

    found = [k for k, f in cfg.input_features.items() if f.type is FeatureType.VISUAL]
    return found if found else list(fallback)


# ---------------------------------------------------------------------------
# Shared CLI
# ---------------------------------------------------------------------------


def add_model_args(parser: argparse.ArgumentParser) -> None:
    repo = pathlib.Path(__file__).resolve().parent.parent
    parser.add_argument("--checkpoint", type=pathlib.Path, default=None,
                        help="a lerobot ACT checkpoint directory; omit to use random weights")
    parser.add_argument("--dataset", type=pathlib.Path,
                        default=repo / ".cache" / "lerobot" / "bimanual_table_setting",
                        help="dataset root, for normalization statistics and calibration")
    parser.add_argument("--repo-id", default="local/bimanual_table_setting")
    parser.add_argument("--batch", type=int, default=1,
                        help="static batch size compiled into the IR")
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                        help="ACT action horizon, when building from scratch")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--state-dim", type=int, default=DEFAULT_STATE_DIM)
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))


def load_for_export(args) -> Tuple[ACTInference, ACTConfig, List[str], str]:
    """The wrapped module ready to trace, plus where its statistics came from."""
    root = args.dataset if args.dataset and (args.dataset / "meta" / "info.json").exists() else None
    meta = dataset_meta(root, args.repo_id)
    policy, cfg, stats, source = build_policy(
        args.checkpoint, meta, chunk=args.chunk, cameras=args.cameras,
        resolution=args.resolution, state_dim=args.state_dim)
    cameras = resolve_cameras(cfg, args.cameras)
    module = ACTInference(policy, stats, cameras).eval()
    return module, cfg, cameras, source


def describe_normalization(module: ACTInference, cfg: ACTConfig) -> None:
    """Show that the action's two unit groups are folded in on separate scales."""
    print(f"  normalization folded in: state={module.state_mode}"
          f" visual={module.visual_mode} action={module.action_mode}")
    if module.action_mode != "MEAN_STD":
        return
    std = module.action_std.reshape(-1)
    mean = module.action_mean.reshape(-1)
    arm = [i for i in range(std.numel()) if i not in GRIPPER_CHANNELS]
    if std.numel() <= max(GRIPPER_CHANNELS):
        return
    print(f"  action scales: {len(arm)} arm channels (rad) std "
          f"{std[arm].min():.4f}-{std[arm].max():.4f}, "
          f"{len(GRIPPER_CHANNELS)} gripper channels (N.m) std "
          f"{min(std[i] for i in GRIPPER_CHANNELS):.4f}-"
          f"{max(std[i] for i in GRIPPER_CHANNELS):.4f}")
    print(f"                 per-channel means range {mean.min():.4f} to {mean.max():.4f}"
          "  (each channel on its own scale, units never pooled)")
