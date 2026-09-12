#!/usr/bin/env python3
"""Language conditioning for ACT, as an extra encoder token.

lerobot 0.4.4's ACT has no language path at all -- the only matches for "language"
in ``modeling_act.py`` and ``configuration_act.py`` are the Apache licence header.
Its inputs are the images and the robot state, and the task strings the dataset
records are never read.  That is why the unconditioned policy only ever goes for the
plate: it cannot be told which prop to move, so it regresses the opening move of the
demonstrations.

What ACT *does* have is a slot for exactly this shape of thing.  Alongside the
latent and the robot-state token it will take one more 1-D observation vector --
``observation.environment_state`` -- give it its own ``nn.Linear`` into ``dim_model``
and its own learned positional embedding, and stack it with the image feature tokens
before the transformer encoder.  A sentence embedding is precisely that: one vector
per observation.  So the language token goes in through that slot and **lerobot is
not modified**.

That choice has three consequences worth knowing:

* The trained result is an ordinary ACT checkpoint.  ``ACTPolicy.from_pretrained``
  loads it with no custom code, which a forked ``ACT.forward`` would have broken --
  and with it the OpenVINO pipeline.
* The token conditions the encoder that feeds the decoder, not the VAE posterior.
  ACT's VAE encoder sees the cls token, the robot state and the action sequence
  only.  The posterior therefore stays language-blind, which is defensible -- it
  already sees the actions the language would be describing -- but it is a real
  difference from conditioning both.
* The IR gains a third input.  ``openvino/act_io.py`` builds its wrapper from the
  state and the cameras, so it needs the extra input wired through before a
  conditioned checkpoint can be converted.  The unconditioned checkpoint is
  unaffected.

The text encoder is small and frozen: MiniLM-L6, 22.7 M parameters, mean-pooled and
L2-normalized to 384 dims.  It is never trained and never part of the checkpoint,
so the only learned language parameters are the 384->512 projection ACT builds.
Because this dataset has just seven task strings, the embeddings are computed once
and looked up thereafter -- training speed is untouched.

With seven strings a one-hot would fit just as well.  The point of a sentence
encoder is what happens off the training set: "pick up the fork" lands near "Place
the fork in the table setting" rather than nowhere.  Measured on the seven, the
structure is right -- each prop's direct and handover phrasings are each other's
nearest neighbours (fork/hand-fork 0.872, spoon/hand-spoon 0.904, mug/hand-mug
0.869) -- so prop identity dominates the embedding, which is what has to reach the
policy.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_DIM = 384
TABLE_FILENAME = "task_embeddings.json"


def _env_state_key() -> str:
    from lerobot.utils.constants import OBS_ENV_STATE

    return OBS_ENV_STATE


class TaskEncoder:
    """Frozen sentence embeddings for task strings, cached per string.

    Loads lazily, so a process that only needs the saved table never pulls the
    model down.
    """

    def __init__(self, model_name: str = TEXT_MODEL, device: str = "cpu",
                 table: Optional[Dict[str, Sequence[float]]] = None):
        self.model_name = model_name
        self.device = device
        self._tok = self._model = None
        self._cache: Dict[str, torch.Tensor] = {}
        if table:
            for task, vec in table.items():
                self._cache[task] = torch.tensor(vec, dtype=torch.float32)
        self.dim = len(next(iter(self._cache.values()))) if self._cache else TEXT_DIM

    # -- model ------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self.dim = int(self._model.config.hidden_size)

    def _embed(self, tasks: Sequence[str]) -> torch.Tensor:
        self._load()
        with torch.no_grad():
            enc = self._tok(list(tasks), padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            hidden = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1).float().cpu()

    # -- use --------------------------------------------------------------
    def warm(self, tasks: Sequence[str]) -> "TaskEncoder":
        """Embed every string now, so training never waits on the text model."""
        missing = sorted({t for t in tasks if t not in self._cache})
        if missing:
            for task, vec in zip(missing, self._embed(missing)):
                self._cache[task] = vec
        return self

    def encode(self, tasks: Sequence[str]) -> torch.Tensor:
        """``(len(tasks), dim)`` embeddings, embedding anything not already cached."""
        self.warm(tasks)
        return torch.stack([self._cache[t] for t in tasks])

    # -- persistence ------------------------------------------------------
    def save(self, directory: pathlib.Path) -> pathlib.Path:
        """Write the table beside a checkpoint, so inference needs no text model."""
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / TABLE_FILENAME
        path.write_text(json.dumps(
            {"model": self.model_name, "dim": self.dim,
             "embeddings": {t: v.tolist() for t, v in sorted(self._cache.items())}},
            indent=1))
        return path

    @classmethod
    def load(cls, directory: pathlib.Path, device: str = "cpu") -> "TaskEncoder":
        path = pathlib.Path(directory) / TABLE_FILENAME
        blob = json.loads(path.read_text())
        return cls(model_name=blob["model"], device=device, table=blob["embeddings"])


def attach_language(cfg, dim: int = TEXT_DIM) -> str:
    """Declare the language token on an ACTConfig.  Returns the batch key it uses.

    Must be called *after* the other features are set, because ACT reads
    ``input_features`` when the model is built and ``make_policy`` overwrites that
    dict from the dataset.  :func:`build_language_policy` does both in order.

    The token is left unnormalized: the embeddings are already L2-normalized, and
    standardizing them against dataset statistics would only distort directions
    that carry the meaning.
    """
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature

    key = _env_state_key()
    cfg.normalization_mapping[FeatureType.ENV] = NormalizationMode.IDENTITY
    cfg.input_features[key] = PolicyFeature(type=FeatureType.ENV, shape=(dim,))
    return key


def build_language_policy(cfg, ds_meta, dim: int = TEXT_DIM):
    """ACT with the language token, other features taken from the dataset."""
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.act.modeling_act import ACTPolicy

    features = dataset_to_policy_features(ds_meta.features)
    cfg.output_features = {k: v for k, v in features.items()
                           if v.type is FeatureType.ACTION}
    cfg.input_features = {k: v for k, v in features.items()
                          if k not in cfg.output_features}
    attach_language(cfg, dim)
    cfg.validate_features()
    return ACTPolicy(cfg)


def inject(batch: dict, encoder: TaskEncoder) -> dict:
    """Add the task embedding to a batch, under the key ACT reads it from.

    Call this on the raw batch, before the preprocessor: the preprocessor is what
    moves tensors to the device, and it will carry this one along with the rest.
    """
    key = _env_state_key()
    if key in batch:
        return batch
    tasks = batch.get("task")
    if tasks is None:
        raise KeyError("batch has no 'task'; language conditioning needs the task string")
    if isinstance(tasks, str):
        tasks = [tasks]
    batch[key] = encoder.encode(list(tasks))
    return batch
