"""
Configuration system.

Every experiment is described by a single `ExperimentConfig` dataclass tree.
Configs are stored as YAML, loaded with `load_config`, and can be overridden
from the command line with dotted keys, e.g.:

    python scripts/train.py --config configs/cifar10_sae_align_frozen_residual.yaml \
        --set optim.epochs=50 --set align.weight=2.0

This is the single source of truth for every knob mentioned in the design
discussion (method, drift-control mode, alignment kind, stop-gradient,
layer, pooling, etc). Nothing is hardcoded in the training loop -- if you
want to test an ambiguity, add/flip a field here.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Optional, Tuple

import yaml

# --------------------------------------------------------------------------
# Sub-configs
# --------------------------------------------------------------------------


@dataclass
class DataConfig:
    dataset: str = "cifar10"          # cifar10 | cifar100 | synthetic
    data_root: str = "./data"
    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    val_fraction: float = 0.02        # sliced off the training set for periodic monitoring
    synthetic_size: int = 2048        # only used when dataset == "synthetic" (fast smoke tests)


@dataclass
class ModelConfig:
    arch: str = "preact_resnet18"     # preact_resnet18 | wideresnet28_10 | wideresnet34_10
    num_classes: int = 10
    # module path (dotted) inside the backbone whose *output* activation is hooked
    # for both the raw-feature-alignment baseline and for the SAE. Sensible
    # defaults are provided per architecture in models/build.py if left as "auto".
    feature_layer: str = "auto"


@dataclass
class AttackConfig:
    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 10
    norm: str = "linf"                # linf | l2
    random_start: bool = True


@dataclass
class SAEConfig:
    reduce: str = "gap"               # gap (global-average-pool to a vector) | spatial (per-position tokens)
    kind: str = "topk"                # topk | l1
    dict_size: int = 4096
    k: int = 32                       # active latents per sample, only used when kind == "topk"
    l1_coef: float = 1e-3             # sparsity coefficient, only used when kind == "l1"
    tied_weights: bool = False
    normalize_decoder: bool = True    # unit-norm decoder columns (standard SAE trick, stabilizes training)
    pretrained_path: Optional[str] = None  # required when align.enabled and align.space == "sae"


@dataclass
class DriftConfig:
    # How the SAE's validity is maintained while the backbone keeps changing
    # under adversarial training. This is the central open question of the
    # project -- run all four and compare.
    mode: str = "frozen_residual"     # frozen | frozen_residual | joint | periodic_refresh
    residual_kind: str = "absolute"   # absolute | delta (delta needs an initial-backbone snapshot, see trainer.py)
    residual_weight: float = 1.0      # weight of L_resid, used by frozen_residual and periodic_refresh
    sae_lr: float = 1e-3              # SAE optimizer LR, used by joint and periodic_refresh (refresh steps)
    sae_recon_weight: float = 1.0     # weight of the SAE's own reconstruction loss, used by joint
    refresh_every: int = 10           # epochs between SAE refreshes, only used by periodic_refresh
    refresh_steps: int = 300          # optimizer steps performed at each refresh
    fvu_alarm_threshold: float = 0.5  # log a warning if running FVU exceeds this (SAE going stale)


@dataclass
class AlignConfig:
    enabled: bool = False             # master switch for the representation-alignment term
    space: str = "sae"                # sae | raw  (raw = ablation baseline, ignores the SAE entirely)
    kind: str = "l2"                  # l2 | l1 | cosine | feature_add
    weight: float = 1.0               # lambda in the total loss
    stop_grad_clean: bool = True      # only pull the adversarial branch toward the (fixed) clean branch
    # EXPERIMENTAL / off by default: let the inner PGD maximization also try to
    # increase the alignment loss. Left in for the "obfuscated gradient" ablation
    # discussed in README.md; default False keeps adversarial-example generation
    # identical to standard PGD-AT, which is what you want for a clean comparison.
    inner_max_uses_alignment: bool = False
    inner_max_align_weight: float = 0.0


@dataclass
class TradesConfig:
    beta: float = 6.0                 # weight of the KL term, only used when base == "trades"


@dataclass
class OptimConfig:
    epochs: int = 100
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "multistep"    # multistep | cosine
    milestones: Tuple[int, ...] = (75, 90)
    lr_gamma: float = 0.1
    warmup_epochs: int = 0


@dataclass
class ExperimentConfig:
    name: str = "baseline_madry"
    # `base` selects how adversarial examples are generated and what the
    # base (non-alignment) loss is:
    #   clean   -- no attack at all, plain CE on unperturbed images. Used to
    #              produce the initial classifier checkpoint that the SAE is
    #              pretrained on (see scripts/pretrain_sae.py) before
    #              adversarial training starts.
    #   madry   -- plain PGD adversarial training (Madry et al. 2018): CE(adv, y)
    #   trades  -- TRADES (Zhang et al. 2019): CE(clean, y) + beta * KL(adv, clean),
    #              with x' generated by maximizing KL instead of CE
    # Independently, `align.enabled` turns on a representation-alignment term
    # on top of any base recipe, and `align.space` chooses whether that term
    # operates on raw backbone activations (ablation baseline) or on a
    # sparse autoencoder's latent code (the proposed method). This makes
    # every combination -- e.g. "trades + sae alignment" -- a config flip
    # rather than a new code path.
    base: str = "madry"
    seed: int = 0
    device: str = "auto"              # auto | cpu | cuda
    out_dir: str = "./runs"
    eval_every: int = 5                # epochs between cheap in-training PGD-10 evals
    save_every: int = 10
    log_every: int = 50                # steps between console/CSV logs
    resume_from: Optional[str] = None  # path to a model checkpoint to initialize the backbone from

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    align: AlignConfig = field(default_factory=AlignConfig)
    trades: TradesConfig = field(default_factory=TradesConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


# --------------------------------------------------------------------------
# (de)serialization helpers
# --------------------------------------------------------------------------


def _dataclass_from_dict(cls, data: dict):
    if not is_dataclass(cls):
        return data
    kwargs = {}
    field_types = {f.name: f.type for f in fields(cls)}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = field_types[f.name]
        # resolve dataclass sub-fields recursively
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if is_dataclass(default) and isinstance(val, dict):
            kwargs[f.name] = _dataclass_from_dict(type(default), val)
        elif isinstance(val, list) and "Tuple" in str(ftype):
            kwargs[f.name] = tuple(val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return dataclasses.asdict(cfg)


def load_config(path: Optional[str] = None, overrides: Optional[list] = None) -> ExperimentConfig:
    """Load a YAML config (or the pure-default config if path is None) and
    apply a list of "dotted.key=value" override strings on top."""
    if path is not None:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = _dataclass_from_dict(ExperimentConfig, raw)
    else:
        cfg = ExperimentConfig()

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Bad override '{ov}', expected dotted.key=value")
        key, val = ov.split("=", 1)
        _set_by_dotted_key(cfg, key.strip(), val.strip())
    return cfg


def _parse_value(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw  # plain string (e.g. a path)


def _set_by_dotted_key(cfg: Any, dotted_key: str, raw_value: str) -> None:
    parts = dotted_key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise AttributeError(f"Unknown config key '{dotted_key}'")
    current = getattr(obj, leaf)
    value = _parse_value(raw_value)
    if isinstance(current, tuple) and isinstance(value, list):
        value = tuple(value)
    setattr(obj, leaf, value)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)


def clone_config(cfg: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(cfg)
