"""Config for Koopman experiments: nested dataclasses, YAML + CLI overrides.

Kept separate from ``ret.config.ExperimentConfig`` on purpose (that loader
warns on unknown keys and has nothing to do with the Koopman stage).

Usage::

    cfg = load_koopman_config("configs/koopman/gptoss20b.yaml",
                              ["model.lift_dim=1024", "train.seed=1"])

Override values are parsed with ``yaml.safe_load`` so lists work:
``loss.horizon_schedule=[[0,1],[2000,8]]``.
"""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass
class DataCfg:
    z_dir: str = ""                    # output of scripts/koopman/export_z.py
    train_split: str = "train"
    val_split: str = "val"             # used for model selection / early stopping
    max_train_docs: int | None = None  # None = all exported docs
    max_val_docs: int | None = 500
    exclude_prompt: bool = False       # drop pairs whose start token is in the prompt
    in_memory: bool = True             # load the z trajectories into RAM (fast window gather)
    data_device: str = "cpu"           # "cuda" keeps trajectories on the GPU


@dataclass
class ClusterCfg:
    # Labels used for the behaviour-level metric, the Ulam baseline and the
    # slow-mode analysis.
    #   ret    = the paper's K-means clusters (scripts/clustering/train_centers.py)
    #   kmeans = K-means fitted here on standardized z (fallback)
    #   none   = skip cluster metrics
    source: str = "kmeans"
    ret_dir: str | None = None         # clustering output dir (contains clustering_meta.json, w1/)
    K: int = 64
    seed: int = 1                      # seed sub-directory of the RET clustering run
    tau: float = 0.1                   # analysis/weighted_tau{tau}
    alpha: float = 1.0                 # grouped_alpha{alpha}
    metric: str | None = None          # None = infer from clustering_meta.json (cosine unless *_centered)
    kmeans_iters: int = 50
    kmeans_sample: int = 200_000


@dataclass
class ModelCfg:
    lift_dim: int = 512                # N, total lifted dimension (including z and the constant)
    psi_hidden: list[int] = field(default_factory=lambda: [512, 512])
    psi_norm: str = "batchnorm"        # batchnorm | layernorm | none (applied to psi outputs, no affine)
    include_state: bool = True         # g = [z, 1, psi(z)]; required for decoder=linear_fixed
    include_const: bool = True
    operator: str = "dense"            # dense | blockdiag | lstsq
    operator_init: str = "identity"    # dense only: identity | random
    blockdiag_rmax: float = 1.0        # blockdiag only: max eigenvalue modulus
    lstsq_ridge: float = 1.0e-4        # lstsq only: ridge, relative to mean diag of G^T G
    lstsq_ema: float = 0.99            # lstsq only: EMA of per-batch K used at eval time
    decoder: str = "linear_fixed"      # linear_fixed (z read off g) | linear_learned | mlp
    decoder_hidden: list[int] = field(default_factory=lambda: [512, 512])


@dataclass
class LossCfg:
    # [[start_step, H], ...]: rollout horizon used for training from start_step on.
    horizon_schedule: list = field(default_factory=lambda: [[0, 1], [3000, 8], [10000, 32]])
    rho: float = 0.9                   # discount on the k-th rollout step
    w_pred: float = 1.0                # multi-step z prediction
    w_lin: float = 0.1                 # lifted-space linearity ||K^k g_t - g_{t+k}||^2 / Var(g)
    w_recon: float = 1.0               # reconstruction (only for learned decoders)
    w_spec: float = 1.0                # penalty on spectral-radius bound above spec_target (dense only)
    spec_target: float = 1.0
    spec_power: int = 16               # bound = ||K^P||_2^(1/P)
    w_var: float = 1.0                 # VICReg variance hinge on psi
    w_cov: float = 0.04                # VICReg covariance on psi
    loss_space: str = "raw"            # raw = weight dims by raw-z variance (matches pooled R^2) | std
    lin_stopgrad: bool = False         # stop-gradient on the g_{t+k} target


@dataclass
class TrainCfg:
    steps: int = 30_000
    batch_size: int = 512              # windows per batch
    lr: float = 1.0e-3
    lr_min_fraction: float = 0.05
    weight_decay: float = 1.0e-4       # not applied to the operator parameters
    warmup_steps: int = 500
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 1000
    eval_max_starts: int = 20_000      # val starts used for model selection
    select_ks: list[int] | None = None # None = powers of two up to the final H
    patience: int = 8                  # evals without improvement, counted once the final H is reached
    final_lstsq_refit: bool = True     # lstsq operator: refit K on train data after training
    refit_max_pairs: int = 2_000_000
    seed: int = 0
    device: str = "cuda"


@dataclass
class EvalCfg:
    split: str = "test"
    max_docs: int | None = 1000
    horizons: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    stride: int = 1                    # stride over start positions for the horizon curves
    max_starts: int | None = 200_000   # random subset of starts (same for every model)
    batch_size: int = 4096
    seed: int = 0
    analysis_max_tokens: int = 200_000 # tokens used for slow-mode / residual analysis
    n_slow_modes: int = 16


@dataclass
class EDMDSpec:
    name: str | None = None
    features: str = "rff"              # rff | rbf
    num_features: int = 512            # size of the extra dictionary (N = d + 1 + num_features)
    bandwidth_scale: float = 1.0
    ridge: float = 1.0e-6


@dataclass
class MLPBaselineCfg:
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: int = 2
    train_horizons: list[int] = field(default_factory=lambda: [1, 32])
    steps: int = 4000
    batch_size: int = 512
    lr: float = 1.0e-3
    eval_every: int = 200
    patience: int = 15
    rho: float = 0.9


@dataclass
class DirectCfg:
    ks: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])
    hidden_dim: int = 512
    num_layers: int = 2
    steps: int = 8000
    batch_size: int = 512
    lr: float = 1.0e-3
    eval_every: int = 500
    patience: int = 10


@dataclass
class UlamCfg:
    smoothing: float = 0.01


@dataclass
class BaselinesCfg:
    run: list[str] = field(default_factory=lambda: [
        "mean", "persistence", "dmd", "edmd", "ulam", "ret_predictor", "mlp", "direct",
    ])
    dmd_ridge: float = 1.0e-6
    max_fit_pairs: int | None = 5_000_000
    edmd: list[EDMDSpec] = field(default_factory=lambda: [
        EDMDSpec(features="rff", num_features=512),
        EDMDSpec(features="rff", num_features=2048),
        EDMDSpec(features="rbf", num_features=256),
    ])
    mlp: MLPBaselineCfg = field(default_factory=MLPBaselineCfg)
    direct: DirectCfg = field(default_factory=DirectCfg)
    ulam: UlamCfg = field(default_factory=UlamCfg)


@dataclass
class KoopmanConfig:
    experiment_dir: str = "outputs/koopman/default"
    run_name: str | None = None        # deep Koopman run name; None = derived from hyperparameters
    ret_checkpoint: str | None = None  # None = read from <z_dir>/export_meta.json
    data: DataCfg = field(default_factory=DataCfg)
    clusters: ClusterCfg = field(default_factory=ClusterCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    baselines: BaselinesCfg = field(default_factory=BaselinesCfg)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _strip_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _build(cls: type, data: dict | None, path: str = "") -> Any:
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data.pop(f.name)
        tp = _strip_optional(hints[f.name])
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            value = _build(tp, value, f"{path}{f.name}.")
        elif typing.get_origin(tp) is list and value is not None:
            (inner,) = typing.get_args(tp) or (Any,)
            inner = _strip_optional(inner)
            if dataclasses.is_dataclass(inner):
                value = [_build(inner, v, f"{path}{f.name}[].") if isinstance(v, dict) else v
                         for v in value]
        kwargs[f.name] = value
    if data:
        raise ValueError(f"Unknown config keys under '{path or '<root>'}': {sorted(data)}")
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _apply_override(d: dict, key: str, raw: str) -> None:
    keys = key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(raw) if raw != "" else None


def load_koopman_config(
    yaml_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> KoopmanConfig:
    base: dict = {}
    if yaml_path is not None:
        with open(yaml_path) as f:
            base = yaml.safe_load(f) or {}
    for ov in overrides or []:
        key, sep, raw = ov.partition("=")
        if not sep:
            raise ValueError(f"Invalid override {ov!r}; expected key=value")
        _apply_override(base, key.strip(), raw.strip())
    cfg = _build(KoopmanConfig, base)
    validate(cfg)
    return cfg


def validate(cfg: KoopmanConfig) -> None:
    m = cfg.model
    if m.operator not in ("dense", "blockdiag", "lstsq"):
        raise ValueError(f"model.operator must be dense|blockdiag|lstsq, got {m.operator!r}")
    if m.decoder not in ("linear_fixed", "linear_learned", "mlp"):
        raise ValueError(f"model.decoder must be linear_fixed|linear_learned|mlp, got {m.decoder!r}")
    if m.decoder == "linear_fixed" and not m.include_state:
        raise ValueError("decoder=linear_fixed reads z off g, so model.include_state must be true")
    if m.operator == "blockdiag" and m.include_state:
        raise ValueError(
            "operator=blockdiag needs model.include_state=false and a learned decoder "
            "(z cannot be constrained to evolve by rotation blocks)"
        )
    if m.psi_norm not in ("batchnorm", "layernorm", "none"):
        raise ValueError(f"model.psi_norm must be batchnorm|layernorm|none, got {m.psi_norm!r}")
    if cfg.loss.loss_space not in ("raw", "std"):
        raise ValueError("loss.loss_space must be raw|std")
    if cfg.clusters.source not in ("ret", "kmeans", "none"):
        raise ValueError("clusters.source must be ret|kmeans|none")
    sched = cfg.loss.horizon_schedule
    if not sched or any(len(s) != 2 for s in sched):
        raise ValueError("loss.horizon_schedule must be a non-empty list of [step, H] pairs")
    if sorted(int(s[0]) for s in sched)[0] != 0:
        raise ValueError("loss.horizon_schedule must start at step 0")


def add_config_args(parser) -> None:
    """Standard CLI: ``script.py config.yaml [key=value ...] [--flags]``."""
    parser.add_argument("config", help="Koopman experiment YAML (configs/koopman/*.yaml)")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. model.lift_dim=1024")


def save_koopman_config(cfg: KoopmanConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)


def final_horizon(cfg: KoopmanConfig) -> int:
    return int(max(cfg.loss.horizon_schedule, key=lambda s: s[0])[1])


def horizon_at(step: int, schedule: list) -> int:
    h = None
    for start, H in sorted(schedule, key=lambda s: s[0]):
        if step >= start:
            h = int(H)
    assert h is not None
    return h


def default_select_ks(H: int) -> list[int]:
    ks, k = [], 1
    while k <= H:
        ks.append(k)
        k *= 2
    return ks
