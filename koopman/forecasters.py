"""Forecasters: every model maps a batch of raw z_t to predictions of z_{t+k}.

Common interface (all in *raw* z units, so every model is scored identically)::

    preds = model.predict(z0, ks)   # z0: (B, d) raw  ->  {k: (B, d) raw}

Models keep the train mean/std as buffers and work in standardized space
internally. ``save``/``load_forecaster`` round-trip any registered model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman.clusters import ClusterAssigner

REGISTRY: dict[str, type] = {}


def register(cls):
    REGISTRY[cls.kind] = cls
    return cls


def rollout(step: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, ks: list[int]) -> dict[int, torch.Tensor]:
    want = set(int(k) for k in ks)
    out: dict[int, torch.Tensor] = {}
    x = x0
    for k in range(1, max(want) + 1):
        x = step(x)
        if k in want:
            out[k] = x
    return out


class Forecaster(nn.Module):
    kind = "base"

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.init_kwargs: dict = {"d": d}
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("std", torch.ones(d))

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> "Forecaster":
        self.mean.copy_(mean.to(self.mean.device).float())
        self.std.copy_(std.to(self.std.device).float())
        return self

    def to_std(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) / self.std

    def from_std(self, s: torch.Tensor) -> torch.Tensor:
        return s * self.std + self.mean

    def available_ks(self, ks: list[int]) -> list[int]:
        return [int(k) for k in ks]

    def predict(self, z0: torch.Tensor, ks: list[int]) -> dict[int, torch.Tensor]:
        raise NotImplementedError

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "kind": self.kind,
            "init_kwargs": self.init_kwargs,
            "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "extra": extra or {},
        }, path)


def load_forecaster(path: str | Path, device: str | torch.device = "cpu") -> tuple[Forecaster, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["kind"] not in REGISTRY:
        import koopman.models  # noqa: F401  (registers deep_koopman; imported lazily to avoid a cycle)
    if ck["kind"] not in REGISTRY:
        raise KeyError(f"Unknown forecaster kind {ck['kind']!r} in {path}")
    cls = REGISTRY[ck["kind"]]
    model = cls(**ck["init_kwargs"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck.get("extra", {})


# ---------------------------------------------------------------------------
# Trivial references
# ---------------------------------------------------------------------------

@register
class MeanForecaster(Forecaster):
    kind = "mean"

    def predict(self, z0, ks):
        m = self.mean.expand_as(z0)
        return {int(k): m for k in ks}


@register
class PersistenceForecaster(Forecaster):
    kind = "persistence"

    def predict(self, z0, ks):
        return {int(k): z0 for k in ks}


# ---------------------------------------------------------------------------
# Linear Koopman with a fixed dictionary: DMD (no features) and EDMD (rff / rbf)
#   g(z) = [z_std, 1, standardized psi(z_std)],  z = first d coordinates of g.
# ---------------------------------------------------------------------------

@register
class LinearKoopmanForecaster(Forecaster):
    kind = "linear_koopman"

    def __init__(self, d: int, features: str = "none", num_features: int = 0):
        super().__init__(d)
        if features not in ("none", "rff", "rbf"):
            raise ValueError(f"features must be none|rff|rbf, got {features!r}")
        self.features = features
        self.m = 0 if features == "none" else int(num_features)
        self.N = d + 1 + self.m
        self.const_index = d
        self.init_kwargs.update(features=features, num_features=self.m)
        self.register_buffer("K", torch.eye(self.N))
        if features == "rff":
            self.register_buffer("W", torch.zeros(self.m, d))
            self.register_buffer("b", torch.zeros(self.m))
        elif features == "rbf":
            self.register_buffer("centers", torch.zeros(self.m, d))
            self.register_buffer("sigma", torch.ones(()))
        self.register_buffer("feat_mean", torch.zeros(self.m))
        self.register_buffer("feat_std", torch.ones(self.m))

    def psi_raw(self, s: torch.Tensor) -> torch.Tensor:
        if self.features == "rff":
            return torch.cos(s @ self.W.T + self.b)
        if self.features == "rbf":
            d2 = (s.pow(2).sum(-1, keepdim=True) - 2 * s @ self.centers.T
                  + self.centers.pow(2).sum(-1)[None]).clamp_min(0)
            return torch.exp(-d2 / (2 * self.sigma ** 2))
        return s[..., :0]

    def lift(self, s: torch.Tensor) -> torch.Tensor:
        parts = [s, torch.ones_like(s[..., :1])]
        if self.m:
            parts.append((self.psi_raw(s) - self.feat_mean) / self.feat_std)
        return torch.cat(parts, -1)

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return g[..., : self.d]

    def koopman_matrix(self) -> torch.Tensor:
        return self.K

    def dynamic_index(self) -> torch.Tensor:
        idx = torch.arange(self.N, device=self.K.device)
        return idx[idx != self.const_index]

    def predict(self, z0, ks):
        g = self.lift(self.to_std(z0))
        KT = self.K.T
        out = rollout(lambda x: x @ KT, g, ks)
        return {k: self.from_std(self.decode(v)) for k, v in out.items()}

    def num_parameters(self) -> int:
        return (self.N - 1) * self.N  # fitted rows of K (the constant row is fixed)


# ---------------------------------------------------------------------------
# Nonlinear one-step map (closure-probe architecture), rolled out
# ---------------------------------------------------------------------------

def _mlp(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    from ret.models.predictor import build_mlp_net  # same residual MLP as RET's T and the closure probes
    return build_mlp_net(input_dim, output_dim, hidden_dim, num_layers)


@register
class MLPForecaster(Forecaster):
    kind = "mlp"

    def __init__(self, d: int, hidden_dim: int = 256, num_layers: int = 2, residual: bool = False):
        super().__init__(d)
        self.init_kwargs.update(hidden_dim=hidden_dim, num_layers=num_layers, residual=residual)
        self.residual = residual
        self.net = _mlp(d, d, hidden_dim, num_layers)

    def step_std(self, s: torch.Tensor) -> torch.Tensor:
        return s + self.net(s) if self.residual else self.net(s)

    def predict(self, z0, ks):
        out = rollout(self.step_std, self.to_std(z0), ks)
        return {k: self.from_std(v) for k, v in out.items()}


@register
class RETPredictorForecaster(Forecaster):
    """RET's own predictor T (trained with L1 against the EMA teacher), rolled out in raw z space."""

    kind = "ret_predictor"

    def __init__(self, d: int, predictor_cfg: dict | None = None):
        super().__init__(d)
        from ret.config import PredictorConfig
        from ret.models.predictor import Predictor
        predictor_cfg = dict(predictor_cfg or {})
        self.init_kwargs.update(predictor_cfg=predictor_cfg)
        pcfg = PredictorConfig(**predictor_cfg)
        if pcfg.mode != "autonomous":
            raise ValueError("Only autonomous RET predictors can be rolled out without token inputs")
        self.predictor = Predictor(d, pcfg)

    def predict(self, z0, ks):
        return rollout(lambda z: self.predictor(z), z0, ks)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, ret_config: str | Path | None = None) -> "RETPredictorForecaster":
        # Read the YAML directly: ret.config.load_config also validates the data section
        # (paths etc.), which is irrelevant for rebuilding the predictor.
        import dataclasses
        import yaml
        from ret.config import EncoderConfig, PredictorConfig
        checkpoint = Path(checkpoint)
        with open(ret_config or checkpoint.parent / "config.yaml") as f:
            raw = yaml.safe_load(f) or {}
        pkeys = {f.name for f in dataclasses.fields(PredictorConfig)}
        pcfg = {k: v for k, v in (raw.get("predictor") or {}).items() if k in pkeys}
        z_dim = (raw.get("encoder") or {}).get("z_dim", EncoderConfig().z_dim)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = cls(int(z_dim), pcfg)
        model.predictor.load_state_dict(ck["predictor"])
        return model


@register
class DirectMLPForecaster(Forecaster):
    """A separate MLP z_t -> z_{t+k} for each k (no rollout, so no compounding error)."""

    kind = "direct_mlp"

    def __init__(self, d: int, ks: list[int] | None = None, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__(d)
        ks = sorted(int(k) for k in (ks or [1]))
        self.ks = ks
        self.init_kwargs.update(ks=ks, hidden_dim=hidden_dim, num_layers=num_layers)
        self.nets = nn.ModuleDict({str(k): _mlp(d, d, hidden_dim, num_layers) for k in ks})

    def available_ks(self, ks):
        return [int(k) for k in ks if int(k) in self.ks]

    def predict_std(self, s: torch.Tensor, k: int) -> torch.Tensor:
        return self.nets[str(k)](s)

    def predict(self, z0, ks):
        s = self.to_std(z0)
        return {int(k): self.from_std(self.predict_std(s, int(k))) for k in ks if int(k) in self.ks}


@register
class UlamForecaster(Forecaster):
    """Markov chain on cluster labels (Koopman projected onto cluster indicators).

    z prediction = E[z_{t+k} | c_t] = onehot(c_t) P^k M, M = per-cluster mean of raw z.
    """

    kind = "ulam"

    def __init__(self, d: int, assigner_kwargs: dict | None = None):
        super().__init__(d)
        assigner_kwargs = dict(assigner_kwargs or {"d": d, "K": 1})
        self.init_kwargs.update(assigner_kwargs=assigner_kwargs)
        self.assigner = ClusterAssigner(**assigner_kwargs)
        C = self.assigner.K
        self.register_buffer("P", torch.eye(C))
        self.register_buffer("M", torch.zeros(C, d))

    def predict(self, z0, ks):
        c = self.assigner.assign(z0)
        p = F.one_hot(c, self.assigner.K).float()
        out = rollout(lambda q: q @ self.P, p, ks)
        return {k: v @ self.M for k, v in out.items()}

    def koopman_matrix(self) -> torch.Tensor:
        return self.P.T  # acts on functions of the cluster label

    def num_parameters(self) -> int:
        C = self.assigner.K
        return C * (C - 1) + C * self.d


def param_count(model: Forecaster) -> int:
    try:
        return int(model.num_parameters())
    except Exception:  # pragma: no cover
        return -1


def is_koopman(model: Forecaster) -> bool:
    return hasattr(model, "lift") and hasattr(model, "koopman_matrix")

