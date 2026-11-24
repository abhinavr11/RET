"""Deep Koopman model: learned lifting psi, linear operator K, readout C.

    g(z) = [z_std ; 1 ; norm(psi(z_std))]      (pieces switchable via config)
    g_{t+1} = K g_t
    z_t    = C g_t   (C = [I 0 0] for decoder=linear_fixed)

Operators
    dense      K is a free N x N matrix (row of the constant fixed to e_const), identity init.
    blockdiag  K = blockdiag of 2x2 blocks r R(omega) with r = rmax * sigmoid(.) <= rmax: stable by
               construction, eigenvalues read off directly. Needs include_state=false + learned decoder.
    lstsq      K is solved per batch by ridge least squares on the lifted pairs (gradients flow into
               psi through the solve); an EMA of those solutions (or a final refit) is used at eval.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from koopman.forecasters import Forecaster, register, rollout


def mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Lifting(nn.Module):
    def __init__(self, d: int, lift_dim: int, psi_hidden: list[int], psi_norm: str,
                 include_state: bool, include_const: bool):
        super().__init__()
        self.d = d
        self.include_state = include_state
        self.include_const = include_const
        self.n_fixed = (d if include_state else 0) + (1 if include_const else 0)
        self.m = lift_dim - self.n_fixed
        if self.m < 0:
            raise ValueError(f"lift_dim={lift_dim} smaller than the fixed part ({self.n_fixed})")
        self.N = lift_dim
        self.const_index = (d if include_state else 0) if include_const else None
        self.psi_start = self.n_fixed
        if self.m > 0:
            self.psi = mlp(d, list(psi_hidden), self.m)
            if psi_norm == "batchnorm":
                self.norm = nn.BatchNorm1d(self.m, affine=False)
            elif psi_norm == "layernorm":
                self.norm = nn.LayerNorm(self.m, elementwise_affine=False)
            else:
                self.norm = nn.Identity()
        else:
            self.psi = None

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        lead = s.shape[:-1]
        parts = []
        if self.include_state:
            parts.append(s)
        if self.include_const:
            parts.append(torch.ones(*lead, 1, device=s.device, dtype=s.dtype))
        if self.psi is not None:
            p = self.norm(self.psi(s.reshape(-1, self.d))).reshape(*lead, self.m)
            parts.append(p)
        return torch.cat(parts, -1)

    def psi_part(self, g: torch.Tensor) -> torch.Tensor:
        return g[..., self.psi_start:]


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class DenseOperator(nn.Module):
    def __init__(self, N: int, const_index: int | None, init: str = "identity"):
        super().__init__()
        self.N, self.const_index = N, const_index
        if init == "identity":
            K = torch.eye(N)
        elif init == "random":
            K = torch.randn(N, N) / math.sqrt(N) * 0.9
        else:
            raise ValueError(f"operator_init must be identity|random, got {init!r}")
        self.weight = nn.Parameter(K)
        mask = torch.ones(N, N)
        fixed = torch.zeros(N, N)
        if const_index is not None:
            mask[const_index] = 0.0
            fixed[const_index, const_index] = 1.0
        self.register_buffer("row_mask", mask)
        self.register_buffer("fixed", fixed)

    def matrix(self) -> torch.Tensor:
        return self.weight * self.row_mask + self.fixed

    def step(self, g: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        return g @ (self.matrix() if K is None else K).T


class BlockDiagOperator(nn.Module):
    """Real block-diagonal K: 2x2 rotation-scaling blocks (+ one real eigenvalue if odd)."""

    def __init__(self, N: int, const_index: int | None, rmax: float = 1.0):
        super().__init__()
        if const_index not in (None, 0):
            raise ValueError("blockdiag expects the constant at index 0 (include_state=false)")
        self.N, self.const_index, self.rmax = N, const_index, rmax
        self.off = 1 if const_index is not None else 0
        n_dyn = N - self.off
        self.P = n_dyn // 2
        self.has_real = n_dyn % 2 == 1
        # radii start in [0.9, 0.999], angles small: slow, mostly non-oscillating modes
        frac = torch.empty(self.P).uniform_(0.9, 0.999) * min(rmax, 1.0) / rmax  # r / rmax in (0, 1)
        self.r_logit = nn.Parameter(torch.log(frac / (1 - frac)))
        self.omega_raw = nn.Parameter(torch.empty(self.P).uniform_(-0.1, 0.1))
        if self.has_real:
            self.real_raw = nn.Parameter(torch.tensor(2.0))

    def params(self):
        r = self.rmax * torch.sigmoid(self.r_logit)
        w = math.pi * torch.tanh(self.omega_raw)
        return r, w

    def step(self, g: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        if K is not None:
            return g @ K.T
        r, w = self.params()
        a, b = r * torch.cos(w), r * torch.sin(w)
        lead = g.shape[:-1]
        parts = []
        if self.off:
            parts.append(g[..., :1])
        pairs = g[..., self.off:self.off + 2 * self.P].reshape(*lead, self.P, 2)
        x, y = pairs[..., 0], pairs[..., 1]
        parts.append(torch.stack([a * x - b * y, b * x + a * y], -1).reshape(*lead, 2 * self.P))
        if self.has_real:
            parts.append(g[..., -1:] * (self.rmax * torch.tanh(self.real_raw)))
        return torch.cat(parts, -1)

    def matrix(self) -> torch.Tensor:
        r, w = self.params()
        K = torch.zeros(self.N, self.N, device=r.device, dtype=r.dtype)
        if self.off:
            K[0, 0] = 1.0
        a, b = r * torch.cos(w), r * torch.sin(w)
        i = torch.arange(self.P, device=r.device) * 2 + self.off
        K[i, i] = a
        K[i, i + 1] = -b
        K[i + 1, i] = b
        K[i + 1, i + 1] = a
        if self.has_real:
            K[-1, -1] = self.rmax * torch.tanh(self.real_raw)
        return K


class LstsqOperator(nn.Module):
    def __init__(self, N: int, const_index: int | None, ridge: float = 1e-4, ema: float = 0.99):
        super().__init__()
        self.N, self.const_index, self.ridge, self.ema = N, const_index, ridge, ema
        self.register_buffer("K", torch.eye(N))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        rows = [j for j in range(N) if j != const_index]
        self.register_buffer("rows", torch.tensor(rows, dtype=torch.long))

    def solve(self, A: torch.Tensor, Bm: torch.Tensor) -> torch.Tensor:
        """K from Gram matrices A = G^T G, Bm = G^T G_+ (float64)."""
        N = self.N
        lam = self.ridge * torch.diagonal(A).mean().detach()
        KT = torch.linalg.solve(A + lam * torch.eye(N, dtype=A.dtype, device=A.device), Bm[:, self.rows])
        K = torch.zeros(N, N, dtype=A.dtype, device=A.device)
        K = K.index_copy(0, self.rows, KT.T)
        if self.const_index is not None:
            e = torch.zeros(N, dtype=A.dtype, device=A.device)
            e[self.const_index] = 1.0
            K = K.index_copy(0, torch.tensor([self.const_index], device=A.device), e[None])
        return K

    def fit_batch(self, G: torch.Tensor, Gp: torch.Tensor) -> torch.Tensor:
        G64, Gp64 = G.double(), Gp.double()
        K = self.solve(G64.T @ G64, G64.T @ Gp64).float()
        with torch.no_grad():
            if not bool(self.initialized):
                self.K.copy_(K.detach())
                self.initialized.fill_(True)
            else:
                self.K.mul_(self.ema).add_(K.detach(), alpha=1 - self.ema)
        return K

    def matrix(self) -> torch.Tensor:
        return self.K

    def step(self, g: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        return g @ (self.K if K is None else K).T


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class FixedStateDecoder(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, g):
        return g[..., : self.d]


@register
class DeepKoopmanForecaster(Forecaster):
    kind = "deep_koopman"

    def __init__(self, d: int, model_cfg: dict | None = None):
        super().__init__(d)
        c = dict(model_cfg or {})
        self.init_kwargs.update(model_cfg=c)
        self.cfg = c
        self.lifting = Lifting(d, c.get("lift_dim", 512), c.get("psi_hidden", [512, 512]),
                               c.get("psi_norm", "batchnorm"), c.get("include_state", True),
                               c.get("include_const", True))
        N, ci = self.lifting.N, self.lifting.const_index
        self.N, self.const_index = N, ci
        op = c.get("operator", "dense")
        if op == "dense":
            self.operator = DenseOperator(N, ci, c.get("operator_init", "identity"))
        elif op == "blockdiag":
            self.operator = BlockDiagOperator(N, ci, c.get("blockdiag_rmax", 1.0))
        elif op == "lstsq":
            self.operator = LstsqOperator(N, ci, c.get("lstsq_ridge", 1e-4), c.get("lstsq_ema", 0.99))
        else:
            raise ValueError(op)
        dec = c.get("decoder", "linear_fixed")
        if dec == "linear_fixed":
            self.decoder = FixedStateDecoder(d)
        elif dec == "linear_learned":
            self.decoder = nn.Linear(N, d)
        elif dec == "mlp":
            self.decoder = mlp(N, list(c.get("decoder_hidden", [512, 512])), d)
        else:
            raise ValueError(dec)
        self.decoder_kind = dec
        self.operator_kind = op

    # --- pieces -----------------------------------------------------------
    def lift(self, s: torch.Tensor) -> torch.Tensor:
        return self.lifting(s)

    def decode(self, g: torch.Tensor) -> torch.Tensor:
        return self.decoder(g)

    def step(self, g: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
        return self.operator.step(g, K)

    def koopman_matrix(self) -> torch.Tensor:
        return self.operator.matrix()

    def dynamic_index(self) -> torch.Tensor:
        idx = torch.arange(self.N, device=self.mean.device)
        return idx if self.const_index is None else idx[idx != self.const_index]

    def readout_matrix(self) -> torch.Tensor | None:
        """C with z_std = C g, when the decoder is linear (None for the MLP decoder)."""
        if self.decoder_kind == "linear_fixed":
            C = torch.zeros(self.d, self.N, device=self.mean.device)
            C[:, : self.d] = torch.eye(self.d, device=self.mean.device)
            return C
        if self.decoder_kind == "linear_learned":
            return self.decoder.weight
        return None

    # --- forecasting ------------------------------------------------------
    def predict(self, z0, ks):
        g = self.lift(self.to_std(z0))
        out = rollout(lambda x: self.step(x), g, ks)
        return {k: self.from_std(self.decode(v)) for k, v in out.items()}

    def num_parameters(self) -> int:
        n = sum(p.numel() for p in self.parameters())
        if self.operator_kind == "lstsq":
            n += (self.N - 1) * self.N
        return n
