"""Losses for the deep Koopman model.

    L = sum_k rho^(k-1) [ w_pred ||z_{t+k} - C K^k g_t||_w^2 + w_lin ||K^k g_t - g_{t+k}||^2 / Var(g) ]
        + w_recon ||z - C g(z)||_w^2          (learned decoders only)
        + w_spec  relu(||K_dd^P||_2^(1/P) - spec_target)^2   (dense operator only)
        + w_var * VICReg-variance(psi) + w_cov * VICReg-covariance(psi)

||.||_w is the dimension-weighted MSE (see baselines.dim_weights) so the z loss
tracks the pooled raw-z R^2 that is reported.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from koopman.baselines import wmse
from koopman.models import DeepKoopmanForecaster


def spectral_radius_bound(K: torch.Tensor, power: int = 16, iters: int = 20) -> torch.Tensor:
    """Differentiable upper bound ||K^P||_2^(1/P) >= rho(K); tight as P grows."""
    A = torch.linalg.matrix_power(K, power)
    with torch.no_grad():
        v = torch.randn(A.shape[1], device=A.device, dtype=A.dtype)
        v = v / v.norm()
        for _ in range(iters):
            v = A.T @ (A @ v)
            v = v / v.norm().clamp_min(1e-30)
    sigma = (A @ v).norm()
    return sigma.clamp_min(1e-30) ** (1.0 / power)


def vicreg_terms(x: torch.Tensor, max_rows: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance hinge (std >= 1) and off-diagonal covariance penalty on rows of x (n, m)."""
    if x.shape[0] > max_rows:
        x = x[torch.randperm(x.shape[0], device=x.device)[:max_rows]]
    x = x - x.mean(0)
    std = torch.sqrt(x.var(0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    n, m = x.shape
    cov = (x.T @ x) / max(n - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / m
    return var_loss, cov_loss


def koopman_loss(model: DeepKoopmanForecaster, W_std: torch.Tensor, H: int, lcfg, w: torch.Tensor,
                 ) -> tuple[torch.Tensor, dict]:
    """W_std: (B, H+1, d) standardized windows."""
    B, L, d = W_std.shape
    assert L == H + 1
    N = model.N
    g_all = model.lift(W_std)  # (B, H+1, N)

    K = None
    if model.operator_kind == "lstsq":
        K = model.operator.fit_batch(g_all[:, :-1].reshape(-1, N), g_all[:, 1:].reshape(-1, N))

    var_g = g_all.reshape(-1, N).var(0).detach().clamp_min(1e-4)
    rho = lcfg.rho
    g = g_all[:, 0]
    pred_loss = W_std.new_zeros(())
    lin_loss = W_std.new_zeros(())
    wsum = 0.0
    per_k = {}
    for k in range(1, H + 1):
        g = model.step(g, K)
        a = rho ** (k - 1)
        pk = wmse(model.decode(g), W_std[:, k], w)
        tgt = g_all[:, k].detach() if lcfg.lin_stopgrad else g_all[:, k]
        lk = ((g - tgt).pow(2) / var_g).mean()
        pred_loss = pred_loss + a * pk
        lin_loss = lin_loss + a * lk
        wsum += a
        if k in (1, H):
            per_k[f"pred_k{k}"] = float(pk.detach())
    pred_loss = pred_loss / wsum
    lin_loss = lin_loss / wsum

    total = lcfg.w_pred * pred_loss + lcfg.w_lin * lin_loss
    metrics = {"pred": float(pred_loss.detach()), "lin": float(lin_loss.detach()), **per_k}

    if model.decoder_kind != "linear_fixed" and lcfg.w_recon > 0:
        recon = wmse(model.decode(g_all), W_std, w)
        total = total + lcfg.w_recon * recon
        metrics["recon"] = float(recon.detach())

    if model.lifting.m > 0 and (lcfg.w_var > 0 or lcfg.w_cov > 0):
        psi = model.lifting.psi_part(g_all).reshape(-1, model.lifting.m)
        v, c = vicreg_terms(psi)
        total = total + lcfg.w_var * v + lcfg.w_cov * c
        metrics["psi_var"] = float(v.detach())
        metrics["psi_cov"] = float(c.detach())

    if model.operator_kind == "dense" and lcfg.w_spec > 0:
        Kd = model.koopman_matrix()
        idx = model.dynamic_index()
        Kdd = Kd.index_select(0, idx).index_select(1, idx)
        bound = spectral_radius_bound(Kdd, power=lcfg.spec_power)
        spec = F.relu(bound - lcfg.spec_target).pow(2)
        total = total + lcfg.w_spec * spec
        metrics["spec_bound"] = float(bound.detach())

    metrics["total"] = float(total.detach())
    return total, metrics
