"""Spectral analysis of a fitted Koopman operator (Phase 4, items 4 and 5).

* eigenvalues of K restricted to the non-constant coordinates, with timescales
  -1/log|lambda| and periods 2*pi/|arg lambda| (in tokens);
* eigenfunctions phi_i(z) = u_i^T g(z) (u_i left eigenvectors) and how well each
  slow one separates the cluster groups (correlation ratio eta^2), next to the same
  number for the top principal components of z as a reference;
* the one-step linearity residual ||g(z_{t+1}) - K g(z_t)|| / ||g(z_{t+1})||,
  overall and per cluster group.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from koopman.data import WindowIndex, ZSplit


def correlation_ratio(x: np.ndarray, labels: np.ndarray) -> float:
    """eta^2 = between-group variance / total variance (0..1)."""
    x = x.astype(np.float64)
    mu = x.mean()
    tot = ((x - mu) ** 2).sum()
    if tot <= 1e-18:
        return 0.0
    between = 0.0
    for c in np.unique(labels):
        sel = labels == c
        between += sel.sum() * (x[sel].mean() - mu) ** 2
    return float(between / tot)


@torch.no_grad()
def koopman_analysis(model, split: ZSplit, assigner=None, *, max_tokens: int = 200_000, n_modes: int = 16,
                     exclude_prompt: bool = False, device="cuda", seed: int = 0, batch_size: int = 8192) -> dict:
    model.eval()
    K = model.koopman_matrix().detach().double().cpu()
    idx = model.dynamic_index().cpu()
    Kdd = K.index_select(0, idx).index_select(1, idx)
    vals, vecs = torch.linalg.eig(Kdd.T)  # columns u_i: u_i^T Kdd = lambda_i u_i^T
    vals_np = vals.numpy()
    order = np.argsort(-np.abs(vals_np))
    vals_np, vecs = vals_np[order], vecs[:, torch.from_numpy(order.copy())]

    absv = np.abs(vals_np)
    ang = np.angle(vals_np)
    with np.errstate(divide="ignore"):
        timescale = np.where(absv < 1, -1.0 / np.log(np.clip(absv, 1e-300, None)), np.inf)
        period = np.where(np.abs(ang) > 1e-9, 2 * math.pi / np.abs(ang), np.inf)

    n_modes = min(n_modes, vecs.shape[1])
    U = vecs[:, :n_modes]  # (N_dyn, n_modes) complex128, CPU
    starts = WindowIndex(split, 1, exclude_prompt).all_starts(1, max_tokens, seed)
    phi_list, res_list, lab_list, zstd_list = [], [], [], []
    K_dev = model.koopman_matrix().detach().float().to(device)
    idx_dev = idx.to(device)
    for i in range(0, len(starts), batch_size):
        W = split.gather(starts[i:i + batch_size], [0, 1]).to(device)
        s0, s1 = model.to_std(W[:, 0]), model.to_std(W[:, 1])
        g0, g1 = model.lift(s0), model.lift(s1)
        r = (g1 - g0 @ K_dev.T).index_select(1, idx_dev)
        denom = g1.index_select(1, idx_dev).norm(dim=1).clamp_min(1e-8)
        res_list.append((r.norm(dim=1) / denom).cpu())
        # eigenfunction values phi_i(z_t) = u_i^T g_dyn(z_t); only the projections are kept
        phi_list.append(g0.index_select(1, idx_dev).double().cpu().to(torch.complex128) @ U)
        zstd_list.append(s0.double().cpu())
        if assigner is not None:
            c = assigner.assign(W[:, 0])
            lab_list.append((assigner.group(c) if assigner.has_groups else c).cpu())
    phi = torch.cat(phi_list).numpy()  # (n, n_modes)
    res = torch.cat(res_list).numpy()
    out = {
        "n_tokens": int(phi.shape[0]),
        "spectral_radius": float(absv.max()) if absv.size else None,
        "n_unstable": int((absv > 1.0 + 1e-6).sum()),
        "eigvals": [[float(v.real), float(v.imag)] for v in vals_np],
        "residual": {"mean": float(res.mean()), "median": float(np.median(res)),
                     "p90": float(np.quantile(res, 0.9))},
    }

    modes = []
    labels = torch.cat(lab_list).numpy() if lab_list else None
    for i in range(n_modes):
        row = {"abs": float(absv[i]), "angle": float(ang[i]), "timescale": float(timescale[i]),
               "period": float(period[i])}
        if labels is not None:
            row["eta2_re"] = correlation_ratio(phi[:, i].real, labels)
            row["eta2_im"] = correlation_ratio(phi[:, i].imag, labels) if abs(ang[i]) > 1e-9 else None
        modes.append(row)
    out["slow_modes"] = modes

    if labels is not None:
        Z = torch.cat(zstd_list)
        Zc = Z - Z.mean(0)
        evals, evecs = torch.linalg.eigh(Zc.T @ Zc / max(Zc.shape[0] - 1, 1))
        pcs = (Zc @ evecs[:, torch.argsort(evals, descending=True)[:n_modes]]).numpy()
        out["pca_eta2"] = [correlation_ratio(pcs[:, i], labels) for i in range(pcs.shape[1])]
        out["label_kind"] = "group" if (assigner is not None and assigner.has_groups) else "cluster"
        by = {}
        for c in np.unique(labels):
            sel = labels == c
            by[int(c)] = {"n": int(sel.sum()), "mean": float(res[sel].mean()), "median": float(np.median(res[sel]))}
        out["residual_by_label"] = by
    return out
