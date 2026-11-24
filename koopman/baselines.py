"""Fitting the Phase-2 baselines: DMD / EDMD (closed form), Ulam chain,
one-step and multi-step MLPs (closure-probe architecture), direct k-step MLPs."""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import torch

from koopman.clusters import ClusterAssigner, kmeans
from koopman.data import WindowIndex, WindowSampler, ZSplit
from koopman.forecasters import (
    DirectMLPForecaster, LinearKoopmanForecaster, MLPForecaster, UlamForecaster,
)
from koopman.metrics import evaluate, select_score

logger = logging.getLogger(__name__)


def dim_weights(std: torch.Tensor, loss_space: str) -> torch.Tensor:
    """Per-dimension weights so that an MSE in standardized space equals (up to a
    constant) the MSE in raw z space — the quantity the pooled R^2 measures."""
    if loss_space == "std":
        return torch.ones_like(std)
    w = std.float() ** 2
    return w / w.mean()


def wmse(pred: torch.Tensor, true: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return ((pred - true) ** 2 * w).mean()


def cosine_lr(step: int, total: int, warmup: int, min_frac: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = min(1.0, (step - warmup) / max(total - warmup, 1))
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# DMD / EDMD
# ---------------------------------------------------------------------------

@torch.no_grad()
def _median_pairwise_distance(x: torch.Tensor, n: int = 2000, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[: min(n, x.shape[0])].to(x.device)
    s = x[idx]
    d = torch.cdist(s, s)
    iu = torch.triu_indices(s.shape[0], s.shape[0], offset=1, device=s.device)
    return float(d[iu[0], iu[1]].median())


@torch.no_grad()
def init_dictionary(model: LinearKoopmanForecaster, sample_std: torch.Tensor, bandwidth_scale: float,
                    seed: int, kmeans_iters: int = 30) -> None:
    """Set the random features / RBF centers, then the feature standardization."""
    if model.features == "none":
        return
    dev = sample_std.device
    g = torch.Generator(device="cpu").manual_seed(seed)
    if model.features == "rff":
        sigma = _median_pairwise_distance(sample_std, seed=seed) * bandwidth_scale
        model.W.copy_((torch.randn(model.m, model.d, generator=g) / sigma).to(dev))
        model.b.copy_((torch.rand(model.m, generator=g) * 2 * math.pi).to(dev))
    else:  # rbf at k-means centers of standardized z
        C = kmeans(sample_std, model.m, iters=kmeans_iters, seed=seed)
        dist = torch.cdist(sample_std[:20000], C)
        sigma = float(dist.median()) * bandwidth_scale
        model.centers.copy_(C)
        model.sigma.fill_(sigma)
    feats = model.psi_raw(sample_std)
    model.feat_mean.copy_(feats.mean(0))
    model.feat_std.copy_(feats.std(0).clamp_min(1e-6))
    logger.info("  dictionary %s m=%d sigma=%.3f", model.features, model.m, sigma)


@torch.no_grad()
def fit_linear_koopman(model: LinearKoopmanForecaster, train: ZSplit, *, ridge: float, exclude_prompt: bool,
                       max_pairs: int | None, batch_size: int = 16384, device="cuda", seed: int = 0) -> dict:
    """Ridge least squares for K on all (or max_pairs) one-step pairs: G_+ ~ G K^T.

    The constant row of K is fixed to e_const; ridge is relative to mean(diag(G^T G)).
    """
    model.to(device)
    N, ci = model.N, model.const_index
    A = torch.zeros(N, N, dtype=torch.float64, device=device)
    Bm = torch.zeros(N, N, dtype=torch.float64, device=device)
    starts = WindowIndex(train, 1, exclude_prompt).all_starts(1, max_pairs, seed)
    for i in range(0, len(starts), batch_size):
        W = train.gather(starts[i:i + batch_size], [0, 1]).to(device)
        G = model.lift(model.to_std(W[:, 0])).double()
        Gp = model.lift(model.to_std(W[:, 1])).double()
        A += G.T @ G
        Bm += G.T @ Gp
    lam = ridge * float(torch.diagonal(A).mean())
    rows = torch.tensor([j for j in range(N) if j != ci], device=device)
    KT = torch.linalg.solve(A + lam * torch.eye(N, dtype=A.dtype, device=device), Bm[:, rows])  # (N, N-1)
    K = torch.zeros(N, N, dtype=torch.float64, device=device)
    K[rows] = KT.T
    K[ci, ci] = 1.0
    model.K.copy_(K.float())
    eig = torch.linalg.eigvals(K[rows][:, rows]).abs()
    info = {"pairs": int(len(starts)), "lambda": lam, "spectral_radius": float(eig.max())}
    logger.info("  fitted K (N=%d) on %d pairs, spectral radius %.4f", N, len(starts), info["spectral_radius"])
    return info


# ---------------------------------------------------------------------------
# Ulam / cluster Markov chain
# ---------------------------------------------------------------------------

@torch.no_grad()
def fit_ulam(train: ZSplit, assigner: ClusterAssigner, mean: torch.Tensor, std: torch.Tensor, *,
             smoothing: float, exclude_prompt: bool, max_pairs: int | None, batch_size: int = 65536,
             device="cuda", seed: int = 0) -> UlamForecaster:
    model = UlamForecaster(train.d, assigner.init_kwargs).to(device)
    model.assigner.load_state_dict(assigner.state_dict())
    model.set_stats(mean, std)
    C = assigner.K
    counts = torch.zeros(C * C, dtype=torch.float64, device=device)
    zsum = torch.zeros(C, train.d, dtype=torch.float64, device=device)
    zcnt = torch.zeros(C, dtype=torch.float64, device=device)
    starts = WindowIndex(train, 1, exclude_prompt).all_starts(1, max_pairs, seed)
    for i in range(0, len(starts), batch_size):
        W = train.gather(starts[i:i + batch_size], [0, 1]).to(device)
        c0 = model.assigner.assign(W[:, 0])
        c1 = model.assigner.assign(W[:, 1])
        counts += torch.bincount(c0 * C + c1, minlength=C * C).double()
        zsum.index_add_(0, c0, W[:, 0].double())
        zcnt += torch.bincount(c0, minlength=C).double()
    counts = counts.view(C, C) + smoothing
    model.P.copy_((counts / counts.sum(1, keepdim=True)).float())
    M = zsum / zcnt.clamp_min(1)[:, None]
    M[zcnt == 0] = mean.to(device).double()
    model.M.copy_(M.float())
    logger.info("  Ulam chain: %d clusters, %d pairs, %d empty clusters", C, len(starts), int((zcnt == 0).sum()))
    return model


# ---------------------------------------------------------------------------
# Gradient-trained baselines
# ---------------------------------------------------------------------------

def _train_loop(model, params, loss_fn, *, steps: int, lr: float, eval_every: int, patience: int,
                eval_fn, warmup: int = 100, weight_decay: float = 0.0, grad_clip: float = 1.0,
                label: str = "") -> dict:
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best, best_state, bad, history = -float("inf"), None, 0, []
    t0, run_loss, n_loss = time.time(), 0.0, 0
    for step in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * cosine_lr(step, steps, warmup, 0.1)
        model.train()
        loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        run_loss += float(loss.detach())
        n_loss += 1
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            score, extra = eval_fn()
            history.append({"step": step + 1, "train_loss": run_loss / n_loss, "val_score": score, **extra})
            run_loss, n_loss = 0.0, 0
            if score > best + 1e-6:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            logger.info("  [%s] step %d loss %.5f val %.4f (best %.4f) %.0fs", label, step + 1,
                        history[-1]["train_loss"], score, best, time.time() - t0)
            if bad >= patience:
                logger.info("  [%s] early stop at step %d", label, step + 1)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_val_score": best, "history": history}


def train_mlp(model: MLPForecaster, train: ZSplit, val: ZSplit, *, H: int, rho: float, w: torch.Tensor,
              steps: int, batch_size: int, lr: float, eval_every: int, patience: int, exclude_prompt: bool,
              eval_max_starts: int, device, seed: int = 0, label: str = "mlp") -> dict:
    """One-step (H=1, closure-probe protocol) or multi-step rollout training."""
    model.to(device)
    w = w.to(device)
    sampler = WindowSampler(train, exclude_prompt, seed=seed)
    weights = torch.tensor([rho ** (k - 1) for k in range(1, H + 1)], device=device)
    weights = weights / weights.sum()
    sel_ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= H]

    def loss_fn():
        W = model.to_std(sampler.sample(batch_size, H, device))
        s, loss = W[:, 0], 0.0
        for k in range(1, H + 1):
            s = model.step_std(s)
            loss = loss + weights[k - 1] * wmse(s, W[:, k], w)
        return loss

    def eval_fn():
        res = evaluate(model, val, sel_ks, max_starts=eval_max_starts, exclude_prompt=exclude_prompt,
                       device=device, seed=seed)
        return select_score(res, sel_ks), {f"val_r2_k{k}": res[k]["r2"] for k in sel_ks}

    return _train_loop(model, list(model.parameters()), loss_fn, steps=steps, lr=lr, eval_every=eval_every,
                       patience=patience, eval_fn=eval_fn, label=label)


def train_direct(model: DirectMLPForecaster, train: ZSplit, val: ZSplit, *, w: torch.Tensor, steps: int,
                 batch_size: int, lr: float, eval_every: int, patience: int, exclude_prompt: bool,
                 eval_max_starts: int, device, seed: int = 0) -> dict:
    model.to(device)
    w = w.to(device)
    sampler = WindowSampler(train, exclude_prompt, seed=seed)
    ks = model.ks
    rel = [0] + ks

    def loss_fn():
        W = model.to_std(sampler.sample_positions(batch_size, rel, device))
        s = W[:, 0]
        return sum(wmse(model.predict_std(s, k), W[:, j + 1], w) for j, k in enumerate(ks)) / len(ks)

    def eval_fn():
        res = evaluate(model, val, ks, max_starts=eval_max_starts, exclude_prompt=exclude_prompt,
                       device=device, seed=seed)
        return select_score(res, ks), {f"val_r2_k{k}": res[k]["r2"] for k in ks}

    return _train_loop(model, list(model.parameters()), loss_fn, steps=steps, lr=lr, eval_every=eval_every,
                       patience=patience, eval_fn=eval_fn, label="direct")


def summarize(res: dict[int, dict]) -> dict:
    return {str(k): {m: (float(v) if isinstance(v, (float, int, np.floating)) else v) for m, v in r.items()}
            for k, r in res.items()}
