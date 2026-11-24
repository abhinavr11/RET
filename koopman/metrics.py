"""Horizon metrics: pooled R^2 (same definition as the paper's closure probes),
skill vs persistence, cosine / norm drift, and cluster-label accuracy."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from koopman.data import WindowIndex, ZSplit


class HorizonAccumulator:
    """Streaming statistics per horizon k.

    R^2 = 1 - SS_res / SS_tot with SS_tot = sum_d sum_i (y_id - mean_d)^2, i.e. the
    pooled definition of evaluation/closure/predictors.py::compute_r2_streaming.
    skill = 1 - SS_res / SS_persistence (persistence predicts z_{t+k} = z_t).
    """

    def __init__(self, ks: list[int], d: int, assigner=None):
        self.ks = list(ks)
        self.assigner = assigner
        self.s = {k: {
            "ss_res": 0.0, "ss_persist": 0.0, "n": 0,
            "sum_y": torch.zeros(d, dtype=torch.float64), "sum_y2": torch.zeros(d, dtype=torch.float64),
            "cos": 0.0, "norm_ratio": 0.0,
            "cl_ok": 0, "cl_persist_ok": 0, "gr_ok": 0, "gr_persist_ok": 0,
        } for k in self.ks}

    @torch.no_grad()
    def update(self, k: int, pred: torch.Tensor, true: torch.Tensor, z0: torch.Tensor) -> None:
        st = self.s[k]
        pred, true, z0 = pred.float(), true.float(), z0.float()
        st["ss_res"] += float((pred - true).double().pow(2).sum())
        st["ss_persist"] += float((z0 - true).double().pow(2).sum())
        y = true.double().cpu()
        st["sum_y"] += y.sum(0)
        st["sum_y2"] += (y * y).sum(0)
        st["n"] += true.shape[0]
        st["cos"] += float(F.cosine_similarity(pred, true, dim=-1).double().sum())
        st["norm_ratio"] += float((pred.norm(dim=-1) / true.norm(dim=-1).clamp_min(1e-8)).double().sum())
        if self.assigner is not None:
            cp, ct, c0 = self.assigner.assign(pred), self.assigner.assign(true), self.assigner.assign(z0)
            st["cl_ok"] += int((cp == ct).sum())
            st["cl_persist_ok"] += int((c0 == ct).sum())
            if self.assigner.has_groups:
                gp, gt, g0 = self.assigner.group(cp), self.assigner.group(ct), self.assigner.group(c0)
                st["gr_ok"] += int((gp == gt).sum())
                st["gr_persist_ok"] += int((g0 == gt).sum())

    def result(self) -> dict[int, dict]:
        out = {}
        for k, st in self.s.items():
            n = max(st["n"], 1)
            ss_tot = float((st["sum_y2"] - st["sum_y"] ** 2 / n).sum())
            d = st["sum_y"].numel()
            r = {
                "n": st["n"],
                "r2": 1.0 - st["ss_res"] / ss_tot if ss_tot > 1e-12 else float("nan"),
                "mse": st["ss_res"] / (n * d),
                "skill": 1.0 - st["ss_res"] / st["ss_persist"] if st["ss_persist"] > 1e-12 else float("nan"),
                "cos": st["cos"] / n,
                "norm_ratio": st["norm_ratio"] / n,
            }
            if self.assigner is not None:
                r["cluster_acc"] = st["cl_ok"] / n
                r["cluster_acc_persist"] = st["cl_persist_ok"] / n
                if self.assigner.has_groups:
                    r["group_acc"] = st["gr_ok"] / n
                    r["group_acc_persist"] = st["gr_persist_ok"] / n
            out[k] = r
        return out


@torch.no_grad()
def evaluate(model, split: ZSplit, ks: list[int], *, max_k: int | None = None, stride: int = 1,
             max_starts: int | None = None, batch_size: int = 4096, exclude_prompt: bool = False,
             assigner=None, seed: int = 0, device: str | torch.device = "cuda") -> dict[int, dict]:
    """Score ``model`` at horizons ``ks``.

    Starts are every position t whose window t..t+max_k lies in one document
    (strided / subsampled deterministically), so for a fixed (split, max_k,
    stride, max_starts, seed) every model is evaluated on identical starts.
    """
    ks = sorted(set(int(k) for k in ks))
    max_k = int(max_k or max(ks))
    ks_avail = model.available_ks(ks)
    if not ks_avail:
        return {}
    starts = WindowIndex(split, max_k, exclude_prompt).all_starts(stride, max_starts, seed)
    rel = [0] + ks_avail
    acc = HorizonAccumulator(ks_avail, split.d, assigner)
    was_training = model.training
    model.eval()
    for i in range(0, len(starts), batch_size):
        W = split.gather(starts[i:i + batch_size], rel).to(device, non_blocking=True)
        z0 = W[:, 0]
        preds = model.predict(z0, ks_avail)
        for j, k in enumerate(ks_avail):
            acc.update(k, preds[k], W[:, j + 1], z0)
    model.train(was_training)
    return acc.result()


def select_score(res: dict[int, dict], ks: list[int]) -> float:
    vals = [res[k]["r2"] for k in ks if k in res and np.isfinite(res[k]["r2"])]
    return float(np.mean(vals)) if vals else float("-inf")


def format_row(name: str, res: dict[int, dict], ks: list[int]) -> str:
    cells = []
    for k in ks:
        r = res.get(k)
        cells.append(f"{r['r2']:.4f}" if r else "   -  ")
    return f"{name:<40s} " + " ".join(f"{c:>8s}" for c in cells)
