"""Cluster labels for z: the paper's K-means clusters, or K-means fitted here."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@torch.no_grad()
def kmeans(x: torch.Tensor, K: int, iters: int = 50, seed: int = 0, chunk: int = 65536) -> torch.Tensor:
    """Plain k-means (k-means++ init + Lloyd). x: (n, d) float32. Returns centers (K, d)."""
    n = x.shape[0]
    if n < K:
        raise ValueError(f"k-means needs at least K={K} points, got {n}")
    g = torch.Generator(device="cpu").manual_seed(seed)

    def sqdist(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (a.pow(2).sum(1, keepdim=True) - 2 * a @ c.T + c.pow(2).sum(1)[None]).clamp_min(0)

    # k-means++ on a subsample (keeps init cheap)
    sub = x[torch.randperm(n, generator=g)[: min(n, 50_000)].to(x.device)]
    first = int(torch.randint(sub.shape[0], (1,), generator=g))
    centers = [sub[first]]
    d2 = sqdist(sub, sub[first:first + 1])[:, 0]
    for _ in range(1, K):
        probs = (d2 / d2.sum().clamp_min(1e-12)).cpu()
        idx = int(torch.multinomial(probs, 1, generator=g))
        centers.append(sub[idx])
        d2 = torch.minimum(d2, sqdist(sub, sub[idx:idx + 1])[:, 0])
    C = torch.stack(centers)

    for _ in range(iters):
        sums = torch.zeros_like(C)
        counts = torch.zeros(K, device=x.device, dtype=x.dtype)
        for s in range(0, n, chunk):
            xb = x[s:s + chunk]
            a = sqdist(xb, C).argmin(1)
            sums.index_add_(0, a, xb)
            counts.index_add_(0, a, torch.ones_like(a, dtype=x.dtype))
        empty = counts == 0
        newC = sums / counts.clamp_min(1)[:, None]
        if empty.any():  # re-seed empty clusters at random points
            ridx = torch.randint(n, (int(empty.sum()),), generator=g).to(x.device)
            newC[empty] = x[ridx]
        shift = (newC - C).pow(2).sum().sqrt() / C.pow(2).sum().sqrt().clamp_min(1e-12)
        C = newC
        if shift < 1e-5:
            break
    return C


class ClusterAssigner(nn.Module):
    """Assigns raw z to clusters.

    x = (z - mu) / scale, L2-normalized when metric == "cosine";
    nearest center by cosine similarity or Euclidean distance.
    For the paper's clusters scale = 1 and mu = the clustering's centering vector.
    """

    def __init__(self, d: int, K: int, metric: str = "cosine", num_groups: int = 0, source: str = "ret"):
        super().__init__()
        if metric not in ("cosine", "euclidean"):
            raise ValueError(f"metric must be cosine|euclidean, got {metric!r}")
        self.init_kwargs = dict(d=d, K=K, metric=metric, num_groups=num_groups, source=source)
        self.d, self.K, self.metric, self.num_groups, self.source = d, K, metric, num_groups, source
        self.register_buffer("centers", torch.zeros(K, d))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("scale", torch.ones(d))
        self.register_buffer("group_labels", torch.zeros(K, dtype=torch.long))

    @property
    def has_groups(self) -> bool:
        return self.num_groups > 0

    def transform(self, z: torch.Tensor) -> torch.Tensor:
        x = (z.float() - self.mu) / self.scale
        if self.metric == "cosine":
            x = F.normalize(x, dim=-1)
        return x

    @torch.no_grad()
    def assign(self, z: torch.Tensor) -> torch.Tensor:
        x = self.transform(z)
        if self.metric == "cosine":
            return (x @ F.normalize(self.centers, dim=-1).T).argmax(-1)
        return (-2 * x @ self.centers.T + self.centers.pow(2).sum(1)[None]).argmin(-1)

    def group(self, c: torch.Tensor) -> torch.Tensor:
        return self.group_labels[c]

    # ------------------------------------------------------------------
    @classmethod
    def from_ret(cls, ret_dir: str | Path, d: int, K: int = 64, seed: int = 1, tau: float = 0.1,
                 alpha: float = 1.0, metric: str | None = None) -> "ClusterAssigner":
        root = Path(ret_dir)
        meta = {}
        if (root / "clustering_meta.json").exists():
            with open(root / "clustering_meta.json") as f:
                meta = json.load(f)
        variant = str(meta.get("variant", "z"))
        if not variant.startswith("z"):
            logger.warning("Clustering variant is %r, not a z variant; labels may be meaningless", variant)
        if metric is None:
            metric = "euclidean" if variant.endswith("_centered") else "cosine"
        w_dir = root / "w1"
        seed_dir = w_dir / f"K{K}" / f"seed{seed}"
        mu = np.load(w_dir / "mu.npy").astype(np.float32).reshape(-1)
        centers = np.load(seed_dir / "centers.npy").astype(np.float32)
        if centers.shape != (K, d) or mu.shape != (d,):
            raise ValueError(f"Clustering shapes centers={centers.shape} mu={mu.shape}, expected ({K},{d}) / ({d},)")
        grp_path = seed_dir / "analysis" / f"weighted_tau{tau}" / f"grouped_alpha{alpha}" / "group_labels.npy"
        groups = None
        if grp_path.exists():
            groups = np.load(grp_path).astype(np.int64).reshape(-1)
        else:
            logger.warning("No group labels at %s; group accuracy will be skipped", grp_path)
        num_groups = int(groups.max()) + 1 if groups is not None else 0
        a = cls(d=d, K=K, metric=metric, num_groups=num_groups, source="ret")
        a.centers.copy_(torch.from_numpy(centers))
        a.mu.copy_(torch.from_numpy(mu))
        if groups is not None:
            a.group_labels.copy_(torch.from_numpy(groups))
        logger.info("Loaded RET clustering from %s (K=%d, metric=%s, groups=%d)", root, K, metric, num_groups)
        return a

    @classmethod
    def fit_kmeans(cls, z_sample: torch.Tensor, K: int, mean: torch.Tensor, std: torch.Tensor,
                   iters: int = 50, seed: int = 0) -> "ClusterAssigner":
        a = cls(d=z_sample.shape[1], K=K, metric="euclidean", num_groups=0, source="kmeans")
        a.mu.copy_(mean.cpu())
        a.scale.copy_(std.cpu())
        x = (z_sample.float() - mean.to(z_sample.device)) / std.to(z_sample.device)
        a.centers.copy_(kmeans(x, K, iters=iters, seed=seed).cpu())
        return a

    def save(self, path: str | Path) -> None:
        torch.save({"init_kwargs": self.init_kwargs, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "ClusterAssigner":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = cls(**ck["init_kwargs"])
        a.load_state_dict(ck["state_dict"])
        return a


def build_assigner(cfg, train_split, mean: torch.Tensor, std: torch.Tensor, cache_path: Path | None,
                   device: str) -> ClusterAssigner | None:
    """Cluster assigner per ``cfg.clusters`` (cached on disk so every script uses the same one)."""
    cc = cfg.clusters
    if cc.source == "none":
        return None
    if cc.source == "ret":
        if not cc.ret_dir:
            raise ValueError("clusters.source=ret requires clusters.ret_dir")
        a = ClusterAssigner.from_ret(cc.ret_dir, d=train_split.d, K=cc.K, seed=cc.seed, tau=cc.tau,
                                     alpha=cc.alpha, metric=cc.metric)
        return a.to(device)
    if cache_path is not None and cache_path.exists():
        a = ClusterAssigner.load(cache_path)
        if a.K == cc.K and a.d == train_split.d:
            logger.info("Loaded k-means clusters from %s", cache_path)
            return a.to(device)
    logger.info("Fitting k-means (K=%d) on %d train tokens ...", cc.K, cc.kmeans_sample)
    sample = train_split.sample_tokens(cc.kmeans_sample, seed=cc.seed).to(device)
    a = ClusterAssigner.fit_kmeans(sample, cc.K, mean.to(device), std.to(device), iters=cc.kmeans_iters,
                                   seed=cc.seed)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        a.save(cache_path)
    return a.to(device)
