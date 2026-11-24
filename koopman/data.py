"""Reading exported z trajectories and sampling windows that stay inside one document."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ZSplit:
    """One exported split (see scripts/koopman/export_z.py for the layout).

    Documents are kept as one flat (total_tokens, d) array plus per-document
    offsets/lengths. With ``in_memory=True`` the first ``max_docs`` documents
    are loaded into a float32 tensor on ``device``; otherwise windows are read
    from the memmap on demand.
    """

    def __init__(self, split_dir: str | Path, max_docs: int | None = None,
                 in_memory: bool = True, device: str = "cpu"):
        self.dir = Path(split_dir)
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found. Export the split first with scripts/koopman/export_z.py"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.d = int(self.meta["z_dim"])
        offsets = np.load(self.dir / "offsets.npy").astype(np.int64)
        lengths = np.load(self.dir / "lengths.npy").astype(np.int64)
        pl_path = self.dir / "prompt_lengths.npy"
        prompt_lengths = (np.load(pl_path).astype(np.int64) if pl_path.exists()
                          else np.zeros_like(lengths))
        n = len(offsets) if max_docs is None else min(int(max_docs), len(offsets))
        offsets, lengths, prompt_lengths = offsets[:n], lengths[:n], prompt_lengths[:n]

        mm = np.memmap(self.dir / "z.bin", dtype=self.meta["dtype"], mode="r",
                       shape=(int(self.meta["total_tokens"]), self.d))
        self.device = torch.device(device)
        self.in_memory = in_memory
        if in_memory:
            # Docs are written back to back, so the first n docs are one contiguous span.
            contiguous = n == 0 or np.array_equal(offsets, np.concatenate([[offsets[0]], offsets[0] + np.cumsum(lengths)[:-1]]))
            if contiguous and n > 0:
                start, end = int(offsets[0]), int(offsets[-1] + lengths[-1])
                arr = np.array(mm[start:end], dtype=np.float32)  # copy into RAM
                offsets = offsets - start
            else:
                arr = np.concatenate([np.asarray(mm[o:o + l], dtype=np.float32)
                                      for o, l in zip(offsets, lengths)]) if n else np.zeros((0, self.d), np.float32)
                offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64) if n else offsets
            self.z = torch.from_numpy(arr).to(self.device)
            self._mm = None
        else:
            self.z = None
            self._mm = mm
        self.offsets = offsets
        self.lengths = lengths
        self.prompt_lengths = np.minimum(prompt_lengths, lengths)
        self.num_docs = n
        self.num_tokens = int(lengths.sum())

    def __repr__(self) -> str:
        return f"ZSplit({self.dir}, docs={self.num_docs}, tokens={self.num_tokens}, d={self.d})"

    def gather(self, starts: torch.Tensor | np.ndarray, rel: torch.Tensor | np.ndarray | list[int]) -> torch.Tensor:
        """Return z at positions start + rel for every start: shape (B, len(rel), d)."""
        if self.in_memory:
            starts_t = torch.as_tensor(starts, dtype=torch.long, device=self.device)
            rel_t = torch.as_tensor(rel, dtype=torch.long, device=self.device)
            return self.z[starts_t[:, None] + rel_t[None, :]]
        starts_n = np.asarray(starts.cpu() if torch.is_tensor(starts) else starts, dtype=np.int64)
        rel_n = np.asarray(rel.cpu() if torch.is_tensor(rel) else rel, dtype=np.int64)
        pos = starts_n[:, None] + rel_n[None, :]
        flat = pos.reshape(-1)
        order = np.argsort(flat)
        vals = np.empty((flat.size, self.d), dtype=np.float32)
        vals[order] = np.asarray(self._mm[flat[order] + 0], dtype=np.float32)  # sorted reads are faster
        return torch.from_numpy(vals.reshape(pos.shape[0], pos.shape[1], self.d))

    def window(self, starts, W: int) -> torch.Tensor:
        return self.gather(starts, np.arange(W))

    def doc(self, i: int) -> torch.Tensor:
        o, l = int(self.offsets[i]), int(self.lengths[i])
        if self.in_memory:
            return self.z[o:o + l]
        return torch.from_numpy(np.asarray(self._mm[o:o + l], dtype=np.float32))

    def iter_token_chunks(self, chunk: int = 1_000_000, max_tokens: int | None = None):
        """Yield all loaded tokens in chunks of rows (float32 tensors)."""
        if self.in_memory:
            total = self.z.shape[0] if max_tokens is None else min(max_tokens, self.z.shape[0])
            for s in range(0, total, chunk):
                yield self.z[s:min(s + chunk, total)]
        else:
            done = 0
            for i in range(self.num_docs):
                x = self.doc(i)
                if max_tokens is not None and done + x.shape[0] > max_tokens:
                    x = x[: max_tokens - done]
                if x.shape[0]:
                    yield x
                done += x.shape[0]
                if max_tokens is not None and done >= max_tokens:
                    break

    def sample_tokens(self, n: int, seed: int = 0) -> torch.Tensor:
        """Random subset of single z vectors (for k-means, bandwidths, ...)."""
        rng = np.random.default_rng(seed)
        n = min(n, self.num_tokens)
        doc = rng.choice(self.num_docs, size=n, p=self.lengths / self.lengths.sum())
        t = (rng.random(n) * self.lengths[doc]).astype(np.int64)
        pos = self.offsets[doc] + t
        return self.gather(pos, [0])[:, 0]


class WindowIndex:
    """All valid window starts (t, ..., t+H inside one document) of a split.

    With ``exclude_prompt`` the start must also satisfy t >= prompt_length,
    mirroring the closure script's --exclude-scaffold.
    """

    def __init__(self, split: ZSplit, H: int, exclude_prompt: bool = False):
        self.split = split
        self.H = int(H)
        start_min = split.prompt_lengths if exclude_prompt else np.zeros_like(split.lengths)
        n_valid = np.clip(split.lengths - start_min - self.H, 0, None)
        self.start_min = start_min
        self.cum = np.concatenate([[0], np.cumsum(n_valid)]).astype(np.int64)
        self.total = int(self.cum[-1])
        if self.total == 0:
            raise ValueError(f"No windows of length {H + 1} in {split}")

    def flat_to_global(self, flat: np.ndarray) -> np.ndarray:
        doc = np.searchsorted(self.cum, flat, side="right") - 1
        t = self.start_min[doc] + (flat - self.cum[doc])
        return self.split.offsets[doc] + t

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.flat_to_global(rng.integers(0, self.total, size=n))

    def all_starts(self, stride: int = 1, max_starts: int | None = None, seed: int = 0) -> np.ndarray:
        flat = np.arange(0, self.total, max(1, int(stride)), dtype=np.int64)
        if max_starts is not None and flat.size > max_starts:
            rng = np.random.default_rng(seed)
            flat = np.sort(rng.choice(flat, size=int(max_starts), replace=False))
        return self.flat_to_global(flat)


class WindowSampler:
    """Random training windows (B, H+1, d) for a given horizon H."""

    def __init__(self, split: ZSplit, exclude_prompt: bool = False, seed: int = 0):
        self.split = split
        self.exclude_prompt = exclude_prompt
        self.rng = np.random.default_rng(seed)
        self._index: dict[int, WindowIndex] = {}

    def index(self, H: int) -> WindowIndex:
        if H not in self._index:
            self._index[H] = WindowIndex(self.split, H, self.exclude_prompt)
        return self._index[H]

    def sample(self, batch_size: int, H: int, device: torch.device | str | None = None) -> torch.Tensor:
        starts = self.index(H).sample(batch_size, self.rng)
        w = self.split.window(starts, H + 1)
        return w.to(device, non_blocking=True) if device is not None else w

    def sample_positions(self, batch_size: int, rel: list[int], device=None) -> torch.Tensor:
        """Sample starts valid for max(rel) and gather only the requested offsets."""
        starts = self.index(max(rel)).sample(batch_size, self.rng)
        w = self.split.gather(starts, rel)
        return w.to(device, non_blocking=True) if device is not None else w


@torch.no_grad()
def compute_stats(split: ZSplit, max_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean and std over the split's tokens (float64 accumulation)."""
    s = torch.zeros(split.d, dtype=torch.float64)
    s2 = torch.zeros(split.d, dtype=torch.float64)
    n = 0
    for x in split.iter_token_chunks(max_tokens=max_tokens):
        x = x.double().cpu()
        s += x.sum(0)
        s2 += (x * x).sum(0)
        n += x.shape[0]
    mean = s / max(n, 1)
    var = (s2 / max(n, 1) - mean ** 2).clamp_min(1e-12)
    return mean.float(), var.sqrt().float()


def open_split(z_dir: str | Path, split: str, max_docs: int | None, in_memory: bool, device: str) -> ZSplit:
    zs = ZSplit(Path(z_dir) / split, max_docs=max_docs, in_memory=in_memory, device=device)
    logger.info("Loaded %s", zs)
    return zs


def read_export_meta(z_dir: str | Path) -> dict:
    p = Path(z_dir) / "export_meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)
