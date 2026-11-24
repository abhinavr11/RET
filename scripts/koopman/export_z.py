#!/usr/bin/env python3
"""Export RET macrostate trajectories z_t (Phase 1 of the Koopman plan).

Runs the frozen *student* encoder of a RET checkpoint over whole documents
(full causal context, never token by token) and writes, per split::

    <out_dir>/<split>/
        meta.json            num_docs, total_tokens, z_dim, dtype
        z.bin                memmap (total_tokens, z_dim)
        input_ids.bin        memmap (total_tokens,) int32
        offsets.npy          (num_docs,) int64  start of each doc in the flat arrays
        lengths.npy          (num_docs,) int64
        prompt_lengths.npy   (num_docs,) int64  0 when unknown
        doc_index.npy        (num_docs,) int64  row / chunk index in the source split
    <out_dir>/export_meta.json
    <out_dir>/ret_config.yaml

Where the documents come from is set by ``--data-config`` (an ordinary RET
ExperimentConfig YAML; default = the checkpoint's own config.yaml):

    cached    RET hidden-state memmaps (cache_hidden_states.py). No LLM is loaded.
    document  Arrow dataset at data.local_dir/<split>. The LLM is loaded to extract h.
    chunked   HF streaming corpus (e.g. the Pile). The LLM is loaded. ``train`` reads
              the shuffled stream exactly like RET training; held-out splits read the
              unshuffled stream after skipping ``--chunked-skip-docs`` documents, so
              they do not overlap with what training saw.

Examples::

    # GPT-OSS-20B / NuminaMath from the cached layer-11 hidden states
    python scripts/koopman/export_z.py \\
        --checkpoint outputs/GptOss20B/numina_math/ret_jepa_auth_l11/checkpoint_final.pt \\
        --out-dir outputs/koopman/GptOss20B_numina_l11/z \\
        --max-docs train=16000 val=500 test=1000 \\
        --prompt-length-source data/numina_math_64k_samples/GptOss20B

    # Pythia-160M / Pile chunks (the checkpoint's own chunked config)
    python scripts/koopman/export_z.py \\
        --checkpoint outputs/Pythia160M/pile/ret_autonomous/checkpoint_final.pt \\
        --out-dir outputs/koopman/Pythia160M_pile_l6/z \\
        --max-docs train=20000 val=500 test=1000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ret.config import ExperimentConfig, load_config, save_config  # noqa: E402
from ret.utils.checkpointing import load_encoder  # noqa: E402

logger = logging.getLogger("export_z")

DEFAULT_CHUNKED_DOCS = {"train": 20_000}
DEFAULT_CHUNKED_HELDOUT = 500


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="RET checkpoint (.pt), e.g. .../checkpoint_final.pt")
    p.add_argument("--ret-config", default=None,
                   help="RET config used to build the encoder (default: config.yaml next to the checkpoint)")
    p.add_argument("--data-config", default=None,
                   help="RET ExperimentConfig YAML whose llm/data sections define the documents "
                        "(default: the RET config)")
    p.add_argument("--data-default", default=None,
                   help="Optional base YAML for --data-config (same as scripts/train.py --default)")
    p.add_argument("--data-override", nargs="*", default=[],
                   help="key=value overrides for the data config, e.g. data.max_seq_length=1024")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--max-docs", nargs="*", default=[],
                   help="Either one int for every split, or split=N entries (e.g. train=16000 val=500)")
    p.add_argument("--batch-size", type=int, default=4, help="documents per encoder/LLM forward")
    p.add_argument("--z-dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--prompt-length-source", default="auto",
                   help="cached mode: Arrow dataset dir with a prompt_length column per split "
                        "(e.g. data/numina_math_64k_samples/GptOss20B); 'none' to skip. "
                        "document mode reads the column directly when present.")
    p.add_argument("--chunked-skip-docs", type=int, default=200_000,
                   help="chunked mode: corpus documents skipped before held-out chunks are taken")
    p.add_argument("--overwrite", action="store_true", help="overwrite splits that already exist")
    return p.parse_args(argv)


def parse_max_docs(items: list[str], splits: list[str]) -> dict[str, int | None]:
    out: dict[str, int | None] = {s: None for s in splits}
    for it in items:
        if "=" in it:
            k, v = it.split("=", 1)
            out[k.strip()] = int(v)
        else:
            for s in splits:
                out[s] = int(it)
    return out


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class SplitWriter:
    def __init__(self, split_dir: Path, z_dim: int, z_dtype: str):
        self.dir = split_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "meta.json"
        if meta.exists():
            meta.unlink()  # meta.json is written last; its presence marks a finished split
        self.z_dim = z_dim
        self.z_dtype = np.dtype(z_dtype)
        self.f_z = open(self.dir / "z.bin", "wb")
        self.f_ids = open(self.dir / "input_ids.bin", "wb")
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.prompt_lengths: list[int] = []
        self.doc_index: list[int] = []
        self.total = 0

    def add(self, z: np.ndarray, ids: np.ndarray, prompt_length: int, doc_index: int) -> None:
        L = int(z.shape[0])
        if L == 0:
            return
        assert z.shape == (L, self.z_dim), (z.shape, self.z_dim)
        assert ids.shape == (L,), (ids.shape, L)
        self.f_z.write(np.ascontiguousarray(z, dtype=self.z_dtype).tobytes())
        self.f_ids.write(np.ascontiguousarray(ids, dtype=np.int32).tobytes())
        self.offsets.append(self.total)
        self.lengths.append(L)
        self.prompt_lengths.append(int(min(max(prompt_length, 0), L)))
        self.doc_index.append(int(doc_index))
        self.total += L

    def close(self, extra: dict) -> dict:
        self.f_z.close()
        self.f_ids.close()
        np.save(self.dir / "offsets.npy", np.asarray(self.offsets, dtype=np.int64))
        np.save(self.dir / "lengths.npy", np.asarray(self.lengths, dtype=np.int64))
        np.save(self.dir / "prompt_lengths.npy", np.asarray(self.prompt_lengths, dtype=np.int64))
        np.save(self.dir / "doc_index.npy", np.asarray(self.doc_index, dtype=np.int64))
        meta = {
            "num_docs": len(self.offsets),
            "total_tokens": int(self.total),
            "z_dim": self.z_dim,
            "dtype": self.z_dtype.name,
            "ids_dtype": "int32",
            "mean_length": float(np.mean(self.lengths)) if self.lengths else 0.0,
            "mean_prompt_length": float(np.mean(self.prompt_lengths)) if self.prompt_lengths else 0.0,
            **extra,
        }
        with open(self.dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        return meta


# ---------------------------------------------------------------------------
# Document sources. Each yields dicts:
#   ids (np.int64, L), prompt_length (int), doc_index (int), and optionally h (np, L x d_h)
# ---------------------------------------------------------------------------

def _load_tokenizer(llm_cfg):
    from transformers import AutoTokenizer
    kw = {"trust_remote_code": True} if llm_cfg.trust_remote_code else {}
    tok = AutoTokenizer.from_pretrained(llm_cfg.name, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _cached_prompt_lengths(source: str | None, split: str, n_docs: int, ds, tokenizer) -> np.ndarray | None:
    """Read prompt_length for cached docs from the Arrow dataset they were cached from."""
    if source in (None, "none"):
        return None
    split_dir = Path(source) / split
    if not split_dir.exists():
        logger.warning("No Arrow split at %s; prompt lengths set to 0 for split %s", split_dir, split)
        return None
    from datasets import load_from_disk
    arrow = load_from_disk(str(split_dir))
    if "prompt_length" not in arrow.column_names:
        logger.warning("%s has no prompt_length column; prompt lengths set to 0", split_dir)
        return None
    if len(arrow) != len(ds):
        logger.warning(
            "Row count mismatch (Arrow %d vs cache %d) for split %s; cannot align prompt lengths, using 0",
            len(arrow), len(ds), split,
        )
        return None
    # Spot-check alignment: re-tokenize a few docs and compare with the cached ids.
    if tokenizer is not None and "text" in arrow.column_names:
        for i in sorted({0, min(1, n_docs - 1), n_docs // 2, n_docs - 1}):
            if i < 0:
                continue
            cached_ids = ds[i]["input_ids"].numpy()
            ids = np.asarray(tokenizer.encode(arrow[i]["text"], add_special_tokens=False)[: len(cached_ids)])
            m = min(len(ids), len(cached_ids), 256)
            if m == 0 or not np.array_equal(ids[:m], cached_ids[:m]):
                logger.warning(
                    "Cached doc %d does not match Arrow row %d after re-tokenization; prompt lengths "
                    "set to 0 for split %s (was the cache built with a chat scaffold?)", i, i, split,
                )
                return None
    pl = np.asarray(arrow["prompt_length"][:n_docs], dtype=np.int64)
    logger.info("Split %s: prompt lengths aligned (mean %.1f tokens)", split, float(pl.mean()) if len(pl) else 0)
    return pl


def iter_cached(data_cfg: ExperimentConfig, split: str, max_docs: int | None,
                prompt_source: str | None) -> tuple[int, Iterator[dict]]:
    from ret.data.dataset import CachedHiddenStateDataset

    cache_dir = Path(data_cfg.data.cached_hidden_states_dir)
    split_dir = cache_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Cached split not found: {split_dir}")
    ds = CachedHiddenStateDataset(split_dir, llm_config=data_cfg.llm,
                                  max_seq_length=data_cfg.data.max_seq_length)
    n = len(ds) if max_docs is None else min(max_docs, len(ds))

    tokenizer = None
    if prompt_source not in (None, "none"):
        try:
            tokenizer = _load_tokenizer(data_cfg.llm)
        except Exception as e:  # tokenizer is only used for a sanity check
            logger.warning("Could not load tokenizer for the alignment check (%s)", e)
    try:
        prompt_lengths = _cached_prompt_lengths(prompt_source, split, n, ds, tokenizer)
    except Exception as e:  # missing/partial Arrow data must not kill the export
        logger.warning("Could not read prompt lengths from %s (%s); using 0 for split %s", prompt_source, e, split)
        prompt_lengths = None

    def gen():
        for i in range(n):
            item = ds[i]
            yield {
                "ids": item["input_ids"].numpy(),
                "h": item["hidden_states"],  # torch float32 (L, d_h)
                "prompt_length": int(prompt_lengths[i]) if prompt_lengths is not None else 0,
                "doc_index": i,
            }
    return n, gen()


def iter_document(data_cfg: ExperimentConfig, split: str, max_docs: int | None,
                  tokenizer) -> tuple[int | None, Iterator[dict]]:
    from datasets import load_from_disk
    from ret.data.dataset import _compute_scaffold_tokens, maybe_apply_chat_template

    dcfg = data_cfg.data
    split_dir = Path(dcfg.local_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Arrow split not found: {split_dir}")
    ds = load_from_disk(str(split_dir))
    max_length = dcfg.max_seq_length or 2048
    has_pl = "prompt_length" in ds.column_names
    scaffold_prefix = 0
    if dcfg.scaffolding == "chatify":
        prefix_ids, _ = _compute_scaffold_tokens(dcfg, tokenizer)
        scaffold_prefix = len(prefix_ids) if prefix_ids is not None else 0
    elif dcfg.scaffolding == "proper_chat":
        logger.warning("scaffolding=proper_chat: prompt lengths are not computed (set to 0)")

    def gen():
        count = 0
        for row_idx in range(len(ds)):
            if max_docs is not None and count >= max_docs:
                break
            row = ds[row_idx]
            if dcfg.scaffolding == "proper_chat":
                formatted = maybe_apply_chat_template(row[dcfg.messages_field], tokenizer, dcfg)
                pl = 0
            else:
                text = row[dcfg.text_field]
                if not isinstance(text, str) or not text.strip():
                    continue  # same filter as build_padded_dataloader
                formatted = maybe_apply_chat_template(text, tokenizer, dcfg)
                if dcfg.scaffolding == "chatify":
                    pl = scaffold_prefix
                else:
                    pl = int(row["prompt_length"]) if has_pl else 0
            ids = tokenizer.encode(formatted, add_special_tokens=False, truncation=True, max_length=max_length)
            if len(ids) < 2:
                continue
            count += 1
            yield {"ids": np.asarray(ids, dtype=np.int64), "prompt_length": pl, "doc_index": row_idx}

    n = len(ds) if max_docs is None else min(max_docs, len(ds))
    return n, gen()


class ChunkStream:
    """Concat-and-chunk over a corpus, matching RET's chunked mode (incl. chat scaffold)."""

    def __init__(self, data_cfg: ExperimentConfig, tokenizer, shuffled: bool, skip_docs: int, seed: int):
        from evaluation.datasets.corpus_source import load_corpus
        from ret.data.dataset import _compute_scaffold_tokens

        dcfg = data_cfg.data
        ds = load_corpus(hf_dataset=dcfg.hf_dataset, local_dir=dcfg.local_dir)
        if shuffled:
            if dcfg.hf_dataset is not None:
                ds = ds.shuffle(buffer_size=dcfg.shuffle_buffer_size, seed=seed)
            else:
                ds = ds.shuffle(seed=seed)
        elif skip_docs > 0:
            if dcfg.hf_dataset is not None:
                ds = ds.skip(skip_docs)
            else:
                ds = ds.select(range(min(skip_docs, len(ds)), len(ds)))
        self.ds = ds
        self.tok = tokenizer
        self.text_field = dcfg.text_field
        self.chunk_length = dcfg.chunk_length
        prefix, suffix = _compute_scaffold_tokens(dcfg, tokenizer)
        self.prefix = prefix.tolist() if prefix is not None else []
        self.suffix = suffix.tolist() if suffix is not None else []
        self._it = self._chunks()
        self.n_emitted = 0

    def _chunks(self) -> Iterator[list[int]]:
        buf: list[int] = []
        for ex in self.ds:
            text = ex[self.text_field]
            if not isinstance(text, str) or not text.strip():
                continue
            buf.extend(self.tok.encode(text, add_special_tokens=False))
            while len(buf) >= self.chunk_length:
                chunk, buf = buf[: self.chunk_length], buf[self.chunk_length:]
                yield self.prefix + chunk + self.suffix

    def take(self, n: int) -> Iterator[dict]:
        for _ in range(n):
            try:
                ids = next(self._it)
            except StopIteration:
                return
            idx = self.n_emitted
            self.n_emitted += 1
            yield {"ids": np.asarray(ids, dtype=np.int64), "prompt_length": len(self.prefix), "doc_index": idx}


def batched(it: Iterable[dict], bs: int) -> Iterator[list[dict]]:
    buf: list[dict] = []
    for x in it:
        buf.append(x)
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_batch(docs: list[dict], encoder, extractor, device: str) -> list[np.ndarray]:
    B = len(docs)
    lengths = [len(d["ids"]) for d in docs]
    T = max(lengths)
    ids = torch.zeros(B, T, dtype=torch.long)
    mask = torch.zeros(B, T, dtype=torch.long)
    for i, d in enumerate(docs):
        ids[i, : lengths[i]] = torch.from_numpy(d["ids"])
        mask[i, : lengths[i]] = 1

    if "h" in docs[0]:
        d_h = docs[0]["h"].shape[-1]
        h = torch.zeros(B, T, d_h, dtype=torch.float32)
        for i, d in enumerate(docs):
            h[i, : lengths[i]] = d["h"]
        h = h.to(device, non_blocking=True)
    else:
        if extractor is not None and getattr(extractor, "tokenizer", None) is not None \
                and extractor.tokenizer.pad_token_id is not None:
            ids = ids.masked_fill(mask == 0, extractor.tokenizer.pad_token_id)
        h = extractor.extract(ids, attention_mask=mask).float().to(device)

    mask_d = mask.to(device)
    z, _aux = encoder(h, attention_mask=mask_d)  # (B, T, d_z) — full causal context
    z = z.float().cpu().numpy()
    return [z[i, : lengths[i]] for i in range(B)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    ret_cfg_path = Path(args.ret_config) if args.ret_config else ckpt_path.parent / "config.yaml"
    if not ret_cfg_path.exists():
        raise FileNotFoundError(f"RET config not found: {ret_cfg_path} (pass --ret-config)")
    ret_cfg = load_config(str(ret_cfg_path))

    if args.data_config or args.data_default:
        data_cfg = load_config(args.data_config, overrides=args.data_override, default_yaml=args.data_default)
    elif args.data_override:
        data_cfg = load_config(str(ret_cfg_path), overrides=args.data_override)
    else:
        data_cfg = ret_cfg

    # The encoder was trained on one LLM layer: the documents must come from the same one.
    for attr in ("name", "layer_index", "hidden_dim"):
        a, b = getattr(ret_cfg.llm, attr), getattr(data_cfg.llm, attr)
        if a != b:
            raise ValueError(f"llm.{attr} differs between RET config ({a!r}) and data config ({b!r})")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")

    logger.info("Loading RET encoder from %s", ckpt_path)
    encoder, _, _ckpt = load_encoder(ckpt_path, device=args.device, config=ret_cfg)
    encoder.eval()
    z_dim = ret_cfg.encoder.z_dim
    mode = data_cfg.data.mode
    logger.info("Data mode: %s | LLM %s layer %s | z_dim %d", mode, data_cfg.llm.name,
                data_cfg.llm.layer_index, z_dim)

    extractor = None
    tokenizer = None
    if mode in ("document", "chunked"):
        from ret.extraction.hooks import HiddenStateExtractor
        logger.info("Loading LLM for hidden-state extraction ...")
        extractor = HiddenStateExtractor(data_cfg.llm, device=args.device)
        tokenizer = extractor.tokenizer
    elif mode != "cached":
        raise ValueError(f"Unsupported data.mode {mode!r}")

    max_docs = parse_max_docs(args.max_docs, args.splits)
    prompt_source = args.prompt_length_source
    if mode == "cached" and prompt_source == "auto":
        # The cached NuminaMath layout is <dataset>/<model>/layerN; the Arrow dataset sits at <dataset>/<model>.
        cand = Path(data_cfg.data.cached_hidden_states_dir).parent
        prompt_source = str(cand) if (cand / "train").exists() or (cand / "val").exists() else "none"
        logger.info("prompt-length-source auto -> %s", prompt_source)

    heldout_stream: ChunkStream | None = None
    split_summaries = {}
    for split in args.splits:
        split_dir = out_dir / split
        if (split_dir / "meta.json").exists() and not args.overwrite:
            logger.info("Split %s already exported at %s (use --overwrite to redo); skipping", split, split_dir)
            with open(split_dir / "meta.json") as f:
                split_summaries[split] = json.load(f)
            continue

        n_max = max_docs.get(split)
        if mode == "cached":
            n_total, docs = iter_cached(data_cfg, split, n_max, prompt_source)
        elif mode == "document":
            n_total, docs = iter_document(data_cfg, split, n_max, tokenizer)
        else:  # chunked
            if split == "train":
                n_total = n_max or DEFAULT_CHUNKED_DOCS["train"]
                stream = ChunkStream(data_cfg, tokenizer, shuffled=True, skip_docs=0,
                                     seed=data_cfg.training.seed)
                docs = stream.take(n_total)
            else:
                if heldout_stream is None:
                    heldout_stream = ChunkStream(data_cfg, tokenizer, shuffled=False,
                                                 skip_docs=args.chunked_skip_docs, seed=0)
                n_total = n_max or DEFAULT_CHUNKED_HELDOUT
                docs = heldout_stream.take(n_total)

        writer = SplitWriter(split_dir, z_dim, args.z_dtype)
        t0 = time.time()
        n_done = 0
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_total, desc=f"export {split}", dynamic_ncols=True)
        except ImportError:  # pragma: no cover
            pbar = None
        for batch in batched(docs, args.batch_size):
            zs = encode_batch(batch, encoder, extractor, args.device)
            for d, z in zip(batch, zs):
                writer.add(z, d["ids"], d["prompt_length"], d["doc_index"])
            n_done += len(batch)
            if pbar is not None:
                pbar.update(len(batch))
        if pbar is not None:
            pbar.close()
        meta = writer.close({
            "split": split,
            "source_mode": mode,
            "seconds": round(time.time() - t0, 1),
        })
        split_summaries[split] = meta
        logger.info("Split %s: %d docs, %d tokens -> %s", split, meta["num_docs"], meta["total_tokens"], split_dir)

    save_config(ret_cfg, out_dir / "ret_config.yaml")
    export_meta = {
        "checkpoint": str(ckpt_path.resolve()),
        "ret_config": str(ret_cfg_path.resolve()),
        "data_config": str(Path(args.data_config).resolve()) if args.data_config else None,
        "data_override": args.data_override,
        "data_mode": mode,
        "data_source": (data_cfg.data.cached_hidden_states_dir if mode == "cached"
                        else data_cfg.data.local_dir or data_cfg.data.hf_dataset),
        "llm": data_cfg.llm.name,
        "layer_index": data_cfg.llm.layer_index,
        "z_dim": z_dim,
        "encoder_type": ret_cfg.encoder.type,
        "encoder": "student",
        "prompt_length_source": prompt_source,
        "splits": {s: {"num_docs": m["num_docs"], "total_tokens": m["total_tokens"]}
                   for s, m in split_summaries.items()},
    }
    with open(out_dir / "export_meta.json", "w") as f:
        json.dump(export_meta, f, indent=2)
    with open(out_dir / "data_config.yaml", "w") as f:
        from dataclasses import asdict
        yaml.safe_dump({"llm": asdict(data_cfg.llm), "data": asdict(data_cfg.data)}, f, sort_keys=False)
    logger.info("Done. Export metadata: %s", out_dir / "export_meta.json")

    if extractor is not None:
        extractor.cleanup()


if __name__ == "__main__":
    main()
