"""CPU tests for the Koopman pipeline (koopman/ + scripts/koopman/).

* export_z on a fake cached hidden-state split with a tiny random RET checkpoint
  (checks that batched, padded encoding equals encoding each document alone);
* baselines, deep Koopman training (dense / lstsq / blockdiag) and evaluation on
  synthetic trajectories with a known nonlinear but Koopman-linearizable law:
      x1' = 0.95 x1 + 0.3 eps,   x2' = 0.9 x2 + 0.5 x1^2 - mean   (mixed into d dims)
  where DMD cannot capture the x1^2 term but a lifted linear model can.

Run: pytest tests/test_koopman.py -v
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D = 8  # z_dim of the synthetic data and the tiny RET encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_z_split(split_dir: Path, docs: list[np.ndarray], prompt_lengths=None) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    z = np.concatenate(docs).astype(np.float32)
    lengths = np.array([len(x) for x in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    z.tofile(split_dir / "z.bin")
    np.zeros(len(z), dtype=np.int32).tofile(split_dir / "input_ids.bin")
    np.save(split_dir / "offsets.npy", offsets)
    np.save(split_dir / "lengths.npy", lengths)
    pl = np.zeros_like(lengths) if prompt_lengths is None else np.asarray(prompt_lengths, dtype=np.int64)
    np.save(split_dir / "prompt_lengths.npy", pl)
    np.save(split_dir / "doc_index.npy", np.arange(len(docs)))
    with open(split_dir / "meta.json", "w") as f:
        json.dump({"num_docs": len(docs), "total_tokens": int(len(z)), "z_dim": D, "dtype": "float32"}, f)


def synthetic_docs(n_docs: int, rng: np.random.Generator, mix: np.ndarray) -> list[np.ndarray]:
    docs = []
    for _ in range(n_docs):
        L = int(rng.integers(150, 300))
        x = np.zeros((L, D))
        x[0] = rng.normal(size=D)
        for t in range(L - 1):
            x1, x2 = x[t, 0], x[t, 1]
            nxt = np.empty(D)
            nxt[0] = 0.95 * x1 + 0.3 * rng.normal()
            nxt[1] = 0.9 * x2 + 0.5 * (x1 ** 2 - 1.0)
            nxt[2:] = 0.8 * x[t, 2:] + 0.2 * rng.normal(size=D - 2)
            x[t + 1] = nxt
        docs.append(x @ mix.T + 0.3)
    return docs


@pytest.fixture(scope="module")
def synth_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("koopman_synth")
    rng = np.random.default_rng(0)
    mix = np.linalg.qr(rng.normal(size=(D, D)))[0]
    z_dir = root / "z"
    write_z_split(z_dir / "train", synthetic_docs(60, rng, mix))
    write_z_split(z_dir / "val", synthetic_docs(12, rng, mix))
    test_docs = synthetic_docs(12, rng, mix)
    write_z_split(z_dir / "test", test_docs, prompt_lengths=[5] * len(test_docs))
    return root


def tiny_ret_checkpoint(root: Path, cache_dir: Path | None = None, llm_name: str = "EleutherAI/pythia-160m",
                        layer: int = 6) -> Path:
    """A random 1-layer causal transformer RET encoder (d_h=16 -> z_dim=8) saved like scripts/train.py."""
    from ret.config import ExperimentConfig, LLMConfig, DataConfig, EncoderConfig, PredictorConfig
    from ret.models.encoders import build_encoder
    from ret.models.predictor import Predictor
    from ret.utils.checkpointing import save_checkpoint

    cfg = ExperimentConfig(
        llm=LLMConfig(name=llm_name, layer_index=layer, hidden_dim=16, dtype="float32"),
        data=DataConfig(mode="cached", hf_dataset=None, cached_hidden_states_dir=str(cache_dir) if cache_dir else None),
        encoder=EncoderConfig(type="transformer", z_dim=D, num_heads=2, num_layers=1, proj_position="end"),
        predictor=PredictorConfig(mode="autonomous", hidden_dim=32, num_layers=2),
    )
    torch.manual_seed(0)
    enc = build_encoder(16, cfg.encoder)
    pred = Predictor(D, cfg.predictor)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()))
    out = root / "ret_run"
    save_checkpoint(out / "checkpoint_final.pt", step=1, encoder=enc, predictor=pred, optimizer=opt, config=cfg)
    return out / "checkpoint_final.pt"


def write_cache_split(split_dir: Path, docs_h: list[np.ndarray]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    h = np.concatenate(docs_h).astype(np.float16)
    lengths = np.array([len(x) for x in docs_h], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    h.tofile(split_dir / "hidden_states.bin")
    np.arange(len(h), dtype=np.int32).tofile(split_dir / "input_ids.bin")
    np.save(split_dir / "offsets.npy", offsets)
    np.save(split_dir / "lengths.npy", lengths)
    with open(split_dir / "metadata.json", "w") as f:
        json.dump({"model_name": "EleutherAI/pythia-160m", "layer_index": 6, "hidden_dim": 16,
                   "dtype": "float16", "num_docs": len(docs_h), "total_tokens": int(len(h))}, f)


def write_cfg(root: Path, z_dir: Path, **over) -> Path:
    cfg = {
        "experiment_dir": str(root / "exp"),
        "data": {"z_dir": str(z_dir), "max_val_docs": None},
        "clusters": {"source": "kmeans", "K": 8, "kmeans_sample": 5000},
        "model": {"lift_dim": 40, "psi_hidden": [64]},
        "loss": {"horizon_schedule": [[0, 1], [150, 4]]},
        "train": {"steps": 400, "batch_size": 128, "eval_every": 100, "eval_max_starts": 1500,
                  "log_every": 100, "warmup_steps": 20, "device": "cpu", "patience": 20},
        "eval": {"split": "test", "max_docs": None, "horizons": [1, 2, 4, 8, 16], "max_starts": 3000,
                 "analysis_max_tokens": 3000, "n_slow_modes": 6},
        "baselines": {
            "run": ["mean", "persistence", "dmd", "edmd", "ulam", "mlp", "direct"],
            "edmd": [{"features": "rff", "num_features": 64}, {"features": "rbf", "num_features": 24}],
            "mlp": {"hidden_dims": [32], "train_horizons": [1, 4], "steps": 300, "eval_every": 100,
                    "patience": 5, "batch_size": 128},
            "direct": {"ks": [1, 4, 16], "hidden_dim": 32, "steps": 300, "eval_every": 100, "batch_size": 128},
        },
    }
    for k, v in over.items():
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = v
    path = root / "cfg.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_roundtrip(tmp_path):
    from koopman.config import load_koopman_config, save_koopman_config
    p = write_cfg(tmp_path, tmp_path / "z")
    cfg = load_koopman_config(p, ["model.lift_dim=64", "loss.horizon_schedule=[[0,1],[10,8]]"])
    assert cfg.model.lift_dim == 64
    assert cfg.loss.horizon_schedule == [[0, 1], [10, 8]]
    assert cfg.baselines.edmd[1].features == "rbf"
    save_koopman_config(cfg, tmp_path / "out.yaml")
    cfg2 = load_koopman_config(tmp_path / "out.yaml")
    assert asdict(cfg2) == asdict(cfg)
    with pytest.raises(ValueError):
        load_koopman_config(p, ["model.bogus=1"])
    with pytest.raises(ValueError):
        load_koopman_config(p, ["model.decoder=linear_fixed", "model.include_state=false"])


def test_window_index_stays_in_doc(synth_root):
    from koopman.data import WindowIndex, ZSplit
    zs = ZSplit(synth_root / "z" / "test")
    for excl in (False, True):
        wi = WindowIndex(zs, 16, exclude_prompt=excl)
        starts = wi.all_starts()
        doc = np.searchsorted(zs.offsets, starts, side="right") - 1
        t = starts - zs.offsets[doc]
        assert (t + 16 < zs.lengths[doc]).all()
        if excl:
            assert (t >= 5).all()
        assert len(starts) == sum(max(0, l - (5 if excl else 0) - 16) for l in zs.lengths)
    # memmap path returns the same windows as the in-memory path
    zm = ZSplit(synth_root / "z" / "test", in_memory=False)
    st = WindowIndex(zs, 4).all_starts(max_starts=50)
    assert torch.allclose(zs.window(st, 5), zm.window(st, 5))


def test_metrics_reference_values(synth_root):
    from koopman.data import ZSplit, compute_stats
    from koopman.forecasters import MeanForecaster, PersistenceForecaster
    from koopman.metrics import evaluate
    zs = ZSplit(synth_root / "z" / "test")
    mean, std = compute_stats(zs)
    pers = PersistenceForecaster(D).set_stats(mean, std)
    res = evaluate(pers, zs, [1, 4], device="cpu")
    assert abs(res[1]["skill"]) < 1e-9 and abs(res[4]["skill"]) < 1e-9
    m = MeanForecaster(D).set_stats(mean, std)
    one = evaluate(m, zs, [1], max_k=1, device="cpu")
    assert abs(one[1]["r2"]) < 0.05  # mean of the same split -> R^2 ~ 0


def test_export_z_cached_matches_per_doc_encoding(tmp_path):
    from ret.utils.checkpointing import load_encoder
    sys.path.insert(0, str(ROOT / "scripts" / "koopman"))
    import export_z

    rng = np.random.default_rng(1)
    cache = tmp_path / "cache"
    docs_h = {s: [rng.normal(size=(int(rng.integers(5, 40)), 16)) for _ in range(n)]
              for s, n in (("train", 7), ("val", 3))}
    for s, dh in docs_h.items():
        write_cache_split(cache / s, dh)
    ckpt = tiny_ret_checkpoint(tmp_path, cache_dir=cache)
    out = tmp_path / "z"
    export_z.main(["--checkpoint", str(ckpt), "--out-dir", str(out), "--splits", "train", "val",
                   "--device", "cpu", "--batch-size", "3", "--prompt-length-source", "none",
                   "--max-docs", "train=5"])
    meta = json.load(open(out / "export_meta.json"))
    assert meta["splits"]["train"]["num_docs"] == 5 and meta["splits"]["val"]["num_docs"] == 3

    encoder, _, _ = load_encoder(ckpt, device="cpu")
    from koopman.data import ZSplit
    for s, n in (("train", 5), ("val", 3)):
        zs = ZSplit(out / s)
        assert zs.num_docs == n
        for i in range(n):
            h = torch.from_numpy(docs_h[s][i].astype(np.float16).astype(np.float32))[None]
            with torch.no_grad():
                z_ref, _ = encoder(h)
            assert torch.allclose(zs.doc(i), z_ref[0], atol=1e-4), (s, i)


def run_script(name: str, *args: str) -> None:
    """Run a scripts/koopman entry point in a fresh interpreter, as on the cluster."""
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "koopman" / name), *args],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, f"{name} {args} failed:\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"


WORDS = [f"w{i}" for i in range(40)]


def tiny_llm(root: Path) -> Path:
    """A random 3-layer GPT-NeoX (hidden 16) + word-level tokenizer, built offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(WORDS)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[PAD]")
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=3,
                                             num_attention_heads=2, intermediate_size=32,
                                             max_position_embeddings=256))
    out = root / "tiny_llm"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


def write_arrow(path: Path, texts: list[str], prompt_lengths: list[int]) -> None:
    from datasets import Dataset
    Dataset.from_dict({"text": texts, "prompt_length": prompt_lengths}).save_to_disk(str(path))


def reference_z(llm_dir: Path, ckpt: Path, ids: np.ndarray) -> torch.Tensor:
    """Encode one document alone (no padding) through the LLM layer and the RET encoder."""
    from ret.config import LLMConfig
    from ret.extraction.hooks import HiddenStateExtractor
    from ret.utils.checkpointing import load_encoder
    ex = HiddenStateExtractor(LLMConfig(name=str(llm_dir), layer_index=1, hidden_dim=16, dtype="float32"),
                              device="cpu")
    enc, _, _ = load_encoder(ckpt, device="cpu")
    with torch.no_grad():
        h = ex.extract(torch.from_numpy(ids.astype(np.int64))[None])
        z, _ = enc(h)
    ex.cleanup()
    return z[0]


@pytest.mark.parametrize("mode", ["document", "chunked"])
def test_export_z_with_llm(tmp_path, mode):
    pytest.importorskip("datasets")
    sys.path.insert(0, str(ROOT / "scripts" / "koopman"))
    import export_z
    from koopman.data import ZSplit

    rng = np.random.default_rng(2)
    llm = tiny_llm(tmp_path)
    ckpt = tiny_ret_checkpoint(tmp_path, cache_dir=tmp_path / "unused", llm_name=str(llm), layer=1)
    texts = {s: [" ".join(rng.choice(WORDS, size=int(rng.integers(8, 90)))) for _ in range(n)]
             for s, n in (("train", 9), ("val", 4))}
    pls = {s: [int(rng.integers(0, 6)) for _ in t] for s, t in texts.items()}
    data = tmp_path / "arrow"
    for s in texts:
        write_arrow(data / s, texts[s], pls[s])
    if mode == "document":
        data_cfg = {"llm": {"name": str(llm), "layer_index": 1, "hidden_dim": 16, "dtype": "float32"},
                    "data": {"mode": "document", "hf_dataset": None, "local_dir": str(data),
                             "max_seq_length": 64, "scaffolding": "raw"}}
        splits, max_docs = ["train", "val"], ["train=7"]
    else:
        data_cfg = {"llm": {"name": str(llm), "layer_index": 1, "hidden_dim": 16, "dtype": "float32"},
                    "data": {"mode": "chunked", "hf_dataset": None, "local_dir": str(data / "train"),
                             "chunk_length": 16, "scaffolding": "raw"}}
        splits, max_docs = ["train", "val", "test"], ["train=5", "val=3", "test=3"]
    with open(tmp_path / "data_cfg.yaml", "w") as f:
        yaml.safe_dump(data_cfg, f)
    out = tmp_path / "z"
    export_z.main(["--checkpoint", str(ckpt), "--data-config", str(tmp_path / "data_cfg.yaml"),
                   "--out-dir", str(out), "--splits", *splits, "--max-docs", *max_docs, "--device", "cpu",
                   "--batch-size", "3", "--chunked-skip-docs", "4"])

    tok = __import__("transformers").AutoTokenizer.from_pretrained(str(llm))
    if mode == "document":
        for s, n in (("train", 7), ("val", 4)):
            zs = ZSplit(out / s)
            ids_all = np.fromfile(out / s / "input_ids.bin", dtype=np.int32)
            assert zs.num_docs == n
            for i in range(n):
                ids = np.asarray(tok.encode(texts[s][i], add_special_tokens=False)[:64])
                o, l = int(zs.offsets[i]), int(zs.lengths[i])
                assert l == len(ids) and np.array_equal(ids_all[o:o + l], ids)
                assert zs.prompt_lengths[i] == min(pls[s][i], l)
                assert torch.allclose(zs.doc(i), reference_z(llm, ckpt, ids), atol=1e-4)
    else:
        seen = {}
        for s, n in (("train", 5), ("val", 3), ("test", 3)):
            zs = ZSplit(out / s)
            ids_all = np.fromfile(out / s / "input_ids.bin", dtype=np.int32)
            assert zs.num_docs == n and (zs.lengths == 16).all()
            seen[s] = [tuple(ids_all[int(o):int(o) + 16]) for o in zs.offsets]
            for i in range(n):
                o = int(zs.offsets[i])
                assert torch.allclose(zs.doc(i), reference_z(llm, ckpt, ids_all[o:o + 16]), atol=1e-4)
        # held-out chunks come from rows after the skipped ones, val and test do not overlap
        heldout_stream = sum((tok.encode(t, add_special_tokens=False) for t in texts["train"][4:]), [])
        assert list(seen["val"][0]) == heldout_stream[:16]
        assert not set(seen["val"]) & set(seen["test"])


def test_export_z_cached_prompt_lengths(tmp_path, caplog):
    pytest.importorskip("datasets")
    sys.path.insert(0, str(ROOT / "scripts" / "koopman"))
    import export_z
    from koopman.data import ZSplit
    from transformers import AutoTokenizer

    rng = np.random.default_rng(3)
    llm = tiny_llm(tmp_path)
    tok = AutoTokenizer.from_pretrained(str(llm))
    texts = [" ".join(rng.choice(WORDS, size=int(rng.integers(8, 30)))) for _ in range(5)]
    pls = [3, 0, 5, 2, 7]
    # cache layout <dataset>/<model>/layer1 next to the Arrow dataset <dataset>/<model>/<split>
    model_dir = tmp_path / "ds" / "Tiny"
    write_arrow(model_dir / "val", texts, pls)
    ids = [np.asarray(tok.encode(t, add_special_tokens=False)) for t in texts]
    split_dir = model_dir / "layer1" / "val"
    write_cache_split(split_dir, [rng.normal(size=(len(x), 16)) for x in ids])
    np.concatenate(ids).astype(np.int32).tofile(split_dir / "input_ids.bin")
    meta = json.load(open(split_dir / "metadata.json"))
    meta.update(model_name=str(llm), layer_index=1)
    json.dump(meta, open(split_dir / "metadata.json", "w"))
    ckpt = tiny_ret_checkpoint(tmp_path, cache_dir=model_dir / "layer1", llm_name=str(llm), layer=1)

    export_z.main(["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "z"), "--splits", "val", "--device", "cpu"])
    assert ZSplit(tmp_path / "z" / "val").prompt_lengths.tolist() == pls  # 'auto' found the Arrow dataset

    write_arrow(model_dir / "val", texts[::-1], pls)  # misaligned rows -> warning, zeros
    export_z.main(["--checkpoint", str(ckpt), "--out-dir", str(tmp_path / "z2"), "--splits", "val", "--device", "cpu"])
    assert ZSplit(tmp_path / "z2" / "val").prompt_lengths.tolist() == [0] * 5


@pytest.fixture(scope="module")
def pipeline(synth_root):
    """Run baselines + four deep Koopman variants + evaluation once; tests inspect the outputs."""
    ckpt = tiny_ret_checkpoint(synth_root)
    cfg = str(write_cfg(synth_root, synth_root / "z", ret_checkpoint=str(ckpt)))
    run_script("fit_baselines.py", cfg, "baselines.run=[mean,persistence,dmd,edmd,ulam,mlp,direct,ret_predictor]")
    run_script("train.py", cfg, "run_name=dense")
    run_script("train.py", cfg, "run_name=lstsq", "model.operator=lstsq", "train.steps=200")
    run_script("train.py", cfg, "run_name=blockdiag", "model.operator=blockdiag", "model.include_state=false",
               "model.decoder=linear_learned", "train.steps=200")
    run_script("train.py", cfg, "run_name=mlpdec", "model.decoder=mlp", "model.decoder_hidden=[32]",
               "train.steps=150")
    run_script("eval_horizon.py", cfg)
    with open(synth_root / "exp" / "eval" / "test" / "results.json") as f:
        return json.load(f)


def test_pipeline_outputs(pipeline, synth_root):
    models = pipeline["models"]
    expected = {"mean", "persistence", "dmd", "edmd_rff64", "edmd_rbf24", "ulam_kmeans8", "mlp_h32_H1",
                "mlp_h32_H4", "direct_mlp", "ret_T", "dense", "lstsq", "blockdiag", "mlpdec"}
    assert expected <= set(models), set(models) ^ expected
    out = synth_root / "exp" / "eval" / "test"
    for f in ("summary.csv", "horizon_long.csv", "r2_vs_k.png", "skill_vs_k.png", "cluster_acc_vs_k.png",
              "summary.md", "eigs_dense.png"):
        assert (out / f).exists(), f
    assert "analysis" in models["dense"] and "analysis" in models["dmd"]
    assert set(models["direct_mlp"]["horizon"]) == {"1", "4", "16"}
    for name in ("dense", "lstsq", "blockdiag", "mlpdec"):
        assert (synth_root / "exp" / "runs" / name / "val_eval.json").exists()
        for k, r in models[name]["horizon"].items():
            assert np.isfinite(r["r2"]), (name, k)


def test_lifting_beats_dmd_on_nonlinear_system(pipeline):
    m = pipeline["models"]
    r1 = {n: m[n]["r2_onestep_allpairs"] for n in m}
    # persistence is the floor; DMD is linear in z and misses the x1^2 term
    assert r1["dmd"] > r1["persistence"]
    assert r1["edmd_rff64"] > r1["dmd"] + 0.005, r1
    assert r1["dense"] > r1["dmd"] + 0.005, r1
    assert m["blockdiag"]["analysis"]["spectral_radius"] <= 1.0 + 1e-6


def test_forecaster_save_load_roundtrip(pipeline, synth_root):
    from koopman.data import ZSplit
    from koopman.forecasters import load_forecaster
    zs = ZSplit(synth_root / "z" / "test")
    z0 = zs.z[:64]
    for p in sorted((synth_root / "exp").rglob("model.pt")):
        m1, _ = load_forecaster(p)
        m2, _ = load_forecaster(p)
        ks = m1.available_ks([1, 4])
        a, b = m1.predict(z0, ks), m2.predict(z0, ks)
        for k in ks:
            assert torch.allclose(a[k], b[k]), p
