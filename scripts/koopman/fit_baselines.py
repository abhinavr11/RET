#!/usr/bin/env python3
"""Fit the Phase-2 baselines on exported z trajectories.

Each baseline is saved to <experiment_dir>/baselines/<name>/model.pt (+ summary.json
with validation R^2) and is picked up later by eval_horizon.py.

    python scripts/koopman/fit_baselines.py configs/koopman/gptoss20b.yaml
    python scripts/koopman/fit_baselines.py configs/koopman/gptoss20b.yaml --only dmd,edmd
    python scripts/koopman/fit_baselines.py configs/koopman/gptoss20b.yaml baselines.mlp.train_horizons=[1]

Baselines (names in config ``baselines.run``):
    mean, persistence    references (R^2 = 0, and the skill denominator)
    dmd                  z_{t+1} = A z_t + b by ridge least squares
    edmd                 fixed dictionaries from ``baselines.edmd`` (rff / rbf), K by ridge LS
    ulam                 Markov chain on cluster labels (needs clusters.source != none)
    ret_predictor        RET's own T from the checkpoint, rolled out
    mlp                  closure-probe MLPs (hidden_dims x train_horizons); H=1 is the paper protocol,
                         H>1 trains through an H-step rollout (the fair multi-step competitor)
    direct               one MLP per horizon k, z_t -> z_{t+k}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koopman.baselines import (  # noqa: E402
    dim_weights, fit_linear_koopman, fit_ulam, init_dictionary, summarize, train_direct, train_mlp,
)
from koopman.clusters import build_assigner  # noqa: E402
from koopman.config import add_config_args, load_koopman_config, save_koopman_config  # noqa: E402
from koopman.data import compute_stats, open_split, read_export_meta  # noqa: E402
from koopman.forecasters import (  # noqa: E402
    DirectMLPForecaster, LinearKoopmanForecaster, MeanForecaster, MLPForecaster, PersistenceForecaster,
    RETPredictorForecaster, param_count,
)
from koopman.metrics import evaluate, format_row  # noqa: E402

logger = logging.getLogger("fit_baselines")


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--only", default=None, help="comma-separated subset of baselines.run")
    p.add_argument("--force", action="store_true", help="refit baselines that already exist")
    args = p.parse_args(argv)
    cfg = load_koopman_config(args.config, args.overrides)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed)
    dc, bc = cfg.data, cfg.baselines
    exp_dir = Path(cfg.experiment_dir)
    base_dir = exp_dir / "baselines"
    base_dir.mkdir(parents=True, exist_ok=True)
    save_koopman_config(cfg, exp_dir / "baselines_config.yaml")

    train = open_split(dc.z_dir, dc.train_split, dc.max_train_docs, dc.in_memory, dc.data_device)
    val = open_split(dc.z_dir, dc.val_split, dc.max_val_docs, dc.in_memory, dc.data_device)
    mean, std = compute_stats(train)
    torch.save({"mean": mean, "std": std, "max_train_docs": dc.max_train_docs}, exp_dir / "stats.pt")
    w = dim_weights(std, cfg.loss.loss_space)
    d = train.d
    logger.info("Train %s | Val %s | device %s", train, val, device)

    todo = [s.strip() for s in args.only.split(",")] if args.only else list(bc.run)
    unknown = set(todo) - {"mean", "persistence", "dmd", "edmd", "ulam", "ret_predictor", "mlp", "direct"}
    if unknown:
        raise ValueError(f"Unknown baselines: {sorted(unknown)}")
    quick_ks = sorted(set(cfg.eval.horizons))
    rows = []

    def exists(name: str) -> bool:
        return (base_dir / name / "model.pt").exists() and not args.force

    def finish(name: str, model, info: dict | None = None, t0: float | None = None) -> None:
        res = evaluate(model, val, quick_ks, max_starts=cfg.train.eval_max_starts, exclude_prompt=dc.exclude_prompt,
                       device=device, seed=cfg.eval.seed)
        info = dict(info or {})
        info.update({"name": name, "num_parameters": param_count(model),
                     "fit_seconds": round(time.time() - t0, 1) if t0 else None})
        model.save(base_dir / name / "model.pt", extra=info)
        with open(base_dir / name / "summary.json", "w") as f:
            json.dump({**{k: v for k, v in info.items() if k != "history"}, "val": summarize(res),
                       "history": info.get("history")}, f, indent=2)
        rows.append(format_row(name, res, quick_ks))
        logger.info("%s\n%s", f"{'':<40s} " + " ".join(f"{('k=' + str(k)):>8s}" for k in quick_ks), rows[-1])

    if "mean" in todo and not exists("mean"):
        finish("mean", MeanForecaster(d).to(device).set_stats(mean, std))
    if "persistence" in todo and not exists("persistence"):
        finish("persistence", PersistenceForecaster(d).to(device).set_stats(mean, std))

    if "dmd" in todo and not exists("dmd"):
        t0 = time.time()
        m = LinearKoopmanForecaster(d).to(device).set_stats(mean, std)
        info = fit_linear_koopman(m, train, ridge=bc.dmd_ridge, exclude_prompt=dc.exclude_prompt,
                                  max_pairs=bc.max_fit_pairs, device=device, seed=cfg.train.seed)
        finish("dmd", m, info, t0)

    if "edmd" in todo:
        sample = None
        for spec in bc.edmd:
            name = spec.name or f"edmd_{spec.features}{spec.num_features}"
            if exists(name):
                continue
            t0 = time.time()
            if sample is None:
                sample = (train.sample_tokens(100_000, seed=cfg.train.seed).to(device) - mean.to(device)) / std.to(device)
            m = LinearKoopmanForecaster(d, spec.features, spec.num_features).to(device).set_stats(mean, std)
            init_dictionary(m, sample, spec.bandwidth_scale, seed=cfg.train.seed)
            info = fit_linear_koopman(m, train, ridge=spec.ridge, exclude_prompt=dc.exclude_prompt,
                                      max_pairs=bc.max_fit_pairs, device=device, seed=cfg.train.seed)
            finish(name, m, {**info, "spec": vars(spec)}, t0)

    if "ulam" in todo:
        name = f"ulam_{cfg.clusters.source}{cfg.clusters.K}"
        if cfg.clusters.source == "none":
            logger.warning("Skipping ulam: clusters.source=none")
        elif not exists(name):
            t0 = time.time()
            assigner = build_assigner(cfg, train, mean, std, exp_dir / "clusters_kmeans.pt", str(device))
            m = fit_ulam(train, assigner, mean, std, smoothing=bc.ulam.smoothing, exclude_prompt=dc.exclude_prompt,
                         max_pairs=bc.max_fit_pairs, device=device, seed=cfg.train.seed)
            finish(name, m, {"clusters": cfg.clusters.source}, t0)

    if "ret_predictor" in todo and not exists("ret_T"):
        ckpt = cfg.ret_checkpoint or read_export_meta(dc.z_dir).get("checkpoint")
        if not ckpt or not Path(ckpt).exists():
            logger.warning("Skipping ret_predictor: RET checkpoint not found (%s); set ret_checkpoint", ckpt)
        else:
            try:
                m = RETPredictorForecaster.from_checkpoint(ckpt).to(device).set_stats(mean, std)
                finish("ret_T", m, {"checkpoint": str(ckpt)})
            except ValueError as e:
                logger.warning("Skipping ret_predictor: %s", e)

    if "mlp" in todo:
        mc = bc.mlp
        for H in mc.train_horizons:
            for hdim in mc.hidden_dims:
                name = f"mlp_h{hdim}_H{H}"
                if exists(name):
                    continue
                t0 = time.time()
                m = MLPForecaster(d, hdim, mc.num_layers).to(device).set_stats(mean, std)
                info = train_mlp(m, train, val, H=int(H), rho=mc.rho, w=w, steps=mc.steps, batch_size=mc.batch_size,
                                 lr=mc.lr, eval_every=mc.eval_every, patience=mc.patience,
                                 exclude_prompt=dc.exclude_prompt, eval_max_starts=cfg.train.eval_max_starts,
                                 device=device, seed=cfg.train.seed, label=name)
                finish(name, m, {**info, "train_horizon": int(H), "hidden_dim": hdim}, t0)

    if "direct" in todo and not exists("direct_mlp"):
        dcfg = bc.direct
        t0 = time.time()
        m = DirectMLPForecaster(d, dcfg.ks, dcfg.hidden_dim, dcfg.num_layers).to(device).set_stats(mean, std)
        info = train_direct(m, train, val, w=w, steps=dcfg.steps, batch_size=dcfg.batch_size, lr=dcfg.lr,
                            eval_every=dcfg.eval_every, patience=dcfg.patience, exclude_prompt=dc.exclude_prompt,
                            eval_max_starts=cfg.train.eval_max_starts, device=device, seed=cfg.train.seed)
        finish("direct_mlp", m, info, t0)

    if rows:
        header = f"{'val R2(k)':<40s} " + " ".join(f"{('k=' + str(k)):>8s}" for k in quick_ks)
        print("\n" + header + "\n" + "\n".join(rows))
    logger.info("Baselines saved under %s", base_dir)


if __name__ == "__main__":
    main()
