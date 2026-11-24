"""Training loop for the deep Koopman model (Phase 3)."""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from koopman.baselines import cosine_lr, dim_weights, summarize
from koopman.config import (
    KoopmanConfig, default_select_ks, final_horizon, horizon_at, save_koopman_config,
)
from koopman.data import WindowIndex, WindowSampler, ZSplit, compute_stats, open_split
from koopman.losses import koopman_loss
from koopman.metrics import evaluate, format_row, select_score
from koopman.models import DeepKoopmanForecaster

logger = logging.getLogger(__name__)


def run_name_for(cfg: KoopmanConfig) -> str:
    if cfg.run_name:
        return cfg.run_name
    m = cfg.model
    return f"N{m.lift_dim}_{m.operator}_{m.decoder}_H{final_horizon(cfg)}_s{cfg.train.seed}"


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def refit_lstsq(model: DeepKoopmanForecaster, train: ZSplit, *, exclude_prompt: bool, max_pairs: int | None,
                device, batch_size: int = 8192, seed: int = 0) -> None:
    """Refit K by ridge least squares on train pairs using the final (eval-mode) lifting."""
    model.eval()
    N = model.N
    A = torch.zeros(N, N, dtype=torch.float64, device=device)
    Bm = torch.zeros(N, N, dtype=torch.float64, device=device)
    starts = WindowIndex(train, 1, exclude_prompt).all_starts(1, max_pairs, seed)
    for i in range(0, len(starts), batch_size):
        W = train.gather(starts[i:i + batch_size], [0, 1]).to(device)
        G = model.lift(model.to_std(W[:, 0])).double()
        Gp = model.lift(model.to_std(W[:, 1])).double()
        A += G.T @ G
        Bm += G.T @ Gp
    model.operator.K.copy_(model.operator.solve(A, Bm).float())
    logger.info("Refit K on %d train pairs", len(starts))


def koopman_spectrum(model) -> dict:
    K = model.koopman_matrix().detach().double().cpu()
    idx = model.dynamic_index().cpu()
    Kdd = K.index_select(0, idx).index_select(1, idx)
    ev = torch.linalg.eigvals(Kdd).numpy()
    return {"eigvals": ev}


def train_deep_koopman(cfg: KoopmanConfig) -> Path:
    seed_all(cfg.train.seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() or cfg.train.device == "cpu" else "cpu")
    if str(device) != cfg.train.device:
        logger.warning("CUDA not available; training on %s", device)

    dc = cfg.data
    train = open_split(dc.z_dir, dc.train_split, dc.max_train_docs, dc.in_memory, dc.data_device)
    val = open_split(dc.z_dir, dc.val_split, dc.max_val_docs, dc.in_memory, dc.data_device)
    mean, std = compute_stats(train)
    w = dim_weights(std, cfg.loss.loss_space).to(device)

    run_name = run_name_for(cfg)
    run_dir = Path(cfg.experiment_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_koopman_config(cfg, run_dir / "config.yaml")

    model = DeepKoopmanForecaster(train.d, asdict(cfg.model)).to(device)
    model.set_stats(mean, std)
    logger.info("Run %s: N=%d (psi dim %d), operator=%s, decoder=%s, %d params",
                run_name, model.N, model.lifting.m, model.operator_kind, model.decoder_kind,
                model.num_parameters())

    op_params = list(model.operator.parameters())
    op_ids = {id(p) for p in op_params}
    other = [p for p in model.parameters() if id(p) not in op_ids]
    groups = [{"params": other, "weight_decay": cfg.train.weight_decay}]
    if op_params:
        groups.append({"params": op_params, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=cfg.train.lr)

    tc = cfg.train
    sampler = WindowSampler(train, dc.exclude_prompt, seed=tc.seed)
    H_final = final_horizon(cfg)
    final_start = max(int(s[0]) for s in cfg.loss.horizon_schedule)
    select_ks = tc.select_ks or default_select_ks(H_final)

    log_path = run_dir / "train_log.csv"
    log_f = open(log_path, "w", newline="")
    log_w = None
    best_score, best_step, bad = -float("inf"), -1, 0
    best_state = None
    t0 = time.time()
    run = {}
    n_run = 0

    for step in range(tc.steps):
        H = horizon_at(step, cfg.loss.horizon_schedule)
        lr = tc.lr * cosine_lr(step, tc.steps, tc.warmup_steps, tc.lr_min_fraction)
        for gp in opt.param_groups:
            gp["lr"] = lr
        model.train()
        W = sampler.sample(tc.batch_size, H, device)
        loss, m = koopman_loss(model, model.to_std(W), H, cfg.loss, w)
        if not torch.isfinite(loss):
            torch.save(model.state_dict(), run_dir / "nan_state.pt")
            raise RuntimeError(f"Non-finite loss at step {step} (H={H}): {m}. State saved to nan_state.pt; "
                               "try a lower lr, larger loss.w_spec, or operator=blockdiag.")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip) if tc.grad_clip else torch.tensor(0.)
        opt.step()
        for k, v in m.items():
            run[k] = run.get(k, 0.0) + v
        n_run += 1

        if (step + 1) % tc.log_every == 0:
            avg = {k: v / n_run for k, v in run.items()}
            logger.info("step %6d H=%2d lr=%.2e loss=%.5f pred=%.5f lin=%.5f %s| %.0fs", step + 1, H, lr,
                        avg["total"], avg["pred"], avg["lin"],
                        f"spec={avg['spec_bound']:.4f} " if "spec_bound" in avg else "", time.time() - t0)

        do_eval = (step + 1) % tc.eval_every == 0 or step + 1 == tc.steps
        if do_eval:
            res = evaluate(model, val, select_ks, max_starts=tc.eval_max_starts, exclude_prompt=dc.exclude_prompt,
                           device=device, seed=cfg.eval.seed)
            score = select_score(res, select_ks)
            avg = {k: v / max(n_run, 1) for k, v in run.items()}
            row = {"step": step + 1, "H": H, "lr": lr, "grad_norm": float(gnorm),
                   **{f"train_{k}": v for k, v in avg.items()},
                   "val_score": score, **{f"val_r2_k{k}": res[k]["r2"] for k in select_ks},
                   **{f"val_skill_k{k}": res[k]["skill"] for k in select_ks}}
            if log_w is None:
                log_w = csv.DictWriter(log_f, fieldnames=list(row.keys()), extrasaction="ignore")
                log_w.writeheader()
            log_w.writerow(row)
            log_f.flush()
            run, n_run = {}, 0
            improved = score > best_score + 1e-6
            if improved:
                best_score, best_step = score, step + 1
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.save(run_dir / "model.pt", extra={"run_name": run_name, "step": step + 1,
                                                        "val_score": score, "select_ks": select_ks})
            if step + 1 > final_start:
                bad = 0 if improved else bad + 1
            logger.info("  eval step %d: val R2 %s  score %.4f (best %.4f @ %d)", step + 1,
                        " ".join(f"k{k}={res[k]['r2']:.4f}" for k in select_ks), score, best_score, best_step)
            if bad >= tc.patience:
                logger.info("Early stopping at step %d", step + 1)
                break
    log_f.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    refit_info = None
    if model.operator_kind == "lstsq" and tc.final_lstsq_refit:
        before = evaluate(model, val, select_ks, max_starts=tc.eval_max_starts, exclude_prompt=dc.exclude_prompt,
                          device=device, seed=cfg.eval.seed)
        K_ema = model.operator.K.clone()
        refit_lstsq(model, train, exclude_prompt=dc.exclude_prompt, max_pairs=tc.refit_max_pairs, device=device,
                    seed=tc.seed)
        after = evaluate(model, val, select_ks, max_starts=tc.eval_max_starts, exclude_prompt=dc.exclude_prompt,
                         device=device, seed=cfg.eval.seed)
        s_before, s_after = select_score(before, select_ks), select_score(after, select_ks)
        if s_after < s_before:
            model.operator.K.copy_(K_ema)
        refit_info = {"score_ema_K": s_before, "score_refit_K": s_after, "used": "refit" if s_after >= s_before else "ema"}
        logger.info("lstsq final refit: EMA K %.4f vs refit K %.4f -> using %s", s_before, s_after, refit_info["used"])

    ks_all = sorted(set(cfg.eval.horizons))
    val_res = evaluate(model, val, ks_all, max_starts=tc.eval_max_starts, exclude_prompt=dc.exclude_prompt,
                       device=device, seed=cfg.eval.seed)
    spec = koopman_spectrum(model)
    ev = spec["eigvals"]
    np.savez(run_dir / "spectrum.npz", eigvals=ev)
    summary = {
        "run_name": run_name,
        "best_step": best_step,
        "best_val_score": best_score,
        "select_ks": select_ks,
        "num_parameters": model.num_parameters(),
        "N": model.N,
        "spectral_radius": float(np.abs(ev).max()) if ev.size else None,
        "lstsq_refit": refit_info,
        "val": summarize(val_res),
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(run_dir / "val_eval.json", "w") as f:
        json.dump(summary, f, indent=2)
    model.save(run_dir / "model.pt", extra={"run_name": run_name, "step": best_step, "val_score": best_score,
                                            "select_ks": select_ks})
    logger.info("Final val R2(k):\n%s\n%s", f"{'':<40s} " + " ".join(f"{('k=' + str(k)):>8s}" for k in ks_all),
                format_row(run_name, val_res, ks_all))
    logger.info("Saved %s", run_dir)
    return run_dir
