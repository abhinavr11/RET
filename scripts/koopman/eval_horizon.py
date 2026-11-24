#!/usr/bin/env python3
"""Evaluate all fitted models on identical held-out starts (Phase 4).

    python scripts/koopman/eval_horizon.py configs/koopman/gptoss20b.yaml
    python scripts/koopman/eval_horizon.py configs/koopman/gptoss20b.yaml eval.split=val \\
        --models outputs/koopman/GptOss20B_numina_l11/runs/N512_dense_linear_fixed_H32_s0

By default every <experiment_dir>/baselines/*/model.pt and runs/*/model.pt is evaluated.

Outputs in <experiment_dir>/eval/<split>[_<tag>]/:
    results.json          everything (one-step all-pairs, horizon metrics, Koopman analysis)
    summary.csv           one row per model
    horizon_long.csv      model, k, metric, value
    r2_vs_k.png, skill_vs_k.png, cluster_acc_vs_k.png, group_acc_vs_k.png, cos_vs_k.png
    eigs_<model>.png, eta2_<model>.png   (Koopman-type models)

Metrics
    r2_onestep_allpairs   pooled R^2 over every (z_t, z_{t+1}) pair of the split — the paper's closure number
    r2 / skill / cos / norm_ratio / cluster_acc / group_acc at each k, on starts whose whole window
    t..t+max(k) lies in one document (strided by eval.stride, capped by eval.max_starts)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koopman.analysis import koopman_analysis  # noqa: E402
from koopman.baselines import summarize  # noqa: E402
from koopman.clusters import ClusterAssigner, build_assigner  # noqa: E402
from koopman.config import add_config_args, load_koopman_config  # noqa: E402
from koopman.data import compute_stats, open_split  # noqa: E402
from koopman.forecasters import is_koopman, load_forecaster, param_count  # noqa: E402
from koopman.metrics import evaluate  # noqa: E402
from koopman.plots import family_of, plot_eigs, plot_eta2, plot_metric_vs_k  # noqa: E402

logger = logging.getLogger("eval_horizon")


def discover(exp_dir: Path, given: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    if given:
        for g in given:
            p = Path(g)
            if p.is_dir():
                found = sorted(p.rglob("model.pt"))
                if not found:
                    logger.warning("No model.pt under %s", p)
                paths += found
            elif p.exists():
                paths.append(p)
            else:
                logger.warning("Model path not found: %s", p)
    else:
        for sub in ("baselines", "runs"):
            paths += sorted((exp_dir / sub).glob("*/model.pt"))
    return paths


def model_name(path: Path) -> str:
    return path.parent.name if path.name == "model.pt" else path.stem


def _clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--models", nargs="*", default=None, help="model.pt files or directories to search")
    p.add_argument("--tag", default=None, help="suffix for the output directory")
    p.add_argument("--no-analysis", action="store_true", help="skip the Koopman spectrum analysis")
    p.add_argument("--skip-onestep", action="store_true", help="skip the all-pairs one-step R^2")
    args = p.parse_args(argv)
    cfg = load_koopman_config(args.config, args.overrides)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    dc, ec = cfg.data, cfg.eval
    exp_dir = Path(cfg.experiment_dir)
    out_dir = exp_dir / "eval" / (ec.split + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    split = open_split(dc.z_dir, ec.split, ec.max_docs, dc.in_memory, dc.data_device)
    assigner = None
    if cfg.clusters.source == "ret":
        assigner = build_assigner(cfg, split, torch.zeros(split.d), torch.ones(split.d), None, str(device))
    elif cfg.clusters.source == "kmeans":
        # The k-means fallback is fitted on the train split (and cached), never on the eval split.
        cache = exp_dir / "clusters_kmeans.pt"
        if cache.exists():
            a = ClusterAssigner.load(cache)
            if a.K == cfg.clusters.K and a.d == split.d:
                assigner = a.to(device)
        if assigner is None:
            train = open_split(dc.z_dir, dc.train_split, dc.max_train_docs, dc.in_memory, dc.data_device)
            mean, std = compute_stats(train)
            if cache.exists():
                cache.unlink()
            assigner = build_assigner(cfg, train, mean, std, cache, str(device))
            del train

    paths = discover(exp_dir, args.models)
    if not paths:
        raise SystemExit(f"No models found under {exp_dir}/baselines or {exp_dir}/runs")
    ks = sorted(set(int(k) for k in ec.horizons))
    logger.info("Evaluating %d models on %s (%d docs, %d tokens), ks=%s", len(paths), ec.split, split.num_docs,
                split.num_tokens, ks)

    results: dict[str, dict] = {}
    for path in paths:
        name = model_name(path)
        t0 = time.time()
        model, extra = load_forecaster(path, device)
        r = {"path": str(path), "kind": model.kind, "family": family_of(name, model.kind),
             "num_parameters": param_count(model)}
        if not args.skip_onestep and 1 in model.available_ks([1]):
            one = evaluate(model, split, [1], max_k=1, stride=1, max_starts=None, batch_size=ec.batch_size,
                           exclude_prompt=dc.exclude_prompt, assigner=None, seed=ec.seed, device=device)
            r["r2_onestep_allpairs"] = one[1]["r2"]
            r["skill_onestep_allpairs"] = one[1]["skill"]
        hor = evaluate(model, split, ks, max_k=max(ks), stride=ec.stride, max_starts=ec.max_starts,
                       batch_size=ec.batch_size, exclude_prompt=dc.exclude_prompt, assigner=assigner,
                       seed=ec.seed, device=device)
        r["horizon"] = summarize(hor)
        if is_koopman(model) and not args.no_analysis and model.kind != "ulam":
            try:
                an = koopman_analysis(model, split, assigner, max_tokens=ec.analysis_max_tokens,
                                      n_modes=ec.n_slow_modes, exclude_prompt=dc.exclude_prompt, device=device,
                                      seed=ec.seed)
                r["analysis"] = an
                ev = np.array([complex(a, b) for a, b in an["eigvals"]])
                plot_eigs(ev, out_dir / f"eigs_{name}.png",
                          f"{name}: rho={an['spectral_radius']:.4f}, residual median={an['residual']['median']:.3f}")
                if "pca_eta2" in an:
                    plot_eta2(an["slow_modes"], an["pca_eta2"], out_dir / f"eta2_{name}.png", name)
            except Exception as e:  # analysis must never kill the evaluation
                logger.warning("Analysis failed for %s: %s", name, e)
        r["eval_seconds"] = round(time.time() - t0, 1)
        results[name] = r
        logger.info("%-36s R2(1,all)=%s  %s  (%.0fs)", name,
                    f"{r.get('r2_onestep_allpairs', float('nan')):.4f}",
                    " ".join(f"k{k}={hor[k]['r2']:.3f}" for k in hor), r["eval_seconds"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    meta = {"split": ec.split, "max_docs": ec.max_docs, "num_docs": split.num_docs, "num_tokens": split.num_tokens,
            "horizons": ks, "stride": ec.stride, "max_starts": ec.max_starts, "exclude_prompt": dc.exclude_prompt,
            "clusters": cfg.clusters.source}
    with open(out_dir / "results.json", "w") as f:
        json.dump(_clean({"meta": meta, "models": results}), f, indent=2)

    # summary.csv (wide) and horizon_long.csv
    metric_names = ["r2", "skill", "cos", "norm_ratio", "cluster_acc", "group_acc"]
    with open(out_dir / "horizon_long.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "family", "k", "metric", "value"])
        for name, r in results.items():
            for k, row in r["horizon"].items():
                for mname in metric_names:
                    if mname in row:
                        wr.writerow([name, r["family"], k, mname, row[mname]])
    cols = ["model", "family", "kind", "num_parameters", "r2_onestep_allpairs", "skill_onestep_allpairs"]
    cols += [f"r2_k{k}" for k in ks] + [f"skill_k{k}" for k in ks] + [f"cluster_acc_k{k}" for k in ks]
    cols += ["spectral_radius", "residual_median"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for name, r in results.items():
            row = {"model": name, "family": r["family"], "kind": r["kind"], "num_parameters": r["num_parameters"],
                   "r2_onestep_allpairs": r.get("r2_onestep_allpairs"),
                   "skill_onestep_allpairs": r.get("skill_onestep_allpairs")}
            for k in ks:
                h = r["horizon"].get(str(k), {})
                row[f"r2_k{k}"] = h.get("r2")
                row[f"skill_k{k}"] = h.get("skill")
                row[f"cluster_acc_k{k}"] = h.get("cluster_acc")
            if "analysis" in r:
                row["spectral_radius"] = r["analysis"]["spectral_radius"]
                row["residual_median"] = r["analysis"]["residual"]["median"]
            wr.writerow(row)

    title = f"{ec.split}: {split.num_docs} docs"
    plot_metric_vs_k(results, "r2", out_dir / "r2_vs_k.png", f"R^2(k) — {title}", ylim=(-0.2, 1.0))
    plot_metric_vs_k(results, "skill", out_dir / "skill_vs_k.png", f"skill vs persistence — {title}", ylim=(-1.0, 1.0))
    plot_metric_vs_k(results, "cos", out_dir / "cos_vs_k.png", f"cosine(pred, true) — {title}")
    if assigner is not None:
        plot_metric_vs_k(results, "cluster_acc", out_dir / "cluster_acc_vs_k.png", f"cluster accuracy — {title}")
        if assigner.has_groups:
            plot_metric_vs_k(results, "group_acc", out_dir / "group_acc_vs_k.png", f"group accuracy — {title}")

    show = [k for k in (1, 8, 32, 128) if k in ks]
    hdr = f"| model | params | R²(1) all pairs | " + " | ".join(f"R²({k})" for k in show) + " | " + \
          " | ".join(f"skill({k})" for k in show) + " |"
    lines = [hdr, "|" + "---|" * (3 + 2 * len(show))]
    for name, r in sorted(results.items(), key=lambda kv: -(kv[1].get("r2_onestep_allpairs") or -9)):
        h = r["horizon"]
        f4 = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) and v == v else "–"  # noqa: E731
        lines.append(f"| {name} | {r['num_parameters']} | {f4(r.get('r2_onestep_allpairs'))} | "
                     + " | ".join(f4(h.get(str(k), {}).get("r2")) for k in show) + " | "
                     + " | ".join(f4(h.get(str(k), {}).get("skill")) for k in show) + " |")
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    print("\n" + table + "\n")
    logger.info("Wrote %s", out_dir)


if __name__ == "__main__":
    main()
