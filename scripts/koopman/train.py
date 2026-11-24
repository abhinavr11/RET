#!/usr/bin/env python3
"""Train the deep Koopman model on exported z trajectories (Phase 3).

    python scripts/koopman/train.py configs/koopman/gptoss20b.yaml
    python scripts/koopman/train.py configs/koopman/gptoss20b.yaml model.lift_dim=1024 train.seed=1
    python scripts/koopman/train.py configs/koopman/gptoss20b.yaml \\
        model.operator=lstsq run_name=N512_lstsq

Writes <experiment_dir>/runs/<run_name>/{model.pt, config.yaml, train_log.csv,
val_eval.json, spectrum.npz}. The run name defaults to
N{lift_dim}_{operator}_{decoder}_H{final H}_s{seed}.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koopman.config import add_config_args, load_koopman_config  # noqa: E402
from koopman.train_koopman import train_deep_koopman  # noqa: E402


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = load_koopman_config(args.config, args.overrides)
    train_deep_koopman(cfg)


if __name__ == "__main__":
    main()
