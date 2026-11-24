"""Koopman models on top of RET macrostates.

Two-stage setup: the RET encoder is frozen and produces z_t; this package learns
a lifting g = phi(z), a linear operator K with g_{t+1} ~ K g_t, and a readout
z = C g, and compares the result against nonlinear predictors.

Pipeline (see KOOPMAN.md for the commands):
  1. scripts/koopman/export_z.py      dump z trajectories from a RET checkpoint
  2. scripts/koopman/fit_baselines.py persistence / DMD / EDMD / Ulam / MLPs / RET T
  3. scripts/koopman/train.py         deep Koopman model
  4. scripts/koopman/eval_horizon.py  R^2(k), skill, cluster accuracy, spectrum
"""
