"""Figures for the Koopman evaluation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FAMILY_STYLE = {
    "persistence": dict(color="black", ls="--"),
    "mean": dict(color="gray", ls=":"),
    "dmd": dict(color="tab:blue"),
    "edmd": dict(color="tab:cyan"),
    "ulam": dict(color="tab:olive"),
    "ret_T": dict(color="tab:red", lw=2),
    "mlp": dict(color="tab:orange"),
    "direct": dict(color="tab:purple", ls="-.", lw=2),
    "deep_koopman": dict(color="tab:green", lw=2),
}


def family_of(name: str, kind: str) -> str:
    if kind == "deep_koopman":
        return "deep_koopman"
    for key in ("persistence", "mean", "edmd", "dmd", "ulam", "ret_T", "direct", "mlp"):
        if name.startswith(key):
            return key
    return kind


LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def plot_metric_vs_k(results: dict, metric: str, path: Path, title: str, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_in_family: dict[str, int] = {}
    for name, r in results.items():
        hor = r["horizon"]
        ks = sorted(int(k) for k in hor if metric in hor[k])
        if not ks:
            continue
        ys = [hor[str(k)][metric] for k in ks]
        fam = r.get("family", "other")
        i = n_in_family.get(fam, 0)
        n_in_family[fam] = i + 1
        style = dict(FAMILY_STYLE.get(fam, {}))
        style.setdefault("lw", 1.2)
        if "ls" not in FAMILY_STYLE.get(fam, {}):
            style["ls"] = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(ks, ys, marker=MARKERS[i % len(MARKERS)], ms=3.5, label=name, alpha=0.9, **style)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("horizon k (tokens)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eigs(eigvals: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="gray", lw=0.8)
    ax.scatter(eigvals.real, eigvals.imag, s=6, alpha=0.7)
    ax.set_aspect("equal")
    lim = max(1.05, float(np.abs(eigvals).max()) * 1.05) if eigvals.size else 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", lw=0.4)
    ax.axvline(0, color="gray", lw=0.4)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eta2(modes: list[dict], pca_eta2: list[float], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    n = len(modes)
    x = np.arange(n)
    eta = [max(m.get("eta2_re") or 0.0, m.get("eta2_im") or 0.0) for m in modes]
    ax.bar(x - 0.2, eta, width=0.4, label="Koopman eigenfunction (slowest first)")
    ax.bar(x[: len(pca_eta2)] + 0.2, pca_eta2, width=0.4, label="principal component of z (largest first)")
    ax.set_xlabel("mode rank")
    ax.set_ylabel("eta^2 w.r.t. cluster labels")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
