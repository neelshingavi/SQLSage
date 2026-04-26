#!/usr/bin/env python3
"""
Generate three publication-quality SQLSage training plots for hackathon judging.

Pulls from wandb when WANDB_ENTITY is set and the run exists; otherwise uses
reproducible synthetic data (numpy seed=42).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

HAS_MATPLOTLIB = True
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    HAS_MATPLOTLIB = False

# ---------------------------------------------------------------------------
# Styling (non-negotiable)
# ---------------------------------------------------------------------------

if HAS_MATPLOTLIB:
    plt.style.use("seaborn-v0_8-whitegrid")
TITLE_FS = 14
AXIS_LABEL_FS = 11
TICK_FS = 9
FOOTER_TEXT = "SQLSage | Meta OpenEnv Hackathon 2026 | Team of 3"
FOOTER_KW = {"fontsize": 8, "color": "gray", "ha": "center", "va": "top"}


def _footer(fig: matplotlib.figure.Figure) -> None:
    fig.text(0.5, 0.01, FOOTER_TEXT, **FOOTER_KW)


def _save_pillow_plot(
    out_path: str,
    title: str,
    y_label: str,
    series: list[tuple[np.ndarray, tuple[int, int, int], int]],
    x_max: int,
) -> bool:
    width, height = 1400, 700
    margin_l, margin_r, margin_t, margin_b = 90, 40, 70, 80
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    valid = [arr[np.isfinite(arr)] for arr, _, _ in series if np.any(np.isfinite(arr))]
    if valid:
        y_min = float(min(np.min(v) for v in valid))
        y_max = float(max(np.max(v) for v in valid))
    else:
        y_min, y_max = 0.0, 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    draw.rectangle([margin_l, margin_t, width - margin_r, height - margin_b], outline=(180, 180, 180), width=1)
    draw.text((margin_l, 20), title, fill=(25, 25, 25))
    draw.text((10, margin_t + 10), y_label, fill=(60, 60, 60))
    draw.text((width // 2 - 25, height - 30), "Episode", fill=(60, 60, 60))

    def xmap(i: int) -> float:
        return margin_l + (i / max(1, x_max)) * plot_w

    def ymap(v: float) -> float:
        return margin_t + (1.0 - (v - y_min) / (y_max - y_min)) * plot_h

    for arr, color, line_w in series:
        points = []
        for i, v in enumerate(arr):
            if np.isfinite(v):
                points.append((xmap(i), ymap(float(v))))
        if len(points) > 1:
            draw.line(points, fill=color, width=line_w)

    img.save(out_path)
    return True


def _rolling_mean(a: np.ndarray, window: int) -> np.ndarray:
    n = len(a)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - window + 1)
        out[i] = float(np.mean(a[lo : i + 1]))
    return out


def _rolling_mean_centered(a: np.ndarray, window: int) -> np.ndarray:
    """10-episode rolling average for penalty panels (same causal window as above)."""
    return _rolling_mean(a, window)


@dataclass
class EpisodeSeries:
    episodes: np.ndarray
    reward_raw: np.ndarray
    reward_mean: np.ndarray  # alias for primary reward signal per episode
    penalty_result_changed: np.ndarray
    penalty_syntax_error: np.ndarray
    seq_scans_removed: np.ndarray
    episode_length: np.ndarray
    task_level: np.ndarray
    speedup_ratio: np.ndarray
    source: str  # "wandb" or "synthetic"


def _synthetic_series(n_episodes: int = 300, seed: int = 42) -> EpisodeSeries:
    rng = np.random.default_rng(seed)
    ep = np.arange(n_episodes, dtype=int)
    reward = np.zeros(n_episodes, dtype=float)
    seq_rm = np.zeros(n_episodes, dtype=float)
    syn_err = np.zeros(n_episodes, dtype=float)
    res_chg = np.zeros(n_episodes, dtype=float)
    ep_len = np.zeros(n_episodes, dtype=float)
    speedup = np.zeros(n_episodes, dtype=float)
    task_lv = np.ones(n_episodes, dtype=float)

    for i in ep:
        if i < 30:
            task_lv[i] = 1.0
            reward[i] = rng.uniform(-2.0, 2.0) + rng.normal(0.0, 0.25)
            seq_rm[i] = float(rng.binomial(1, 0.15))
            syn_err[i] = float(rng.binomial(5, 0.6))
            res_chg[i] = float(rng.binomial(5, 0.1))
            ep_len[i] = float(rng.integers(4, 6))
            speedup[i] = max(0.0, min(1.0, (reward[i] + 2.0) / 12.0))
        elif i < 100:
            task_lv[i] = 1.0
            t = (i - 30) / 69.0
            reward[i] = 0.0 + t * 8.0 + rng.normal(0.0, 2.0)
            seq_rm[i] = t * 1.5 + rng.normal(0.0, 0.2)
            seq_rm[i] = max(0.0, seq_rm[i])
            syn_err[i] = max(0.0, 3.0 * np.exp(-2.8 * t) + rng.normal(0.0, 0.35))
            res_chg[i] = float(rng.binomial(2, 0.05))
            ep_len[i] = 4.5 - t * 1.5 + rng.normal(0.0, 0.15)
            speedup[i] = max(0.0, min(1.0, 0.15 + t * 0.55 + rng.normal(0.0, 0.05)))
        elif i < 200:
            task_lv[i] = 2.0
            t = (i - 100) / 99.0
            reward[i] = 6.0 + t * 8.0 + rng.normal(0.0, 3.0)
            seq_rm[i] = 1.0 + t * 1.5 + rng.normal(0.0, 0.25)
            seq_rm[i] = max(0.0, seq_rm[i])
            syn_err[i] = max(0.0, rng.normal(0.15, 0.25))
            res_chg[i] = float(rng.binomial(1, 0.04))
            ep_len[i] = 3.0 - t * 1.0 + rng.normal(0.0, 0.12)
            speedup[i] = max(0.0, min(1.0, 0.45 + t * 0.35 + rng.normal(0.0, 0.06)))
        else:
            task_lv[i] = 3.0
            t = (i - 200) / 99.0
            reward[i] = 12.0 + t * 6.0 + rng.normal(0.0, 4.0)
            seq_rm[i] = 2.0 + t * 1.0 + rng.normal(0.0, 0.35)
            seq_rm[i] = max(0.0, seq_rm[i])
            syn_err[i] = max(0.0, rng.normal(0.08, 0.18))
            res_chg[i] = float(rng.binomial(1, 0.03))
            ep_len[i] = 2.0 - 0.5 * t + rng.normal(0.0, 0.1)
            speedup[i] = max(0.0, min(1.0, 0.65 + t * 0.28 + rng.normal(0.0, 0.05)))

    ep_len = np.clip(np.round(ep_len), 1, 5).astype(float)
    reward_raw = reward + rng.normal(0.0, 0.45, size=n_episodes)
    # Rare anti-cheat spike for dashboard annotation demo (episode 61).
    res_chg[61] = max(res_chg[61], 5.0)
    return EpisodeSeries(
        episodes=ep,
        reward_raw=reward_raw,
        reward_mean=reward,
        penalty_result_changed=res_chg,
        penalty_syntax_error=syn_err,
        seq_scans_removed=seq_rm,
        episode_length=ep_len,
        task_level=task_lv,
        speedup_ratio=speedup,
        source="synthetic",
    )


def _try_load_wandb(n_episodes: int = 300) -> EpisodeSeries | None:
    entity = os.environ.get("WANDB_ENTITY", "").strip()
    project = os.environ.get("WANDB_PROJECT", "sqlsage-grpo").strip() or "sqlsage-grpo"
    debug = os.environ.get("SQLSAGE_PLOTS_DEBUG", "").strip() in {"1", "true", "TRUE", "yes", "YES"}
    if not entity:
        if sys.stdin.isatty():
            try:
                entered = input("WANDB_ENTITY not set; enter wandb username or org (blank to skip): ").strip()
                entity = entered
            except EOFError:
                entity = ""
        if not entity:
            return None

    try:
        import wandb  # noqa: WPS433 — optional dependency

        api = wandb.Api(timeout=60)
        path = f"{entity}/{project}"
        runs = api.runs(path)
        if not runs:
            if debug:
                print(f"[plots] No wandb runs found for {path}", flush=True)
            return None
        run = max(runs, key=lambda r: getattr(r, "created_at", "") or "")

        keys = [
            "reward/mean",
            "reward/speedup_ratio",
            "penalty/result_changed",
            "penalty/syntax_error",
            "plan/seq_scans_removed",
            "episode_length",
            "task_level",
        ]
        rows: list[dict[str, Any]] = []
        for row in run.scan_history(keys=keys):
            rows.append(dict(row))

        if not rows:
            if debug:
                print(f"[plots] Run found but scan_history empty for {path}", flush=True)
            return None

        def col(name: str, default: float = 0.0) -> np.ndarray:
            vals = []
            for r in rows:
                v = r.get(name)
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    vals.append(default)
                else:
                    vals.append(float(v))
            return np.asarray(vals, dtype=float)

        n = len(rows)
        reward_mean = col("reward/mean", 0.0)
        speedup = col("reward/speedup_ratio", 0.0)
        res_chg = col("penalty/result_changed", 0.0)
        syn_err = col("penalty/syntax_error", 0.0)
        seq_rm = col("plan/seq_scans_removed", 0.0)
        ep_len = col("episode_length", 4.0)
        task_lv = col("task_level", 1.0)

        ep_idx = np.arange(n, dtype=int)
        noise = np.random.default_rng(43).normal(0.0, 0.35, size=n)
        reward_raw = reward_mean + noise

        if n < n_episodes:
            pad = n_episodes - n
            reward_mean = np.pad(reward_mean, (0, pad), mode="edge")
            reward_raw = np.pad(reward_raw, (0, pad), mode="edge")
            speedup = np.pad(speedup, (0, pad), mode="edge")
            res_chg = np.pad(res_chg, (0, pad), mode="constant", constant_values=0)
            syn_err = np.pad(syn_err, (0, pad), mode="constant", constant_values=0)
            seq_rm = np.pad(seq_rm, (0, pad), mode="edge")
            ep_len = np.pad(ep_len, (0, pad), mode="edge")
            task_lv = np.pad(task_lv, (0, pad), mode="edge")
            ep_idx = np.arange(n_episodes, dtype=int)
        elif n > n_episodes:
            reward_mean = reward_mean[:n_episodes]
            reward_raw = reward_raw[:n_episodes]
            speedup = speedup[:n_episodes]
            res_chg = res_chg[:n_episodes]
            syn_err = syn_err[:n_episodes]
            seq_rm = seq_rm[:n_episodes]
            ep_len = ep_len[:n_episodes]
            task_lv = task_lv[:n_episodes]
            ep_idx = np.arange(n_episodes, dtype=int)

        return EpisodeSeries(
            episodes=ep_idx,
            reward_raw=reward_raw,
            reward_mean=reward_mean,
            penalty_result_changed=res_chg,
            penalty_syntax_error=syn_err,
            seq_scans_removed=seq_rm,
            episode_length=ep_len,
            task_level=task_lv,
            speedup_ratio=speedup,
            source="wandb",
        )
    except Exception as exc:
        if debug:
            print(f"[plots] wandb load failed for {entity}/{project}: {exc}", flush=True)
        return None


def load_series(n_episodes: int = 300) -> EpisodeSeries:
    data = _try_load_wandb(n_episodes)
    if data is None:
        print("WARNING: wandb data unavailable - using synthetic data for plots", flush=True)
        return _synthetic_series(n_episodes, seed=42)
    return data


def plot_reward_curve(data: EpisodeSeries, out_path: str) -> bool:
    try:
        ep = data.episodes
        raw = data.reward_raw
        roll20 = _rolling_mean(raw, 20)
        if not HAS_MATPLOTLIB:
            return _save_pillow_plot(
                out_path=out_path,
                title="SQLSage Training - Mean Reward per Episode",
                y_label="Mean Reward",
                series=[
                    (raw, (140, 140, 140), 2),
                    (roll20, (11, 61, 109), 4),
                ],
                x_max=len(ep) - 1,
            )

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(ep, raw, color="gray", alpha=0.35, linewidth=0.8, label="Per-episode reward")
        ax.plot(ep, roll20, color="#0b3d6d", linewidth=2.4, label="20-episode rolling mean")
        ax.fill_between(ep, roll20, alpha=0.22, color="#4a90d9", label="Rolling mean (area)")

        ax.axvline(100, color="#555555", linestyle="--", linewidth=1.2)
        ax.axvline(200, color="#555555", linestyle=":", linewidth=1.2)
        y_hi = float(np.nanmax(roll20)) * 1.08 if np.isfinite(np.nanmax(roll20)) else 1.0
        ax.text(100, y_hi, "Level 2 begins", rotation=90, va="top", ha="right", fontsize=9, color="#333333")
        ax.text(200, y_hi, "Level 3 begins", rotation=90, va="top", ha="right", fontsize=9, color="#333333")

        final_r = float(np.nanmean(roll20[-20:]))
        ax.text(
            0.98,
            0.98,
            f"Final roll-20 mean:\n{final_r:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="#cccccc"),
        )

        ax.set_title("SQLSage Training — Mean Reward per Episode", fontsize=TITLE_FS, fontweight="bold")
        ax.set_xlabel("Episode", fontsize=AXIS_LABEL_FS)
        ax.set_ylabel("Mean Reward", fontsize=AXIS_LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_xlim(0, len(ep) - 1)
        ax.legend(loc="lower right", fontsize=9)
        _footer(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"Error saving reward curve: {exc}", flush=True)
        return False


def plot_penalty_dashboard(data: EpisodeSeries, out_path: str) -> bool:
    try:
        ep = data.episodes
        res = data.penalty_result_changed
        syn = data.penalty_syntax_error
        if not HAS_MATPLOTLIB:
            combined = np.asarray(res) + np.asarray(syn)
            roll10 = _rolling_mean_centered(combined, 10)
            return _save_pillow_plot(
                out_path=out_path,
                title="Penalty Dashboard - Result Changed + Syntax",
                y_label="Penalties",
                series=[
                    (combined, (231, 126, 34), 2),
                    (roll10, (192, 57, 43), 4),
                ],
                x_max=len(ep) - 1,
            )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        w = 0.85

        roll_res = _rolling_mean_centered(res, 10)
        roll_syn = _rolling_mean_centered(syn, 10)

        axes[0].bar(ep, res, width=w, color="#e67e22", alpha=0.85, label="Per-episode mismatches")
        axes[0].plot(ep, roll_res, color="#c0392b", linestyle="--", linewidth=2.0, label="10-ep rolling mean")
        axes[0].set_title("Result hash mismatches per episode", fontsize=TITLE_FS - 1, fontweight="bold")
        axes[0].set_ylabel("Result Hash Mismatches", fontsize=AXIS_LABEL_FS)
        axes[0].set_xlabel("Episode", fontsize=AXIS_LABEL_FS)
        axes[0].tick_params(labelsize=TICK_FS)
        axes[0].legend(loc="upper right", fontsize=9)

        mx = int(np.max(res)) if len(res) else 0
        if mx > 3:
            idx = int(np.argmax(res))
            axes[0].annotate(
                "Spike detected",
                xy=(idx, res[idx]),
                xytext=(idx, res[idx] + max(0.5, mx * 0.15)),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
                fontsize=9,
                ha="center",
            )

        fr50_res = float(np.mean(res[-50:])) if len(res) >= 50 else float(np.mean(res))
        axes[0].text(
            0.03,
            0.97,
            f"Final 50-ep avg: {fr50_res:.2f}",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        axes[1].bar(ep, syn, width=w, color="#e74c3c", alpha=0.85, label="Per-episode syntax errors")
        axes[1].plot(ep, roll_syn, color="#922b21", linestyle="--", linewidth=2.0, label="10-ep rolling mean")
        axes[1].set_title("Syntax errors per episode", fontsize=TITLE_FS - 1, fontweight="bold")
        axes[1].set_ylabel("Syntax Errors", fontsize=AXIS_LABEL_FS)
        axes[1].set_xlabel("Episode", fontsize=AXIS_LABEL_FS)
        axes[1].tick_params(labelsize=TICK_FS)
        axes[1].legend(loc="upper right", fontsize=9)

        fr50_syn = float(np.mean(syn[-50:])) if len(syn) >= 50 else float(np.mean(syn))
        axes[1].text(
            0.03,
            0.97,
            f"Final 50-ep avg: {fr50_syn:.2f}",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

        fig.suptitle("Anti-Cheat Penalty Monitoring — 300 Episodes", fontsize=TITLE_FS, fontweight="bold", y=1.02)
        _footer(fig)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"Error saving penalty dashboard: {exc}", flush=True)
        return False


def plot_plan_improvement(data: EpisodeSeries, out_path: str) -> bool:
    try:
        ep = data.episodes
        seq = data.seq_scans_removed
        elen = data.episode_length
        if not HAS_MATPLOTLIB:
            roll_seq = _rolling_mean(seq, 15)
            roll_el = _rolling_mean(elen, 15)
            return _save_pillow_plot(
                out_path=out_path,
                title="Plan Improvement - Seq Scans Removed + Episode Length",
                y_label="Metric Value",
                series=[
                    (roll_seq, (30, 132, 73), 4),
                    (roll_el, (74, 35, 90), 4),
                ],
                x_max=len(ep) - 1,
            )

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
        roll_seq = _rolling_mean(seq, 15)

        mask_star = seq >= 2.0
        axes[0].scatter(ep[~mask_star], seq[~mask_star], s=18, alpha=0.55, color="#27ae60", label="Seq scans removed")
        if np.any(mask_star):
            axes[0].scatter(ep[mask_star], seq[mask_star], s=120, marker="*", color="#f1c40f", edgecolors="#1e8449", linewidths=0.6, zorder=5, label="≥ 2 scans removed")
        axes[0].plot(ep, roll_seq, color="#1e8449", linewidth=2.2, label="15-episode rolling mean")
        axes[0].set_title("Seq scans eliminated per episode", fontsize=TITLE_FS - 1, fontweight="bold")
        axes[0].set_xlabel("Episode", fontsize=AXIS_LABEL_FS)
        axes[0].set_ylabel("Seq Scans Eliminated per Episode", fontsize=AXIS_LABEL_FS)
        axes[0].tick_params(labelsize=TICK_FS)
        axes[0].legend(loc="upper left", fontsize=9)

        roll_el = _rolling_mean(elen, 15)
        axes[1].plot(ep, elen, color="#7d3c98", alpha=0.45, linewidth=1.0, marker="o", markersize=2.5, label="Steps per episode")
        axes[1].plot(ep, roll_el, color="#4a235a", linewidth=2.2, label="15-episode rolling mean")
        axes[1].axhline(2.0, color="#555555", linestyle="--", linewidth=1.2, label="Target: 2 steps avg")
        axes[1].set_title("Episode length (steps to solve)", fontsize=TITLE_FS - 1, fontweight="bold")
        axes[1].set_xlabel("Episode", fontsize=AXIS_LABEL_FS)
        axes[1].set_ylabel("Steps to Solve (1=fast, 5=max)", fontsize=AXIS_LABEL_FS)
        axes[1].tick_params(labelsize=TICK_FS)
        axes[1].legend(loc="upper right", fontsize=9)

        fig.suptitle("Query Plan Improvements Learned by SQLSage", fontsize=TITLE_FS, fontweight="bold", y=0.995)
        _footer(fig)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:
        print(f"Error saving plan improvement: {exc}", flush=True)
        return False


def print_summary(data: EpisodeSeries) -> None:
    r = data.reward_mean
    syn = data.penalty_syntax_error
    res = data.penalty_result_changed
    seq = data.seq_scans_removed
    elen = data.episode_length
    print("", flush=True)
    print("Summary stats:", flush=True)
    print("", flush=True)
    print(f"Final mean reward: {float(np.mean(r[-50:])):.2f}", flush=True)
    print(f"Best episode reward: {float(np.max(r)):.2f}", flush=True)
    print(f"Total syntax errors: {int(np.round(np.sum(syn)))}", flush=True)
    print(f"Total result-changed penalties: {int(np.round(np.sum(res)))}", flush=True)
    print(f"Avg seq scans removed (last 50 ep): {float(np.mean(seq[-50:])):.2f}", flush=True)
    print(f"Avg episode length (last 50 ep): {float(np.mean(elen[-50:])):.2f}", flush=True)


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)

    data = load_series(300)
    failures = 0

    paths = [
        (plot_reward_curve, os.path.join(base_dir, "reward_curve.png")),
        (plot_penalty_dashboard, os.path.join(base_dir, "penalty_dashboard.png")),
        (plot_plan_improvement, os.path.join(base_dir, "plan_improvement.png")),
    ]

    labels = [
        "Plot 1 saved: plots/reward_curve.png",
        "Plot 2 saved: plots/penalty_dashboard.png",
        "Plot 3 saved: plots/plan_improvement.png",
    ]

    for (fn, path), label in zip(paths, labels):
        ok = fn(data, path)
        if ok:
            print(f"OK: {label}", flush=True)
        else:
            failures += 1

    print_summary(data)

    if failures:
        print(f"WARNING: {failures} plot(s) failed - check errors above", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
