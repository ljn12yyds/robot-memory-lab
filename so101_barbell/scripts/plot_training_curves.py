#!/usr/bin/env python3
"""Plot reproducible training curves from the project's plain-text logs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_RE = re.compile(
    r"INFO (?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}).*?"
    r"step:(?P<display_step>\S+)\s+smpl:(?P<samples>\S+)\s+ep:(?P<episodes>\S+)\s+"
    r"epch:(?P<epoch>[\d.]+)\s+loss:(?P<loss>[-+\deE.]+)\s+"
    r"grdn:(?P<grad_norm>[-+\deE.]+)\s+lr:(?P<lr>[-+\deE.]+)\s+"
    r"updt_s:(?P<update_s>[-+\deE.]+)\s+data_s:(?P<data_s>[-+\deE.]+)"
)
LOG_FREQ_RE = re.compile(r"'log_freq':\s*(\d+)")


@dataclass(frozen=True)
class Run:
    key: str
    title: str
    family: str
    log_path: Path
    color: str


RUNS = (
    Run(
        "smolvla_sft",
        "SmolVLA SFT (10K)",
        "policy",
        Path("logs/smolvla_barbell_10k_v1.log"),
        "#2A6FBB",
    ),
    Run(
        "pi05_sft",
        "PI0.5 SFT (30K)",
        "policy",
        Path("logs/pi05_barbell_expert_quantileclip_effb8_30k_v2.log"),
        "#E07A1F",
    ),
    Run(
        "pistar06_value",
        "Pistar06 Value (8K)",
        "value",
        Path("logs/pi05_hil_pistar06_8k_v1.log"),
        "#6A4C93",
    ),
    Run(
        "pi05_acp",
        "PI0.5 ACP Policy Update (30K)",
        "policy",
        Path("logs/pi05_barbell_hil_acp_30k_v1.log"),
        "#2A9D66",
    ),
)


def human_number(token: str) -> float:
    suffixes = {"K": 1_000.0, "M": 1_000_000.0, "B": 1_000_000_000.0}
    if token[-1:].upper() in suffixes:
        return float(token[:-1]) * suffixes[token[-1].upper()]
    return float(token)


def parse_log(path: Path) -> list[dict[str, float | str]]:
    text = path.read_text(errors="replace")
    freq_match = LOG_FREQ_RE.search(text)
    if freq_match is None:
        raise RuntimeError(f"log_freq not found in {path}")
    log_freq = int(freq_match.group(1))

    rows: list[dict[str, float | str]] = []
    first_time: datetime | None = None
    for index, match in enumerate(METRIC_RE.finditer(text), start=1):
        timestamp = datetime.fromisoformat(f"{match['date']} {match['time']}")
        if first_time is None:
            first_time = timestamp
        rows.append(
            {
                # The logger rounds displayed steps (e.g. many rows become "30K").
                # A completed run emits exactly one row every log_freq steps, so this
                # sequence is the lossless training-step reconstruction.
                "step": float(index * log_freq),
                "display_step": match["display_step"],
                "samples_display": match["samples"],
                "samples_approx": human_number(match["samples"]),
                "episodes_display": match["episodes"],
                "epoch": float(match["epoch"]),
                "loss": float(match["loss"]),
                "grad_norm": float(match["grad_norm"]),
                "lr": float(match["lr"]),
                "update_s": float(match["update_s"]),
                "data_s": float(match["data_s"]),
                "timestamp": timestamp.isoformat(sep=" "),
                "elapsed_hours": (timestamp - first_time).total_seconds() / 3600.0,
            }
        )
    if not rows:
        raise RuntimeError(f"no training metric rows found in {path}")
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, min(window, len(values)))
    if window == 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    valid = np.convolve(values, kernel, mode="valid")
    prefix = np.full(window - 1, np.nan)
    return np.concatenate((prefix, valid))


def arrays(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color="#D8DEE9", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_dashboard(run: Run, rows: list[dict[str, float | str]], output: Path) -> None:
    step = arrays(rows, "step")
    window = max(5, len(rows) // 40)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    specs = (
        ("loss", "Training loss", False),
        ("grad_norm", "Gradient norm", False),
        ("lr", "Learning rate", True),
        ("update_s", "Update time per step (s)", False),
    )
    for ax, (key, label, scientific) in zip(axes.flat, specs, strict=True):
        value = arrays(rows, key)
        smooth = rolling_mean(value, window)
        ax.plot(step, value, color=run.color, alpha=0.20, linewidth=0.8, label="logged")
        ax.plot(step, smooth, color=run.color, linewidth=2.0, label=f"moving mean ({window} logs)")
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(label)
        if scientific:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(run.title, fontsize=16, weight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_policy_comparison(
    parsed: dict[str, list[dict[str, float | str]]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for run in RUNS:
        if run.family != "policy":
            continue
        rows = parsed[run.key]
        step = arrays(rows, "step")
        loss = arrays(rows, "loss")
        smooth = rolling_mean(loss, max(5, len(rows) // 40))
        axes[0].plot(step, smooth, color=run.color, linewidth=2.2, label=run.title)
        normalized_step = step / step[-1]
        axes[1].plot(normalized_step, smooth, color=run.color, linewidth=2.2, label=run.title)

    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Training loss (objective-specific)")
    axes[0].set_title("Policy training loss")
    axes[1].set_xlabel("Fraction of scheduled training")
    axes[1].set_ylabel("Training loss (objective-specific)")
    axes[1].set_title("Training-progress normalized")
    for ax in axes:
        style_axis(ax)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Policy Loss Comparison\nLoss scales differ across model/objective; compare trends, not absolute values",
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rl_summary(parsed: dict[str, list[dict[str, float | str]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    pairs = (
        ("pistar06_value", "loss", "Value-model cross-entropy loss"),
        ("pistar06_value", "grad_norm", "Value-model gradient norm"),
        ("pi05_acp", "loss", "ACP policy flow-matching loss"),
        ("pi05_acp", "grad_norm", "ACP policy gradient norm"),
    )
    run_lookup = {run.key: run for run in RUNS}
    for ax, (key, metric, title) in zip(axes.flat, pairs, strict=True):
        run = run_lookup[key]
        rows = parsed[key]
        step = arrays(rows, "step")
        value = arrays(rows, metric)
        smooth = rolling_mean(value, max(5, len(rows) // 40))
        ax.plot(step, value, color=run.color, alpha=0.18, linewidth=0.8)
        ax.plot(step, smooth, color=run.color, linewidth=2.2)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title)
        style_axis(ax)
    fig.suptitle("Evo-RL Reproduction: Value Training + ACP Policy Update", fontsize=16, weight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path, parsed: dict[str, list[dict[str, float | str]]], output_dir: Path
) -> None:
    lines = [
        "# Training curve summary",
        "",
        "All values below were parsed from the original training logs. The final-window statistics use the last 10% of logged points.",
        "",
        "| Run | Steps | Logged points | Initial loss | Final loss | Final 10% mean | Minimum loss | Elapsed hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        rows = parsed[run.key]
        loss = arrays(rows, "loss")
        tail = loss[-max(1, len(loss) // 10) :]
        lines.append(
            f"| {run.title} | {int(float(rows[-1]['step'])):,} | {len(rows):,} | "
            f"{loss[0]:.4f} | {loss[-1]:.4f} | {tail.mean():.4f} | {loss.min():.4f} | "
            f"{float(rows[-1]['elapsed_hours']):.2f} |"
        )
    lines += [
        "",
        "## Interpretation notes",
        "",
        "- SmolVLA, PI0.5 SFT, Pistar06 Value, and ACP policy losses are different objectives/scales; their absolute values must not be ranked directly.",
        "- A decreasing training loss shows optimization fit, not real-robot success or generalization.",
        "- There is no held-out validation loss in these logs, so overfitting cannot be diagnosed from training loss alone.",
        "- The ACP 25K checkpoint performed better than 30K in the earlier small real-robot screen; this is compatible with training loss continuing to decrease while task performance degrades.",
        "",
        "## Files",
        "",
    ]
    for file in sorted(output_dir.glob("*.png")):
        lines.append(f"- `{file.name}`")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reports" / "training_curves_20260830",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parsed: dict[str, list[dict[str, float | str]]] = {}
    for index, run in enumerate(RUNS, start=1):
        resolved = args.project_root / run.log_path
        rows = parse_log(resolved)
        parsed[run.key] = rows
        write_csv(args.output_dir / f"{run.key}_metrics.csv", rows)
        plot_dashboard(run, rows, args.output_dir / f"{index:02d}_{run.key}_dashboard.png")

    plot_policy_comparison(parsed, args.output_dir / "05_policy_loss_comparison.png")
    plot_rl_summary(parsed, args.output_dir / "06_evorl_training_summary.png")
    write_summary(args.output_dir / "README.md", parsed, args.output_dir)
    print(f"Wrote training curves to {args.output_dir}")


if __name__ == "__main__":
    main()
