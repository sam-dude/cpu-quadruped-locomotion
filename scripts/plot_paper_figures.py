from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import ticker

import paper_plot_constants as C


@dataclass
class FigureRecord:
    figure_id: str
    title: str
    status: str
    output_png: str | None
    output_svg: str | None
    note: str = ""


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, comment="#")
    except Exception:
        return None


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": C.FIGURE_DPI,
            "savefig.dpi": C.FIGURE_DPI,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


def _save(fig: plt.Figure, file_stem: str) -> tuple[str, str]:
    C.PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = C.PAPER_FIGURES_DIR / f"{file_stem}.png"
    svg = C.PAPER_FIGURES_DIR / f"{file_stem}.svg"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return str(png), str(svg)


def _policy_math(policy: str) -> str:
    mapping = {
        "pi_C": r"$\pi_C$",
        "pi_D": r"$\pi_D$",
    }
    return mapping.get(policy, policy)


def _load_training_runs() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for label, path in {
        "pi_C": C.PI_C_TRAINING_CSV,
        "pi_D": C.PI_D_TRAINING_CSV,
    }.items():
        df = _safe_read_csv(path)
        if df is not None and not df.empty:
            out[label] = df.sort_values("timestep").reset_index(drop=True)
    return out


def _load_locked_triplet() -> dict[str, dict] | None:
    rows: dict[str, dict] = {}
    for _, policy, environment, path in [*C.LOCKED_BASELINE_SUMMARIES, *C.LOCKED_COMPLETED_SUMMARIES]:
        loaded = _safe_read_json(path)
        if loaded is None:
            return None
        key = f"{policy}_{environment}".replace("@", "").replace(" ", "_")
        rows[key] = loaded
    if "pi_C_M_C" not in rows or "pi_C_M_D" not in rows or "pi_D_M_D" not in rows:
        return None
    return rows


def _load_terrain_rows() -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for terrain, pair in C.TERRAIN_GROUP_SUMMARIES.items():
        terrain_rows: dict[str, dict] = {}
        for policy, path in pair.items():
            loaded = _safe_read_json(path)
            if loaded is not None:
                terrain_rows[policy] = loaded
        out[terrain] = terrain_rows
    return out


def _load_reward_diagnostic_rows() -> list[tuple[str, str, dict]]:
    rows: list[tuple[str, str, dict]] = []
    for run_key, label, path in C.REWARD_DIAGNOSTIC_SUMMARIES:
        loaded = _safe_read_json(path)
        if loaded is not None:
            rows.append((run_key, label, loaded))
    return rows


def _rolling_mean(values: pd.Series, window: int = 9) -> pd.Series:
    return values.rolling(window=window, min_periods=1, center=True).mean()


def _million_tick(value: float, _: int) -> str:
    return f"{value / 1_000_000:.1f}M"


def _format_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _metric_delta(before: float, after: float, lower_is_better: bool) -> float:
    if abs(before) < 1e-12:
        return np.nan
    if lower_is_better:
        return (before - after) / before * 100.0
    return (after - before) / before * 100.0


def _box_strip(
    ax: plt.Axes,
    series_list: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    title: str,
) -> None:
    positions = np.arange(1, len(series_list) + 1)
    box = ax.boxplot(
        series_list,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 5},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(42)
    for idx, (vals, color) in enumerate(zip(series_list, colors), start=1):
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), idx) + jitter, vals, s=20, alpha=0.55, color=color, edgecolor="none")
        ax.text(idx, float(np.mean(vals)), f"{np.mean(vals):.2f}", ha="center", va="bottom", fontsize=9, color=color)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, label: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=color,
        facecolor=color,
        alpha=0.16,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=11)


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.6, color=color)
    ax.add_patch(arrow)


def _write_pipeline_mermaid() -> None:
    mermaid = """flowchart LR
    classDef data fill:#eef2f7,stroke:#5f6c7b,stroke-width:1.5px,color:#1f2933;
    classDef clean fill:#e7f0fb,stroke:#3465a4,stroke-width:1.5px,color:#12344d;
    classDef degraded fill:#fff0da,stroke:#d17c0f,stroke-width:1.5px,color:#5f3b00;
    classDef eval fill:#edf7ea,stroke:#6a994e,stroke-width:1.5px,color:#1d3b1a;

    A[Analytic trot reference]
    B[Perturbation data collection]
    C[BC dataset filtering]
    D[BC initialization]
    E[PPO retrain: pi_C in M_C]
    F[PPO retrain: pi_D in M_D]
    G[Eval: pi_C @ M_C]
    H[Eval: pi_C @ M_D]
    I[Eval: pi_D @ M_D]
    J[OOD terrains: laterite, debris, heightfield]
    K[Hardware degradation: latency, IMU noise and bias, torque scale, friction]

    A --> B --> C --> D
    D --> E
    D --> F
    K --> F
    E --> G
    E --> H
    F --> I
    F --> J

    class A,B,C,D data;
    class E,G,H clean;
    class F,I degraded;
    class J,K eval;
"""
    (C.PAPER_FIGURES_DIR / "A1_training_evaluation_pipeline.mmd").write_text(mermaid, encoding="utf-8")


def _figure_a1_pipeline(records: list[FigureRecord]) -> None:
    _write_pipeline_mermaid()
    fig, ax = plt.subplots(figsize=C.FIGURE_SIZE_PIPELINE)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    lane_specs = [
        (0.77, 0.16, "Data", "#f4f6f8"),
        (0.49, 0.18, "Training", "#f7f8fa"),
        (0.18, 0.20, "Evaluation", "#f4f7f2"),
    ]
    for y, h, label, fill in lane_specs:
        ax.add_patch(Rectangle((0.03, y), 0.94, h, facecolor=fill, edgecolor="none", zorder=0))
        ax.text(0.02, y + h / 2, label, ha="right", va="center", fontsize=12, color="#4a5560", fontweight="bold")

    _draw_box(ax, (0.06, 0.80), 0.16, 0.10, "Analytic trot\nreference", "#5f6c7b")
    _draw_box(ax, (0.28, 0.80), 0.16, 0.10, "Perturbation data\ncollection", "#5f6c7b")
    _draw_box(ax, (0.50, 0.80), 0.16, 0.10, "BC dataset\nfiltering", "#5f6c7b")
    _draw_box(ax, (0.72, 0.80), 0.16, 0.10, "BC\ninitialization", "#4f7cac")

    _draw_box(ax, (0.22, 0.53), 0.20, 0.11, "PPO retrain\n$\\pi_C$ in $M_C$", C.POLICY_COLORS["pi_C"])
    _draw_box(ax, (0.52, 0.53), 0.20, 0.11, "PPO retrain\n$\\pi_D$ in $M_D$", C.POLICY_COLORS["pi_D"])
    _draw_box(ax, (0.77, 0.52), 0.16, 0.13, "Hardware degradation\nlatency | IMU | torque | friction", "#b56576")

    _draw_box(ax, (0.05, 0.22), 0.18, 0.10, "Clean eval\n$\\pi_C$ @ $M_C$", C.POLICY_COLORS["pi_C"])
    _draw_box(ax, (0.29, 0.22), 0.18, 0.10, "Transfer eval\n$\\pi_C$ @ $M_D$", "#7a7f87")
    _draw_box(ax, (0.53, 0.22), 0.18, 0.10, "Locked eval\n$\\pi_D$ @ $M_D$", C.POLICY_COLORS["pi_D"])
    _draw_box(ax, (0.77, 0.22), 0.17, 0.10, "OOD eval\nlaterite | debris | heightfield", "#6a994e")

    _draw_arrow(ax, (0.22, 0.85), (0.28, 0.85))
    _draw_arrow(ax, (0.44, 0.85), (0.50, 0.85))
    _draw_arrow(ax, (0.66, 0.85), (0.72, 0.85))
    _draw_arrow(ax, (0.80, 0.80), (0.32, 0.64))
    _draw_arrow(ax, (0.80, 0.80), (0.62, 0.64))

    _draw_arrow(ax, (0.32, 0.53), (0.14, 0.32))
    _draw_arrow(ax, (0.32, 0.53), (0.38, 0.32))
    _draw_arrow(ax, (0.62, 0.53), (0.62, 0.32))
    _draw_arrow(ax, (0.62, 0.53), (0.86, 0.32))
    _draw_arrow(ax, (0.77, 0.58), (0.72, 0.58), color="#8d3d54")

    ax.text(0.5, 0.96, "Training and Evaluation Pipeline", ha="center", va="center", fontsize=15, fontweight="bold")

    png, svg = _save(fig, "A1_training_evaluation_pipeline")
    records.append(FigureRecord("A1", "Training and evaluation pipeline", "generated", png, svg))


def _figure_g1_training_dynamics(records: list[FigureRecord], training: dict[str, pd.DataFrame]) -> None:
    if not training:
        records.append(FigureRecord("G1", "Training dynamics", "skipped", None, None, "missing rollout csv files"))
        return

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    specs = [
        ("sim_steps_per_second", "Simulation steps/s", "Throughput"),
        ("policy_loss", "Policy loss", "Policy Loss"),
        ("entropy", "Entropy", "Entropy"),
    ]

    for ax, (column, ylabel, title) in zip(axes, specs):
        for policy, df in training.items():
            valid = df[["timestep", column]].dropna()
            if valid.empty:
                continue
            color = C.POLICY_COLORS[policy]
            ax.plot(valid["timestep"], valid[column], color=color, alpha=0.18, linewidth=1.0)
            smooth = _rolling_mean(valid[column], window=11)
            ax.plot(valid["timestep"], smooth, color=color, linewidth=2.3, label=_policy_math(policy))
            ax.scatter(valid["timestep"].iloc[-1], smooth.iloc[-1], color=color, s=36, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")

    axes[-1].set_xlabel("PPO timesteps")
    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(_million_tick))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Training Dynamics", y=1.02, fontsize=15)
    png, svg = _save(fig, "G1_training_dynamics")
    records.append(FigureRecord("G1", "Training dynamics", "generated", png, svg))


def _figure_g2_degradation_profile(records: list[FigureRecord]) -> None:
    clean_cfg = _safe_read_json(C.CLEAN_CONFIG_PATH)
    degraded_cfg = _safe_read_json(C.DEGRADED_CONFIG_PATH)
    if clean_cfg is None or degraded_cfg is None:
        records.append(FigureRecord("G2", "Degradation model profile", "skipped", None, None, "missing config json"))
        return

    clean = clean_cfg.get("go1_env", {})
    degraded = degraded_cfg.get("go1_env", {})
    fig, axes = plt.subplots(2, 2, figsize=C.FIGURE_SIZE_GRID)
    axes = axes.flatten()

    def range_panel(ax: plt.Axes, title: str, xmin: float, xmax: float, clean_value: float, xlim: tuple[float, float], unit: str) -> None:
        ax.barh([0], [xmax - xmin], left=xmin, height=0.35, color=C.POLICY_COLORS["pi_D"], alpha=0.35)
        ax.scatter([xmin, xmax], [0, 0], color=C.POLICY_COLORS["pi_D"], s=36, zorder=3)
        ax.axvline(clean_value, color=C.POLICY_COLORS["pi_C"], linestyle="--", linewidth=2.0)
        ax.text((xmin + xmax) / 2, 0.22, f"{xmin:g} to {xmax:g} {unit}".strip(), ha="center", va="bottom", fontsize=10)
        ax.text(clean_value, -0.32, f"clean: {clean_value:g}", ha="center", va="top", fontsize=9, color=C.POLICY_COLORS["pi_C"])
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_title(title, loc="left")

    range_panel(
        axes[0],
        "Motor strength scale",
        float(degraded["MOTOR_STRENGTH_MIN"]),
        float(degraded["MOTOR_STRENGTH_MAX"]),
        1.0,
        (0.0, 1.05),
        "",
    )
    range_panel(
        axes[1],
        "Control latency (ms)",
        float(degraded["LATENCY_MS_MIN"]),
        float(degraded["LATENCY_MS_MAX"]),
        0.0,
        (0.0, 45.0),
        "ms",
    )

    imu_ax = axes[2]
    noise_std = float(degraded["IMU_NOISE_STD"])
    bias_range = float(degraded["IMU_BIAS_RANGE"])
    imu_ax.barh([1, 0], [noise_std, bias_range], left=0.0, height=0.35, color=C.POLICY_COLORS["pi_D"], alpha=0.35)
    imu_ax.scatter([noise_std, bias_range], [1, 0], color=C.POLICY_COLORS["pi_D"], s=36, zorder=3)
    imu_ax.axvline(0.0, color=C.POLICY_COLORS["pi_C"], linestyle="--", linewidth=2.0)
    imu_ax.set_xlim(0.0, 0.06)
    imu_ax.set_yticks([1, 0])
    imu_ax.set_yticklabels(["per-step noise std", "episode bias range"])
    imu_ax.set_title("IMU perturbation (rad)", loc="left")
    imu_ax.text(noise_std, 1.2, f"{noise_std:.2f}", ha="center", fontsize=9)
    imu_ax.text(bias_range, 0.2, f"+/-{bias_range:.2f}", ha="center", fontsize=9)

    range_panel(
        axes[3],
        "Floor friction",
        float(degraded["FRICTION_MIN"]),
        float(degraded["FRICTION_MAX"]),
        1.0,
        (0.0, 1.05),
        "",
    )

    fig.suptitle("Hardware Degradation Model", y=1.02, fontsize=15)
    png, svg = _save(fig, "G2_degradation_model_profile")
    records.append(FigureRecord("G2", "Degradation model profile", "generated", png, svg))


def _figure_g3_core_metrics(records: list[FigureRecord], locked_triplet: dict[str, dict] | None) -> None:
    if not locked_triplet:
        records.append(FigureRecord("G3", "Core metric transition figure", "skipped", None, None, "missing locked summary json"))
        return

    refs = locked_triplet["pi_C_M_C"].get("reference_benchmarks", {})
    ordered = [
        ("pi_C_M_C", r"$\pi_C$ @ $M_C$"),
        ("pi_C_M_D", r"$\pi_C$ @ $M_D$"),
        ("pi_D_M_D", r"$\pi_D$ @ $M_D$"),
    ]
    panel_specs = [
        ("survival_success_rate", "Success Rate (%)", False),
        ("mean_distance_m", "Forward Distance (m)", False),
        ("mean_velocity_tracking_error_mps", "Velocity Error (m/s)", True),
        ("mean_true_slip_ratio", "Slip Ratio", True),
        ("mean_full_contact_ratio", "Full Contact Ratio", False),
        ("mean_mechanical_power_w", "Mechanical Power (W)", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.4))
    axes = axes.flatten()

    for ax, (metric, title, lower_is_better) in zip(axes, panel_specs):
        xs = np.arange(len(ordered))
        ys = np.array([float(locked_triplet[key][metric]) for key, _ in ordered], dtype=float)
        if metric == "survival_success_rate":
            ys = ys * 100.0
        ax.plot(xs, ys, color=C.POLICY_COLORS["locked"], linewidth=1.8, alpha=0.65)
        for x, y, (key, label) in zip(xs, ys, ordered):
            ax.scatter(x, y, s=70, color=C.CONDITION_COLORS[key], zorder=3)
            ax.text(x, y, f" {y:.2f}", va="center", ha="left", fontsize=9)
        delta = _metric_delta(ys[1], ys[2], lower_is_better)
        if np.isfinite(delta):
            ax.text(1.5, (ys[1] + ys[2]) / 2.0, _format_pct(delta), color=C.POLICY_COLORS["pi_D"], fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([label for _, label in ordered])
        ax.set_title(title, loc="left")
        ax.set_xlim(-0.15, 2.45)

        if metric == "survival_success_rate" and "uddin_2026_success_rate_clean" in refs:
            benchmark = 100.0 * float(refs["uddin_2026_success_rate_clean"])
            ax.axhline(benchmark, color="#7a7f87", linestyle="--", linewidth=1.4)
            ax.text(2.05, benchmark, " Uddin 2026", va="center", fontsize=8, color="#555555")
            ax.set_ylim(90.0, 101.0)
        elif metric == "mean_velocity_tracking_error_mps" and "hwangbo_2019_velocity_error_mps" in refs:
            benchmark = float(refs["hwangbo_2019_velocity_error_mps"])
            ax.axhline(benchmark, color="#c44e52", linestyle="--", linewidth=1.4)
            ax.text(2.05, benchmark, " Hwangbo 2019", va="center", fontsize=8, color="#555555")
        elif metric == "mean_true_slip_ratio":
            if "uddin_2026_slippage_clean" in refs:
                bench_clean = 100.0 * float(refs["uddin_2026_slippage_clean"])
                ax.axhline(bench_clean, color="#4c78a8", linestyle="--", linewidth=1.2)
                ax.text(2.05, bench_clean, " Uddin clean", va="center", fontsize=8, color="#555555")
            if "uddin_2026_slippage_degraded" in refs:
                bench_deg = 100.0 * float(refs["uddin_2026_slippage_degraded"])
                ax.axhline(bench_deg, color="#f58518", linestyle=":", linewidth=1.2)
                ax.text(2.05, bench_deg, " Uddin degraded", va="center", fontsize=8, color="#555555")
        elif metric == "mean_mechanical_power_w":
            if "tan_2018_power_w" in refs:
                bench_tan = float(refs["tan_2018_power_w"])
                ax.axhline(bench_tan, color="#54a24b", linestyle="--", linewidth=1.3)
                ax.text(2.05, bench_tan, " Tan 2018", va="center", fontsize=8, color="#555555")
            if "hwangbo_2019_power_w" in refs:
                bench_hwangbo = float(refs["hwangbo_2019_power_w"])
                ax.axhline(bench_hwangbo, color="#e45756", linestyle=":", linewidth=1.3)
                ax.text(2.05, bench_hwangbo, " Hwangbo 2019", va="center", fontsize=8, color="#555555")

    fig.suptitle("Core Evaluation Metrics", y=1.02, fontsize=15)
    png, svg = _save(fig, "G3_core_metric_transitions")
    records.append(FigureRecord("G3", "Core metric transition figure", "generated", png, svg))


def _figure_g4_episode_distributions(records: list[FigureRecord]) -> None:
    pi_c_df = _safe_read_csv(C.PI_C_MD_EPISODE_CSV)
    pi_d_df = _safe_read_csv(C.PI_D_MD_EPISODE_CSV)
    if pi_c_df is None or pi_d_df is None:
        records.append(FigureRecord("G4", "Degraded episode distributions", "skipped", None, None, "missing episode csv"))
        return

    required = [
        "forward_distance_m",
        "avg_velocity_tracking_error_mps",
        "full_contact_ratio",
        "contact_transition_count",
    ]
    if not all(col in pi_c_df.columns and col in pi_d_df.columns for col in required):
        records.append(FigureRecord("G4", "Degraded episode distributions", "skipped", None, None, "missing episode-level columns"))
        return

    fig, axes = plt.subplots(2, 2, figsize=C.FIGURE_SIZE_GRID)
    axes = axes.flatten()
    labels = [r"$\pi_C$ @ $M_D$", r"$\pi_D$ @ $M_D$"]
    colors = [C.POLICY_COLORS["pi_C"], C.POLICY_COLORS["pi_D"]]

    _box_strip(
        axes[0],
        [
            pd.to_numeric(pi_c_df["forward_distance_m"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(pi_d_df["forward_distance_m"], errors="coerce").dropna().to_numpy(dtype=float),
        ],
        labels,
        colors,
        "Forward distance per episode (m)",
        "Distance",
    )

    _box_strip(
        axes[1],
        [
            pd.to_numeric(pi_c_df["avg_velocity_tracking_error_mps"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(pi_d_df["avg_velocity_tracking_error_mps"], errors="coerce").dropna().to_numpy(dtype=float),
        ],
        labels,
        colors,
        "Velocity error per episode (m/s)",
        "Velocity Error",
    )

    _box_strip(
        axes[2],
        [
            pd.to_numeric(pi_c_df["full_contact_ratio"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(pi_d_df["full_contact_ratio"], errors="coerce").dropna().to_numpy(dtype=float),
        ],
        labels,
        colors,
        "Full contact ratio per episode",
        "Full Contact Ratio",
    )
    axes[2].set_ylim(0.0, 1.0)

    _box_strip(
        axes[3],
        [
            pd.to_numeric(pi_c_df["contact_transition_count"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(pi_d_df["contact_transition_count"], errors="coerce").dropna().to_numpy(dtype=float),
        ],
        labels,
        colors,
        "Contact transitions per episode",
        "Contact Transitions",
    )

    fig.suptitle("Degraded Episode Distributions", y=1.02, fontsize=15)
    png, svg = _save(fig, "G4_degraded_episode_distributions")
    records.append(FigureRecord("G4", "Degraded episode distributions", "generated", png, svg))


def _figure_g5_ood_generalization(records: list[FigureRecord]) -> None:
    terrain_rows = _load_terrain_rows()
    if not terrain_rows:
        records.append(FigureRecord("G5", "OOD terrain generalization", "skipped", None, None, "missing terrain summaries"))
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8))
    terrains = C.OOD_TERRAINS
    display = [C.TERRAIN_DISPLAY_NAMES[t] for t in terrains]
    x = np.arange(len(terrains))
    width = 0.35

    distance_c: list[float] = []
    distance_d: list[float] = []
    gains: list[float] = []
    heatmap_rows: list[list[float]] = []

    for terrain in terrains:
        pair = terrain_rows.get(terrain, {})
        pi_c = pair.get("pi_C")
        pi_d = pair.get("pi_D")
        if pi_c is None or pi_d is None:
            records.append(FigureRecord("G5", "OOD terrain generalization", "skipped", None, None, f"missing summaries for {terrain}"))
            plt.close(fig)
            return
        distance_c.append(float(pi_c["mean_distance_m"]))
        distance_d.append(float(pi_d["mean_distance_m"]))
        gains.append(_metric_delta(float(pi_c["mean_distance_m"]), float(pi_d["mean_distance_m"]), False))
        heatmap_rows.append(
            [
                _metric_delta(float(pi_c[metric]), float(pi_d[metric]), lower_is_better)
                for metric, _, lower_is_better in C.OOD_HEATMAP_METRICS
            ]
        )

    axes[0].bar(x - width / 2, distance_c, width=width, color=C.POLICY_COLORS["pi_C"], label=_policy_math("pi_C"))
    axes[0].bar(x + width / 2, distance_d, width=width, color=C.POLICY_COLORS["pi_D"], label=_policy_math("pi_D"))
    for idx, gain in enumerate(gains):
        axes[0].text(x[idx], max(distance_c[idx], distance_d[idx]) + 0.18, _format_pct(gain), ha="center", fontsize=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display)
    axes[0].set_ylabel("Mean forward distance (m)")
    axes[0].set_title("Distance", loc="left")
    axes[0].legend(frameon=False)

    heatmap = np.asarray(heatmap_rows, dtype=float)
    vmax = float(np.nanmax(np.abs(heatmap))) if np.isfinite(heatmap).any() else 1.0
    image = axes[1].imshow(heatmap, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[1].set_xticks(np.arange(len(C.OOD_HEATMAP_METRICS)))
    axes[1].set_xticklabels([name for _, name, _ in C.OOD_HEATMAP_METRICS], rotation=20, ha="right")
    axes[1].set_yticks(np.arange(len(display)))
    axes[1].set_yticklabels(display)
    axes[1].set_title("Relative Improvement (%)", loc="left")
    for row_idx in range(heatmap.shape[0]):
        for col_idx in range(heatmap.shape[1]):
            val = heatmap[row_idx, col_idx]
            text_color = "white" if abs(val) > 0.55 * vmax else "#222222"
            axes[1].text(col_idx, row_idx, f"{val:.1f}", ha="center", va="center", fontsize=9, color=text_color)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04, label="Improvement (%)")

    fig.suptitle("OOD Generalization", y=1.02, fontsize=15)
    png, svg = _save(fig, "G5_ood_generalization")
    records.append(FigureRecord("G5", "OOD terrain generalization", "generated", png, svg))


def _figure_g6_reward_hacking(records: list[FigureRecord]) -> None:
    rows = _load_reward_diagnostic_rows()
    if len(rows) < 3:
        records.append(FigureRecord("G6", "Reward-hacking diagnostic", "skipped", None, None, "missing reward diagnostic summaries"))
        return

    fig, axes = plt.subplots(2, 2, figsize=C.FIGURE_SIZE_GRID)
    axes = axes.flatten()
    xs = np.arange(len(rows))
    labels = [label for _, label, _ in rows]
    color_map = {
        "pi_D_locked": C.POLICY_COLORS["pi_D"],
        "r1_explore": C.POLICY_COLORS["exploratory"],
        "r1_resume": C.POLICY_COLORS["hacked"],
    }

    for ax, (metric, title, lower_is_better) in zip(axes, C.REWARD_DIAGNOSTIC_METRICS):
        ys = np.array([float(row[2][metric]) for row in rows], dtype=float)
        ax.plot(xs, ys, color=C.POLICY_COLORS["locked"], linewidth=1.6, alpha=0.65)
        for x, (run_key, _, row), y in zip(xs, rows, ys):
            ax.scatter(x, y, s=70, color=color_map[run_key], zorder=3)
            ax.text(x, y, f" {y:.2f}", va="center", ha="left", fontsize=9)
        exploratory_to_hacked = _metric_delta(ys[1], ys[2], lower_is_better)
        if np.isfinite(exploratory_to_hacked):
            ax.text(1.5, (ys[1] + ys[2]) / 2.0, _format_pct(exploratory_to_hacked), color=C.POLICY_COLORS["hacked"], fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=10)
        ax.set_title(title, loc="left")

    fig.suptitle("Reward Exploitation", y=1.02, fontsize=15)
    png, svg = _save(fig, "G6_reward_hacking_diagnostic")
    records.append(FigureRecord("G6", "Reward-hacking diagnostic", "generated", png, svg))


def _figure_g7_compute(records: list[FigureRecord], locked_triplet: dict[str, dict] | None) -> None:
    if not locked_triplet:
        records.append(FigureRecord("G7", "Compute benchmark", "skipped", None, None, "missing locked summary json"))
        return

    pi_c = locked_triplet["pi_C_M_C"]
    pi_d = locked_triplet["pi_D_M_D"]
    hours = [
        float(pi_c["training_total_wall_clock_s"]) / 3600.0,
        float(pi_d["training_total_wall_clock_s"]) / 3600.0,
    ]
    steps = [
        float(pi_c["training_mean_steps_per_second"]),
        float(pi_d["training_mean_steps_per_second"]),
        float(C.RUDIN_2022_STEPS_PER_SECOND),
    ]

    fig, axes = plt.subplots(1, 2, figsize=C.FIGURE_SIZE_WIDE)
    labels_hours = [r"$\pi_C$", r"$\pi_D$"]
    colors_hours = [C.POLICY_COLORS["pi_C"], C.POLICY_COLORS["pi_D"]]
    axes[0].barh(labels_hours, hours, color=colors_hours)
    for idx, value in enumerate(hours):
        axes[0].text(value + 0.05, idx, f"{value:.2f} h", va="center", fontsize=10)
    axes[0].set_xlabel("Training wall-clock time (hours)")
    axes[0].set_title("CPU training time", loc="left")

    labels_steps = [r"$\pi_C$", r"$\pi_D$", "Rudin 2022 GPU"]
    colors_steps = [C.POLICY_COLORS["pi_C"], C.POLICY_COLORS["pi_D"], "#5f6c7b"]
    axes[1].barh(labels_steps, steps, color=colors_steps)
    axes[1].set_xscale("log")
    for idx, value in enumerate(steps):
        label = f"{value:,.1f}" if value < 10_000 else f"{value:,.0f}"
        axes[1].text(value * 1.08, idx, label, va="center", fontsize=10)
    axes[1].set_xlabel("Simulation steps per second (log scale)")
    axes[1].set_title("Throughput reference", loc="left")

    fig.suptitle("Compute", y=1.02, fontsize=15)
    png, svg = _save(fig, "G7_compute_benchmark")
    records.append(FigureRecord("G7", "Compute benchmark", "generated", png, svg))


def _write_core_tables(locked_triplet: dict[str, dict] | None) -> None:
    if not locked_triplet:
        return

    rows = []
    for key, label in [
        ("pi_C_M_C", "pi_C @ M_C"),
        ("pi_C_M_D", "pi_C @ M_D"),
        ("pi_D_M_D", "pi_D @ M_D"),
    ]:
        row = locked_triplet[key]
        rows.append(
            {
                "condition": label,
                "survival_success_rate": row["survival_success_rate"],
                "mean_distance_m": row["mean_distance_m"],
                "mean_velocity_tracking_error_mps": row["mean_velocity_tracking_error_mps"],
                "mean_mechanical_power_w": row["mean_mechanical_power_w"],
                "mean_true_slip_ratio": row["mean_true_slip_ratio"],
                "mean_full_contact_ratio": row["mean_full_contact_ratio"],
                "mean_contact_transition_count": row["mean_contact_transition_count"],
                "training_mean_steps_per_second": row["training_mean_steps_per_second"],
                "training_total_wall_clock_h": float(row["training_total_wall_clock_s"]) / 3600.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(C.PAPER_FIGURES_DIR / "paper_metrics_table.csv", index=False)

    power_table = pd.DataFrame(
        [
            {"condition": "pi_C @ M_C", "power_w": locked_triplet["pi_C_M_C"]["mean_mechanical_power_w"]},
            {"condition": "pi_C @ M_D", "power_w": locked_triplet["pi_C_M_D"]["mean_mechanical_power_w"]},
            {"condition": "pi_D @ M_D", "power_w": locked_triplet["pi_D_M_D"]["mean_mechanical_power_w"]},
            {"condition": "Tan18", "power_w": C.TAN_2018_POWER_W},
            {"condition": "Hwangbo19", "power_w": C.HWANGBO_2019_POWER_W},
        ]
    )
    power_table.to_csv(C.PAPER_FIGURES_DIR / "paper_mechanical_power_table.csv", index=False)


def _write_terrain_gain_table() -> None:
    terrain_rows = _load_terrain_rows()
    rows: list[dict] = []
    for terrain in C.OOD_TERRAINS:
        pair = terrain_rows.get(terrain, {})
        pi_c = pair.get("pi_C")
        pi_d = pair.get("pi_D")
        if pi_c is None or pi_d is None:
            continue
        row = {"terrain": terrain}
        for metric, _, lower_is_better in C.OOD_HEATMAP_METRICS:
            row[f"{metric}_pi_C"] = pi_c[metric]
            row[f"{metric}_pi_D"] = pi_d[metric]
            row[f"{metric}_improvement_pct"] = _metric_delta(float(pi_c[metric]), float(pi_d[metric]), lower_is_better)
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(C.PAPER_FIGURES_DIR / "paper_terrain_gain_table.csv", index=False)


def _write_reward_hacking_table() -> None:
    rows = _load_reward_diagnostic_rows()
    if not rows:
        return
    out_rows = []
    for run_key, label, row in rows:
        out_rows.append(
            {
                "run_key": run_key,
                "label": label,
                "mean_distance_m": row["mean_distance_m"],
                "mean_velocity_tracking_error_mps": row["mean_velocity_tracking_error_mps"],
                "mean_mechanical_power_w": row["mean_mechanical_power_w"],
                "mean_true_slip_ratio": row["mean_true_slip_ratio"],
                "mean_full_contact_ratio": row["mean_full_contact_ratio"],
                "mean_contact_transition_count": row["mean_contact_transition_count"],
            }
        )
    pd.DataFrame(out_rows).to_csv(C.PAPER_FIGURES_DIR / "paper_reward_hacking_table.csv", index=False)


def _write_compute_table(locked_triplet: dict[str, dict] | None) -> None:
    if not locked_triplet:
        return
    df = pd.DataFrame(
        [
            {
                "label": "pi_C",
                "training_total_wall_clock_h": float(locked_triplet["pi_C_M_C"]["training_total_wall_clock_s"]) / 3600.0,
                "training_mean_steps_per_second": locked_triplet["pi_C_M_C"]["training_mean_steps_per_second"],
            },
            {
                "label": "pi_D",
                "training_total_wall_clock_h": float(locked_triplet["pi_D_M_D"]["training_total_wall_clock_s"]) / 3600.0,
                "training_mean_steps_per_second": locked_triplet["pi_D_M_D"]["training_mean_steps_per_second"],
            },
            {
                "label": "Rudin 2022 GPU",
                "training_total_wall_clock_h": np.nan,
                "training_mean_steps_per_second": C.RUDIN_2022_STEPS_PER_SECOND,
            },
        ]
    )
    df.to_csv(C.PAPER_FIGURES_DIR / "paper_compute_table.csv", index=False)


def main() -> None:
    _setup_style()
    C.PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    records: list[FigureRecord] = []
    training = _load_training_runs()
    locked_triplet = _load_locked_triplet()

    _figure_a1_pipeline(records)
    _figure_g1_training_dynamics(records, training)
    _figure_g2_degradation_profile(records)
    _figure_g3_core_metrics(records, locked_triplet)
    _figure_g4_episode_distributions(records)
    _figure_g5_ood_generalization(records)
    _figure_g6_reward_hacking(records)
    _figure_g7_compute(records, locked_triplet)

    _write_core_tables(locked_triplet)
    _write_terrain_gain_table()
    _write_reward_hacking_table()
    _write_compute_table(locked_triplet)

    manifest = [record.__dict__ for record in records]
    (C.PAPER_FIGURES_DIR / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    generated = [record.figure_id for record in records if record.status == "generated"]
    skipped = [record.figure_id for record in records if record.status == "skipped"]
    print(f"Generated figures: {generated}")
    print(f"Skipped figures: {skipped}")
    print(f"Output directory: {C.PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()
