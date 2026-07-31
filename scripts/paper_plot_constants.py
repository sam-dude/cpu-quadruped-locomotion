from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
TRAINING_RUNS_DIR = WORKSPACE_ROOT / "training_runs"
IMAGES_DIR = WORKSPACE_ROOT / "images"
PAPER_FIGURES_DIR = Path(os.getenv("PAPER_FIGURES_OUT_DIR", str(IMAGES_DIR / "paper_figures")))

RUN_PI_C = "run_clean_policy_from_bc_20260328_162902"
RUN_PI_D = "run_degraded_policy_from_bc_20260402_221943"
RUN_EXPLORE_R1 = "run_deg_robust_r1_20260409_175127"
RUN_EXPLORE_R1_RESUME = "run_deg_robust_r1_resume_20260410_020019"

PI_C_RUN_DIR = TRAINING_RUNS_DIR / RUN_PI_C
PI_D_RUN_DIR = TRAINING_RUNS_DIR / RUN_PI_D
R1_RUN_DIR = TRAINING_RUNS_DIR / RUN_EXPLORE_R1
R1_RESUME_RUN_DIR = TRAINING_RUNS_DIR / RUN_EXPLORE_R1_RESUME

PI_C_TRAINING_CSV = PI_C_RUN_DIR / "rollout_metrics_clean.csv"
PI_D_TRAINING_CSV = PI_D_RUN_DIR / "rollout_metrics_clean.csv"

PI_C_MC_SUMMARY = PI_C_RUN_DIR / "evaluation_paper_summary_clean_20260330_185546.json"
PI_C_MD_SUMMARY_20260403 = PI_C_RUN_DIR / "evaluation_paper_summary_degraded_20260403_105422.json"
PI_C_MD_SUMMARY = PI_C_RUN_DIR / "evaluation_paper_summary_degraded_20260408_203800.json"
PI_D_MD_SUMMARY_20260403 = PI_D_RUN_DIR / "evaluation_paper_summary_degraded_20260403_103009.json"
PI_D_MD_SUMMARY = PI_D_RUN_DIR / "evaluation_paper_summary_degraded_20260409_110921.json"

PI_C_MD_SUMMARY_20260405 = PI_C_RUN_DIR / "evaluation_paper_summary_degraded_20260405_145728.json"
PI_C_MD_SUMMARY_20260407 = PI_C_RUN_DIR / "evaluation_paper_summary_degraded_20260407_145400.json"
PI_C_MD_SUMMARY_20260408_124551 = PI_C_RUN_DIR / "evaluation_paper_summary_degraded_20260408_124551.json"
PI_D_MD_SUMMARY_20260403 = PI_D_RUN_DIR / "evaluation_paper_summary_degraded_20260403_103009.json"
PI_D_MD_SUMMARY_20260405 = PI_D_RUN_DIR / "evaluation_paper_summary_degraded_20260405_153457.json"
PI_D_MD_SUMMARY_20260408_125619 = PI_D_RUN_DIR / "evaluation_paper_summary_degraded_20260408_125619.json"
PI_D_MD_SUMMARY_20260408_210131 = PI_D_RUN_DIR / "evaluation_paper_summary_degraded_20260408_210131.json"

LOCKED_COMPLETED_SUMMARIES = [
    ("EXP-007A", "pi_C", "M_D", PI_C_MD_SUMMARY_20260403),
    ("EXP-007B", "pi_D", "M_D", PI_D_MD_SUMMARY_20260403),
]

LOCKED_BASELINE_SUMMARIES = [
    ("EXP-007A-BL", "pi_C", "M_C", PI_C_MC_SUMMARY),
]

PI_C_MC_EPISODE_CSV = PI_C_RUN_DIR / "evaluation_paper_clean_20260330_185546.csv"
PI_C_MD_EPISODE_CSV = PI_C_RUN_DIR / "evaluation_paper_degraded_20260403_105422.csv"
PI_D_MD_EPISODE_CSV = PI_D_RUN_DIR / "evaluation_paper_degraded_20260403_103009.csv"

LOCKED_COMPLETED_EPISODE_CSVS = [
    ("EXP-007A", "pi_C @ M_D", PI_C_MD_EPISODE_CSV),
    ("EXP-007B", "pi_D @ M_D", PI_D_MD_EPISODE_CSV),
]

TERRAIN_GROUP_SUMMARIES = {
    "flat_clean": {
        "pi_C": PI_C_MC_SUMMARY,
        "pi_D": PI_D_MD_SUMMARY_20260403,
    },
    "flat_degraded": {
        "pi_C": PI_C_MD_SUMMARY_20260403,
        "pi_D": PI_D_MD_SUMMARY_20260403,
    },
    "laterite": {
        "pi_C": PI_C_MD_SUMMARY_20260405,
        "pi_D": PI_D_MD_SUMMARY_20260405,
    },
    "debris": {
        "pi_C": PI_C_MD_SUMMARY_20260408_124551,
        "pi_D": PI_D_MD_SUMMARY_20260408_125619,
    },
    "heightfield": {
        "pi_C": PI_C_MD_SUMMARY,
        "pi_D": PI_D_MD_SUMMARY_20260408_210131,
    },
}

OOD_TERRAINS = ["laterite", "debris", "heightfield"]

R1_MD_SUMMARY = R1_RUN_DIR / "evaluation_paper_summary_degraded_20260409_223625.json"
R1_RESUME_MD_SUMMARY = R1_RESUME_RUN_DIR / "evaluation_paper_summary_degraded_20260410_071626.json"

REWARD_DIAGNOSTIC_SUMMARIES = [
    ("pi_D_locked", r"$\pi_{D}$ locked", PI_D_MD_SUMMARY_20260403),
    ("r1_explore", "exploratory run", R1_MD_SUMMARY),
    ("r1_resume", "power-hacked run", R1_RESUME_MD_SUMMARY),
]

CLEAN_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "training" / "clean" / "clean_policy_from_bc.json"
DEGRADED_CONFIG_PATH = WORKSPACE_ROOT / "configs" / "training" / "degraded" / "degraded_policy_from_bc.json"

SUMMARY_GLOB = "evaluation_paper_summary_*.json"
EPISODE_CSV_GLOB = "evaluation_paper_*.csv"

FIGURE_DPI = 300
FIGURE_SIZE_WIDE = (11.5, 6.5)
FIGURE_SIZE_MEDIUM = (9.5, 5.8)
FIGURE_SIZE_TALL = (11.0, 7.6)
FIGURE_SIZE_GRID = (12.0, 8.0)
FIGURE_SIZE_PIPELINE = (14.0, 7.0)

RUDIN_2022_STEPS_PER_SECOND = 900000.0
TAN_2018_POWER_W = 71.78
HWANGBO_2019_POWER_W = 78.10

CURRICULUM_START = 0.25
CURRICULUM_END = 1.0
CURRICULUM_EPISODES = 500

POLICY_COLORS = {
    "pi_C": "#3465a4",
    "pi_D": "#d17c0f",
    "locked": "#30343f",
    "exploratory": "#2a9d8f",
    "hacked": "#b24c63",
}

CONDITION_COLORS = {
    "pi_C_M_C": "#3465a4",
    "pi_C_M_D": "#7a7f87",
    "pi_D_M_D": "#d17c0f",
}

TERRAIN_DISPLAY_NAMES = {
    "flat_clean": "Flat clean",
    "flat_degraded": "Flat degraded",
    "laterite": "Laterite",
    "debris": "Debris",
    "heightfield": "Heightfield",
}

CORE_METRICS = [
    ("mean_distance_m", "Forward distance (m)", False),
    ("mean_velocity_tracking_error_mps", "Velocity error (m/s)", True),
    ("mean_full_contact_ratio", "Full contact ratio", False),
    ("mean_contact_transition_count", "Contact transitions", False),
    ("mean_mechanical_power_w", "Mechanical power (W)", True),
    ("mean_true_slip_ratio", "True slip ratio", True),
]

OOD_HEATMAP_METRICS = [
    ("mean_distance_m", "Distance", False),
    ("mean_velocity_tracking_error_mps", "Vel. error", True),
    ("mean_full_contact_ratio", "Full contact", False),
    ("mean_contact_transition_count", "Transitions", False),
    ("mean_true_slip_ratio", "Slip", True),
    ("mean_mechanical_power_w", "Power", True),
]

REWARD_DIAGNOSTIC_METRICS = [
    ("mean_distance_m", "Distance (m)", False),
    ("mean_mechanical_power_w", "Power (W)", True),
    ("mean_full_contact_ratio", "Full contact", False),
    ("mean_contact_transition_count", "Transitions", False),
]
