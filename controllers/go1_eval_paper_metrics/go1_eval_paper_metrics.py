import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

sys.path.append(r"C:\Program Files\Webots\lib\controller\python")

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from utils.envs.go1_env import Go1Env
from utils.logging.paper_metrics import utc_now_iso

FAILURE_REASONS = {"fall", "base_contact", "tilt_limit_exceeded"}

RUN_FOLDER_NAME = "run_clean_policy_from_bc_20260328_162902"
# RUN_FOLDER_NAME = "run_degraded_policy_from_bc_20260402_221943"
CHECKPOINT_PATH = None
EPISODES = 50
DETERMINISTIC = True
CONDITION = "degraded"
MAX_STEPS = 1000
REQUIRE_VALID_SLIP_SAMPLES = True
LOG_TOUCH_PER_STEP = True
PRINT_TOUCH_STEP_TO_CONSOLE = False
TOUCH_CONSOLE_EVERY_N_STEPS = 1
CONTACT_SOURCE = "auto"
FAIL_LOUD_ZERO_PHYSICS_CONTACT_STEPS = 120
WARN_TOUCH_SATURATION_RATIO = 0.98
FORCE_CLEAN_ENV = True
FORCE_DEGRADED_ENV = False
SKIP_CONFIG_CONDITION_GATE = False

DEGRADED_STAGE_CONFIG_PATH = (
    workspace_root / "configs" / "training" / "degraded" / "degraded_policy_from_bc.json"
)
DEGRADATION_GO1_KEYS = [
    "DEGRADATION_MODEL_ACTIVE",
    "MOTOR_STRENGTH_MIN",
    "MOTOR_STRENGTH_MAX",
    "LATENCY_MS_MIN",
    "LATENCY_MS_MAX",
    "IMU_NOISE_STD",
    "IMU_BIAS_RANGE",
    "FRICTION_MIN",
    "FRICTION_MAX",
]


def _find_model_path(run_folder_name: str | None) -> Path:
    if run_folder_name:
        model_path = workspace_root / "training_runs" / run_folder_name / "go1_ppo_final.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for run folder '{run_folder_name}': {model_path}")
        return model_path

    candidates = sorted(
        workspace_root.glob("training_runs/run_*/go1_ppo_final.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No trained model found in training_runs/run_*/go1_ppo_final.zip")
    return candidates[0]


def _resolve_model_path_and_run_dir(run_folder_name: str | None, checkpoint_path: str | None) -> tuple[Path, Path]:
    if checkpoint_path:
        model_path = Path(checkpoint_path).expanduser()
        if not model_path.is_absolute():
            model_path = (workspace_root / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")
        return model_path, model_path.parent

    model_path = _find_model_path(run_folder_name)
    return model_path, model_path.parent


def _read_training_compute_metrics(run_dir: Path) -> dict[str, float | None]:
    rollout_candidates = sorted(run_dir.glob("rollout_metrics_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not rollout_candidates:
        return {
            "training_total_wall_clock_s": None,
            "training_last_checkpoint_steps_per_second": None,
            "training_mean_steps_per_second": None,
        }

    csv_path = rollout_candidates[0]
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {
            "training_total_wall_clock_s": None,
            "training_last_checkpoint_steps_per_second": None,
            "training_mean_steps_per_second": None,
        }

    sps_values = []
    for row in rows:
        try:
            sps_values.append(float(row.get("sim_steps_per_second", "nan")))
        except Exception:
            continue

    last_row = rows[-1]
    training_total_wall_clock_s = None
    training_last_checkpoint_steps_per_second = None
    training_mean_steps_per_second = None

    try:
        training_total_wall_clock_s = float(last_row.get("cumulative_wall_clock_s", "nan"))
    except Exception:
        pass

    try:
        training_last_checkpoint_steps_per_second = float(last_row.get("sim_steps_per_second", "nan"))
    except Exception:
        pass

    if sps_values:
        training_mean_steps_per_second = float(np.nanmean(np.asarray(sps_values, dtype=np.float64)))

    return {
        "training_total_wall_clock_s": training_total_wall_clock_s,
        "training_last_checkpoint_steps_per_second": training_last_checkpoint_steps_per_second,
        "training_mean_steps_per_second": training_mean_steps_per_second,
    }


def _load_run_config(run_dir: Path) -> dict | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None

    with run_config_path.open("r", encoding="utf-8") as f:
        run_config = json.load(f)
    if not isinstance(run_config, dict):
        raise ValueError(f"Invalid run config format: {run_config_path}")
    return run_config


def _apply_go1_env_from_run_config(run_config: dict | None) -> dict[str, str]:
    if run_config is None:
        return {}
    applied = run_config.get("applied_go1_environment", {})
    if not isinstance(applied, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, value in applied.items():
        env_key = str(key)
        env_value = str(value)
        os.environ[env_key] = env_value
        normalized[env_key] = env_value
    return normalized


def _read_degradation_active_from_run_config(run_config: dict) -> bool:
    applied = run_config.get("applied_go1_environment", {})
    if isinstance(applied, dict) and "GO1_DEGRADATION_MODEL_ACTIVE" in applied:
        value = str(applied.get("GO1_DEGRADATION_MODEL_ACTIVE", "0")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    stage_config = run_config.get("stage_config", {})
    if isinstance(stage_config, dict):
        go1_env = stage_config.get("go1_env", {})
        if isinstance(go1_env, dict) and "DEGRADATION_MODEL_ACTIVE" in go1_env:
            return bool(go1_env.get("DEGRADATION_MODEL_ACTIVE"))

    # Clean-stage configs may omit DEGRADATION_MODEL_ACTIVE entirely; treat omission as disabled.
    return False


def _apply_clean_environment_override() -> dict[str, str]:
    overrides = {
        "GO1_DEGRADATION_MODEL_ACTIVE": "0",
        "GO1_MOTOR_STRENGTH_MIN": "1.0",
        "GO1_MOTOR_STRENGTH_MAX": "1.0",
        "GO1_LATENCY_MS_MIN": "0.0",
        "GO1_LATENCY_MS_MAX": "0.0",
        "GO1_IMU_NOISE_STD": "0.0",
        "GO1_IMU_BIAS_RANGE": "0.0",
        "GO1_FRICTION_MIN": "1.0",
        "GO1_FRICTION_MAX": "1.0",
    }
    for key, value in overrides.items():
        os.environ[key] = value
    return overrides


def _as_env_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _apply_degraded_environment_override() -> dict[str, str]:
    defaults = {
        "GO1_DEGRADATION_MODEL_ACTIVE": "1",
        "GO1_MOTOR_STRENGTH_MIN": "0.6",
        "GO1_MOTOR_STRENGTH_MAX": "0.8",
        "GO1_LATENCY_MS_MIN": "15",
        "GO1_LATENCY_MS_MAX": "40",
        "GO1_IMU_NOISE_STD": "0.05",
        "GO1_IMU_BIAS_RANGE": "0.05",
        "GO1_FRICTION_MIN": "0.2",
        "GO1_FRICTION_MAX": "1.0",
    }

    overrides = dict(defaults)
    if DEGRADED_STAGE_CONFIG_PATH.exists():
        try:
            with DEGRADED_STAGE_CONFIG_PATH.open("r", encoding="utf-8") as f:
                stage_config = json.load(f)
            go1_env = stage_config.get("go1_env", {}) if isinstance(stage_config, dict) else {}
            if isinstance(go1_env, dict):
                for key in DEGRADATION_GO1_KEYS:
                    if key in go1_env:
                        overrides[f"GO1_{key}"] = _as_env_value(go1_env[key])
        except Exception:
            pass

    for key, value in overrides.items():
        os.environ[key] = value
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-grade Go1 policy evaluation with TensorBoard logging")
    parser.add_argument("--run-folder", type=str, default=RUN_FOLDER_NAME, help="Training run folder name under training_runs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT_PATH,
        help="Direct path to PPO checkpoint zip; when set, this takes precedence over --run-folder",
    )
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Number of evaluation episodes")
    parser.add_argument(
        "--condition",
        type=str,
        default=CONDITION,
        help="Condition label for outputs: clean, degraded, or auto (infer from run_config)",
    )
    parser.add_argument("--deterministic", action="store_true", default=DETERMINISTIC, help="Use deterministic actions")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Episode horizon used for survival metric")
    parser.add_argument(
        "--force-clean-env",
        action="store_true",
        default=FORCE_CLEAN_ENV,
        help="Override runtime GO1 env variables so evaluation runs under clean dynamics.",
    )
    parser.add_argument(
        "--force-degraded-env",
        action=argparse.BooleanOptionalAction,
        default=FORCE_DEGRADED_ENV,
        help=(
            "Override runtime GO1 env variables so evaluation runs under degraded dynamics. "
            "Ignored when --force-clean-env is set."
        ),
    )
    parser.add_argument(
        "--skip-config-gate",
        action="store_true",
        default=SKIP_CONFIG_CONDITION_GATE,
        help="Skip condition vs run_config degradation consistency assertion.",
    )
    args = parser.parse_args()

    model_path, run_dir = _resolve_model_path_and_run_dir(args.run_folder, args.checkpoint)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_config = _load_run_config(run_dir)
    _apply_go1_env_from_run_config(run_config)
    clean_env_overrides: dict[str, str] = {}
    degraded_env_overrides: dict[str, str] = {}
    if bool(args.force_clean_env):
        clean_env_overrides = _apply_clean_environment_override()
    elif bool(args.force_degraded_env):
        degraded_env_overrides = _apply_degraded_environment_override()
    os.environ["GO1_CONTACT_SOURCE"] = str(CONTACT_SOURCE)

    requested_condition = str(args.condition).strip().lower()
    if requested_condition not in {"clean", "degraded", "auto"}:
        raise ValueError(
            f"Unsupported --condition '{args.condition}'. Expected one of: clean, degraded, auto."
        )

    force_clean_env = bool(args.force_clean_env)
    force_degraded_env = bool(args.force_degraded_env) and (not force_clean_env)

    eval_condition = requested_condition
    if force_degraded_env and requested_condition == "auto":
        eval_condition = "degraded"
    config_degradation_active = None
    if run_config is not None:
        config_degradation_active = _read_degradation_active_from_run_config(run_config)
        if requested_condition == "auto" and (not force_degraded_env):
            eval_condition = "degraded" if config_degradation_active else "clean"
        if force_clean_env:
            eval_condition = "clean"
        intended_degraded = eval_condition.startswith("degrad")
        gate_bypassed_for_degraded_override = force_degraded_env
        if not (bool(args.skip_config_gate) or gate_bypassed_for_degraded_override):
            assert config_degradation_active == intended_degraded, (
                "Condition/config mismatch: "
                f"condition='{eval_condition}' implies degraded={intended_degraded}, "
                f"but run config DEGRADATION_MODEL_ACTIVE={config_degradation_active}."
            )
    else:
        if requested_condition == "auto" and (not force_degraded_env):
            eval_condition = "clean"
        intended_degraded = eval_condition.startswith("degrad")
        warnings.warn(
            "run_config.json not found next to checkpoint; skipping condition/config gate and environment replay from training metadata.",
            RuntimeWarning,
            stacklevel=2,
        )

    eval_csv_path = run_dir / f"evaluation_paper_{eval_condition}_{timestamp}.csv"
    summary_json_path = run_dir / f"evaluation_paper_summary_{eval_condition}_{timestamp}.json"
    tb_dir = run_dir / "tensorboard" / f"evaluation_{eval_condition}_{timestamp}"
    touch_step_csv_path = run_dir / f"evaluation_touch_steps_{eval_condition}_{timestamp}.csv"

    model = PPO.load(str(model_path), device="cpu")
    env = Go1Env()
    writer = SummaryWriter(log_dir=str(tb_dir))

    if REQUIRE_VALID_SLIP_SAMPLES:
        if not bool(getattr(env, "has_foot_contact_sensors", False)):
            raise RuntimeError(
                "Slip metric invalid: Go1Env has no foot contact sensors enabled. "
                "Ensure FL/FR/RL/RR TouchSensor devices are present and enabled."
            )
        if not bool(getattr(env, "has_foot_world_nodes", False)):
            raise RuntimeError(
                "Slip metric invalid: Go1Env cannot resolve foot world nodes (FL_foot/FR_foot/RL_foot/RR_foot)."
            )

    print(f"Evaluating model: {model_path}")
    print(f"Episode metrics CSV: {eval_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"TensorBoard logdir: {tb_dir}")
    if LOG_TOUCH_PER_STEP:
        print(f"Touch step CSV: {touch_step_csv_path}")
    if LOG_TOUCH_PER_STEP and PRINT_TOUCH_STEP_TO_CONSOLE:
        print(
            "Touch console logging enabled | "
            f"every_n_steps={max(1, int(TOUCH_CONSOLE_EVERY_N_STEPS))}"
        )
    if config_degradation_active is not None:
        print(
            "Config gate | "
            f"condition={eval_condition} "
            f"intended_degraded={intended_degraded} "
            f"config_DEGRADATION_MODEL_ACTIVE={config_degradation_active}"
        )
    else:
        print(
            "Config gate | "
            f"condition={eval_condition} "
            f"intended_degraded={intended_degraded} "
            "config_DEGRADATION_MODEL_ACTIVE=unknown (missing run_config.json)"
        )
    if bool(args.skip_config_gate):
        print("Config gate bypassed | skip_config_gate=1")
    elif bool(args.force_degraded_env) and (not bool(args.force_clean_env)):
        print("Config gate bypassed | reason=force_degraded_env")
    if clean_env_overrides:
        print(
            "Clean env override | "
            f"enabled=1 keys={','.join(sorted(clean_env_overrides.keys()))}"
        )
    if degraded_env_overrides:
        print(
            "Degraded env override | "
            f"enabled=1 keys={','.join(sorted(degraded_env_overrides.keys()))}"
        )
    print(
        "Environment gate | "
        f"env_DEGRADATION_MODEL_ACTIVE={int(getattr(env, 'degradation_model_active', False))}"
    )
    world_path = "unknown"
    try:
        world_path = str(env.supervisor.getWorldPath())
    except Exception:
        pass
    print(f"Preflight world_path={world_path}")
    print(
        "Preflight | "
        f"has_foot_contact_sensors={bool(getattr(env, 'has_foot_contact_sensors', False))} "
        f"has_foot_world_nodes={bool(getattr(env, 'has_foot_world_nodes', False))} "
        f"contact_source_mode={str(getattr(env, 'contact_source_mode', 'unknown'))} "
        f"enable_contact_points_tracking={int(bool(getattr(env, 'enable_contact_points_tracking', False)))} "
        f"contact_tracking_enabled_count={int(getattr(env, 'contact_tracking_enabled_count', len(getattr(env, 'contact_tracking_enabled_nodes', []))))} "
        f"contact_tracking_failed_count={int(getattr(env, 'contact_tracking_failed_count', len(getattr(env, 'contact_tracking_failed_nodes', []))))} "
        f"contact_sensor_reliable_initial={bool(getattr(env, 'contact_sensor_reliable', False))} "
        f"contact_stale_steps={int(getattr(env, 'contact_stale_steps', -1))} "
        f"contact_threshold={float(getattr(env, 'contact_threshold', float('nan'))):.2e}"
    )
    failed_nodes = getattr(env, "contact_tracking_failed_nodes", [])
    if failed_nodes:
        print(f"Preflight contact-tracking failures: {failed_nodes}")

    global_start = time.perf_counter()
    valid_slip_sample_count = 0
    total_step_count = 0
    total_stance_slip_distance_m = 0.0

    episode_rows: list[dict[str, float | int | str | bool]] = []

    touch_step_file = None
    touch_step_writer = None
    if LOG_TOUCH_PER_STEP:
        touch_step_file = touch_step_csv_path.open("w", encoding="utf-8", newline="")
        touch_step_writer = csv.writer(touch_step_file)
        touch_step_writer.writerow(
            [
                "timestamp_utc",
                "episode",
                "step",
                "termination_reason",
                "contact_sensor_reliable",
                "contact_source",
                "foot_contact_vector_json",
                "foot_contact_strengths_json",
                "foot_contact_vector_touch_json",
                "foot_contact_strengths_touch_json",
                "foot_contact_vector_physics_json",
                "foot_contact_counts_physics_json",
                "foot_world_z_json",
                "robot_contact_count",
                "trunk_contact_count",
                "calf_contact_counts_json",
                "true_stance_slip_distance_step_m",
                "true_stance_intended_distance_step_m",
                "forward_distance_m",
            ]
        )

    for episode_idx in range(1, int(args.episodes) + 1):
        obs, _ = env.reset()
        episode_start = time.perf_counter()

        step_count = 0
        survival = False
        fell = False
        no_progress = False

        forward_error_sum = 0.0
        mechanical_energy_j = 0.0
        mechanical_power_accum_w = 0.0
        true_stance_intended_distance_m = 0.0
        true_stance_slip_distance_m = 0.0
        episode_slip_valid_steps = 0
        episode_touch_rows = 0
        episode_consecutive_zero_physics_steps = 0
        episode_consecutive_zero_selected_steps = 0
        episode_full_contact_steps = 0
        episode_no_contact_steps = 0
        episode_contact_transition_count = 0
        prev_selected_contacts = None

        while True:
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, _, terminated, truncated, info = env.step(action)
            step_count += 1
            total_step_count += 1

            forward_error_sum += float(info.get("velocity_tracking_error", 0.0))
            mechanical_power_w = float(info.get("step_mechanical_power_w", 0.0))
            mechanical_energy_j += float(info.get("step_mechanical_work_j", 0.0))
            mechanical_power_accum_w += mechanical_power_w

            step_true_stance_slip_distance = float(info.get("true_stance_slip_distance_step_m", np.nan))
            step_true_stance_intended_distance = float(info.get("true_stance_intended_distance_step_m", np.nan))
            if np.isfinite(step_true_stance_slip_distance) and np.isfinite(step_true_stance_intended_distance):
                true_stance_slip_distance_m += step_true_stance_slip_distance
                true_stance_intended_distance_m += step_true_stance_intended_distance
                valid_slip_sample_count += 1
                episode_slip_valid_steps += 1
                total_stance_slip_distance_m += step_true_stance_slip_distance

            if touch_step_writer is not None:
                contact_source = str(info.get("contact_source", "none"))
                contacts_selected = info.get("foot_contact_vector", None)
                contacts_phys = info.get("foot_contact_vector_physics", None)

                if isinstance(contacts_selected, list) and len(contacts_selected) == 4:
                    selected_sum = float(np.sum(np.asarray(contacts_selected, dtype=np.float64)))
                    if selected_sum >= 3.999:
                        episode_full_contact_steps += 1
                    if selected_sum <= 0.001:
                        episode_no_contact_steps += 1

                    if prev_selected_contacts is not None and contacts_selected != prev_selected_contacts:
                        episode_contact_transition_count += 1
                    prev_selected_contacts = list(contacts_selected)

                if contact_source == "physics" and isinstance(contacts_phys, list) and len(contacts_phys) == 4:
                    if float(np.sum(np.asarray(contacts_phys, dtype=np.float64))) <= 0.0:
                        episode_consecutive_zero_physics_steps += 1
                    else:
                        episode_consecutive_zero_physics_steps = 0

                    if episode_consecutive_zero_physics_steps >= int(FAIL_LOUD_ZERO_PHYSICS_CONTACT_STEPS):
                        raise RuntimeError(
                            "Physics contact stream remained all-zero for "
                            f"{episode_consecutive_zero_physics_steps} consecutive steps "
                            f"(episode={episode_idx}, step={step_count}). "
                            "This indicates missing/invalid foot-floor collision contact reporting in Webots scene."
                        )

                if isinstance(contacts_selected, list) and len(contacts_selected) == 4:
                    if float(np.sum(np.asarray(contacts_selected, dtype=np.float64))) <= 0.0:
                        episode_consecutive_zero_selected_steps += 1
                    else:
                        episode_consecutive_zero_selected_steps = 0

                    if episode_consecutive_zero_selected_steps >= int(FAIL_LOUD_ZERO_PHYSICS_CONTACT_STEPS):
                        raise RuntimeError(
                            "Selected contact stream remained all-zero for "
                            f"{episode_consecutive_zero_selected_steps} consecutive steps "
                            f"(source={contact_source}, episode={episode_idx}, step={step_count}). "
                            "Slip metric is invalid without stance-contact samples."
                        )

                touch_step_writer.writerow(
                    [
                        utc_now_iso(),
                        episode_idx,
                        step_count,
                        str(info.get("termination_reason", "none")),
                        int(bool(info.get("contact_sensor_reliable", False))),
                        str(info.get("contact_source", "none")),
                        json.dumps(info.get("foot_contact_vector", None)),
                        json.dumps(info.get("foot_contact_strengths", None)),
                        json.dumps(info.get("foot_contact_vector_touch", None)),
                        json.dumps(info.get("foot_contact_strengths_touch", None)),
                        json.dumps(info.get("foot_contact_vector_physics", None)),
                        json.dumps(info.get("foot_contact_counts_physics", None)),
                        json.dumps(info.get("foot_world_z", None)),
                        info.get("robot_contact_count", None),
                        info.get("trunk_contact_count", None),
                        json.dumps(info.get("calf_contact_counts", None)),
                        step_true_stance_slip_distance,
                        step_true_stance_intended_distance,
                        float(info.get("forward_distance", 0.0)),
                    ]
                )
                episode_touch_rows += 1

                if PRINT_TOUCH_STEP_TO_CONSOLE and (
                    step_count % max(1, int(TOUCH_CONSOLE_EVERY_N_STEPS)) == 0
                ):
                    print(
                        "touch_step | "
                        f"ep={episode_idx:02d} step={step_count:04d} "
                        f"reliable={int(bool(info.get('contact_sensor_reliable', False)))} "
                        f"source={info.get('contact_source', 'none')} "
                        f"contacts_sel={info.get('foot_contact_vector', None)} "
                        f"contacts_touch={info.get('foot_contact_vector_touch', None)} "
                        f"contacts_phys={info.get('foot_contact_vector_physics', None)} "
                        f"counts_phys={info.get('foot_contact_counts_physics', None)} "
                        f"foot_z={info.get('foot_world_z', None)} "
                        f"robot_cnt={info.get('robot_contact_count', None)} "
                        f"trunk_cnt={info.get('trunk_contact_count', None)} "
                        f"calf_cnt={info.get('calf_contact_counts', None)} "
                        f"slip_step_m={step_true_stance_slip_distance:.6f} "
                        f"intend_step_m={step_true_stance_intended_distance:.6f} "
                        f"fwd_m={float(info.get('forward_distance', 0.0)):.3f}"
                    )

            if terminated or truncated:
                reason = str(info.get("termination_reason", "unknown"))
                fell = bool(reason in FAILURE_REASONS)
                no_progress = bool(reason == "no_progress")
                survival = bool(step_count >= int(args.max_steps) or reason == "truncation") and (not fell)

                avg_velocity_tracking_error = forward_error_sum / max(1, step_count)
                avg_mechanical_power_w = mechanical_power_accum_w / max(1, step_count)
                true_slip_ratio = np.nan
                if episode_slip_valid_steps > 0:
                    true_slip_ratio = true_stance_slip_distance_m / max(true_stance_intended_distance_m, 1e-9)
                episode_wall_clock_s = time.perf_counter() - episode_start
                full_contact_ratio = float(episode_full_contact_steps / max(1, step_count))
                no_contact_ratio = float(episode_no_contact_steps / max(1, step_count))

                row = {
                    "timestamp_utc": utc_now_iso(),
                    "condition": eval_condition,
                    "episode": episode_idx,
                    "steps": step_count,
                    "termination_reason": reason,
                    "fell": int(fell),
                    "survival_success": int(survival),
                    "no_progress": int(no_progress),
                    "forward_distance_m": float(info.get("forward_distance", 0.0)),
                    "avg_velocity_tracking_error_mps": float(avg_velocity_tracking_error),
                    "mechanical_energy_j": float(mechanical_energy_j),
                    "avg_mechanical_power_w": float(avg_mechanical_power_w),
                    "stance_intended_distance_m": float(true_stance_intended_distance_m),
                    "stance_slip_distance_m": float(true_stance_slip_distance_m),
                    "true_slip_ratio": float(true_slip_ratio) if np.isfinite(true_slip_ratio) else np.nan,
                    "slip_valid_steps": int(episode_slip_valid_steps),
                    "full_contact_ratio": float(full_contact_ratio),
                    "no_contact_ratio": float(no_contact_ratio),
                    "contact_transition_count": int(episode_contact_transition_count),
                    "episode_wall_clock_s": float(episode_wall_clock_s),
                }
                episode_rows.append(row)

                writer.add_scalar("eval/episode/survival_success", float(row["survival_success"]), episode_idx)
                writer.add_scalar("eval/episode/fell", float(row["fell"]), episode_idx)
                writer.add_scalar(
                    "eval/episode/avg_velocity_tracking_error_mps",
                    float(row["avg_velocity_tracking_error_mps"]),
                    episode_idx,
                )
                writer.add_scalar("eval/episode/avg_mechanical_power_w", float(row["avg_mechanical_power_w"]), episode_idx)
                writer.add_scalar(
                    "eval/episode/true_slip_ratio",
                    float(row["true_slip_ratio"]) if np.isfinite(float(row["true_slip_ratio"])) else np.nan,
                    episode_idx,
                )
                writer.add_scalar("eval/episode/slip_valid_steps", float(row["slip_valid_steps"]), episode_idx)
                writer.add_scalar("eval/episode/steps", float(row["steps"]), episode_idx)
                writer.add_scalar("eval/episode/full_contact_ratio", float(row["full_contact_ratio"]), episode_idx)
                writer.add_scalar("eval/episode/contact_transition_count", float(row["contact_transition_count"]), episode_idx)
                print(
                    f"Episode {episode_idx:02d}/{int(args.episodes)} | "
                    f"steps={step_count} | reason={reason} | "
                    f"forward_distance_m={float(info.get('forward_distance', 0.0)):.3f}"
                )
                if (
                    str(info.get("contact_source", "none")) == "touch"
                    and full_contact_ratio >= float(WARN_TOUCH_SATURATION_RATIO)
                ):
                    warnings.warn(
                        "Touch-contact saturation warning: selected contact indicates near-constant all-feet contact "
                        f"(episode={episode_idx}, full_contact_ratio={full_contact_ratio:.3f}, "
                        f"transitions={episode_contact_transition_count}). "
                        "Slip metric may reflect continuous dragging rather than alternating stance events.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if LOG_TOUCH_PER_STEP:
                    print(
                        f"Touch rows written | episode={episode_idx:02d} "
                        f"rows={episode_touch_rows} expected_steps={step_count}"
                    )
                break

    if touch_step_file is not None:
        touch_step_file.close()

    with eval_csv_path.open("w", encoding="utf-8", newline="") as f:
        if episode_rows:
            fieldnames = list(episode_rows[0].keys())
        else:
            fieldnames = [
                "timestamp_utc",
                "condition",
                "episode",
                "steps",
                "termination_reason",
                "fell",
                "survival_success",
                "no_progress",
                "forward_distance_m",
                "avg_velocity_tracking_error_mps",
                "mechanical_energy_j",
                "avg_mechanical_power_w",
                "stance_intended_distance_m",
                "stance_slip_distance_m",
                "true_slip_ratio",
                "slip_valid_steps",
                "full_contact_ratio",
                "no_contact_ratio",
                "contact_transition_count",
                "episode_wall_clock_s",
            ]
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in episode_rows:
            writer_csv.writerow(row)

    arr_survival = np.asarray([float(row["survival_success"]) for row in episode_rows], dtype=np.float64)
    arr_fall = np.asarray([float(row["fell"]) for row in episode_rows], dtype=np.float64)
    arr_no_progress = np.asarray([float(row["no_progress"]) for row in episode_rows], dtype=np.float64)
    arr_velocity_error = np.asarray([float(row["avg_velocity_tracking_error_mps"]) for row in episode_rows], dtype=np.float64)
    arr_power = np.asarray([float(row["avg_mechanical_power_w"]) for row in episode_rows], dtype=np.float64)
    arr_slip = np.asarray([float(row["true_slip_ratio"]) for row in episode_rows], dtype=np.float64)
    arr_distance = np.asarray([float(row["forward_distance_m"]) for row in episode_rows], dtype=np.float64)
    arr_full_contact_ratio = np.asarray([float(row["full_contact_ratio"]) for row in episode_rows], dtype=np.float64)
    arr_no_contact_ratio = np.asarray([float(row["no_contact_ratio"]) for row in episode_rows], dtype=np.float64)
    arr_contact_transitions = np.asarray([float(row["contact_transition_count"]) for row in episode_rows], dtype=np.float64)

    mean_true_slip_ratio = None
    if valid_slip_sample_count == 0:
        warnings.warn(
            "Slip metric warning: 0 valid slip samples collected across evaluation. "
            "True slip ratio is omitted from summary.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif total_stance_slip_distance_m <= 1e-12:
        warnings.warn(
            "Slip metric warning: valid slip samples exist but total stance slip distance is effectively zero. "
            "True slip ratio is omitted from summary pending slip calculation validation.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        mean_true_slip_ratio = float(np.nanmean(arr_slip)) if arr_slip.size > 0 else 0.0

    compute_metrics = _read_training_compute_metrics(run_dir)

    summary = {
        "timestamp_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "condition": eval_condition,
        "episodes": int(args.episodes),
        "horizon_steps": int(args.max_steps),
        "survival_success_rate": float(np.mean(arr_survival) if arr_survival.size > 0 else 0.0),
        "fall_rate": float(np.mean(arr_fall) if arr_fall.size > 0 else 0.0),
        "no_progress_rate": float(np.mean(arr_no_progress) if arr_no_progress.size > 0 else 0.0),
        "mean_distance_m": float(np.mean(arr_distance) if arr_distance.size > 0 else 0.0),
        "mean_velocity_tracking_error_mps": float(np.mean(arr_velocity_error) if arr_velocity_error.size > 0 else 0.0),
        "mean_mechanical_power_w": float(np.mean(arr_power) if arr_power.size > 0 else 0.0),
        "mean_true_slip_ratio": mean_true_slip_ratio,
        "mean_full_contact_ratio": float(np.mean(arr_full_contact_ratio) if arr_full_contact_ratio.size > 0 else 0.0),
        "mean_no_contact_ratio": float(np.mean(arr_no_contact_ratio) if arr_no_contact_ratio.size > 0 else 0.0),
        "mean_contact_transition_count": float(np.mean(arr_contact_transitions) if arr_contact_transitions.size > 0 else 0.0),
        "evaluation_wall_clock_s": float(time.perf_counter() - global_start),
        "slip_valid_steps": int(valid_slip_sample_count),
        "valid_slip_sample_count": int(valid_slip_sample_count),
        "total_step_count": int(total_step_count),
        "valid_slip_sample_ratio": float(valid_slip_sample_count / max(1, total_step_count)),
        "training_total_wall_clock_s": compute_metrics["training_total_wall_clock_s"],
        "training_last_checkpoint_steps_per_second": compute_metrics["training_last_checkpoint_steps_per_second"],
        "training_mean_steps_per_second": compute_metrics["training_mean_steps_per_second"],
        "reference_benchmarks": {
            "uddin_2026_success_rate_clean": 0.946,
            "tan_2018_power_w": 71.78,
            "hwangbo_2019_power_w": 78.1,
            "uddin_2026_slippage_clean": 0.012,
            "uddin_2026_slippage_degraded": 0.077,
            "hwangbo_2019_velocity_error_mps": 0.143,
            "rudin_2022_steps_per_second": 900000.0,
        },
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    writer.add_text("eval/run_dir", str(run_dir))
    writer.add_scalar("eval/summary/survival_success_rate", summary["survival_success_rate"], 0)
    writer.add_scalar("eval/summary/fall_rate", summary["fall_rate"], 0)
    writer.add_scalar("eval/summary/no_progress_rate", summary["no_progress_rate"], 0)
    writer.add_scalar("eval/summary/mean_velocity_tracking_error_mps", summary["mean_velocity_tracking_error_mps"], 0)
    writer.add_scalar("eval/summary/mean_mechanical_power_w", summary["mean_mechanical_power_w"], 0)
    if summary["mean_true_slip_ratio"] is not None:
        writer.add_scalar("eval/summary/mean_true_slip_ratio", float(summary["mean_true_slip_ratio"]), 0)
    writer.add_scalar("eval/summary/valid_slip_sample_ratio", summary["valid_slip_sample_ratio"], 0)
    writer.add_scalar("eval/summary/valid_slip_sample_count", float(summary["valid_slip_sample_count"]), 0)
    writer.add_scalar("eval/summary/mean_full_contact_ratio", float(summary["mean_full_contact_ratio"]), 0)
    writer.add_scalar("eval/summary/mean_contact_transition_count", float(summary["mean_contact_transition_count"]), 0)

    if summary["training_total_wall_clock_s"] is not None:
        writer.add_scalar("train/summary/total_wall_clock_s", float(summary["training_total_wall_clock_s"]), 0)
    if summary["training_last_checkpoint_steps_per_second"] is not None:
        writer.add_scalar(
            "train/summary/last_checkpoint_steps_per_second",
            float(summary["training_last_checkpoint_steps_per_second"]),
            0,
        )
    if summary["training_mean_steps_per_second"] is not None:
        writer.add_scalar("train/summary/mean_steps_per_second", float(summary["training_mean_steps_per_second"]), 0)

    writer.flush()
    writer.close()
    env.close()

    print("Paper evaluation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
