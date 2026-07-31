import csv
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(r"C:\Program Files\Webots\lib\controller\python")

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from stable_baselines3 import PPO

from utils.envs.go1_env import Go1Env
from utils.logging.paper_metrics import (
    EVALUATION_EPISODE_COLUMNS,
    EVALUATION_SUMMARY_COLUMNS,
    utc_now_iso,
)

RUN_FOLDER_NAME = "run_degraded_policy_from_bc_20260326_174359"
EPISODES = 50
DETERMINISTIC = True
CONDITION = "clean"
FAILURE_REASONS = {"fall", "base_contact", "tilt_limit_exceeded"}

def find_model_path(run_folder_name: str | None = None) -> Path:
    if run_folder_name:
        model_path = workspace_root / "training_runs" / run_folder_name / "go1_ppo_final.zip"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found for run folder '{run_folder_name}'. Expected: {model_path}"
            )
        return model_path

    candidates = sorted(
        workspace_root.glob("training_runs/run_*/go1_ppo_final.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No trained model found. Run training first to create training_runs/run_*/go1_ppo_final.zip"
        )
    return candidates[0]


def main() -> None:
    model_path = find_model_path(RUN_FOLDER_NAME)
    run_dir = model_path.parent

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_csv_path = run_dir / f"evaluation_{CONDITION}_{timestamp}.csv"
    summary_csv_path = run_dir / f"evaluation_summary_{CONDITION}_{timestamp}.csv"

    model = PPO.load(str(model_path), device="cpu")
    env = Go1Env()

    print(f"Evaluating policy: {model_path}")
    print(f"Writing evaluation CSV: {eval_csv_path}")
    print(f"Writing evaluation summary CSV: {summary_csv_path}")

    start_time = time.perf_counter()

    success_count = 0
    fall_count = 0
    truncation_count = 0
    no_progress_count = 0
    success_steps_total = 0.0
    total_forward_distance = 0.0
    total_mean_body_forward_velocity = 0.0
    total_mean_yaw_rate = 0.0
    total_mean_hind_ratio = 0.0
    total_mean_gait_sync = 0.0
    total_mean_velocity_tracking_error = 0.0
    total_mean_mechanical_work = 0.0
    total_mean_foot_slip_ratio = 0.0
    total_mean_action_smoothness = 0.0

    with eval_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(EVALUATION_EPISODE_COLUMNS)

        for episode_number in range(1, EPISODES + 1):
            obs, _ = env.reset()
            episode_body_forward_sum = 0.0
            episode_yaw_rate_sum = 0.0
            episode_hind_ratio_sum = 0.0
            episode_rear_joint_velocity_sum = 0.0
            episode_front_joint_velocity_sum = 0.0
            episode_gait_sync_sum = 0.0
            episode_velocity_tracking_error_sum = 0.0
            episode_mechanical_work_sum = 0.0
            episode_foot_slip_ratio_sum = 0.0
            episode_action_smoothness_sum = 0.0
            step_count = 0
            episode_start_time = time.perf_counter()

            while True:
                action, _ = model.predict(obs, deterministic=DETERMINISTIC)
                obs, _, terminated, truncated, info = env.step(action)

                episode_body_forward_sum += float(info.get("body_forward_velocity", 0.0))
                episode_yaw_rate_sum += abs(float(info.get("yaw_rate", 0.0)))
                episode_hind_ratio_sum += float(info.get("hind_motion_ratio", 0.0))
                episode_rear_joint_velocity_sum += float(info.get("rear_joint_velocity_magnitude", 0.0))
                episode_front_joint_velocity_sum += float(info.get("front_joint_velocity_magnitude", 0.0))
                episode_gait_sync_sum += float(info.get("gait_sync_score", 0.0))
                episode_velocity_tracking_error_sum += float(info.get("velocity_tracking_error", 0.0))
                episode_mechanical_work_sum += float(info.get("mean_mechanical_work", 0.0))
                episode_foot_slip_ratio_sum += float(info.get("foot_slip_ratio", 0.0))
                episode_action_smoothness_sum += float(info.get("action_delta_mean", 0.0))
                step_count += 1

                if terminated or truncated:
                    termination_reason = info.get("termination_reason", "unknown")
                    success = int(termination_reason not in FAILURE_REASONS)
                    if success:
                        success_count += 1
                        success_steps_total += step_count
                    elif termination_reason in FAILURE_REASONS:
                        fall_count += 1
                    elif termination_reason == "no_progress":
                        no_progress_count += 1
                    else:
                        truncation_count += 1

                    mean_body_forward_velocity = episode_body_forward_sum / max(1, step_count)
                    mean_yaw_rate = episode_yaw_rate_sum / max(1, step_count)
                    mean_hind_ratio = episode_hind_ratio_sum / max(1, step_count)
                    mean_rear_joint_velocity = episode_rear_joint_velocity_sum / max(1, step_count)
                    mean_front_joint_velocity = episode_front_joint_velocity_sum / max(1, step_count)
                    mean_gait_sync_score = episode_gait_sync_sum / max(1, step_count)
                    mean_velocity_tracking_error = episode_velocity_tracking_error_sum / max(1, step_count)
                    mean_mechanical_work = episode_mechanical_work_sum / max(1, step_count)
                    mean_foot_slip_ratio = episode_foot_slip_ratio_sum / max(1, step_count)
                    mean_action_smoothness = episode_action_smoothness_sum / max(1, step_count)
                    forward_distance = float(info.get("forward_distance", 0.0))

                    total_forward_distance += forward_distance
                    total_mean_body_forward_velocity += mean_body_forward_velocity
                    total_mean_yaw_rate += mean_yaw_rate
                    total_mean_hind_ratio += mean_hind_ratio
                    total_mean_gait_sync += mean_gait_sync_score
                    total_mean_velocity_tracking_error += mean_velocity_tracking_error
                    total_mean_mechanical_work += mean_mechanical_work
                    total_mean_foot_slip_ratio += mean_foot_slip_ratio
                    total_mean_action_smoothness += mean_action_smoothness

                    writer.writerow(
                        [
                            utc_now_iso(),
                            CONDITION,
                            episode_number,
                            success,
                            forward_distance,
                            step_count,
                            mean_body_forward_velocity,
                            mean_velocity_tracking_error,
                            mean_mechanical_work,
                            mean_foot_slip_ratio,
                            mean_action_smoothness,
                            mean_yaw_rate,
                            mean_hind_ratio,
                            mean_rear_joint_velocity,
                            mean_front_joint_velocity,
                            mean_gait_sync_score,
                            termination_reason,
                            time.perf_counter() - episode_start_time,
                            time.perf_counter() - start_time,
                        ]
                    )

                    print(
                        f"Episode {episode_number:02d}/{EPISODES} | "
                        f"success={success} | "
                        f"distance={forward_distance:.3f} m | "
                        f"mean_body_v={mean_body_forward_velocity:.3f} m/s | "
                        f"hind_ratio={mean_hind_ratio:.3f} | "
                        f"reason={termination_reason}"
                    )
                    break

    with summary_csv_path.open("w", newline="", encoding="utf-8") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(EVALUATION_SUMMARY_COLUMNS)
        summary_writer.writerow(
            [
                utc_now_iso(),
                CONDITION,
                EPISODES,
                success_count / max(1, EPISODES),
                fall_count / max(1, EPISODES),
                truncation_count / max(1, EPISODES),
                no_progress_count / max(1, EPISODES),
                total_forward_distance / max(1, EPISODES),
                total_mean_body_forward_velocity / max(1, EPISODES),
                total_mean_velocity_tracking_error / max(1, EPISODES),
                total_mean_mechanical_work / max(1, EPISODES),
                total_mean_foot_slip_ratio / max(1, EPISODES),
                total_mean_action_smoothness / max(1, EPISODES),
                total_mean_yaw_rate / max(1, EPISODES),
                total_mean_hind_ratio / max(1, EPISODES),
                total_mean_gait_sync / max(1, EPISODES),
                success_steps_total / max(1, success_count),
            ]
        )

    print(
        "Evaluation summary | "
        f"success_rate={success_count / max(1, EPISODES):.2%} "
        f"fall_rate={fall_count / max(1, EPISODES):.2%} "
        f"no_progress_rate={no_progress_count / max(1, EPISODES):.2%} "
        f"truncation_rate={truncation_count / max(1, EPISODES):.2%} "
        f"mean_distance={total_forward_distance / max(1, EPISODES):.3f}m "
        f"mean_hind_ratio={total_mean_hind_ratio / max(1, EPISODES):.3f} "
        f"mean_gait_sync={total_mean_gait_sync / max(1, EPISODES):.3f}"
    )

    env.close()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
