import csv
import time
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from utils.logging.paper_metrics import EPISODE_METRICS_COLUMNS, utc_now_iso


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, run_dir: Path, condition: str = "clean", verbose=0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.condition = condition
        self.episode_csv_path = run_dir / f"episode_metrics_{condition}.csv"
        self.rollout_csv_path = run_dir / f"rollout_metrics_{condition}.csv"
        self.episode_file = None
        self.rollout_file = None
        self.episode_writer = None
        self.rollout_writer = None
        self.start_time = 0.0
        self.last_wall_time = 0.0
        self.last_timestep = 0
        self.episode_counter = 0
        self.env_accumulators: list[dict] = []

    def _new_accumulator(self) -> dict:
        return {
            "joint_velocity_sum": 0.0,
            "action_magnitude_sum": 0.0,
            "action_delta_sum": 0.0,
            "action_magnitude_penalty_sum": 0.0,
            "action_smoothness_penalty_sum": 0.0,
            "action_jerk_penalty_sum": 0.0,
            "yaw_rate_sum": 0.0,
            "joint_velocity_delta_penalty_sum": 0.0,
            "vertical_velocity_penalty_sum": 0.0,
            "vertical_velocity_abs_sum": 0.0,
            "feet_gait_reward_sum": 0.0,
            "joint_mirror_penalty_sum": 0.0,
            "feet_air_time_variance_penalty_sum": 0.0,
            "target_progress_reward_sum": 0.0,
            "speed_tracking_reward_sum": 0.0,
            "stride_amplitude_reward_sum": 0.0,
            "hind_engagement_reward_sum": 0.0,
            "foot_clearance_reward_sum": 0.0,
            "foot_slip_penalty_sum": 0.0,
            "speed_target_sum": 0.0,
            "front_joint_velocity_sum": 0.0,
            "rear_joint_velocity_sum": 0.0,
            "hind_motion_ratio_sum": 0.0,
            "contact_sensor_reliable_sum": 0.0,
            "using_contact_sensors_sum": 0.0,
            "no_progress_count": 0.0,
            "velocity_tracking_error_sum": 0.0,
            "mechanical_work_sum": 0.0,
            "torque_effort_penalty_sum": 0.0,
            "mechanical_power_penalty_sum": 0.0,
            "penalty_curriculum_factor_sum": 0.0,
            "foot_slip_ratio_sum": 0.0,
            "step_count": 0,
        }

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.episode_file = self.episode_csv_path.open("w", newline="", encoding="utf-8")
        self.rollout_file = self.rollout_csv_path.open("w", newline="", encoding="utf-8")

        self.episode_writer = csv.writer(self.episode_file)
        self.rollout_writer = csv.writer(self.rollout_file)

        self.episode_writer.writerow(EPISODE_METRICS_COLUMNS)

        self.rollout_writer.writerow([
            "timestamp_utc",
            "condition",
            "timestep",
            "mean_episode_reward_last_rollout",
            "mean_episode_length_last_rollout",
            "policy_loss",
            "value_loss",
            "entropy",
            "wall_clock_s_since_last_checkpoint",
            "cumulative_wall_clock_s",
            "sim_steps_per_second",
        ])

        self.start_time = time.perf_counter()
        self.last_wall_time = self.start_time
        self.last_timestep = int(self.num_timesteps)

        num_envs = 1
        if self.training_env is not None and hasattr(self.training_env, "num_envs"):
            num_envs = int(self.training_env.num_envs)
        self.env_accumulators = [self._new_accumulator() for _ in range(num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            if env_idx >= len(self.env_accumulators):
                self.env_accumulators.append(self._new_accumulator())

            accumulator = self.env_accumulators[env_idx]
            accumulator["joint_velocity_sum"] += float(info.get("mean_joint_velocity_magnitude", 0.0))
            accumulator["action_magnitude_sum"] += float(info.get("action_magnitude", 0.0))
            accumulator["action_delta_sum"] += float(info.get("action_delta_mean", 0.0))
            accumulator["action_magnitude_penalty_sum"] += float(info.get("action_magnitude_penalty", 0.0))
            accumulator["action_smoothness_penalty_sum"] += float(info.get("action_smoothness_penalty", 0.0))
            accumulator["action_jerk_penalty_sum"] += float(info.get("action_jerk_penalty", 0.0))
            accumulator["yaw_rate_sum"] += abs(float(info.get("yaw_rate", 0.0)))
            accumulator["joint_velocity_delta_penalty_sum"] += float(
                info.get("joint_velocity_delta_penalty", 0.0)
            )
            accumulator["vertical_velocity_penalty_sum"] += float(
                info.get("vertical_velocity_penalty", 0.0)
            )
            accumulator["vertical_velocity_abs_sum"] += abs(float(info.get("vertical_velocity_z", 0.0)))
            accumulator["feet_gait_reward_sum"] += float(info.get("feet_gait_reward", 0.0))
            accumulator["joint_mirror_penalty_sum"] += float(info.get("joint_mirror_penalty", 0.0))
            accumulator["feet_air_time_variance_penalty_sum"] += float(
                info.get("feet_air_time_variance_penalty", 0.0)
            )
            accumulator["target_progress_reward_sum"] += float(info.get("target_progress_reward", 0.0))
            accumulator["speed_tracking_reward_sum"] += float(info.get("speed_tracking_reward", 0.0))
            accumulator["stride_amplitude_reward_sum"] += float(info.get("stride_amplitude_reward", 0.0))
            accumulator["hind_engagement_reward_sum"] += float(info.get("hind_engagement_reward", 0.0))
            accumulator["foot_clearance_reward_sum"] += float(info.get("foot_clearance_reward", 0.0))
            accumulator["foot_slip_penalty_sum"] += float(info.get("foot_slip_penalty", 0.0))
            accumulator["speed_target_sum"] += float(info.get("speed_target", 0.0))
            accumulator["front_joint_velocity_sum"] += float(info.get("front_joint_velocity_magnitude", 0.0))
            accumulator["rear_joint_velocity_sum"] += float(info.get("rear_joint_velocity_magnitude", 0.0))
            accumulator["hind_motion_ratio_sum"] += float(info.get("hind_motion_ratio", 0.0))
            accumulator["contact_sensor_reliable_sum"] += float(info.get("contact_sensor_reliable", 0.0))
            accumulator["using_contact_sensors_sum"] += float(info.get("using_contact_sensors", 0.0))
            accumulator["no_progress_count"] += 1.0 if bool(info.get("no_progress", False)) else 0.0
            accumulator["velocity_tracking_error_sum"] += float(info.get("velocity_tracking_error", 0.0))
            accumulator["mechanical_work_sum"] += float(info.get("mean_mechanical_work", 0.0))
            accumulator["torque_effort_penalty_sum"] += float(info.get("torque_effort_penalty", 0.0))
            accumulator["mechanical_power_penalty_sum"] += float(info.get("mechanical_power_penalty", 0.0))
            accumulator["penalty_curriculum_factor_sum"] += float(info.get("penalty_curriculum_factor", 1.0))
            accumulator["foot_slip_ratio_sum"] += float(info.get("foot_slip_ratio", 0.0))
            accumulator["step_count"] += 1

            episode = info.get("episode")
            if episode is not None:
                self.episode_counter += 1
                steps = max(1, int(accumulator["step_count"]))
                cumulative_wall_clock = time.perf_counter() - self.start_time

                self.episode_writer.writerow([
                    utc_now_iso(),
                    self.condition,
                    self.episode_counter,
                    self.num_timesteps,
                    float(episode.get("r", np.nan)),
                    float(info.get("forward_distance", np.nan)),
                    int(episode.get("l", 0)),
                    info.get("termination_reason", "unknown"),
                    accumulator["joint_velocity_sum"] / steps,
                    accumulator["action_magnitude_sum"] / steps,
                    accumulator["action_delta_sum"] / steps,
                    accumulator["action_magnitude_penalty_sum"] / steps,
                    accumulator["action_smoothness_penalty_sum"] / steps,
                    accumulator["action_jerk_penalty_sum"] / steps,
                    accumulator["velocity_tracking_error_sum"] / steps,
                    accumulator["mechanical_work_sum"] / steps,
                    accumulator["torque_effort_penalty_sum"] / steps,
                    accumulator["mechanical_power_penalty_sum"] / steps,
                    accumulator["penalty_curriculum_factor_sum"] / steps,
                    accumulator["foot_slip_ratio_sum"] / steps,
                    float(info.get("body_forward_velocity", np.nan)),
                    accumulator["yaw_rate_sum"] / steps,
                    accumulator["joint_velocity_delta_penalty_sum"] / steps,
                    accumulator["vertical_velocity_penalty_sum"] / steps,
                    accumulator["vertical_velocity_abs_sum"] / steps,
                    accumulator["feet_gait_reward_sum"] / steps,
                    accumulator["joint_mirror_penalty_sum"] / steps,
                    accumulator["feet_air_time_variance_penalty_sum"] / steps,
                    accumulator["target_progress_reward_sum"] / steps,
                    accumulator["speed_tracking_reward_sum"] / steps,
                    accumulator["stride_amplitude_reward_sum"] / steps,
                    accumulator["hind_engagement_reward_sum"] / steps,
                    accumulator["foot_clearance_reward_sum"] / steps,
                    accumulator["foot_slip_penalty_sum"] / steps,
                    accumulator["speed_target_sum"] / steps,
                    accumulator["front_joint_velocity_sum"] / steps,
                    accumulator["rear_joint_velocity_sum"] / steps,
                    accumulator["hind_motion_ratio_sum"] / steps,
                    accumulator["contact_sensor_reliable_sum"] / steps,
                    accumulator["using_contact_sensors_sum"] / steps,
                    int(accumulator["no_progress_count"] > 0),
                    int(info.get("success", False)),
                    float(episode.get("t", np.nan)),
                    cumulative_wall_clock,
                ])
                self.env_accumulators[env_idx] = self._new_accumulator()
        return True

    def _on_rollout_end(self) -> None:
        values = self.model.logger.name_to_value
        current_wall_time = time.perf_counter()
        cumulative_wall_clock = current_wall_time - self.start_time
        wall_clock_delta = current_wall_time - self.last_wall_time
        timestep_delta = int(self.num_timesteps) - self.last_timestep
        sim_steps_per_second = float(timestep_delta) / max(wall_clock_delta, 1e-9)

        self.rollout_writer.writerow([
            utc_now_iso(),
            self.condition,
            int(self.num_timesteps),
            float(values.get("rollout/ep_rew_mean", np.nan)),
            float(values.get("rollout/ep_len_mean", np.nan)),
            float(values.get("train/policy_gradient_loss", np.nan)),
            float(values.get("train/value_loss", np.nan)),
            float(values.get("train/entropy_loss", np.nan)),
            wall_clock_delta,
            cumulative_wall_clock,
            sim_steps_per_second,
        ])

        self.last_wall_time = current_wall_time
        self.last_timestep = int(self.num_timesteps)

    def _on_training_end(self) -> None:
        if self.episode_file is not None:
            self.episode_file.close()
        if self.rollout_file is not None:
            self.rollout_file.close()
