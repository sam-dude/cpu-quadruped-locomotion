from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


EPISODE_METRICS_COLUMNS = [
    "timestamp_utc",
    "condition",
    "episode_number",
    "timestep",
    "total_reward",
    "forward_distance_m",
    "episode_length_steps",
    "termination_reason",
    "mean_joint_velocity_magnitude",
    "mean_action_magnitude",
    "mean_action_smoothness",
    "mean_action_magnitude_penalty",
    "mean_action_smoothness_penalty",
    "mean_action_jerk_penalty",
    "mean_velocity_tracking_error",
    "mean_mechanical_work",
    "mean_torque_effort_penalty",
    "mean_mechanical_power_penalty",
    "mean_penalty_curriculum_factor",
    "mean_foot_slip_ratio",
    "end_body_forward_velocity",
    "mean_yaw_rate",
    "mean_joint_velocity_delta_penalty",
    "mean_vertical_velocity_penalty",
    "mean_abs_vertical_velocity",
    "mean_feet_gait_reward",
    "mean_joint_mirror_penalty",
    "mean_feet_air_time_variance_penalty",
    "mean_target_progress_reward",
    "mean_speed_tracking_reward",
    "mean_stride_amplitude_reward",
    "mean_hind_engagement_reward",
    "mean_foot_clearance_reward",
    "mean_foot_slip_penalty",
    "mean_speed_target",
    "mean_front_joint_velocity_magnitude",
    "mean_rear_joint_velocity_magnitude",
    "mean_hind_motion_ratio",
    "mean_contact_sensor_reliable",
    "mean_using_contact_sensors",
    "no_progress_termination",
    "success",
    "episode_wall_clock_s",
    "cumulative_wall_clock_s",
]


EVALUATION_EPISODE_COLUMNS = [
    "timestamp_utc",
    "condition",
    "episode_number",
    "success",
    "forward_distance_m",
    "episode_survival_length",
    "mean_body_forward_velocity",
    "mean_velocity_tracking_error",
    "mean_mechanical_work",
    "mean_foot_slip_ratio",
    "mean_action_smoothness",
    "mean_yaw_rate",
    "mean_hind_motion_ratio",
    "mean_rear_joint_velocity_magnitude",
    "mean_front_joint_velocity_magnitude",
    "mean_gait_sync_score",
    "termination_reason",
    "episode_wall_clock_s",
    "cumulative_wall_clock_s",
]


EVALUATION_SUMMARY_COLUMNS = [
    "timestamp_utc",
    "condition",
    "episodes",
    "success_rate",
    "fall_rate",
    "truncation_rate",
    "no_progress_rate",
    "mean_forward_distance_m",
    "mean_body_forward_velocity",
    "mean_velocity_tracking_error",
    "mean_mechanical_work",
    "mean_foot_slip_ratio",
    "mean_action_smoothness",
    "mean_yaw_rate",
    "mean_hind_motion_ratio",
    "mean_gait_sync_score",
    "mean_steps_to_success",
]
