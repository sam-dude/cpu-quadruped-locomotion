from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Go1RewardWeights:
    base: float = 0.25
    alive_penalty: float = 0.01
    upright: float = 1.2
    height: float = 0.6
    pose: float = 0.6

    forward_scale: float = 1.5
    forward_clip_max: float = 1.5

    heading: float = 0.3
    yaw_rate: float = 0.05
    action_magnitude: float = 0.03
    action_smoothness: float = 0.03
    action_jerk: float = 0.015
    joint_velocity_delta: float = 2.5e-4
    vertical_velocity: float = 2.0

    feet_gait: float = 0.5
    joint_mirror: float = 0.05
    feet_air_time_variance: float = 2.5
    target_progress: float = 8.0
    speed_tracking: float = 0.8
    stride_amplitude: float = 0.35
    hind_engagement: float = 0.4
    foot_clearance: float = 0.45
    foot_slip: float = 0.25
    torque_effort: float = 0.012
    mechanical_power: float = 0.06

    speed_target: float = 0.85
    speed_tracking_sigma: float = 0.35
    target_progress_clip: float = 0.15
    stride_amplitude_scale: float = 2.0
    hind_engagement_target_ratio: float = 0.5
    hind_engagement_sigma: float = 0.18
    clearance_target: float = 0.22
    clearance_sigma: float = 0.10
    slip_velocity_scale: float = 4.5
    torque_scale: float = 12.0
    mechanical_power_scale: float = 20.0

    upright_exp_scale: float = 4.0
    height_exp_scale: float = 35.0
    pose_exp_scale: float = 2.5


class Go1RewardComputer:
    def __init__(
        self,
        dt: float,
        default_pose_array: np.ndarray,
        target_height: float,
        weights: Go1RewardWeights | None = None,
    ) -> None:
        self.dt = float(max(dt, 1e-6))
        self.default_pose_array = np.asarray(default_pose_array, dtype=np.float64)
        self.target_height = float(target_height)
        self.weights = weights if weights is not None else Go1RewardWeights()

        self._prev_airborne = np.zeros(4, dtype=bool)
        self._air_time_accum = np.zeros(4, dtype=np.float64)
        self._last_air_times = np.zeros(4, dtype=np.float64)
        self._prev_target_distance: float | None = None

    @staticmethod
    def formula_text() -> str:
        return (
            "reward = 0.25 - 0.01 + M_prog*(1.2*upright + 0.6*height + 0.6*pose) + forward_reward "
            "+ k_c*(0.5*feet_gait_sync) "
            "+ 8.0*target_progress + 0.8*speed_tracking + 0.35*stride_amplitude + 0.4*hind_engagement "
            "+ k_c*(0.45*foot_clearance) "
            "- 0.3*heading_error_to_target - 0.05*abs(yaw_rate) "
            "- k_c*(0.03*||action||^2 + 0.03*||action-prev_action||^2 + 0.015*||action-2*prev+prev_prev||^2) "
            "- k_c*(2.5e-4*||joint_vel-prev_joint_vel||^2 + 2.0*vz^2) "
            "- k_c*(0.05*joint_mirror_error + 2.5*feet_air_time_variance_proxy + 0.25*foot_slip "
            "+ 0.012*torque_effort + 0.06*mechanical_power)"
        )

    @staticmethod
    def simple_formula_text() -> str:
        return (
            "reward = 0.25 + 1.2*upright + 0.8*height + 0.6*pose + forward_reward "
            "- w_vy*(v_y_cmd_axis-v_cmd_y)^2 - w_wz*(w_z-w_cmd_z)^2 "
            "- k_c*(w_smooth*||action-prev_action||^2 + w_tau*tau_effort + w_slip*foot_slip)"
        )

    def reset_state(self, initial_target_distance: float | None = None) -> None:
        self._prev_airborne[:] = False
        self._air_time_accum[:] = 0.0
        self._last_air_times[:] = 0.0
        if initial_target_distance is None:
            self._prev_target_distance = None
        else:
            self._prev_target_distance = float(initial_target_distance)

    @staticmethod
    def _wrapped_abs_angle_diff(a: float, b: float) -> float:
        diff = a - b
        return abs(float(np.arctan2(np.sin(diff), np.cos(diff))))

    @staticmethod
    def _pair_sync_score(signal_a: float, signal_b: float) -> float:
        value = 1.0 - 0.5 * ((signal_a - signal_b) ** 2)
        return float(np.clip(value, 0.0, 1.0))

    def _gait_terms(
        self,
        current_pose: np.ndarray,
        joint_velocities: np.ndarray,
        foot_contacts: np.ndarray | None = None,
    ) -> tuple[float, float, float, float, float, float, float]:
        calf_velocity_indices = [2, 5, 8, 11]
        calf_position_indices = [2, 5, 8, 11]

        calf_vel = np.asarray(joint_velocities[calf_velocity_indices], dtype=np.float64)
        calf_pos = np.asarray(current_pose[calf_position_indices], dtype=np.float64)
        calf_default = np.asarray(self.default_pose_array[calf_position_indices], dtype=np.float64)

        if foot_contacts is not None and foot_contacts.size == 4:
            contacts = np.asarray(foot_contacts, dtype=np.float64)
            fl_rr_sync = 1.0 - abs(float(contacts[0] - contacts[3]))
            fr_rl_sync = 1.0 - abs(float(contacts[1] - contacts[2]))
            gait_sync_score = float(np.clip(0.5 * (fl_rr_sync + fr_rl_sync), 0.0, 1.0))
            airborne = contacts < 0.5
            stance = np.logical_not(airborne)
            using_contact_sensors = 1.0
        else:
            phase_signal = np.tanh(0.15 * calf_vel)
            fl_rr_sync = self._pair_sync_score(float(phase_signal[0]), float(phase_signal[3]))
            fr_rl_sync = self._pair_sync_score(float(phase_signal[1]), float(phase_signal[2]))
            gait_sync_score = 0.5 * (fl_rr_sync + fr_rl_sync)
            airborne = calf_pos > (calf_default + 0.20)
            stance = np.logical_not(airborne)
            using_contact_sensors = 0.0

        fl = current_pose[0:3]
        fr = current_pose[3:6]
        rl = current_pose[6:9]
        rr = current_pose[9:12]
        joint_mirror_error = float(np.mean(np.square(fl - rr)) + np.mean(np.square(fr - rl)))

        touchdown = np.logical_and(self._prev_airborne, np.logical_not(airborne))

        self._air_time_accum[airborne] += self.dt
        if np.any(touchdown):
            self._last_air_times[touchdown] = self._air_time_accum[touchdown]
            self._air_time_accum[touchdown] = 0.0
        self._prev_airborne = airborne

        valid_air_times = self._last_air_times[self._last_air_times > 0.0]
        if valid_air_times.size >= 2:
            air_time_variance = float(np.var(valid_air_times))
        else:
            air_time_variance = 0.0

        stride_amplitude_raw = float(np.mean(np.abs(calf_pos - calf_default)))
        stride_amplitude_score = float(
            np.clip(self.weights.stride_amplitude_scale * stride_amplitude_raw, 0.0, 1.0)
        )

        if np.any(airborne):
            swing_clearance = float(np.mean(np.maximum(calf_pos[airborne] - calf_default[airborne], 0.0)))
        else:
            swing_clearance = 0.0
        clearance_error = (
            (swing_clearance - self.weights.clearance_target)
            / max(self.weights.clearance_sigma, 1e-6)
        )
        foot_clearance_term = float(np.exp(-(clearance_error * clearance_error)))

        if np.any(stance):
            foot_slip_term = float(
                np.mean(np.abs(calf_vel[stance])) / max(self.weights.slip_velocity_scale, 1e-6)
            )
        else:
            foot_slip_term = 0.0

        return (
            gait_sync_score,
            joint_mirror_error,
            air_time_variance,
            stride_amplitude_score,
            using_contact_sensors,
            foot_clearance_term,
            foot_slip_term,
        )

    def compute(
        self,
        *,
        roll: float,
        pitch: float,
        yaw: float,
        initial_yaw: float,
        heading_reference_yaw: float | None = None,
        yaw_rate: float,
        height: float,
        current_pose: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        prev_prev_action: np.ndarray,
        joint_velocities: np.ndarray,
        prev_joint_velocities: np.ndarray,
        vz: float,
        forward_velocity: float,
        target_distance_remaining: float,
        foot_contacts: np.ndarray | None = None,
        speed_target_override: float | None = None,
        measured_torques: np.ndarray | None = None,
        penalty_curriculum_factor: float = 1.0,
    ) -> dict[str, float]:
        curriculum_factor = float(np.clip(penalty_curriculum_factor, 0.0, 1.0))
        heading_reference = float(initial_yaw) if heading_reference_yaw is None else float(heading_reference_yaw)
        heading_error = self._wrapped_abs_angle_diff(yaw, heading_reference)

        upright_term = float(np.exp(-self.weights.upright_exp_scale * (roll * roll + pitch * pitch)))
        height_term = float(np.exp(-self.weights.height_exp_scale * abs(height - self.target_height)))
        pose_error = float(np.mean(np.abs(current_pose - self.default_pose_array)))
        pose_term = float(np.exp(-self.weights.pose_exp_scale * pose_error))

        forward_reward = self.weights.forward_scale * float(
            np.clip(forward_velocity, 0.0, self.weights.forward_clip_max)
        )

        heading_penalty = self.weights.heading * heading_error
        yaw_rate_penalty = self.weights.yaw_rate * abs(float(yaw_rate))
        action_magnitude_penalty = curriculum_factor * self.weights.action_magnitude * float(np.sum(np.square(action)))
        action_smoothness_penalty = curriculum_factor * self.weights.action_smoothness * float(
            np.sum(np.square(action - prev_action))
        )
        action_jerk_penalty = curriculum_factor * self.weights.action_jerk * float(
            np.sum(np.square(action - 2.0 * prev_action + prev_prev_action))
        )
        joint_velocity_delta_penalty = curriculum_factor * self.weights.joint_velocity_delta * float(
            np.sum(np.square(joint_velocities - prev_joint_velocities))
        )
        vertical_velocity_penalty = curriculum_factor * self.weights.vertical_velocity * float(vz * vz)

        if measured_torques is None:
            torque_values = np.zeros_like(joint_velocities)
        else:
            torque_values = np.asarray(measured_torques, dtype=np.float64)

        mean_torque_effort = float(np.mean(np.square(torque_values)))
        mean_mechanical_work = float(np.mean(np.abs(torque_values * joint_velocities)))
        torque_effort_penalty = curriculum_factor * self.weights.torque_effort * (
            mean_torque_effort / max(self.weights.torque_scale, 1e-6)
        )
        mechanical_power_penalty = curriculum_factor * self.weights.mechanical_power * (
            mean_mechanical_work / max(self.weights.mechanical_power_scale, 1e-6)
        )

        (
            gait_sync_score,
            joint_mirror_error,
            air_time_variance,
            stride_amplitude_score,
            using_contact_sensors,
            foot_clearance_term,
            foot_slip_term,
        ) = self._gait_terms(
            current_pose=current_pose,
            joint_velocities=joint_velocities,
            foot_contacts=foot_contacts,
        )

        target_progress_delta = 0.0
        if self._prev_target_distance is not None:
            target_progress_delta = float(self._prev_target_distance - target_distance_remaining)
        self._prev_target_distance = float(target_distance_remaining)

        target_progress_term = float(
            np.clip(
                target_progress_delta,
                -self.weights.target_progress_clip,
                self.weights.target_progress_clip,
            )
        )
        target_progress_reward = self.weights.target_progress * target_progress_term

        speed_target = (
            float(speed_target_override)
            if speed_target_override is not None
            else float(self.weights.speed_target)
        )
        progress_multiplier = float(np.clip(forward_velocity / max(speed_target, 1e-6), 0.0, 1.0))
        speed_error = (forward_velocity - speed_target) / max(self.weights.speed_tracking_sigma, 1e-6)
        speed_tracking_term = float(np.exp(-(speed_error * speed_error)))
        speed_tracking_reward = self.weights.speed_tracking * speed_tracking_term

        stride_amplitude_reward = self.weights.stride_amplitude * stride_amplitude_score

        front_joint_velocity_magnitude = float(np.mean(np.abs(joint_velocities[0:6])))
        rear_joint_velocity_magnitude = float(np.mean(np.abs(joint_velocities[6:12])))
        hind_ratio = rear_joint_velocity_magnitude / max(
            front_joint_velocity_magnitude + rear_joint_velocity_magnitude,
            1e-9,
        )
        hind_balance_error = (
            (hind_ratio - self.weights.hind_engagement_target_ratio)
            / max(self.weights.hind_engagement_sigma, 1e-6)
        )
        hind_engagement_term = float(np.exp(-(hind_balance_error * hind_balance_error)))
        hind_engagement_reward = self.weights.hind_engagement * hind_engagement_term
        foot_clearance_reward = curriculum_factor * self.weights.foot_clearance * foot_clearance_term

        feet_gait_reward = curriculum_factor * self.weights.feet_gait * gait_sync_score
        joint_mirror_penalty = curriculum_factor * self.weights.joint_mirror * joint_mirror_error
        feet_air_time_variance_penalty = curriculum_factor * self.weights.feet_air_time_variance * air_time_variance
        foot_slip_penalty = curriculum_factor * self.weights.foot_slip * foot_slip_term
        foot_slip_ratio = float(np.clip(foot_slip_term, 0.0, 1.0))

        reward = (
            self.weights.base
            + progress_multiplier * self.weights.upright * upright_term
            + progress_multiplier * self.weights.height * height_term
            + progress_multiplier * self.weights.pose * pose_term
            + forward_reward
            + target_progress_reward
            + speed_tracking_reward
            + feet_gait_reward
            + stride_amplitude_reward
            + hind_engagement_reward
            + foot_clearance_reward
            - self.weights.alive_penalty
            - heading_penalty
            - yaw_rate_penalty
            - action_magnitude_penalty
            - action_smoothness_penalty
            - action_jerk_penalty
            - joint_velocity_delta_penalty
            - vertical_velocity_penalty
            - joint_mirror_penalty
            - feet_air_time_variance_penalty
            - foot_slip_penalty
            - torque_effort_penalty
            - mechanical_power_penalty
        )

        return {
            "reward": float(reward),
            "heading_error": float(heading_error),
            "heading_reference_yaw": float(heading_reference),
            "upright_term": float(upright_term),
            "height_term": float(height_term),
            "pose_term": float(pose_term),
            "pose_error": float(pose_error),
            "forward_reward": float(forward_reward),
            "target_progress_delta": float(target_progress_delta),
            "target_progress_reward": float(target_progress_reward),
            "speed_tracking_term": float(speed_tracking_term),
            "speed_tracking_reward": float(speed_tracking_reward),
            "speed_target": float(speed_target),
            "progress_multiplier": float(progress_multiplier),
            "stride_amplitude_score": float(stride_amplitude_score),
            "stride_amplitude_reward": float(stride_amplitude_reward),
            "hind_engagement_term": float(hind_engagement_term),
            "hind_engagement_reward": float(hind_engagement_reward),
            "foot_clearance_term": float(foot_clearance_term),
            "foot_clearance_reward": float(foot_clearance_reward),
            "foot_slip_term": float(foot_slip_term),
            "foot_slip_penalty": float(foot_slip_penalty),
            "foot_slip_ratio": float(foot_slip_ratio),
            "front_joint_velocity_magnitude": float(front_joint_velocity_magnitude),
            "rear_joint_velocity_magnitude": float(rear_joint_velocity_magnitude),
            "hind_motion_ratio": float(hind_ratio),
            "using_contact_sensors": float(using_contact_sensors),
            "mean_torque_effort": float(mean_torque_effort),
            "mean_mechanical_work": float(mean_mechanical_work),
            "torque_effort_penalty": float(torque_effort_penalty),
            "mechanical_power_penalty": float(mechanical_power_penalty),
            "penalty_curriculum_factor": float(curriculum_factor),
            "heading_penalty": float(heading_penalty),
            "yaw_rate_penalty": float(yaw_rate_penalty),
            "action_magnitude_penalty": float(action_magnitude_penalty),
            "action_smoothness_penalty": float(action_smoothness_penalty),
            "action_jerk_penalty": float(action_jerk_penalty),
            "joint_velocity_delta_penalty": float(joint_velocity_delta_penalty),
            "vertical_velocity_penalty": float(vertical_velocity_penalty),
            "feet_gait_reward": float(feet_gait_reward),
            "gait_sync_score": float(gait_sync_score),
            "joint_mirror_penalty": float(joint_mirror_penalty),
            "joint_mirror_error": float(joint_mirror_error),
            "feet_air_time_variance_penalty": float(feet_air_time_variance_penalty),
            "feet_air_time_variance": float(air_time_variance),
        }
