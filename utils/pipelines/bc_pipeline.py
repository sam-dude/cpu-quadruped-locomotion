from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from utils.callbacks.training_callbacks import TrainingMetricsCallback

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRAINING_RUNS_DIR = WORKSPACE_ROOT / "training_runs"


def get_trot_reference(
    t: float,
    frequency: float = 1.5,
    amplitude: float = 0.4,
    calf_lift_scale: float = 1.25,
    hip_swing_scale: float = 0.85,
    thigh_stride_scale: float = 1.0,
    thigh_stride_direction: float = -1.0,
    lift_on_positive_stride: bool = True,
    thigh_extension_scale: float = 0.65,
    left_leg_gain: float = 1.0,
    right_leg_gain: float = 1.0,
) -> np.ndarray:
    phase_fl_rr = 2 * np.pi * frequency * t
    phase_fr_rl = phase_fl_rr + np.pi

    # Forward-biased trot reference:
    # - Thigh stride drives fore-aft stepping on ground contact.
    # - Lift (unipolar) occurs during swing to avoid dragging.
    # - Hip swing is kept for robustness but mirrored across left/right legs
    #   to avoid cumulative lateral drift.
    hip_swing_amplitude = float(amplitude) * max(0.0, float(hip_swing_scale))
    thigh_stride_amplitude = float(amplitude) * max(0.0, float(thigh_stride_scale))
    extension_amplitude = float(amplitude)
    calf_extension_scale = max(0.0, float(calf_lift_scale))
    thigh_extension_scale = max(0.0, float(thigh_extension_scale))

    stride_direction = 1.0 if float(thigh_stride_direction) >= 0.0 else -1.0

    def leg_targets(phase: float, hip_side_sign: float, leg_gain: float) -> tuple[float, float, float]:
        stride = stride_direction * np.sin(phase)
        lift_phase_value = stride if lift_on_positive_stride else -stride
        lift = max(0.0, lift_phase_value)
        gain = float(np.clip(leg_gain, 0.7, 1.3))

        hip = gain * hip_side_sign * hip_swing_amplitude * stride
        thigh = 0.8 + gain * (
            thigh_stride_amplitude * stride
            + thigh_extension_scale * extension_amplitude * lift
        )
        calf = -1.5 - gain * (calf_extension_scale * extension_amplitude * lift)
        return hip, thigh, calf

    fl_hip, fl_thigh, fl_calf = leg_targets(phase_fl_rr, hip_side_sign=+1.0, leg_gain=left_leg_gain)
    fr_hip, fr_thigh, fr_calf = leg_targets(phase_fr_rl, hip_side_sign=-1.0, leg_gain=right_leg_gain)
    rl_hip, rl_thigh, rl_calf = leg_targets(phase_fr_rl, hip_side_sign=+1.0, leg_gain=left_leg_gain)
    rr_hip, rr_thigh, rr_calf = leg_targets(phase_fl_rr, hip_side_sign=-1.0, leg_gain=right_leg_gain)

    return np.array([
        fl_hip, fl_thigh, fl_calf,
        fr_hip, fr_thigh, fr_calf,
        rl_hip, rl_thigh, rl_calf,
        rr_hip, rr_thigh, rr_calf,
    ], dtype=np.float64)


def to_normalized_action(env, target_joint_angles: np.ndarray) -> np.ndarray:
    normalized = (target_joint_angles - env.default_pose_array) / env.action_scale
    return np.clip(normalized, env.action_space.low, env.action_space.high)


def create_run_dir(prefix: str) -> Path:
    run_name = datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S")
    run_dir = TRAINING_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def find_artifact(run_folder_name: str | None, artifact_name: str, run_prefix: str) -> Path:
    if run_folder_name:
        artifact = TRAINING_RUNS_DIR / run_folder_name / artifact_name
        if not artifact.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact}")
        return artifact

    candidates = sorted(
        TRAINING_RUNS_DIR.glob(f"{run_prefix}_*/{artifact_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No artifact found for prefix '{run_prefix}' and file '{artifact_name}'."
        )
    return candidates[0]


def collect_reference_trajectory(
    env,
    output_path: Path,
    n_steps: int = 10_000,
    frequency: float = 1.5,
    amplitude: float = 0.4,
    calf_lift_scale: float = 1.25,
    hip_swing_scale: float = 0.85,
    thigh_stride_scale: float = 1.0,
    thigh_stride_direction: float = -1.0,
    lift_on_positive_stride: bool = True,
    thigh_extension_scale: float = 0.65,
    left_leg_gain: float = 1.0,
    right_leg_gain: float = 1.0,
    auto_select_stride_direction: bool = True,
    auto_select_lift_phase: bool = True,
    direction_probe_steps: int = 220,
    probe_heading_penalty_weight: float = 0.35,
    probe_lateral_penalty_weight: float = 0.15,
    probe_clearance_reward_weight: float = 0.10,
    recording_warmup_steps: int = 200,
    per_reset_warmup_steps: int = 120,
    skip_initial_episodes: int = 0,
    push_start_after_steps: int = 250,
    sanity_forward_check_steps: int = 160,
    sanity_min_forward_m: float = 0.02,
    auto_recover_direction_on_negative_sanity: bool = True,
    enable_random_pushes: bool = True,
    push_interval_seconds: float = 3.0,
    push_duration_steps: int = 2,
    push_force_min_n: float = 12.0,
    push_force_max_n: float = 24.0,
    push_forward_bias: float = 0.85,
    push_lateral_ratio: float = 0.1,
    balance_lateral_pushes: bool = True,
    push_vertical_force_n: float = 0.0,
    rng_seed: int | None = 42,
) -> Path:
    observations = []
    actions = []
    dt = env.timestep / 1000.0
    rng = np.random.default_rng(rng_seed)

    def _phase_aligned_start_time(direction_value: float, lift_on_positive_value: bool) -> float:
        # Choose a reset phase that starts in a stance-like forward-driving segment
        # to avoid backward transients at episode boundaries.
        stride_direction_local = 1.0 if float(direction_value) >= 0.0 else -1.0
        desired_stride_sign = -1.0 if bool(lift_on_positive_value) else 1.0
        sin_target = float(np.clip(desired_stride_sign / stride_direction_local, -1.0, 1.0))
        phase = 0.5 * np.pi if sin_target >= 0.0 else 1.5 * np.pi
        return float(phase / (2.0 * np.pi * max(float(frequency), 1e-6)))

    def _probe_stride_direction(direction_value: float, lift_on_positive_value: bool) -> tuple[float, float, float, float, float]:
        probe_obs, _ = env.reset()
        probe_t = _phase_aligned_start_time(direction_value, lift_on_positive_value)
        probe_distance = 0.0
        heading_error_abs_sum = 0.0
        lateral_velocity_abs_sum = 0.0
        clearance_reward_sum = 0.0
        measured_steps = 0
        max_steps = int(max(40, direction_probe_steps))
        for _ in range(max_steps):
            probe_pose = get_trot_reference(
                probe_t,
                frequency=frequency,
                amplitude=amplitude,
                calf_lift_scale=calf_lift_scale,
                hip_swing_scale=hip_swing_scale,
                thigh_stride_scale=thigh_stride_scale,
                thigh_stride_direction=direction_value,
                lift_on_positive_stride=lift_on_positive_value,
                thigh_extension_scale=thigh_extension_scale,
                left_leg_gain=left_leg_gain,
                right_leg_gain=right_leg_gain,
            )
            probe_action = to_normalized_action(env, probe_pose)
            probe_obs, _, probe_terminated, probe_truncated, probe_info = env.step(probe_action)
            probe_t += dt
            probe_distance = float(probe_info.get("forward_distance", probe_distance))
            heading_error_abs_sum += abs(float(probe_info.get("heading_error", 0.0)))
            lateral_velocity_abs_sum += abs(float(probe_info.get("lateral_velocity_command_axis", 0.0)))
            clearance_reward_sum += float(probe_info.get("foot_clearance_reward", 0.0))
            measured_steps += 1
            if probe_terminated or probe_truncated:
                break
        avg_heading_error_abs = heading_error_abs_sum / max(1, measured_steps)
        avg_lateral_velocity_abs = lateral_velocity_abs_sum / max(1, measured_steps)
        avg_clearance_reward = clearance_reward_sum / max(1, measured_steps)
        probe_score = (
            probe_distance
            - probe_heading_penalty_weight * avg_heading_error_abs
            - probe_lateral_penalty_weight * avg_lateral_velocity_abs
            + probe_clearance_reward_weight * avg_clearance_reward
        )
        return (
            probe_score,
            probe_distance,
            avg_heading_error_abs,
            avg_lateral_velocity_abs,
            avg_clearance_reward,
        )

    probe_results: list[tuple[float, bool, float, float, float, float, float]] = []
    if auto_select_stride_direction or auto_select_lift_phase:
        candidate_pairs = [(-1.0, True), (+1.0, True), (-1.0, False), (+1.0, False)]
        if not auto_select_stride_direction:
            candidate_pairs = [(thigh_stride_direction, True), (thigh_stride_direction, False)]
        elif not auto_select_lift_phase:
            candidate_pairs = [(-1.0, lift_on_positive_stride), (+1.0, lift_on_positive_stride)]

        best_score = -np.inf
        best_direction = thigh_stride_direction
        best_lift_on_positive = lift_on_positive_stride
        probe_results = []

        for cand_direction, cand_lift_on_positive in candidate_pairs:
            (
                cand_score,
                cand_forward_distance,
                cand_heading_error_abs,
                cand_lateral_velocity_abs,
                cand_clearance_reward,
            ) = _probe_stride_direction(cand_direction, cand_lift_on_positive)
            probe_results.append(
                (
                    cand_direction,
                    cand_lift_on_positive,
                    cand_score,
                    cand_forward_distance,
                    cand_heading_error_abs,
                    cand_lateral_velocity_abs,
                    cand_clearance_reward,
                )
            )
            if cand_score > best_score:
                best_score = cand_score
                best_direction = cand_direction
                best_lift_on_positive = cand_lift_on_positive

        thigh_stride_direction = best_direction
        lift_on_positive_stride = best_lift_on_positive

        probe_result_text = " ".join(
            [
                (
                    f"d={cand_direction:+.1f}/lift_pos={int(cand_lift_on_positive)}"
                    f":score={cand_score:.4f},fwd={cand_forward_distance:.4f},"
                    f"head={cand_heading_error_abs:.3f},lat={cand_lateral_velocity_abs:.3f},"
                    f"clr={cand_clearance_reward:.3f}"
                )
                for (
                    cand_direction,
                    cand_lift_on_positive,
                    cand_score,
                    cand_forward_distance,
                    cand_heading_error_abs,
                    cand_lateral_velocity_abs,
                    cand_clearance_reward,
                ) in probe_results
            ]
        )
        print(
            "[Trajectory] Direction probe | "
            f"{probe_result_text} "
            f"selected=d{thigh_stride_direction:+.1f}/lift_pos={int(lift_on_positive_stride)}"
        )
        best_forward_distance = max(
            [entry[3] for entry in probe_results],
            default=0.0,
        )
        if best_forward_distance <= 0.0:
            print(
                "[Trajectory] Warning: all probe candidates show non-positive forward progress. "
                "Consider increasing TROT_THIGH_STRIDE_SCALE or reducing TROT_HIP_SWING_SCALE."
            )

    def _sanity_forward_distance(direction_value: float, lift_on_positive_value: bool) -> float:
        sanity_obs, _ = env.reset()
        sanity_t = _phase_aligned_start_time(direction_value, lift_on_positive_value)
        sanity_forward_distance = 0.0
        max_steps = int(max(40, sanity_forward_check_steps))
        for _ in range(max_steps):
            sanity_pose = get_trot_reference(
                sanity_t,
                frequency=frequency,
                amplitude=amplitude,
                calf_lift_scale=calf_lift_scale,
                hip_swing_scale=hip_swing_scale,
                thigh_stride_scale=thigh_stride_scale,
                thigh_stride_direction=direction_value,
                lift_on_positive_stride=lift_on_positive_value,
                thigh_extension_scale=thigh_extension_scale,
                left_leg_gain=left_leg_gain,
                right_leg_gain=right_leg_gain,
            )
            sanity_action = to_normalized_action(env, sanity_pose)
            sanity_obs, _, sanity_terminated, sanity_truncated, sanity_info = env.step(sanity_action)
            sanity_t += dt
            sanity_forward_distance = float(
                sanity_info.get("forward_distance", sanity_forward_distance)
            )
            if sanity_terminated or sanity_truncated:
                break
        return sanity_forward_distance

    sanity_forward_distance = _sanity_forward_distance(
        thigh_stride_direction,
        lift_on_positive_stride,
    )
    if (
        auto_recover_direction_on_negative_sanity
        and sanity_forward_distance < float(sanity_min_forward_m)
        and probe_results
    ):
        best_by_forward = max(probe_results, key=lambda item: item[3])
        recovered_direction = float(best_by_forward[0])
        recovered_lift_pos = bool(best_by_forward[1])
        recovered_forward = float(best_by_forward[3])
        if recovered_forward > sanity_forward_distance:
            thigh_stride_direction = recovered_direction
            lift_on_positive_stride = recovered_lift_pos
            sanity_forward_distance = _sanity_forward_distance(
                thigh_stride_direction,
                lift_on_positive_stride,
            )

    print(
        "[Trajectory] Sanity check | "
        f"forward_m={sanity_forward_distance:.4f} "
        f"selected=d{thigh_stride_direction:+.1f}/lift_pos={int(lift_on_positive_stride)}"
    )
    if sanity_forward_distance < float(sanity_min_forward_m):
        print(
            "[Trajectory] Warning: sanity forward distance is low/non-positive. "
            "The first collected steps may still contain weak or backward motion."
        )

    obs, _ = env.reset()
    t = _phase_aligned_start_time(thigh_stride_direction, lift_on_positive_stride)
    steps_since_reset = 0
    completed_episodes = 0

    push_api_available = bool(getattr(env, "robot_node", None) is not None)
    if push_api_available:
        push_api_available = callable(getattr(env.robot_node, "addForce", None))

    push_enabled = bool(enable_random_pushes and push_api_available)
    interval_steps = int(max(1, round(push_interval_seconds / max(dt, 1e-6))))
    jitter_steps = max(1, int(round(0.35 * interval_steps)))
    next_push_step = interval_steps
    active_push_steps = 0
    current_push = np.zeros(3, dtype=np.float64)
    push_events = 0
    push_steps_applied = 0
    lateral_impulse_accumulator = 0.0
    action_clip_ratio_sum = 0.0
    action_clip_ratio_max = 0.0

    if enable_random_pushes and not push_api_available:
        print("[Trajectory] Random pushes requested but unavailable: env.robot_node.addForce not found.")

    rollout_steps = int(n_steps + max(0, int(recording_warmup_steps)))
    for step_idx in range(rollout_steps):
        reference_pose = get_trot_reference(
            t,
            frequency=frequency,
            amplitude=amplitude,
            calf_lift_scale=calf_lift_scale,
            hip_swing_scale=hip_swing_scale,
            thigh_stride_scale=thigh_stride_scale,
            thigh_stride_direction=thigh_stride_direction,
            lift_on_positive_stride=lift_on_positive_stride,
            thigh_extension_scale=thigh_extension_scale,
            left_leg_gain=left_leg_gain,
            right_leg_gain=right_leg_gain,
        )
        reference_action = to_normalized_action(env, reference_pose)

        if (
            push_enabled
            and completed_episodes >= int(max(0, skip_initial_episodes))
            and step_idx >= int(max(0, push_start_after_steps))
            and steps_since_reset >= int(max(0, per_reset_warmup_steps))
        ):
            if active_push_steps <= 0 and step_idx >= next_push_step:
                magnitude = float(rng.uniform(push_force_min_n, push_force_max_n))
                lateral_ratio = float(np.clip(push_lateral_ratio, 0.0, 0.49))
                forward_bias = float(np.clip(push_forward_bias, 0.0, 1.0))

                forward_sign = 1.0 if float(rng.random()) < forward_bias else -1.0
                if balance_lateral_pushes and abs(lateral_impulse_accumulator) > 1e-9:
                    lateral_sign = -float(np.sign(lateral_impulse_accumulator))
                else:
                    lateral_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=np.float64)))

                forward_force = forward_sign * magnitude * (1.0 - lateral_ratio)
                lateral_force = lateral_sign * magnitude * lateral_ratio
                current_push = np.array(
                    [forward_force, lateral_force, float(push_vertical_force_n)],
                    dtype=np.float64,
                )
                active_push_steps = int(max(1, push_duration_steps))
                push_events += 1

                interval_jitter = int(rng.integers(-jitter_steps, jitter_steps + 1))
                next_push_step = step_idx + int(max(1, interval_steps + interval_jitter))

            if active_push_steps > 0:
                try:
                    env.robot_node.addForce(current_push.tolist(), False)
                    push_steps_applied += 1
                    lateral_impulse_accumulator += float(current_push[1])
                except Exception:
                    push_enabled = False
                active_push_steps -= 1

        if (
            step_idx >= int(max(0, recording_warmup_steps))
            and completed_episodes >= int(max(0, skip_initial_episodes))
            and steps_since_reset >= int(max(0, per_reset_warmup_steps))
        ):
            observations.append(obs)
            actions.append(reference_action)

        obs, _, terminated, truncated, info = env.step(reference_action)
        t += dt

        clip_ratio = float(info.get("action_clip_ratio", 0.0))
        action_clip_ratio_sum += clip_ratio
        action_clip_ratio_max = max(action_clip_ratio_max, clip_ratio)

        if terminated or truncated:
            completed_episodes += 1
            obs, _ = env.reset()
            t = _phase_aligned_start_time(thigh_stride_direction, lift_on_positive_stride)
            steps_since_reset = 0
        else:
            steps_since_reset += 1

        if len(observations) >= int(n_steps):
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        frequency_hz=np.array([frequency], dtype=np.float32),
        amplitude=np.array([amplitude], dtype=np.float32),
        calf_lift_scale=np.array([calf_lift_scale], dtype=np.float32),
        hip_swing_scale=np.array([hip_swing_scale], dtype=np.float32),
        thigh_stride_scale=np.array([thigh_stride_scale], dtype=np.float32),
        thigh_stride_direction=np.array([thigh_stride_direction], dtype=np.float32),
        lift_on_positive_stride=np.array([int(lift_on_positive_stride)], dtype=np.int32),
        thigh_extension_scale=np.array([thigh_extension_scale], dtype=np.float32),
        left_leg_gain=np.array([left_leg_gain], dtype=np.float32),
        right_leg_gain=np.array([right_leg_gain], dtype=np.float32),
        mean_action_clip_ratio=np.array([action_clip_ratio_sum / max(1, rollout_steps)], dtype=np.float32),
        max_action_clip_ratio=np.array([action_clip_ratio_max], dtype=np.float32),
        recording_warmup_steps=np.array([recording_warmup_steps], dtype=np.int32),
        per_reset_warmup_steps=np.array([per_reset_warmup_steps], dtype=np.int32),
        skip_initial_episodes=np.array([skip_initial_episodes], dtype=np.int32),
        push_start_after_steps=np.array([push_start_after_steps], dtype=np.int32),
        sanity_forward_check_steps=np.array([sanity_forward_check_steps], dtype=np.int32),
        sanity_forward_distance_m=np.array([sanity_forward_distance], dtype=np.float32),
        reset_phase_aligned=np.array([1], dtype=np.int32),
        random_pushes_enabled=np.array([int(push_enabled)], dtype=np.int32),
        random_push_event_count=np.array([push_events], dtype=np.int32),
        random_push_steps_applied=np.array([push_steps_applied], dtype=np.int32),
        random_push_forward_bias=np.array([push_forward_bias], dtype=np.float32),
        random_push_lateral_ratio=np.array([push_lateral_ratio], dtype=np.float32),
        random_push_lateral_impulse_sum=np.array([lateral_impulse_accumulator], dtype=np.float32),
    )
    mean_clip_ratio = action_clip_ratio_sum / max(1, rollout_steps)
    print(
        "[Trajectory] Collection summary | "
        f"samples={len(observations)} "
        f"warmup_steps={int(max(0, recording_warmup_steps))} "
        f"per_reset_warmup_steps={int(max(0, per_reset_warmup_steps))} "
        f"skip_initial_episodes={int(max(0, skip_initial_episodes))} "
        "reset_phase_aligned=1 "
        f"push_start_after_steps={int(max(0, push_start_after_steps))} "
        f"mean_clip={mean_clip_ratio:.3f} "
        f"max_clip={action_clip_ratio_max:.3f} "
        f"hip_swing_scale={hip_swing_scale:.2f} "
        f"thigh_stride_scale={thigh_stride_scale:.2f} "
        f"thigh_stride_direction={thigh_stride_direction:+.1f} "
        f"lift_on_positive_stride={int(lift_on_positive_stride)} "
        f"left_leg_gain={left_leg_gain:.2f} "
        f"right_leg_gain={right_leg_gain:.2f} "
        f"calf_lift_scale={calf_lift_scale:.2f} "
        f"push_enabled={int(push_enabled)} "
        f"push_forward_bias={float(push_forward_bias):.2f} "
        f"push_lateral_ratio={float(push_lateral_ratio):.2f} "
        f"push_lateral_impulse_sum={lateral_impulse_accumulator:.3f} "
        f"push_events={push_events} "
        f"push_steps_applied={push_steps_applied}"
    )
    if mean_clip_ratio > 0.02:
        print(
            "[Trajectory] Warning: reference actions are being clipped by GO1_RESIDUAL_MAX_RAD. "
            "Consider reducing amplitudes or increasing residual limit for collection."
        )
    return output_path


def train_bc_from_dataset(
    env,
    dataset_path: Path,
    output_model_path: Path,
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> Path:
    dataset = np.load(dataset_path)
    obs_data = dataset["observations"]
    action_data = dataset["actions"]

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)

    obs_tensor = torch.tensor(obs_data, dtype=torch.float32)
    action_tensor = torch.tensor(action_data, dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(obs_tensor, action_tensor)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(model.device)
            batch_actions = batch_actions.to(model.device)

            optimizer.zero_grad()
            distribution = model.policy.get_distribution(batch_obs)
            predicted_actions = distribution.distribution.mean
            loss = nn.MSELoss()(predicted_actions, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            avg_loss = total_loss / max(1, len(loader))
            print(f"BC epoch {epoch + 1}/{n_epochs} | loss={avg_loss:.6f}")

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_model_path))
    return output_model_path


def build_ppo_callbacks(
    run_dir: Path,
    checkpoint_freq: int = 5_000,
    condition: str = "clean",
) -> CallbackList:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="go1_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = TrainingMetricsCallback(run_dir=run_dir, condition=condition)
    return CallbackList([checkpoint_callback, metrics_callback])


def make_monitored_env(env, run_dir: Path):
    return Monitor(env, filename=str(run_dir / "monitor.csv"))
