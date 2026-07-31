import sys
import os
from pathlib import Path

sys.path.append(r"C:\Program Files\Webots\lib\controller\python")

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from utils.pipelines.bc_pipeline import collect_reference_trajectory, create_run_dir
from utils.envs.go1_env import Go1Env

TRAJECTORY_STEPS = 10_000
TROT_FREQUENCY = 2.2
TROT_AMPLITUDE = 0.38
TROT_HIP_SWING_SCALE = 0.08
TROT_THIGH_STRIDE_SCALE = 1.25
TROT_THIGH_STRIDE_DIRECTION = -1.0
LIFT_ON_POSITIVE_STRIDE = False
TROT_THIGH_EXTENSION_SCALE = 0.75
TROT_CALF_LIFT_SCALE = 1.30
LEFT_LEG_GAIN = 1.00
RIGHT_LEG_GAIN = 1.06
RECORDING_WARMUP_STEPS = 420
PER_RESET_WARMUP_STEPS = 120
SKIP_INITIAL_EPISODES = 2
PUSH_START_AFTER_STEPS = 600
SANITY_FORWARD_CHECK_STEPS = 180
SANITY_MIN_FORWARD_M = 0.03
TRAJECTORY_RESIDUAL_MAX_RAD = 0.45
TRAJECTORY_MAX_EPISODE_STEPS = 1400
TRAJECTORY_NO_PROGRESS_PATIENCE_STEPS = 1800
ENABLE_RANDOM_PUSHES = True
PUSH_INTERVAL_SECONDS = 3.0
PUSH_DURATION_STEPS = 2
PUSH_FORCE_MIN_N = 6.0
PUSH_FORCE_MAX_N = 12.0
PUSH_FORWARD_BIAS = 1.00
PUSH_LATERAL_RATIO = 0.00


def main() -> None:
    os.environ["GO1_RESIDUAL_MAX_RAD"] = str(TRAJECTORY_RESIDUAL_MAX_RAD)
    os.environ["GO1_MAX_EPISODE_STEPS"] = str(TRAJECTORY_MAX_EPISODE_STEPS)
    os.environ["GO1_NO_PROGRESS_PATIENCE_STEPS"] = str(TRAJECTORY_NO_PROGRESS_PATIENCE_STEPS)

    run_dir = create_run_dir("run_traj")
    output_path = run_dir / "trajectory_data.npz"

    env = Go1Env()
    collect_reference_trajectory(
        env,
        output_path=output_path,
        n_steps=TRAJECTORY_STEPS,
        frequency=TROT_FREQUENCY,
        amplitude=TROT_AMPLITUDE,
        hip_swing_scale=TROT_HIP_SWING_SCALE,
        thigh_stride_scale=TROT_THIGH_STRIDE_SCALE,
        thigh_stride_direction=TROT_THIGH_STRIDE_DIRECTION,
        lift_on_positive_stride=LIFT_ON_POSITIVE_STRIDE,
        thigh_extension_scale=TROT_THIGH_EXTENSION_SCALE,
        calf_lift_scale=TROT_CALF_LIFT_SCALE,
        left_leg_gain=LEFT_LEG_GAIN,
        right_leg_gain=RIGHT_LEG_GAIN,
        recording_warmup_steps=RECORDING_WARMUP_STEPS,
        per_reset_warmup_steps=PER_RESET_WARMUP_STEPS,
        skip_initial_episodes=SKIP_INITIAL_EPISODES,
        push_start_after_steps=PUSH_START_AFTER_STEPS,
        sanity_forward_check_steps=SANITY_FORWARD_CHECK_STEPS,
        sanity_min_forward_m=SANITY_MIN_FORWARD_M,
        enable_random_pushes=ENABLE_RANDOM_PUSHES,
        push_interval_seconds=PUSH_INTERVAL_SECONDS,
        push_duration_steps=PUSH_DURATION_STEPS,
        push_force_min_n=PUSH_FORCE_MIN_N,
        push_force_max_n=PUSH_FORCE_MAX_N,
        push_forward_bias=PUSH_FORWARD_BIAS,
        push_lateral_ratio=PUSH_LATERAL_RATIO,
    )

    print(f"Trajectory collection complete: {output_path}")


if __name__ == "__main__":
    main()
