import sys
sys.path.append(r"C:\Program Files\Webots\lib\controller\python")
import os
from collections import deque

import gymnasium as gym
import numpy as np
from controller import Supervisor  # pyright: ignore[reportMissingImports]
from gymnasium.spaces import Box

from utils.rewards.go1_reward import Go1RewardComputer


class Go1Env(gym.Env):
    @staticmethod
    def reward_formula() -> str:
        reward_mode = os.getenv("GO1_REWARD_MODE", "advanced").strip().lower()
        if reward_mode == "simple":
            return Go1RewardComputer.simple_formula_text()
        return Go1RewardComputer.formula_text()

    def __init__(self, settle_steps: int = 20):
        super().__init__()
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.settle_steps = max(0, int(settle_steps))

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(34,), dtype=np.float64
        )
        self.action_space = Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float64
        )

        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]

        self.joints = []
        for name in joint_names:
            motor = self.supervisor.getDevice(name)
            motor.setPosition(float("inf"))
            motor.setVelocity(0.0)
            try:
                motor.enableTorqueFeedback(self.timestep)
            except Exception:
                pass
            self.joints.append(motor)

        self.default_pose = [
            0.0, 0.8, -1.5,
            0.0, 0.8, -1.5,
            0.0, 0.8, -1.5,
            0.0, 0.8, -1.5,
        ]
        self.default_pose_array = np.array(self.default_pose, dtype=np.float64)
        self.action_scale = 0.6
        self.residual_max_rad = float(os.getenv("GO1_RESIDUAL_MAX_RAD", "0.25"))
        self.residual_action_max = float(
            np.clip(self.residual_max_rad / max(self.action_scale, 1e-6), 0.0, 1.0)
        )
        self.max_joint_velocity = 8.0
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.prev_prev_action = np.zeros(12, dtype=np.float64)
        self.dt = self.timestep / 1000.0
        self.prev_joint_positions = None
        self.prev_joint_velocities = np.zeros(12, dtype=np.float64)
        self.episode_start_xy = np.zeros(2, dtype=np.float64)
        self.initial_yaw = 0.0
        self.initial_heading = np.array([1.0, 0.0], dtype=np.float64)
        self.heading_flip = os.getenv("GO1_HEADING_FLIP", "0") == "1"
        self.reward_mode = os.getenv("GO1_REWARD_MODE", "advanced").strip().lower()
        self.direction_command_mode = os.getenv("GO1_DIRECTION_COMMAND_MODE", "0") == "1"
        self.command_vx = float(os.getenv("GO1_CMD_VX", "0.8"))
        self.command_vy = float(os.getenv("GO1_CMD_VY", "0.0"))
        self.command_wz = float(os.getenv("GO1_CMD_WZ", "0.0"))
        self.command_lateral_penalty_weight = float(os.getenv("GO1_LATERAL_PENALTY_WEIGHT", "0.8"))
        self.command_yaw_penalty_weight = float(os.getenv("GO1_YAW_PENALTY_WEIGHT", "0.08"))
        self.command_energy_penalty_weight = float(os.getenv("GO1_ENERGY_PENALTY_WEIGHT", "0.012"))
        self.command_slip_penalty_weight = float(os.getenv("GO1_SLIP_PENALTY_WEIGHT", "0.25"))
        self.command_smoothness_penalty_weight = float(os.getenv("GO1_SMOOTHNESS_PENALTY_WEIGHT", "0.03"))
        self.max_episode_steps = int(os.getenv("GO1_MAX_EPISODE_STEPS", "3000"))
        self.use_target_guidance = os.getenv("GO1_USE_TARGET_GUIDANCE", "1") == "1"
        self.target_distance_m = float(os.getenv("GO1_TARGET_DISTANCE_M", "5.0"))
        self.target_success_radius_m = float(os.getenv("GO1_TARGET_SUCCESS_RADIUS_M", "0.35"))
        self.target_success_bonus = float(os.getenv("GO1_TARGET_SUCCESS_BONUS", "12.0"))
        self.target_def_name = os.getenv("GO1_TARGET_DEF", "TRAINING_TARGET")
        self.no_progress_patience_steps = int(os.getenv("GO1_NO_PROGRESS_PATIENCE_STEPS", "420"))
        self.no_progress_epsilon_m = float(os.getenv("GO1_NO_PROGRESS_EPS_M", "0.12"))
        self.no_progress_penalty = float(os.getenv("GO1_NO_PROGRESS_PENALTY", "2.5"))
        self.speed_target_start = float(os.getenv("GO1_SPEED_TARGET_START", "0.7"))
        self.speed_target_end = float(os.getenv("GO1_SPEED_TARGET_END", "1.0"))
        self.speed_curriculum_episodes = int(os.getenv("GO1_SPEED_CURRICULUM_EPISODES", "300"))
        self.penalty_curriculum_start = float(os.getenv("GO1_PENALTY_CURRICULUM_START", "0.1"))
        self.penalty_curriculum_end = float(os.getenv("GO1_PENALTY_CURRICULUM_END", "1.0"))
        self.penalty_curriculum_episodes = int(os.getenv("GO1_PENALTY_CURRICULUM_EPISODES", "800"))
        self.debug_diagnostics = os.getenv("GO1_DEBUG_DIAGNOSTICS", "0") == "1"
        self.debug_steps_to_print = int(os.getenv("GO1_DEBUG_STEPS", "10"))
        self.debug_print_counter = 0
        self.degradation_model_active = os.getenv("GO1_DEGRADATION_MODEL_ACTIVE", "0") == "1"
        self.motor_strength_min = float(os.getenv("GO1_MOTOR_STRENGTH_MIN", "1.0"))
        self.motor_strength_max = float(os.getenv("GO1_MOTOR_STRENGTH_MAX", "1.0"))
        self.latency_ms_min = float(os.getenv("GO1_LATENCY_MS_MIN", "0.0"))
        self.latency_ms_max = float(os.getenv("GO1_LATENCY_MS_MAX", "0.0"))
        self.imu_noise_std = float(os.getenv("GO1_IMU_NOISE_STD", "0.0"))
        self.imu_bias_range = float(os.getenv("GO1_IMU_BIAS_RANGE", "0.0"))
        self.friction_min = float(os.getenv("GO1_FRICTION_MIN", "1.0"))
        self.friction_max = float(os.getenv("GO1_FRICTION_MAX", "1.0"))
        self.episode_motor_strength = 1.0
        self.episode_latency_ms = 0.0
        self.episode_latency_steps = 0
        self.episode_friction = 1.0
        self.episode_imu_bias_rpy = np.zeros(3, dtype=np.float64)
        self.episode_imu_bias_gyro = np.zeros(3, dtype=np.float64)
        self.obs_history: deque[np.ndarray] = deque(maxlen=1)
        self.target_xy = np.zeros(2, dtype=np.float64)
        self.target_node = None
        self.target_translation_field = None
        self.target_z = 0.0
        self._target_node_warned = False
        self._friction_field_warned = False

        self.joint_min = np.array([
            -0.9, 0.1, -2.5,
            -0.9, 0.1, -2.5,
            -0.9, 0.1, -2.5,
            -0.9, 0.1, -2.5,
        ], dtype=np.float64)
        self.joint_max = np.array([
            0.9, 1.4, -0.7,
            0.9, 1.4, -0.7,
            0.9, 1.4, -0.7,
            0.9, 1.4, -0.7,
        ], dtype=np.float64)

        self.joint_sensors = []
        for name in joint_names:
            sensor = self.supervisor.getDevice(name + "_sensor")
            sensor.enable(self.timestep)
            self.joint_sensors.append(sensor)

        self.foot_touch_sensor_names = [
            "FL_foot_touch",
            "FR_foot_touch",
            "RL_foot_touch",
            "RR_foot_touch",
        ]
        self.foot_node_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.calf_node_names = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
        self.foot_touch_sensors = []
        self.foot_nodes = []
        self.calf_nodes = []
        self.trunk_node = None
        self.has_foot_world_nodes = True
        self.prev_foot_xy = np.zeros((4, 2), dtype=np.float64)
        self.prev_foot_xy_valid = False
        self.contact_sensors_enabled = os.getenv("GO1_CONTACT_SENSORS", "1") == "1"
        self.has_foot_contact_sensors = self.contact_sensors_enabled
        self.contact_sensor_reliable = True
        self.contact_zero_steps = 0
        self.contact_source_mode = os.getenv("GO1_CONTACT_SOURCE", "auto").strip().lower()
        self.last_contact_source = "none"
        self.contact_stale_steps = int(os.getenv("GO1_CONTACT_STALE_STEPS", "120"))
        self.contact_threshold = float(os.getenv("GO1_CONTACT_THRESHOLD", "1e-5"))
        self.contact_auto_touch_grace_steps = int(os.getenv("GO1_CONTACT_AUTO_TOUCH_GRACE_STEPS", "40"))
        self.enable_contact_points_tracking = os.getenv("GO1_ENABLE_CONTACT_POINTS_TRACKING", "1") == "1"
        self.contact_tracking_enabled_nodes: list[str] = []
        self.contact_tracking_failed_nodes: list[str] = []
        self.touch_zero_steps = 0
        self.physics_zero_steps = 0
        if self.contact_sensors_enabled:
            for sensor_name in self.foot_touch_sensor_names:
                try:
                    sensor = self.supervisor.getDevice(sensor_name)
                except Exception:
                    sensor = None
                if sensor is None:
                    self.has_foot_contact_sensors = False
                    break
                sensor.enable(self.timestep)
                self.foot_touch_sensors.append(sensor)
            if not self.has_foot_contact_sensors:
                self.foot_touch_sensors = []

        self.imu = self.supervisor.getDevice("trunk_imu inertial")
        self.imu.enable(self.timestep)

        self.gyro = self.supervisor.getDevice("trunk_imu gyro")
        self.gyro.enable(self.timestep)

        self.robot_node = self.supervisor.getSelf()
        self.trunk_node = self._find_named_node_in_subtree(self.robot_node, "trunk")
        for foot_name in self.foot_node_names:
            foot_node = self._find_named_node_in_subtree(self.robot_node, foot_name)
            if foot_node is None:
                self.has_foot_world_nodes = False
                break
            self.foot_nodes.append(foot_node)
        if not self.has_foot_world_nodes:
            self.foot_nodes = []
        for calf_name in self.calf_node_names:
            calf_node = self._find_named_node_in_subtree(self.robot_node, calf_name)
            if calf_node is not None:
                self.calf_nodes.append(calf_node)

        self._enable_contact_tracking_if_requested()

        self.floor_node = self._find_floor_node()
        translation_field = self.robot_node.getField("translation")
        self.initial_translation = list(translation_field.getSFVec3f())
        rotation_field = self.robot_node.getField("rotation")
        self.initial_rotation = list(rotation_field.getSFRotation())
        self.target_height = float(self.initial_translation[2])
        self.reward_computer = Go1RewardComputer(
            dt=self.dt,
            default_pose_array=self.default_pose_array,
            target_height=self.target_height,
        )
        self.target_node = self.supervisor.getFromDef(self.target_def_name)
        if self.target_node is not None:
            self.target_translation_field = self.target_node.getField("translation")
            if self.target_translation_field is not None:
                target_position = self.target_translation_field.getSFVec3f()
                self.target_z = float(target_position[2])
        self.steps_alive = 0
        self.episode_count = 0
        self.best_target_distance = np.inf
        self.best_forward_distance = 0.0
        self.steps_since_progress = 0
        self._sample_episode_degradation()
        self._configure_observation_delay_buffer()
        self._apply_episode_floor_friction()
        self._apply_motor_strength_to_actuators()

    @staticmethod
    def _safe_touch_scalar(sensor) -> float:
        try:
            return abs(float(sensor.getValue()))
        except Exception:
            return 0.0

    def _enable_contact_tracking_for_node(self, node, label: str) -> None:
        if node is None:
            return
        try:
            node.enableContactPointsTracking(self.timestep)
            self.contact_tracking_enabled_nodes.append(label)
            return
        except Exception:
            pass
        try:
            node.enableContactPointsTracking(self.timestep, True)
            self.contact_tracking_enabled_nodes.append(label)
            return
        except Exception as exc:
            self.contact_tracking_failed_nodes.append(f"{label}:{type(exc).__name__}")

    def _enable_contact_tracking_if_requested(self) -> None:
        if not self.enable_contact_points_tracking:
            return

        self.contact_tracking_enabled_nodes = []
        self.contact_tracking_failed_nodes = []

        self._enable_contact_tracking_for_node(self.robot_node, "robot")
        self._enable_contact_tracking_for_node(self.trunk_node, "trunk")
        for idx, node in enumerate(self.foot_nodes):
            self._enable_contact_tracking_for_node(node, f"foot_{idx}")
        for idx, node in enumerate(self.calf_nodes):
            self._enable_contact_tracking_for_node(node, f"calf_{idx}")

    @staticmethod
    def _safe_contact_count(foot_node) -> int:
        if foot_node is None:
            return 0
        max_count = 0
        try:
            max_count = max(max_count, int(foot_node.getNumberOfContactPoints(True)))
        except Exception:
            pass
        try:
            max_count = max(max_count, int(foot_node.getNumberOfContactPoints(False)))
        except Exception:
            pass
        try:
            max_count = max(max_count, int(foot_node.getNumberOfContactPoints()))
        except Exception:
            pass
        return max_count

    def _read_touch_contacts(self) -> tuple[np.ndarray | None, list[float] | None]:
        if not self.has_foot_contact_sensors or len(self.foot_touch_sensors) != 4:
            return None, None

        touch_strengths: list[float] = []
        touch_contacts: list[float] = []
        for sensor in self.foot_touch_sensors:
            strength = self._safe_touch_scalar(sensor)
            touch_strengths.append(strength)
            touch_contacts.append(1.0 if strength > self.contact_threshold else 0.0)

        return np.asarray(touch_contacts, dtype=np.float64), touch_strengths

    def _read_physics_contacts(self) -> tuple[np.ndarray | None, list[float] | None]:
        if not self.has_foot_world_nodes or len(self.foot_nodes) != 4:
            return None, None

        physics_counts: list[float] = []
        physics_contacts: list[float] = []
        for foot_node in self.foot_nodes:
            count = self._safe_contact_count(foot_node)
            physics_counts.append(float(count))
            physics_contacts.append(1.0 if count > 0 else 0.0)

        return np.asarray(physics_contacts, dtype=np.float64), physics_counts

    def _read_foot_contacts(
        self,
    ) -> tuple[np.ndarray | None, list[float] | None, str, np.ndarray | None, list[float] | None, np.ndarray | None, list[float] | None]:
        touch_array, touch_strengths = self._read_touch_contacts()
        physics_array, physics_counts = self._read_physics_contacts()

        touch_has_contact = bool(touch_array is not None and float(np.sum(touch_array)) > 0.0)
        physics_has_contact = bool(physics_array is not None and float(np.sum(physics_array)) > 0.0)

        if touch_array is not None:
            self.touch_zero_steps = 0 if touch_has_contact else (self.touch_zero_steps + 1)
        else:
            self.touch_zero_steps = 0

        if physics_array is not None:
            self.physics_zero_steps = 0 if physics_has_contact else (self.physics_zero_steps + 1)
        else:
            self.physics_zero_steps = 0

        selected_array: np.ndarray | None = None
        selected_strengths: list[float] | None = None
        source = "none"

        mode = self.contact_source_mode
        if mode == "touch":
            selected_array, selected_strengths, source = touch_array, touch_strengths, "touch"
        elif mode == "physics":
            selected_array, selected_strengths, source = physics_array, physics_counts, "physics"
        else:
            if physics_has_contact:
                selected_array, selected_strengths, source = physics_array, physics_counts, "physics"
            elif touch_has_contact:
                selected_array, selected_strengths, source = touch_array, touch_strengths, "touch"
            elif physics_array is not None and self.touch_zero_steps >= self.contact_auto_touch_grace_steps:
                # Do not stay pinned to a dead touch stream when physics stream exists.
                selected_array, selected_strengths, source = physics_array, physics_counts, "physics"
            elif touch_array is not None:
                selected_array, selected_strengths, source = touch_array, touch_strengths, "touch"
            elif physics_array is not None:
                selected_array, selected_strengths, source = physics_array, physics_counts, "physics"

        if selected_array is not None:
            if float(np.sum(selected_array)) <= 0.0:
                self.contact_zero_steps += 1
            else:
                self.contact_zero_steps = 0

        if mode == "touch" and self.touch_zero_steps >= self.contact_stale_steps:
            self.contact_sensor_reliable = False
        elif mode == "physics" and self.physics_zero_steps >= self.contact_stale_steps:
            self.contact_sensor_reliable = False
        elif mode == "auto" and self.contact_zero_steps >= self.contact_stale_steps:
            self.contact_sensor_reliable = False

        self.last_contact_source = source
        return selected_array, selected_strengths, source, touch_array, touch_strengths, physics_array, physics_counts

    def _current_speed_target(self) -> float:
        if self.speed_curriculum_episodes <= 1:
            return float(self.speed_target_end)
        alpha = float(np.clip((self.episode_count - 1) / max(self.speed_curriculum_episodes - 1, 1), 0.0, 1.0))
        return float(self.speed_target_start + alpha * (self.speed_target_end - self.speed_target_start))

    def _current_penalty_curriculum_factor(self) -> float:
        if self.penalty_curriculum_episodes <= 1:
            return float(self.penalty_curriculum_end)
        alpha = float(
            np.clip(
                (self.episode_count - 1) / max(self.penalty_curriculum_episodes - 1, 1),
                0.0,
                1.0,
            )
        )
        return float(
            self.penalty_curriculum_start
            + alpha * (self.penalty_curriculum_end - self.penalty_curriculum_start)
        )

    @staticmethod
    def _safe_torque_scalar(motor) -> float:
        try:
            return float(motor.getTorqueFeedback())
        except Exception:
            return 0.0

    def _read_joint_torques(self) -> np.ndarray:
        return np.asarray([self._safe_torque_scalar(motor) for motor in self.joints], dtype=np.float64)

    def _sample_uniform(self, minimum: float, maximum: float) -> float:
        lo = float(min(minimum, maximum))
        hi = float(max(minimum, maximum))
        if hi - lo <= 1e-12:
            return lo
        return float(np.random.uniform(lo, hi))

    def _sample_episode_degradation(self) -> None:
        if not self.degradation_model_active:
            self.episode_motor_strength = 1.0
            self.episode_latency_ms = 0.0
            self.episode_latency_steps = 0
            self.episode_friction = 1.0
            self.episode_imu_bias_rpy = np.zeros(3, dtype=np.float64)
            self.episode_imu_bias_gyro = np.zeros(3, dtype=np.float64)
            return

        self.episode_motor_strength = self._sample_uniform(self.motor_strength_min, self.motor_strength_max)
        self.episode_latency_ms = self._sample_uniform(self.latency_ms_min, self.latency_ms_max)
        self.episode_latency_steps = int(max(0, round(self.episode_latency_ms / max(float(self.timestep), 1e-6))))
        self.episode_friction = self._sample_uniform(self.friction_min, self.friction_max)
        self.episode_imu_bias_rpy = np.random.uniform(
            -self.imu_bias_range,
            self.imu_bias_range,
            size=3,
        ).astype(np.float64)
        self.episode_imu_bias_gyro = np.random.uniform(
            -self.imu_bias_range,
            self.imu_bias_range,
            size=3,
        ).astype(np.float64)

    def _configure_observation_delay_buffer(self) -> None:
        delay_steps = int(max(0, self.episode_latency_steps))
        self.obs_history = deque(maxlen=delay_steps + 1)

    def _push_observation(self, obs: np.ndarray) -> None:
        self.obs_history.append(np.asarray(obs, dtype=np.float64).copy())

    def _get_delayed_observation(self) -> np.ndarray:
        if not self.obs_history:
            return np.zeros(self.observation_space.shape, dtype=np.float64)
        return self.obs_history[0].copy()

    def _apply_motor_strength_to_actuators(self) -> None:
        scale = float(np.clip(self.episode_motor_strength, 0.0, 1.0))
        if not self.degradation_model_active:
            scale = 1.0

        for motor in self.joints:
            try:
                max_torque = float(motor.getMaxTorque())
            except Exception:
                continue
            try:
                motor.setAvailableTorque(max_torque * scale)
            except Exception:
                pass

    @staticmethod
    def _safe_get_field(node, field_name: str):
        if node is None:
            return None
        try:
            return node.getField(field_name)
        except Exception:
            return None

    @staticmethod
    def _safe_field_count(field) -> int:
        if field is None:
            return 0
        try:
            return int(field.getCount())
        except Exception:
            return 0

    @staticmethod
    def _safe_get_mf_node(field, index: int):
        if field is None:
            return None
        try:
            return field.getMFNode(index)
        except Exception:
            return None

    @staticmethod
    def _safe_get_sf_node(field):
        if field is None:
            return None
        try:
            return field.getSFNode()
        except Exception:
            return None

    def _find_floor_node(self):
        root = self.supervisor.getRoot()
        if root is None:
            return None
        children_field = self._safe_get_field(root, "children")
        if children_field is None:
            return None

        for i in range(self._safe_field_count(children_field)):
            node = self._safe_get_mf_node(children_field, i)
            if node is None:
                continue
            node_children = self._safe_get_field(node, "children")
            if node_children is None:
                continue
            for j in range(self._safe_field_count(node_children)):
                child = self._safe_get_mf_node(node_children, j)
                if child is None:
                    continue
                name_field = self._safe_get_field(child, "name")
                if name_field is None:
                    continue
                try:
                    if name_field.getSFString() == "floor":
                        return child
                except Exception:
                    continue
        return None

    def _find_named_node_in_subtree(self, node, target_name: str, max_depth: int = 10):
        if node is None or max_depth < 0:
            return None

        name_field = self._safe_get_field(node, "name")
        if name_field is not None:
            try:
                if name_field.getSFString() == target_name:
                    return node
            except Exception:
                pass

        children_field = self._safe_get_field(node, "children")
        if children_field is not None:
            for i in range(self._safe_field_count(children_field)):
                child = self._safe_get_mf_node(children_field, i)
                found = self._find_named_node_in_subtree(child, target_name, max_depth=max_depth - 1)
                if found is not None:
                    return found

        end_point_field = self._safe_get_field(node, "endPoint")
        if end_point_field is not None:
            end_node = self._safe_get_sf_node(end_point_field)
            found = self._find_named_node_in_subtree(end_node, target_name, max_depth=max_depth - 1)
            if found is not None:
                return found

        return None

    def _read_foot_xy(self) -> np.ndarray | None:
        if not self.has_foot_world_nodes or len(self.foot_nodes) != 4:
            return None

        foot_xy = np.zeros((4, 2), dtype=np.float64)
        for idx, foot_node in enumerate(self.foot_nodes):
            try:
                position = foot_node.getPosition()
                foot_xy[idx, 0] = float(position[0])
                foot_xy[idx, 1] = float(position[1])
            except Exception:
                return None
        return foot_xy

    def _read_foot_z(self) -> list[float] | None:
        if not self.has_foot_world_nodes or len(self.foot_nodes) != 4:
            return None

        foot_z: list[float] = []
        for foot_node in self.foot_nodes:
            try:
                position = foot_node.getPosition()
                foot_z.append(float(position[2]))
            except Exception:
                return None
        return foot_z

    def _read_extra_contact_counts(self) -> tuple[float | None, list[float] | None]:
        trunk_contact_count: float | None = None
        calf_contact_counts: list[float] | None = None

        if self.trunk_node is not None:
            trunk_contact_count = float(self._safe_contact_count(self.trunk_node))

        if self.calf_nodes:
            calf_contact_counts = [float(self._safe_contact_count(node)) for node in self.calf_nodes]

        return trunk_contact_count, calf_contact_counts

    def _apply_episode_floor_friction(self) -> None:
        if not self.degradation_model_active:
            return

        if self.floor_node is None:
            if self.debug_diagnostics and not self._friction_field_warned:
                print("[GO1 DEBUG] Floor node not found; direct floor friction field update unavailable.")
                self._friction_field_warned = True
            return

        friction_field = self.floor_node.getField("coulombFriction")
        if friction_field is not None:
            try:
                friction_field.setSFFloat(float(self.episode_friction))
            except Exception:
                pass
        elif self.debug_diagnostics and not self._friction_field_warned:
            print("[GO1 DEBUG] Floor node has no coulombFriction field; sampled friction is not applied elsewhere.")
            self._friction_field_warned = True

    def _read_imu_with_degradation(self, apply_degradation: bool) -> tuple[np.ndarray, np.ndarray]:
        rpy = np.asarray(self.imu.getRollPitchYaw(), dtype=np.float64)
        gyro = np.asarray(self.gyro.getValues(), dtype=np.float64)

        if apply_degradation and self.degradation_model_active:
            if self.imu_noise_std > 0.0:
                rpy = rpy + np.random.normal(0.0, self.imu_noise_std, size=3)
                gyro = gyro + np.random.normal(0.0, self.imu_noise_std, size=3)
            rpy = rpy + self.episode_imu_bias_rpy
            gyro = gyro + self.episode_imu_bias_gyro

        return rpy, gyro

    def get_observations(self, apply_degradation: bool = True):
        obs = []
        current_joint_positions = np.array(
            [sensor.getValue() for sensor in self.joint_sensors],
            dtype=np.float64,
        )

        if self.prev_joint_positions is None:
            joint_velocities = np.zeros_like(current_joint_positions)
        else:
            joint_velocities = (current_joint_positions - self.prev_joint_positions) / max(self.dt, 1e-6)

        self.prev_joint_positions = current_joint_positions.copy()

        for value in current_joint_positions:
            obs.append(value)
        for value in joint_velocities:
            obs.append(value)

        rpy, gyro = self._read_imu_with_degradation(apply_degradation=apply_degradation)
        roll, pitch, yaw = rpy.tolist()
        
        # Calculate relative roll, pitch, yaw to prevent absolute-frame axis-swapping (gimbal effects)
        theta = getattr(self, "initial_yaw", 0.0)
        c1 = np.cos(yaw)
        s1 = np.sin(yaw)
        c2 = np.cos(pitch)
        s2 = np.sin(pitch)
        c3 = np.cos(roll)
        s3 = np.sin(roll)
        
        R = np.array([
            [c1*c2, c1*s2*s3 - s1*c3, c1*s2*c3 + s1*s3],
            [s1*c2, s1*s2*s3 + c1*c3, s1*s2*c3 - c1*s3],
            [-s2,   c2*s3,            c2*c3]
        ])
        
        R_z_inv = np.array([
            [np.cos(theta),  np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0,              0,             1]
        ])
        
        R_rel = R_z_inv @ R
        
        pitch_rel = float(np.arcsin(np.clip(-R_rel[2, 0], -1.0, 1.0)))
        if abs(np.cos(pitch_rel)) > 1e-6:
            roll_rel = float(np.arctan2(R_rel[2, 1], R_rel[2, 2]))
            yaw_rel = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
        else:
            roll_rel = 0.0
            yaw_rel = float(np.arctan2(-R_rel[0, 1], R_rel[1, 1]))
            
        obs.extend([roll_rel, pitch_rel, yaw_rel])

        wx, wy, wz = gyro.tolist()
        obs.extend([wx, wy, wz])

        touch_contacts, _ = self._read_touch_contacts()
        if touch_contacts is None:
            touch_contacts = np.zeros(4, dtype=np.float64)
        obs.extend(touch_contacts.tolist())

        return np.array(obs, dtype=np.float64)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        bounded_action = np.clip(action, -self.residual_action_max, self.residual_action_max)
        actuation_scale = (
            float(np.clip(self.episode_motor_strength, 0.0, 1.0))
            if self.degradation_model_active
            else 1.0
        )
        target_pose = self.default_pose_array + self.action_scale * actuation_scale * bounded_action
        target_pose = np.clip(target_pose, self.joint_min, self.joint_max)

        self._apply_motor_strength_to_actuators()

        for i, motor in enumerate(self.joints):
            motor.setVelocity(self.max_joint_velocity)
            motor.setPosition(float(target_pose[i]))

        self.supervisor.step(self.timestep)
        self.steps_alive += 1

        current_policy_obs = self.get_observations(apply_degradation=True)
        self._push_observation(current_policy_obs)
        obs = self._get_delayed_observation()
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        angular_rate = np.array(self.gyro.getValues(), dtype=np.float64)
        current_pose = np.array([sensor.getValue() for sensor in self.joint_sensors], dtype=np.float64)
        robot_position = self.robot_node.getField("translation").getSFVec3f()
        x_position = float(robot_position[0])
        y_position = float(robot_position[1])
        height = float(robot_position[2])

        velocity = self.robot_node.getVelocity()
        vx = float(velocity[0])
        vy = float(velocity[1])
        vz = float(velocity[2])
        body_forward_velocity = vx * float(np.cos(yaw)) + vy * float(np.sin(yaw))

        velocity_xy = np.array([vx, vy], dtype=np.float64)
        command_heading = self.initial_heading
        lateral_heading = np.array([-command_heading[1], command_heading[0]], dtype=np.float64)
        forward_velocity_command_axis = float(np.dot(velocity_xy, command_heading))
        lateral_velocity_command_axis = float(np.dot(velocity_xy, lateral_heading))
        to_target_xy = self.target_xy - np.array([x_position, y_position], dtype=np.float64)
        target_distance_remaining = float(np.linalg.norm(to_target_xy))
        if target_distance_remaining > 1e-6:
            target_direction = to_target_xy / target_distance_remaining
        else:
            target_direction = self.initial_heading

        if self.direction_command_mode:
            forward_velocity = forward_velocity_command_axis
        elif self.use_target_guidance:
            forward_velocity = float(np.dot(velocity_xy, target_direction))
        else:
            forward_velocity = (
                vx * float(self.initial_heading[0])
                + vy * float(self.initial_heading[1])
            )
        heading_target_yaw = (
            float(self.initial_yaw)
            if self.direction_command_mode
            else float(np.arctan2(target_direction[1], target_direction[0]))
        )
        forward_reward = 1.5 * float(np.clip(forward_velocity, 0.0, 1.5))

        joint_velocities = current_policy_obs[12:24]
        (
            raw_foot_contacts,
            foot_contact_strengths,
            contact_source,
            touch_contact_array,
            touch_contact_strengths,
            physics_contact_array,
            physics_contact_counts,
        ) = self._read_foot_contacts()
        if self.contact_sensor_reliable:
            foot_contacts = raw_foot_contacts
        else:
            foot_contacts = None
        foot_contact_vector = (
            raw_foot_contacts.astype(np.int32).tolist() if raw_foot_contacts is not None else None
        )
        speed_target = self._current_speed_target()
        penalty_curriculum_factor = self._current_penalty_curriculum_factor()
        joint_torques = self._read_joint_torques()
        step_mechanical_power_w = float(np.sum(np.abs(joint_torques * joint_velocities)))
        step_mechanical_work_j = float(step_mechanical_power_w * self.dt)
        true_stance_slip_distance_step_m = np.nan
        true_stance_intended_distance_step_m = np.nan
        true_stance_contact_count_step = 0
        foot_world_z = self._read_foot_z()
        trunk_contact_count, calf_contact_counts = self._read_extra_contact_counts()
        robot_contact_count = float(self._safe_contact_count(self.robot_node))

        current_foot_xy = self._read_foot_xy()
        if (
            current_foot_xy is not None
            and self.prev_foot_xy_valid
            and raw_foot_contacts is not None
            and raw_foot_contacts.size == 4
        ):
            foot_step_disp = np.linalg.norm(current_foot_xy - self.prev_foot_xy, axis=1)
            stance_mask = raw_foot_contacts > 0.5
            true_stance_contact_count_step = int(np.sum(stance_mask))
            if true_stance_contact_count_step > 0:
                true_stance_slip_distance_step_m = float(np.sum(foot_step_disp[stance_mask]))
                stance_fraction = float(true_stance_contact_count_step) / 4.0
                true_stance_intended_distance_step_m = float(abs(speed_target) * self.dt * stance_fraction)

        if current_foot_xy is not None:
            self.prev_foot_xy = current_foot_xy
            self.prev_foot_xy_valid = True

        if self.reward_mode == "simple":
            upright_term = float(np.exp(-4.0 * (roll * roll + pitch * pitch)))
            height_term = float(np.exp(-35.0 * abs(height - self.target_height)))
            pose_error = float(np.mean(np.abs(current_pose - self.default_pose_array)))
            pose_term = float(np.exp(-2.5 * pose_error))
            action_magnitude_penalty = 0.0
            action_smoothness_penalty = (
                penalty_curriculum_factor
                * self.command_smoothness_penalty_weight
                * float(np.sum(np.square(bounded_action - self.prev_action)))
            )
            lateral_error = lateral_velocity_command_axis - self.command_vy
            lateral_drift_penalty = self.command_lateral_penalty_weight * float(lateral_error * lateral_error)
            yaw_error = float(angular_rate[2]) - self.command_wz
            yaw_rate_penalty = self.command_yaw_penalty_weight * float(yaw_error * yaw_error)
            mean_torque_effort = float(np.mean(np.square(joint_torques)))
            torque_effort_penalty = (
                penalty_curriculum_factor
                * self.command_energy_penalty_weight
                * (mean_torque_effort / 12.0)
            )
            calf_velocity_indices = [2, 5, 8, 11]
            calf_vel_abs = np.abs(joint_velocities[calf_velocity_indices])
            if foot_contacts is not None and foot_contacts.size == 4:
                stance_mask = foot_contacts > 0.5
            else:
                stance_mask = np.ones(4, dtype=bool)
            if np.any(stance_mask):
                foot_slip_ratio = float(np.mean(calf_vel_abs[stance_mask]) / 4.5)
            else:
                foot_slip_ratio = 0.0
            foot_slip_penalty = (
                penalty_curriculum_factor
                * self.command_slip_penalty_weight
                * foot_slip_ratio
            )
            heading_error = 0.0
            reward = (
                0.25
                + 1.2 * upright_term
                + 0.8 * height_term
                + 0.6 * pose_term
                + forward_reward
                - lateral_drift_penalty
                - action_magnitude_penalty
                - action_smoothness_penalty
                - yaw_rate_penalty
                - torque_effort_penalty
                - foot_slip_penalty
            )

            front_joint_velocity_magnitude = float(np.mean(np.abs(joint_velocities[0:6])))
            rear_joint_velocity_magnitude = float(np.mean(np.abs(joint_velocities[6:12])))
            hind_ratio = rear_joint_velocity_magnitude / max(
                front_joint_velocity_magnitude + rear_joint_velocity_magnitude,
                1e-9,
            )
            reward_terms = {
                "reward": float(reward),
                "heading_error": float(heading_error),
                "heading_reference_yaw": float(heading_target_yaw if self.use_target_guidance else self.initial_yaw),
                "heading_penalty": 0.0,
                "yaw_rate_penalty": float(yaw_rate_penalty),
                "action_magnitude_penalty": float(action_magnitude_penalty),
                "action_smoothness_penalty": float(action_smoothness_penalty),
                "action_jerk_penalty": 0.0,
                "joint_velocity_delta_penalty": 0.0,
                "vertical_velocity_penalty": 0.0,
                "feet_gait_reward": 0.0,
                "gait_sync_score": 0.0,
                "joint_mirror_penalty": 0.0,
                "joint_mirror_error": 0.0,
                "feet_air_time_variance_penalty": 0.0,
                "feet_air_time_variance": 0.0,
                "target_progress_delta": 0.0,
                "target_progress_reward": 0.0,
                "speed_tracking_term": 0.0,
                "speed_tracking_reward": 0.0,
                "speed_target": float(self.command_vx),
                "stride_amplitude_score": 0.0,
                "stride_amplitude_reward": 0.0,
                "hind_engagement_term": 0.0,
                "hind_engagement_reward": 0.0,
                "foot_clearance_term": 0.0,
                "foot_clearance_reward": 0.0,
                "foot_slip_term": float(foot_slip_ratio),
                "foot_slip_penalty": float(foot_slip_penalty),
                "foot_slip_ratio": float(np.clip(foot_slip_ratio, 0.0, 1.0)),
                "mean_torque_effort": float(mean_torque_effort),
                "mean_mechanical_work": 0.0,
                "torque_effort_penalty": float(torque_effort_penalty),
                "mechanical_power_penalty": 0.0,
                "penalty_curriculum_factor": float(penalty_curriculum_factor),
                "front_joint_velocity_magnitude": float(front_joint_velocity_magnitude),
                "rear_joint_velocity_magnitude": float(rear_joint_velocity_magnitude),
                "hind_motion_ratio": float(hind_ratio),
                "using_contact_sensors": float(1.0 if (foot_contacts is not None) else 0.0),
            }
        else:
            reward_terms = self.reward_computer.compute(
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
                initial_yaw=float(self.initial_yaw),
                heading_reference_yaw=heading_target_yaw if self.use_target_guidance else float(self.initial_yaw),
                yaw_rate=float(angular_rate[2]),
                height=height,
                current_pose=current_pose,
                action=bounded_action,
                prev_action=self.prev_action,
                prev_prev_action=self.prev_prev_action,
                joint_velocities=joint_velocities,
                prev_joint_velocities=self.prev_joint_velocities,
                vz=vz,
                forward_velocity=forward_velocity,
                target_distance_remaining=target_distance_remaining,
                foot_contacts=foot_contacts,
                speed_target_override=speed_target,
                measured_torques=joint_torques,
                penalty_curriculum_factor=penalty_curriculum_factor,
            )

        heading_error = reward_terms["heading_error"]
        heading_penalty = reward_terms["heading_penalty"]
        yaw_rate_penalty = reward_terms["yaw_rate_penalty"]
        action_magnitude_penalty = reward_terms["action_magnitude_penalty"]
        action_smoothness_penalty = reward_terms["action_smoothness_penalty"]
        action_jerk_penalty = reward_terms["action_jerk_penalty"]
        joint_velocity_delta_penalty = reward_terms["joint_velocity_delta_penalty"]
        vertical_velocity_penalty = reward_terms["vertical_velocity_penalty"]

        self.prev_joint_velocities = joint_velocities.copy()
        action_magnitude = float(np.mean(np.abs(bounded_action)))
        action_delta_mean = float(np.mean(np.abs(bounded_action - self.prev_action)))
        mean_joint_velocity_magnitude = float(np.mean(np.abs(joint_velocities)))
        velocity_tracking_error = abs(forward_velocity - reward_terms["speed_target"])

        reward = reward_terms["reward"]
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = bounded_action

        delta_xy = np.array(
            [x_position, y_position],
            dtype=np.float64,
        ) - self.episode_start_xy
        forward_distance = float(np.dot(delta_xy, self.initial_heading))

        tipped = abs(roll) > 1.0 or abs(pitch) > 1.0
        too_low = height < (self.target_height - 0.12)
        if self.direction_command_mode:
            success = False
            if forward_distance > self.best_forward_distance + self.no_progress_epsilon_m:
                self.best_forward_distance = forward_distance
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1
        else:
            success = target_distance_remaining <= self.target_success_radius_m
            if target_distance_remaining + self.no_progress_epsilon_m < self.best_target_distance:
                self.best_target_distance = target_distance_remaining
                self.steps_since_progress = 0
            else:
                self.steps_since_progress += 1

        no_progress = (
            (self.steps_since_progress >= self.no_progress_patience_steps)
            and (not success)
            and (self.steps_alive > max(20, self.no_progress_patience_steps // 3))
        )

        terminated = bool(tipped or too_low or success or no_progress)
        termination_reason = "none"
        if terminated:
            if tipped or too_low:
                reward -= 5.0
                termination_reason = "fall"
            elif success:
                progress_fraction = 1.0 - float(np.clip(target_distance_remaining / max(self.target_distance_m, 1e-6), 0.0, 1.0))
                reward += self.target_success_bonus * (0.5 + 0.5 * progress_fraction)
                termination_reason = "success"
            elif no_progress:
                reward -= self.no_progress_penalty
                termination_reason = "no_progress"

        truncated = bool(self.steps_alive > self.max_episode_steps)
        if truncated and not terminated:
            termination_reason = "truncation"

        info = {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
            "initial_yaw": float(self.initial_yaw),
            "heading_target_yaw": float(reward_terms["heading_reference_yaw"]),
            "heading_error": float(heading_error),
            "height": height,
            "forward_velocity_world_x": vx,
            "lateral_velocity_world_y": vy,
            "forward_velocity_command_axis": forward_velocity_command_axis,
            "lateral_velocity_command_axis": lateral_velocity_command_axis,
            "body_forward_velocity": float(body_forward_velocity),
            "forward_velocity_initial_heading": float(forward_velocity),
            "forward_velocity_to_target": float(np.dot(velocity_xy, target_direction)),
            "target_distance_remaining": target_distance_remaining,
            "target_x": float(self.target_xy[0]),
            "target_y": float(self.target_xy[1]),
            "yaw_rate": float(angular_rate[2]),
            "vertical_velocity_z": vz,
            "mean_joint_velocity_magnitude": mean_joint_velocity_magnitude,
            "front_joint_velocity_magnitude": reward_terms["front_joint_velocity_magnitude"],
            "rear_joint_velocity_magnitude": reward_terms["rear_joint_velocity_magnitude"],
            "hind_motion_ratio": reward_terms["hind_motion_ratio"],
            "has_foot_contact_sensors": int(self.has_foot_contact_sensors),
            "using_contact_sensors": reward_terms["using_contact_sensors"],
            "contact_sensor_reliable": int(self.contact_sensor_reliable),
            "contact_zero_steps": int(self.contact_zero_steps),
            "contact_source_mode": str(self.contact_source_mode),
            "contact_source": str(contact_source),
            "enable_contact_points_tracking": int(self.enable_contact_points_tracking),
            "contact_tracking_enabled_count": int(len(self.contact_tracking_enabled_nodes)),
            "contact_tracking_failed_count": int(len(self.contact_tracking_failed_nodes)),
            "contact_tracking_failed_nodes": list(self.contact_tracking_failed_nodes),
            "foot_contact_strengths": foot_contact_strengths,
            "foot_contact_vector": foot_contact_vector,
            "foot_contact_vector_touch": (
                touch_contact_array.astype(np.int32).tolist()
                if touch_contact_array is not None
                else None
            ),
            "foot_contact_strengths_touch": touch_contact_strengths,
            "foot_contact_vector_physics": (
                physics_contact_array.astype(np.int32).tolist()
                if physics_contact_array is not None
                else None
            ),
            "foot_contact_counts_physics": physics_contact_counts,
            "foot_world_z": foot_world_z,
            "trunk_contact_count": trunk_contact_count,
            "calf_contact_counts": calf_contact_counts,
            "robot_contact_count": robot_contact_count,
            "action_magnitude": action_magnitude,
            "action_delta_mean": action_delta_mean,
            "action_clip_ratio": float(np.mean(np.abs(action - bounded_action))),
            "residual_action_max": self.residual_action_max,
            "residual_max_rad": self.residual_max_rad,
            "degradation_model_active": int(self.degradation_model_active),
            "degradation_motor_strength": float(self.episode_motor_strength),
            "degradation_latency_ms": float(self.episode_latency_ms),
            "degradation_latency_steps": int(self.episode_latency_steps),
            "degradation_imu_noise_std": float(self.imu_noise_std),
            "degradation_friction": float(self.episode_friction),
            "direction_command_mode": int(self.direction_command_mode),
            "command_vx": float(self.command_vx),
            "command_vy": float(self.command_vy),
            "command_wz": float(self.command_wz),
            "action_magnitude_penalty": action_magnitude_penalty,
            "action_smoothness_penalty": action_smoothness_penalty,
            "action_jerk_penalty": action_jerk_penalty,
            "joint_velocity_delta_penalty": joint_velocity_delta_penalty,
            "vertical_velocity_penalty": vertical_velocity_penalty,
            "feet_gait_reward": reward_terms["feet_gait_reward"],
            "gait_sync_score": reward_terms["gait_sync_score"],
            "joint_mirror_penalty": reward_terms["joint_mirror_penalty"],
            "joint_mirror_error": reward_terms["joint_mirror_error"],
            "feet_air_time_variance_penalty": reward_terms["feet_air_time_variance_penalty"],
            "feet_air_time_variance": reward_terms["feet_air_time_variance"],
            "forward_distance": float(forward_distance),
            "forward_reward": forward_reward,
            "target_progress_delta": reward_terms["target_progress_delta"],
            "target_progress_reward": reward_terms["target_progress_reward"],
            "speed_tracking_term": reward_terms["speed_tracking_term"],
            "speed_tracking_reward": reward_terms["speed_tracking_reward"],
            "speed_target": reward_terms["speed_target"],
            "commanded_forward_velocity": float(reward_terms["speed_target"]),
            "velocity_tracking_error": velocity_tracking_error,
            "stride_amplitude_score": reward_terms["stride_amplitude_score"],
            "stride_amplitude_reward": reward_terms["stride_amplitude_reward"],
            "hind_engagement_term": reward_terms["hind_engagement_term"],
            "hind_engagement_reward": reward_terms["hind_engagement_reward"],
            "foot_clearance_term": reward_terms["foot_clearance_term"],
            "foot_clearance_reward": reward_terms["foot_clearance_reward"],
            "foot_slip_term": reward_terms["foot_slip_term"],
            "foot_slip_penalty": reward_terms["foot_slip_penalty"],
            "foot_slip_ratio": reward_terms["foot_slip_ratio"],
            "mean_torque_effort": reward_terms["mean_torque_effort"],
            "mean_mechanical_work": reward_terms["mean_mechanical_work"],
            "step_mechanical_power_w": step_mechanical_power_w,
            "step_mechanical_work_j": step_mechanical_work_j,
            "true_stance_slip_distance_step_m": true_stance_slip_distance_step_m,
            "true_stance_intended_distance_step_m": true_stance_intended_distance_step_m,
            "true_stance_contact_count_step": int(true_stance_contact_count_step),
            "torque_effort_penalty": reward_terms["torque_effort_penalty"],
            "mechanical_power_penalty": reward_terms["mechanical_power_penalty"],
            "penalty_curriculum_factor": reward_terms["penalty_curriculum_factor"],
            "heading_penalty": heading_penalty,
            "yaw_rate_penalty": yaw_rate_penalty,
            "success": bool(success),
            "no_progress": bool(no_progress),
            "termination_reason": termination_reason,
            "steps_alive": self.steps_alive,
            "max_episode_steps": int(self.max_episode_steps),
        }

        if self.debug_diagnostics and self.debug_print_counter < self.debug_steps_to_print:
            self.debug_print_counter += 1
            print(
                "[GO1 DEBUG] "
                f"step={self.steps_alive:04d} "
                f"yaw0={self.initial_yaw:+.3f} yaw={yaw:+.3f} "
                f"pos=({x_position:+.3f},{y_position:+.3f}) "
                f"vxy=({vx:+.3f},{vy:+.3f}) "
                f"forward_v={forward_velocity:+.3f} "
                f"target_dist={target_distance_remaining:.3f}"
            )

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_alive = 0
        self.episode_count += 1
        self._sample_episode_degradation()
        self._configure_observation_delay_buffer()
        self.prev_action = np.zeros(12, dtype=np.float64)
        self.prev_prev_action = np.zeros(12, dtype=np.float64)
        self.prev_joint_positions = None
        self.prev_joint_velocities = np.zeros(12, dtype=np.float64)
        self.debug_print_counter = 0
        self.contact_zero_steps = 0
        self.touch_zero_steps = 0
        self.physics_zero_steps = 0
        self.contact_sensor_reliable = self.has_foot_contact_sensors
        self.prev_foot_xy_valid = False
        self.steps_since_progress = 0
        self.best_forward_distance = 0.0

        self.robot_node.resetPhysics()
        translation_field = self.robot_node.getField("translation")
        translation_field.setSFVec3f(self.initial_translation)
        rotation_field = self.robot_node.getField("rotation")
        rotation_field.setSFRotation(self.initial_rotation)

        self._apply_episode_floor_friction()
        self._apply_motor_strength_to_actuators()

        for i, motor in enumerate(self.joints):
            motor.setVelocity(self.max_joint_velocity)
            motor.setPosition(self.default_pose[i])

        for _ in range(self.settle_steps):
            self.supervisor.step(self.timestep)

        foot_xy = self._read_foot_xy()
        if foot_xy is not None:
            self.prev_foot_xy = foot_xy
            self.prev_foot_xy_valid = True

        _, _, self.initial_yaw = self.imu.getRollPitchYaw()
        if self.heading_flip:
            self.initial_yaw = float(
                np.arctan2(np.sin(self.initial_yaw + np.pi), np.cos(self.initial_yaw + np.pi))
            )
        self.initial_heading = np.array(
            [np.cos(self.initial_yaw), np.sin(self.initial_yaw)],
            dtype=np.float64,
        )

        start_position = self.robot_node.getField("translation").getSFVec3f()
        self.episode_start_xy = np.array(
            [float(start_position[0]), float(start_position[1])],
            dtype=np.float64,
        )

        self.target_xy = self.episode_start_xy + self.target_distance_m * self.initial_heading
        self.best_target_distance = float(np.linalg.norm(self.target_xy - self.episode_start_xy))
        self.reward_computer.reset_state(
            initial_target_distance=self.best_target_distance
        )

        if self.target_translation_field is not None:
            self.target_translation_field.setSFVec3f(
                [float(self.target_xy[0]), float(self.target_xy[1]), float(self.target_z)]
            )
        elif self.use_target_guidance and self.debug_diagnostics and not self._target_node_warned:
            print(
                "[GO1 DEBUG] "
                f"Target DEF '{self.target_def_name}' not found. "
                "Using virtual target only (reward still works, marker not visible)."
            )
            self._target_node_warned = True

        if self.debug_diagnostics:
            print(
                "[GO1 DEBUG] "
                f"reset episode={self.episode_count} "
                f"heading_flip={self.heading_flip} "
                f"use_target_guidance={self.use_target_guidance} "
                f"degradation_active={int(self.degradation_model_active)} "
                f"motor_strength={self.episode_motor_strength:.3f} "
                f"latency_ms={self.episode_latency_ms:.1f} "
                f"latency_steps={self.episode_latency_steps} "
                f"friction={self.episode_friction:.3f} "
                f"yaw_0={self.initial_yaw:+.3f} "
                f"heading=({self.initial_heading[0]:+.3f},{self.initial_heading[1]:+.3f}) "
                f"start_xy=({self.episode_start_xy[0]:+.3f},{self.episode_start_xy[1]:+.3f}) "
                f"target_xy=({self.target_xy[0]:+.3f},{self.target_xy[1]:+.3f}) "
                f"target_distance_m={self.target_distance_m:.2f}"
            )

        policy_obs = self.get_observations(apply_degradation=True)
        warmup_count = int(max(1, self.episode_latency_steps + 1))
        for _ in range(warmup_count):
            self._push_observation(policy_obs)

        return self._get_delayed_observation(), {}

    def render(self):
        pass
