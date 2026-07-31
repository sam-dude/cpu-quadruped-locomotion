import sys
import os
import json
import random
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

sys.path.append(r"C:\Program Files\Webots\lib\controller\python")

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from stable_baselines3 import PPO

from utils.pipelines.bc_pipeline import (
    build_ppo_callbacks,
    create_run_dir,
    find_artifact,
    make_monitored_env,
)
from utils.envs.go1_env import Go1Env

STAGE_NAME = os.getenv("GO1_TRAIN_STAGE", "stage_b")
STAGE_CONFIG_DIR = workspace_root / "configs" / "training"
ACTIVE_STAGE_CONFIG_PATH = STAGE_CONFIG_DIR / "active_stage.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _load_stage_config(stage_name: str) -> tuple[Path, dict[str, Any]]:
    normalized_stage = stage_name.replace("\\", "/").strip("/")
    candidates: list[Path] = []

    if normalized_stage.lower().endswith(".json"):
        stage_path = Path(normalized_stage)
        if stage_path.is_absolute():
            candidates.append(stage_path)
        else:
            candidates.append(workspace_root / stage_path)

    candidates.extend(
        [
            STAGE_CONFIG_DIR / f"{normalized_stage}.json",
            STAGE_CONFIG_DIR / normalized_stage / "index.json",
        ]
    )

    config_path = None
    for candidate in candidates:
        if candidate.exists():
            config_path = candidate
            break

    if config_path is None:
        raise FileNotFoundError(
            "Stage config not found. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid stage config format: {config_path}")
    return config_path, config


def _resolve_stage_name() -> tuple[str, str]:
    env_stage = os.getenv("GO1_TRAIN_STAGE")
    if env_stage:
        return str(env_stage), "environment"

    if ACTIVE_STAGE_CONFIG_PATH.exists():
        with ACTIVE_STAGE_CONFIG_PATH.open("r", encoding="utf-8") as f:
            active_config = json.load(f)

        if not isinstance(active_config, dict):
            raise ValueError(f"Invalid active stage config format: {ACTIVE_STAGE_CONFIG_PATH}")

        active_config_path = active_config.get("active_config")
        if isinstance(active_config_path, str) and active_config_path.strip():
            return active_config_path.strip(), "active_stage_file:active_config"

        stage = active_config.get("stage")
        if isinstance(stage, str) and stage.strip():
            return stage.strip(), "active_stage_file"

        raise ValueError(
            f"Missing 'active_config' or 'stage' string in active stage config: {ACTIVE_STAGE_CONFIG_PATH}"
        )

    return STAGE_NAME, "default"


def _set_go1_environment(env_config: dict[str, Any]) -> dict[str, str]:
    applied: dict[str, str] = {}
    for key, value in env_config.items():
        env_key = key if key.startswith("GO1_") else f"GO1_{key}"
        env_value = _as_env_value(value)
        os.environ[env_key] = env_value
        applied[env_key] = env_value
    return applied


def _resolve_init_model_path(init_config: dict[str, Any]) -> Path:
    source = str(init_config.get("source", "bc")).lower()
    run_folder_name = init_config.get("run_folder_name")

    if source == "bc":
        return find_artifact(
            run_folder_name=run_folder_name,
            artifact_name=str(init_config.get("artifact_name", "go1_bc_policy.zip")),
            run_prefix=str(init_config.get("run_prefix", "run_bc")),
        )
    if source == "ppo":
        return find_artifact(
            run_folder_name=run_folder_name,
            artifact_name=str(init_config.get("artifact_name", "go1_ppo_final.zip")),
            run_prefix=str(init_config.get("run_prefix", "run_ppo_stage_a")),
        )

    raise ValueError(f"Unsupported init source '{source}'. Expected one of: bc, ppo.")


def _write_reproducibility_metadata(
    run_dir: Path,
    *,
    stage_name: str,
    stage_config_path: Path,
    stage_config: dict[str, Any],
    init_model_path: Path,
    applied_go1_env: dict[str, str],
    total_timesteps: int,
    checkpoint_freq: int,
    seed: int,
    stage_source: str,
    model: str,
) -> None:
    metadata = {
        "created_utc": _utc_now_iso(),
        "stage": stage_name,
        "stage_config_path": str(stage_config_path),
        "run_dir": str(run_dir),
        "model": model,
        "init_model_path": str(init_model_path),
        "total_timesteps": int(total_timesteps),
        "checkpoint_freq": int(checkpoint_freq),
        "seed": int(seed),
        "stage_source": stage_source,
        "reward_formula": Go1Env.reward_formula(),
        "applied_go1_environment": applied_go1_env,
        "stage_config": stage_config,
        "controller": str(Path(__file__).resolve()),
    }

    metadata_path = run_dir / "run_config.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reproducibility metadata: {metadata_path}")

    applied_env_path = run_dir / "applied_go1_environment.json"
    with applied_env_path.open("w", encoding="utf-8") as f:
        json.dump(applied_go1_env, f, indent=2, sort_keys=True)
    print(f"Saved applied GO1 environment config: {applied_env_path}")


def main() -> None:
    stage_name, stage_source = _resolve_stage_name()
    stage_config_path, stage_config = _load_stage_config(stage_name)

    env_config = stage_config.get("go1_env", {})
    if not isinstance(env_config, dict):
        raise ValueError(f"Invalid 'go1_env' section in {stage_config_path}")
    applied_go1_env = _set_go1_environment(env_config)

    init_model = stage_config.get("init_model", {})
    if not isinstance(init_model, dict):
        raise ValueError(f"Invalid 'init_model' section in {stage_config_path}")
    init_model_path = _resolve_init_model_path(init_model)

    total_timesteps = int(stage_config.get("total_timesteps", 500_000))
    checkpoint_freq = int(stage_config.get("checkpoint_freq", 5_000))
    seed = int(stage_config.get("seed", 42))
    print_training_config = bool(stage_config.get("print_training_config", True))
    run_prefix = str(stage_config.get("run_prefix", f"run_ppo_{stage_name}"))

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    run_dir = create_run_dir(run_prefix)
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model_name = "PPO(MlpPolicy)"

    print(
        "PPO training start | "
        f"date_utc={_utc_now_iso()} "
        f"run_folder={run_dir} "
        f"model={model_name}"
    )

    base_env = Go1Env()
    env = make_monitored_env(base_env, run_dir)
    callbacks = build_ppo_callbacks(run_dir, checkpoint_freq=checkpoint_freq, condition="clean")

    model = PPO.load(str(init_model_path), env=env, device="cpu")
    model.tensorboard_log = str(tensorboard_dir)
    _write_reproducibility_metadata(
        run_dir,
        stage_name=stage_name,
        stage_source=stage_source,
        stage_config_path=stage_config_path,
        stage_config=stage_config,
        init_model_path=init_model_path,
        applied_go1_env=applied_go1_env,
        total_timesteps=total_timesteps,
        checkpoint_freq=checkpoint_freq,
        seed=seed,
        model=model_name,
    )

    if print_training_config:
        print(
            "Training config | "
            f"stage={stage_name} "
            f"stage_source={stage_source} "
            f"init={init_model_path} "
            f"timesteps={total_timesteps} "
            f"checkpoint_freq={checkpoint_freq} "
            f"seed={seed}"
        )
        for key in sorted(applied_go1_env):
            print(f"{key}={applied_go1_env[key]}")
        print(f"Reward formula | {Go1Env.reward_formula()}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=run_dir.name)
    model.save(str(run_dir / "go1_ppo_final"))

    env.close()
    print(f"PPO training complete ({stage_name}): {run_dir}")


if __name__ == "__main__":
    main()
