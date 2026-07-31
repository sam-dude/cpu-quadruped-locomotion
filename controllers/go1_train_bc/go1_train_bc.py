import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.append(r"C:\Program Files\Webots\lib\controller\python")

workspace_root = Path(__file__).resolve().parents[2]
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

from utils.pipelines.bc_pipeline import (
    create_run_dir,
    find_artifact,
    train_bc_from_dataset,
)
from utils.envs.go1_env import Go1Env

TRAJECTORY_RUN_FOLDER_NAME = None
BC_EPOCHS = 50
BC_BATCH_SIZE = 64
BC_LEARNING_RATE = 1e-3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    trajectory_path = find_artifact(
        run_folder_name=TRAJECTORY_RUN_FOLDER_NAME,
        artifact_name="trajectory_data.npz",
        run_prefix="run_traj",
    )

    run_dir = create_run_dir("run_bc")
    output_model_path = run_dir / "go1_bc_policy"
    model_name = "PPO(MlpPolicy)"
    started_utc = _utc_now_iso()

    print(
        "BC training start | "
        f"date_utc={started_utc} "
        f"run_folder={run_dir} "
        f"model={model_name}"
    )
    print(f"BC expert data folder: {trajectory_path.parent}")
    print(f"BC expert data file: {trajectory_path}")

    metadata = {
        "created_utc": started_utc,
        "run_dir": str(run_dir),
        "model": model_name,
        "dataset_file": str(trajectory_path),
        "dataset_folder": str(trajectory_path.parent),
        "controller": str(Path(__file__).resolve()),
    }
    metadata_path = run_dir / "run_config.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved reproducibility metadata: {metadata_path}")

    env = Go1Env()
    train_bc_from_dataset(
        env,
        dataset_path=trajectory_path,
        output_model_path=output_model_path,
        n_epochs=BC_EPOCHS,
        batch_size=BC_BATCH_SIZE,
        learning_rate=BC_LEARNING_RATE,
    )

    print(f"BC training complete: {output_model_path}.zip")
    print(f"Source trajectory: {trajectory_path}")


if __name__ == "__main__":
    main()
