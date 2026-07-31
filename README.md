# CPU-Only Quadrupedal Locomotion Under Hardware Degradation

---

## 📌 Overview

This repository provides an open-source, CPU-only reinforcement learning pipeline for quadrupedal locomotion (Unitree Go1) in Webots. It addresses compute limitations in resource-constrained research environments by eliminating GPU requirements while explicitly modeling hardware imperfections characteristic of low-cost robotic builds:
- **Control Latency** (15–20 ms communication delays)
- **Sensor Noise & Drift** (IMU angular velocity & acceleration noise)
- **Actuator Limits** (Motor torque limits & thermal saturation)

The pipeline uses a two-stage approach:
1. **Behavior Cloning ($\pi_{BC}$)**: Initialises quadruped gait using synthetic trot trajectory demonstrations.
2. **PPO Fine-Tuning ($\pi_C$ & $\pi_D$)**: Fine-tunes the policy using Proximal Policy Optimisation under idealised clean ($\pi_C$) or hardware-degraded ($\pi_D$) environments.

---

## 📂 Repository Structure

```
cpu-quadruped-locomotion/
├── README.md                      # Documentation & Quickstart
├── requirements.txt                # Python package dependencies
├── configs/                        # Reproducible stage presets
│   └── training/
│       ├── active_stage.json
│       ├── clean/
│       │   └── clean_policy_from_bc.json
│       └── degraded/
│           └── degraded_policy_from_bc.json
├── controllers/                    # Webots robot controllers
│   ├── go1_collect_trajectory/    # Expert gait dataset generation
│   ├── go1_train_bc/               # Stage 1: Behavior Cloning initialisation
│   ├── go1_train_ppo_from_bc/      # Stage 2: CPU PPO fine-tuning
│   ├── go1_eval_paper_metrics/     # Metric logging & benchmark evaluation
│   └── go1_eval_policy/            # Visual policy evaluation controller
├── dataset/                        # Expert Trajectory Dataset for Behavior Cloning
│   ├── README.md                   # Hugging Face Dataset Card
│   ├── trajectory_data.npz         # 10,000 steps Go1 trot trajectory (36-dim obs)
│   └── trajectory_data_window3.npz # 10,000 steps Go1 trot trajectory (108-dim obs)
├── utils/                          # Core Python classes
│   ├── envs/go1_env.py             # Gymnasium Webots Env + Degradation Model
│   ├── rewards/go1_reward.py       # Multi-objective reward formulation
│   ├── pipelines/bc_pipeline.py    # BC model architecture & trainer
│   ├── callbacks/training_callbacks.py # Metric logging & curriculum callbacks
│   └── logging/paper_metrics.py    # Robustness metrics collector
├── worlds/                         # Simulation environments
│   ├── world.wbt                   # Flat benchmark terrain
│   ├── world_laterite.wbt          # Unpaved laterite soil
│   ├── world_debris.wbt            # Construction debris terrain
│   └── world_uneven.wbt            # Erosion depressions & slopes
├── protos/                         # Robot 3D assets & URDF definitions
│   ├── go1.proto
│   └── meshes/
├── pretrained/                     # Pre-trained model checkpoints
│   ├── pi_bc.zip                   # Behavior cloning policy model
│   ├── pi_clean.zip                # Clean baseline policy (pi_C)
│   └── pi_degraded.zip             # Hardware-degraded policy (pi_D)
└── scripts/                        # Reproducible plotting & dataset tools
    ├── plot_paper_figures.py
    ├── paper_plot_constants.py
    └── upload_to_huggingface.py
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

* **Webots**: Download & install [Webots R2023b+](https://cyberbotics.com/).
* **Python**: Python 3.10+ recommended.

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Set Webots environment variables (adjust path according to your OS):
```powershell
# Windows PowerShell Example
$env:WEBOTS_HOME = "C:\Program Files\Webots"
$env:PYTHONPATH = "$env:WEBOTS_HOME\lib\controller\python"
```

---

### 2. Expert Trajectory Dataset (Hugging Face)

The expert dataset is hosted on Hugging Face at 🤗 **[sam-dude/go1-expert-trajectories](https://huggingface.co/datasets/sam-dude/go1-expert-trajectories)**.

#### Downloading Dataset from Hugging Face:
```python
from huggingface_hub import hf_hub_download
import numpy as np

# Download trajectory dataset directly from Hugging Face
file_path = hf_hub_download(
    repo_id="sam-dude/go1-expert-trajectories",
    filename="trajectory_data.npz",
    repo_type="dataset"
)

data = np.load(file_path)
print(f"Loaded {len(data['observations'])} expert steps.")
```

#### Uploading/Updating Dataset on Hugging Face:
Run the included upload script:
```bash
python scripts/upload_to_huggingface.py sam-dude/go1-expert-trajectories
```

---

### 3. Evaluating Pre-trained Policies

Pre-trained model checkpoints are included in `pretrained/`.

To visually evaluate the degraded policy ($\pi_D$) in Webots:
```bash
python controllers/go1_eval_policy/go1_eval_policy.py
```

To run complete paper evaluation metrics across clean and degraded regimes:
```bash
python controllers/go1_eval_paper_metrics/go1_eval_paper_metrics.py
```

---

### 4. Training Pipeline Reproducibility

#### Step 1: Expert Demonstration & Behavior Cloning ($\pi_{BC}$)
Generate gait trajectory demonstrations (or use the dataset in `dataset/`) and train the initial BC policy:
```bash
# 1. Collect expert trot trajectory (optional if using dataset/)
python controllers/go1_collect_trajectory/go1_collect_trajectory.py

# 2. Train Behavior Cloning policy
python controllers/go1_train_bc/go1_train_bc.py
```

#### Step 2: CPU PPO Policy Optimization ($\pi_C$ and $\pi_D$)

Select the active regime in `configs/training/active_stage.json`:

* **For Clean Baseline Policy ($\pi_C$)**:
  ```json
  {
    "stage": "clean/clean_policy_from_bc"
  }
  ```

* **For Hardware-Degraded Robust Policy ($\pi_D$)**:
  ```json
  {
    "stage": "degraded/degraded_policy_from_bc"
  }
  ```

Run PPO policy training:
```bash
python controllers/go1_train_ppo_from_bc/go1_train_ppo_from_bc.py
```

---

### 5. Reproducing Paper Figures

To re-generate all publication-quality graphs:
```bash
python scripts/plot_paper_figures.py
```

---

## 📜 Citation

If you find this codebase or dataset useful in your research, please cite our working paper:

```bibtex
@article{ibiyemi2026cpu,
  title={Training Walking Robots on Low-Cost Hardware: A CPU-Only Approach for African Urban Deployment},
  author={Ibiyemi, Samuel and Akinremi, Bunmi},
  journal={Preprint / Working Paper},
  year={2026}
}
```
