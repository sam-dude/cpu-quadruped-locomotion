---
annotations_creators:
- expert-demonstration
language:
- en
license: mit
task_categories:
- robotics
- reinforcement-learning
tags:
- quadruped
- webots
- unitree-go1
- behavior-cloning
- trajectory-data
pretty_name: Unitree Go1 Quadruped Trot Gait Expert Dataset
---

# 🤖 Unitree Go1 Quadruped Expert Trajectory Dataset

This dataset contains expert trajectory demonstrations for the Unitree Go1 quadruped robot operating in Webots. It serves as the initialisation dataset for Behavior Cloning ($\pi_{BC}$) in the paper:

> **"Training Walking Robots on Low-Cost Hardware: A CPU-Only Approach for African Urban Deployment"**

---

## 📊 Dataset Structure & Contents

The dataset is packaged in NumPy compressed archive format (`.npz`):

* **`trajectory_data.npz`** (1.3 MB):
  * **`observations`**: `(10000, 36)` array containing normalized robot states (roll, pitch, angular velocities, joint positions, joint velocities).
  * **`actions`**: `(10000, 12)` array containing target joint angle motor commands for the 12 Go1 leg actuators.

* **`trajectory_data_window3.npz`** (3.25 MB):
  * **`observations`**: `(10000, 108)` array incorporating a 3-step temporal observation window for sequence modeling.
  * **`actions`**: `(10000, 12)` array containing target joint angle motor commands.

---

## 💻 Usage & Loading in Python

### Using NumPy
```python
import numpy as np

# Load trajectory dataset
data = np.load("trajectory_data.npz")

observations = data["observations"] # Shape: (10000, 36)
actions = data["actions"]           # Shape: (10000, 12)

print(f"Loaded {len(observations)} expert state-action pairs.")
```

### Downloading via Hugging Face Hub
```python
from huggingface_hub import hf_hub_download
import numpy as np

file_path = hf_hub_download(
    repo_id="sam-dude/go1-expert-trajectories",
    filename="trajectory_data.npz",
    repo_type="dataset"
)

data = np.load(file_path)
print("Dataset loaded successfully!")
```

---

## 📜 Citation

```bibtex
@inproceedings{ibiyemi2026cpu,
  title={Training Walking Robots on Low-Cost Hardware: A CPU-Only Approach for African Urban Deployment},
  author={Ibiyemi, Samuel and Akinremi, Bunmi},
  year={2026}
}
```
