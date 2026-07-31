#!/usr/bin/env python3
"""
Upload Expert Trajectory Dataset to Hugging Face Hub
"""

import sys
import os
from pathlib import Path
from huggingface_hub import HfApi, login

DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"
DEFAULT_REPO_ID = "sam-dude/go1-expert-trajectories"

def upload_dataset(repo_id: str = DEFAULT_REPO_ID, token: str = None) -> None:
    api = HfApi()

    if token:
        login(token=token)
    else:
        try:
            user = api.whoami()
            print(f"Authenticated as Hugging Face user: {user['name']}")
        except Exception:
            print("Not authenticated. Please enter your Hugging Face Access Token.")
            print("Get token from: https://huggingface.co/settings/tokens")
            token = input("HF Access Token: ").strip()
            if not token:
                print("Error: Token required to create and upload dataset.")
                sys.exit(1)
            login(token=token)

    print(f"Creating Hugging Face dataset repository: {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=False
    )

    print(f"Uploading files from {DATASET_DIR} to Hugging Face...")
    api.upload_folder(
        folder_path=str(DATASET_DIR),
        repo_id=repo_id,
        repo_type="dataset"
    )

    dataset_url = f"https://huggingface.co/datasets/{repo_id}"
    print("\n✅ Dataset upload complete!")
    print(f"🔗 Hugging Face Dataset URL: {dataset_url}\n")

if __name__ == "__main__":
    repo_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO_ID
    upload_dataset(repo_id=repo_name)
