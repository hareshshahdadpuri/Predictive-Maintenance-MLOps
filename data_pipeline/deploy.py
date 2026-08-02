"""Pushes the deployment folder (app.py, Dockerfile, requirements.txt, README.md) into the
Hugging Face Space, so the live app always serves the latest trained model."""
import os
from huggingface_hub import HfApi

HF_USERNAME = os.environ.get("HF_USERNAME", "shahdadpuri")
HF_SPACE_REPO = os.environ.get("HF_SPACE_REPO", f"{HF_USERNAME}/predictive-maintenance-app")
HF_TOKEN = os.environ["HF_TOKEN"]

api = HfApi(token=HF_TOKEN)
api.upload_folder(
    folder_path="deployment",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    commit_message="Automated deployment update from GitHub Actions",
)
print("Deployment files pushed to https://huggingface.co/spaces/" + HF_SPACE_REPO)
