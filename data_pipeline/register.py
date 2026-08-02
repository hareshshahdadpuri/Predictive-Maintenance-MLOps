"""Re-downloads the raw dataset from the Hugging Face dataset space and validates its schema."""
import os
import pandas as pd
from huggingface_hub import HfApi, create_repo, hf_hub_download

HF_USERNAME = os.environ.get("HF_USERNAME", "shahdadpuri")
HF_DATASET_REPO = f"{HF_USERNAME}/predictive-maintenance-engine-data"
HF_TOKEN = os.environ["HF_TOKEN"]

api = HfApi(token=HF_TOKEN)
create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

local_path = hf_hub_download(
    repo_id=HF_DATASET_REPO, filename="engine_data.csv", repo_type="dataset", token=HF_TOKEN
)
df = pd.read_csv(local_path)

expected_columns = [
    "Timestamp", "Engine rpm", "Lub oil pressure", "Fuel pressure",
    "Coolant pressure", "lub oil temp", "Coolant temp", "Engine Condition",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset verified on Hugging Face:", HF_DATASET_REPO)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(df["Engine Condition"].value_counts())
