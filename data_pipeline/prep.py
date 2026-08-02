"""Loads the raw dataset from Hugging Face, cleans it, splits it, and re-uploads train/test sets."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

HF_USERNAME = os.environ.get("HF_USERNAME", "shahdadpuri")
HF_DATASET_REPO = f"{HF_USERNAME}/predictive-maintenance-engine-data"
HF_TOKEN = os.environ["HF_TOKEN"]
RANDOM_STATE = 42

local_path = hf_hub_download(
    repo_id=HF_DATASET_REPO, filename="engine_data.csv", repo_type="dataset", token=HF_TOKEN
)
df = pd.read_csv(local_path)

df_clean = df.drop(columns=["Timestamp"]).drop_duplicates().dropna()

X = df_clean.drop(columns=["Engine Condition"])
y = df_clean["Engine Condition"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

train_df = X_train.copy()
train_df["Engine Condition"] = y_train.values
test_df = X_test.copy()
test_df["Engine Condition"] = y_test.values

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

api = HfApi(token=HF_TOKEN)
api.upload_file(path_or_fileobj="train.csv", path_in_repo="train.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")
api.upload_file(path_or_fileobj="test.csv", path_in_repo="test.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")

print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
print("train.csv and test.csv uploaded to", HF_DATASET_REPO)
