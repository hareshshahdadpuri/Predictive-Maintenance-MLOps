"""Downloads train/test data, tunes four model types, evaluates them, and registers the best model."""
import os
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from huggingface_hub import HfApi, create_repo

HF_USERNAME = os.environ.get("HF_USERNAME", "shahdadpuri")
HF_MODEL_REPO = f"{HF_USERNAME}/predictive-maintenance-engine-model"
HF_TOKEN = os.environ["HF_TOKEN"]
RANDOM_STATE = 42

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X_train = train_df.drop(columns=["Engine Condition"])
y_train = train_df["Engine Condition"]
X_test = test_df.drop(columns=["Engine Condition"])
y_test = test_df["Engine Condition"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grids = {
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_leaf": [1, 5]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
    ),
    "XGBoost": (
        XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
    ),
}

experiment_log = []
fitted_models = {}
for name, (estimator, params) in grids.items():
    grid = GridSearchCV(estimator, params, scoring="f1", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    fitted_models[name] = grid.best_estimator_
    for i in range(len(grid.cv_results_["params"])):
        experiment_log.append({
            "model": name,
            "params": grid.cv_results_["params"][i],
            "mean_cv_f1": grid.cv_results_["mean_test_score"][i],
        })
    print(f"{name}: best CV F1 = {grid.best_score_:.3f}, params = {grid.best_params_}")

pd.DataFrame(experiment_log).sort_values("mean_cv_f1", ascending=False).to_csv("experiment_log.csv", index=False)

results = []
for name, model in fitted_models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    })
results_df = pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
results_df.to_csv("model_comparison.csv", index=False)
print(results_df.round(3))

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
best_metrics = results_df.iloc[0].to_dict()
print("Best model:", best_model_name)

joblib.dump(best_model, "model.joblib")
with open("model_info.json", "w") as f:
    json.dump(
        {"algorithm": best_model_name, **{k: round(v, 4) for k, v in best_metrics.items() if k != "Model"}},
        f, indent=2,
    )

api = HfApi(token=HF_TOKEN)
create_repo(repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN, exist_ok=True)
api.upload_file(path_or_fileobj="model.joblib", path_in_repo="model.joblib", repo_id=HF_MODEL_REPO, repo_type="model")
api.upload_file(path_or_fileobj="model_info.json", path_in_repo="model_info.json", repo_id=HF_MODEL_REPO, repo_type="model")
print("Best model registered at https://huggingface.co/" + HF_MODEL_REPO)
