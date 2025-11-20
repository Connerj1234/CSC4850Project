import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt

DATASET_SUMMARIES = []

def get_base_dir() -> Path:
    """
    Returns the directory that contains this file (classification folder).
    """
    return Path(__file__).resolve().parent


def load_dataset(dataset_id: int):
    """
    Load TrainData{k}.txt, TrainLabel{k}.txt, TestData{k}.txt for given k.

    Returns:
        X_train: np.ndarray, shape (n_train_samples, n_features)
        y_train: np.ndarray, shape (n_train_samples,)
        X_test: np.ndarray, shape (n_test_samples, n_features)
    """
    base = get_base_dir()

    train_data_path = base / f"TrainData{dataset_id}.txt"
    train_label_path = base / f"TrainLabel{dataset_id}.txt"
    test_data_path = base / f"TestData{dataset_id}.txt"

    X_train = np.loadtxt(train_data_path)
    y_train = np.loadtxt(train_label_path).astype(int)
    X_test = np.loadtxt(test_data_path)

    return X_train, y_train, X_test


def get_model_configs(dataset_id: int):
    """
    Define model pipelines and hyperparameter grids for this dataset.

    We always:
      - Impute missing values (1e99) with mean
      - For linear models and MLP: standardize features

    Returns:
        Dict of model_name -> (pipeline, param_grid)
    """
    imputer = SimpleImputer(missing_values=1e99, strategy="mean")

    # Some light dataset specific tweaks
    if dataset_id in (1, 2):
        # Small sample, high dimensional: keep models simpler and heavily regularized
        mlp_hidden = (50,)
        rf_n_estimators = 100
        rf_max_depth = 5
    else:
        # More samples, fewer features: can afford slightly richer models
        mlp_hidden = (128, 64)
        rf_n_estimators = 200
        rf_max_depth = None

    models = {}

    # 1. Logistic Regression
    logreg_pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    logreg_param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }
    models["logistic_regression"] = (logreg_pipeline, logreg_param_grid)

    # 2. Linear SVM
    svm_pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=10000, dual="auto")),
        ]
    )
    svm_param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }
    models["linear_svm"] = (svm_pipeline, svm_param_grid)

    # 3. Random Forest (trees do not need scaling)
    rf_pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("clf", RandomForestClassifier(n_estimators=rf_n_estimators, random_state=42)),
        ]
    )
    rf_param_grid = {
        "clf__max_depth": [rf_max_depth, 10, 20] if rf_max_depth is None else [rf_max_depth, 3, 7],
        "clf__min_samples_split": [2, 5],
    }
    models["random_forest"] = (rf_pipeline, rf_param_grid)

    # 4. MLP (neural network)
    mlp_pipeline = Pipeline(
        steps=[
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=mlp_hidden,
                    activation="relu",
                    solver="adam",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    mlp_param_grid = {
        "clf__alpha": [0.0005, 0.001, 0.01],
    }
    models["mlp"] = (mlp_pipeline, mlp_param_grid)

    return models


def save_predictions(preds: np.ndarray, dataset_id: int):
    """
    Save predicted labels as one integer per line in results/LastNameClassification{k}.txt
    """
    base = get_base_dir()
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"JamisonClassification{dataset_id}.txt"
    out_path = results_dir / filename

    with out_path.open("w") as f:
        for label in preds:
            f.write(f"{int(label)}\n")

    print(f"Saved predictions for dataset {dataset_id} to {out_path}")


def run_dataset(dataset_id: int):
    """
    Run full model selection for a single dataset:
      - Load data
      - For each candidate model:
          * Run GridSearchCV with stratified CV
          * Track best model and score
      - Refit best overall model on all training data
      - Predict on test data
      - Save predictions
      - Store stats for plotting
    """
    print("=" * 60)
    print(f"Dataset {dataset_id}")

    X_train, y_train, X_test = load_dataset(dataset_id)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model_configs = get_model_configs(dataset_id)

    # choose a safe number of CV splits given smallest class size
    class_counts = np.bincount(y_train)
    min_class_count = class_counts[class_counts > 0].min()
    n_splits = min(5, int(min_class_count))
    if n_splits < 2:
        n_splits = 2  # fallback to at least 2 folds

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_overall = None
    best_overall_name = None
    best_overall_score = -np.inf

    # store per model scores for plotting
    model_scores = {}

    for name, (pipeline, param_grid) in model_configs.items():
        print(f"\nRunning model: {name}")

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(X_train, y_train)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV accuracy: {grid.best_score_:.4f}")

        # record this model's score
        model_scores[name] = grid.best_score_

        if grid.best_score_ > best_overall_score:
            best_overall_score = grid.best_score_
            best_overall = grid.best_estimator_
            best_overall_name = name

    print("\n" + "-" * 60)
    print(f"Best model for dataset {dataset_id}: {best_overall_name}")
    print(f"Best CV accuracy: {best_overall_score:.4f}")

    # predict on test data with best model
    y_pred = best_overall.predict(X_test)

    # Save predictions
    save_predictions(y_pred, dataset_id)

    # store stats for plotting later
    DATASET_SUMMARIES.append(
        {
            "dataset_id": dataset_id,
            "n_train": int(X_train.shape[0]),
            "n_features": int(X_train.shape[1]),
            "model_scores": model_scores,
            "best_model_name": best_overall_name,
            "best_score": float(best_overall_score),
        }
    )

def make_classification_plots():
    """
    Use DATASET_SUMMARIES to generate:
      - one combined figure with samples and features per dataset
      - one bar chart per dataset with CV accuracy by model
    """
    if not DATASET_SUMMARIES:
        print("No dataset summaries available, skipping plots.")
        return

    base = get_base_dir()
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # sort by dataset id for consistent order
    summaries = sorted(DATASET_SUMMARIES, key=lambda d: d["dataset_id"])

    # combined samples and features figure
    labels = [f"Dataset {s['dataset_id']}" for s in summaries]
    n_train = [s["n_train"] for s in summaries]
    n_features = [s["n_features"] for s in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # left: samples
    axes[0].bar(labels, n_train)
    axes[0].set_title("Number of Samples per Dataset")
    axes[0].set_ylabel("Samples")
    axes[0].set_xlabel("Dataset")

    # right: features
    axes[1].bar(labels, n_features)
    axes[1].set_title("Number of Features per Dataset")
    axes[1].set_ylabel("Features")
    axes[1].set_xlabel("Dataset")

    fig.tight_layout()
    combined_path = results_dir / "classification_overview.png"
    fig.savefig(combined_path, dpi=300)
    plt.close(fig)

    print(f"Saved combined samples/features figure to {combined_path}")

    # per dataset model comparison plots
    for summary in summaries:
        dataset_id = summary["dataset_id"]
        model_scores = summary["model_scores"]

        model_names = list(model_scores.keys())
        scores = [model_scores[m] for m in model_names]

        plt.figure(figsize=(6, 4))
        plt.bar(model_names, scores)
        plt.ylim(0.0, 1.0)
        plt.ylabel("CV Accuracy")
        plt.xlabel("Model")
        plt.title(f"Dataset {dataset_id} – Cross validated accuracy by model")
        plt.tight_layout()

        out_path = results_dir / f"dataset{dataset_id}_models.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved model comparison figure for dataset {dataset_id} to {out_path}")

def main():
    # Run all four classification datasets
    for dataset_id in [1, 2, 3, 4]:
        run_dataset(dataset_id)
        
    try:
        make_classification_plots()
    except Exception as e:
        print(f"Could not create plots: {e}")


if __name__ == "__main__":
    main()
