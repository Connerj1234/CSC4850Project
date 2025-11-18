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
            ("clf", LinearSVC(max_iter=5000)),
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
                    max_iter=400,
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
          * Run GridSearchCV with 5 fold stratified CV
          * Track best model and score
      - Refit best overall model on all training data
      - Predict on test data
      - Save predictions
    """
    print("=" * 60)
    print(f"Dataset {dataset_id}")

    X_train, y_train, X_test = load_dataset(dataset_id)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model_configs = get_model_configs(dataset_id)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_overall = None
    best_overall_name = None
    best_overall_score = -np.inf

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

        if grid.best_score_ > best_overall_score:
            best_overall_score = grid.best_score_
            best_overall = grid.best_estimator_
            best_overall_name = name

    print("\n" + "-" * 60)
    print(f"Best model for dataset {dataset_id}: {best_overall_name}")
    print(f"Best CV accuracy: {best_overall_score:.4f}")

    # Refit is already done on full training data inside GridSearchCV for best params,
    # but to be explicit we can refit the chosen best model
    best_overall.fit(X_train, y_train)

    # Predict on test data
    y_pred = best_overall.predict(X_test)

    # Save predictions
    save_predictions(y_pred, dataset_id)


def main():
    # Run all four classification datasets
    for dataset_id in [1, 2, 3, 4]:
        run_dataset(dataset_id)


if __name__ == "__main__":
    main()
