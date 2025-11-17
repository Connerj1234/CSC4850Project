import numpy as np
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Set this to your last name or group name
LAST_NAME = "Jamison"  # change if needed


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

    # Load data; files are whitespace separated
    X_train = np.loadtxt(train_data_path)
    y_train = np.loadtxt(train_label_path).astype(int)
    X_test = np.loadtxt(test_data_path)

    return X_train, y_train, X_test


def build_model():
    """
    Build a pipeline:
      - Impute missing values (1e99) using feature-wise mean
      - Standardize features
      - Logistic Regression classifier

    Wrapped in a GridSearchCV to tune regularization strength C.
    """
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(missing_values=1e99, strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # Small grid for C; you can expand this if you want more tuning
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )

    return grid


def save_predictions(preds: np.ndarray, dataset_id: int):
    """
    Save predicted labels as one integer per line in results/LastNameClassification{k}.txt
    """
    base = get_base_dir()
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{LAST_NAME}Classification{dataset_id}.txt"
    out_path = results_dir / filename

    with out_path.open("w") as f:
        for label in preds:
            f.write(f"{int(label)}\n")

    print(f"Saved predictions for dataset {dataset_id} to {out_path}")


def run_dataset(dataset_id: int):
    """
    Run the full pipeline for a single dataset:
      - Load data
      - Fit GridSearchCV
      - Print best params and CV score
      - Refit best model on all training data
      - Predict on test data
      - Save predictions
    """
    print("=" * 60)
    print(f"Dataset {dataset_id}")

    X_train, y_train, X_test = load_dataset(dataset_id)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = build_model()

    # Fit with cross validation to select best C
    model.fit(X_train, y_train)

    print(f"Best params for dataset {dataset_id}: {model.best_params_}")
    print(
        f"Best CV accuracy for dataset {dataset_id}: "
        f"{model.best_score_:.4f}"
    )

    # Best estimator already refit on full training data by default
    best_model = model.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_test)

    # Save predictions
    save_predictions(y_pred, dataset_id)


def main():
    # Run all four classification datasets
    for dataset_id in [1, 2, 3, 4]:
        run_dataset(dataset_id)


if __name__ == "__main__":
    main()
