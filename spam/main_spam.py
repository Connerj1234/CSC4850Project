# main_spam.py

import re
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB

RANDOM_STATE = 42
LAST_NAME = "Jamison"

def get_base_dir() -> Path:
    """
    Spam folder containing spam_train1.csv, spam_train2.csv, spam_test.csv.
    """
    return Path(__file__).resolve().parent


def normalize_labels(y):
    """
    Map various label encodings (e.g., 'spam', 'ham', 0/1) into {0, 1}.
    0 = ham / non-spam
    1 = spam
    """
    out = []
    for v in y:
        s = str(v).strip().lower()
        if s in ("ham", "0", "false", "no"):
            out.append(0)
        elif s in ("spam", "1", "true", "yes"):
            out.append(1)
        else:
            # fall back to int if possible
            try:
                out.append(1 if int(s) == 1 else 0)
            except Exception:
                raise ValueError(f"Unrecognized label value: {v}")
    return np.array(out, dtype=int)


# Precompiled regex patterns for cleaning
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_text(text: str) -> str:
    """
    Basic text normalization for spam messages:
      - lowercase
      - replace URLs with 'url'
      - replace phone numbers with 'phone'
      - remove punctuation / non-alphanumeric characters
      - collapse multiple spaces
    """
    s = str(text).lower()

    # Replace URLs and phone numbers with placeholders
    s = URL_RE.sub(" url ", s)
    s = PHONE_RE.sub(" phone ", s)

    # Remove punctuation / non-alphanumeric characters
    s = NON_ALNUM_RE.sub(" ", s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


def infer_text_label_columns(df: pd.DataFrame, is_test: bool = False):
    """
    Infer which columns are text and label based on common spam datasets.
    For the classic SMS spam dataset, columns are often:
      - 'v1' -> label
      - 'v2' -> message text
    For test data, there may be only a text column.
    """
    cols = [c.lower() for c in df.columns]

    if is_test:
        # Prefer 'v2' or 'text' or the last column as message text
        if "v2" in cols:
            text_col = df.columns[cols.index("v2")]
        elif "text" in cols:
            text_col = df.columns[cols.index("text")]
        else:
            text_col = df.columns[-1]
        return text_col, None

    # Training: need label + text
    if "v1" in cols and "v2" in cols:
        label_col = df.columns[cols.index("v1")]
        text_col = df.columns[cols.index("v2")]
    else:
        # Fallback guesses
        label_col = df.columns[0]
        text_col = df.columns[1]

    return text_col, label_col


def load_train(path: Path):
    """
    Load a training CSV and return (cleaned_text, normalized_labels).
    """
    df = pd.read_csv(path)
    text_col, label_col = infer_text_label_columns(df, is_test=False)

    X_raw = df[text_col].astype(str).fillna("")
    X_clean = X_raw.apply(clean_text)

    y_raw = df[label_col]
    y = normalize_labels(y_raw)

    return X_clean, y


def load_test(path: Path):
    """
    Load test CSV and return cleaned text series.
    """
    df = pd.read_csv(path)
    text_col, _ = infer_text_label_columns(df, is_test=True)

    X_raw = df[text_col].astype(str).fillna("")
    X_clean = X_raw.apply(clean_text)

    return X_clean


def build_features():
    """
    Build a FeatureUnion combining:
      - word-level TF-IDF (unigrams + bigrams)
      - character-level TF-IDF
    """
    word = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.99,
    )

    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.99,
    )

    feats = FeatureUnion(
        transformer_list=[
            ("word", word),
            ("char", char),
        ]
    )

    return feats


def candidate_models():
    """
    Return model objects for the four model families:
      - Logistic Regression
      - Linear SVM
      - ComplementNB
      - MultinomialNB
    """
    return {
        "logreg": LogisticRegression(
            max_iter=5000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "linsvm": LinearSVC(
            dual=False,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "cnb": ComplementNB(),
        "mnb": MultinomialNB(),
    }


def grids():
    """
    Hyperparameter grids for each model.
    We also allow the TF-IDF min_df parameters to move a bit.
    """
    base = {
        "feats__word__min_df": [2, 5],
        "feats__char__min_df": [2, 5],
    }

    return {
        "logreg": {**base, "clf__C": [0.5, 1.0, 2.0]},
        "linsvm": {**base, "clf__C": [0.5, 1.0, 2.0]},
        "cnb":    {**base, "clf__alpha": [0.5, 1.0, 2.0]},
        "mnb":    {**base, "clf__alpha": [0.5, 1.0, 2.0]},
    }


def main():
    base_dir = get_base_dir()
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load and combine the two training sets
    train1_path = base_dir / "spam_train1.csv"
    train2_path = base_dir / "spam_train2.csv"
    test_path = base_dir / "spam_test.csv"

    X1, y1 = load_train(train1_path)
    X2, y2 = load_train(train2_path)

    X = pd.concat([X1, X2], axis=0).reset_index(drop=True)
    y = np.concatenate([y1, y2], axis=0)

    print(f"Combined training shape: {X.shape[0]} samples")

    feats = build_features()
    models = candidate_models()
    param_grids = grids()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_score = -np.inf
    best_name = None
    best_estimator = None

    # Evaluate each candidate model using F1 score
    for name, clf in models.items():
        print(f"\nRunning model: {name}")

        pipe = Pipeline(
            steps=[
                ("feats", feats),
                ("clf", clf),
            ]
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[name],
            scoring="f1",          # binary spam detection, F1 is appropriate
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(X, y)

        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV F1: {grid.best_score_:.4f}")

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_name = name
            best_estimator = grid.best_estimator_

    print(f"\nSelected model: {best_name}  CV_F1={best_score:.4f}")

    # Fit best model on all training data (GridSearchCV already refits, but be explicit)
    best_estimator.fit(X, y)

    # Predict on test set
    Xte = load_test(test_path)
    preds = best_estimator.predict(Xte)

    out_path = results_dir / f"{LAST_NAME}Spam.txt"
    with out_path.open("w") as f:
        for v in preds:
            f.write(str(int(v)) + "\n")

    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
