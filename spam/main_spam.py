# main_spam.py

import os
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import precision_recall_fscore_support

RANDOM_STATE = 42

def normalize_labels(y):
    out = []
    for v in y:
        s = str(v).strip().lower()
        if s in ("ham", "0", "false", "no"):
            out.append(0)
        elif s in ("spam", "1", "true", "yes"):
            out.append(1)
        else:
            # try int
            try:
                out.append(1 if int(s) == 1 else 0)
            except Exception:
                raise ValueError(f"Unrecognized label value: {v}")
    return np.array(out, dtype=int)

def infer_text_label_columns(df, is_test=False):
    cols = [c.lower() for c in df.columns]
    mapping = {c.lower(): c for c in df.columns}

    # Training sets
    if not is_test:
        # Case A (spam_train1.csv): v1=label, v2=text (SMS spam format)
        if "v1" in cols and "v2" in cols:
            return mapping["v2"], mapping["v1"]
        # Case B (spam_train2.csv): label, text
        if "label" in cols and "text" in cols:
            return mapping["text"], mapping["label"]
        # Fallback common names
        for tcol in ["text", "message", "content", "body"]:
            if tcol in cols:
                # try find label
                for lcol in ["label", "target", "class", "y"]:
                    if lcol in cols:
                        return mapping[tcol], mapping[lcol]
        raise ValueError(f"Cannot infer text/label columns from: {df.columns}")
    else:
        # Test set
        for tcol in ["text", "message", "content", "body", "v2"]:
            if tcol in cols:
                return mapping[tcol], None
        raise ValueError(f"Cannot infer text column from test columns: {df.columns}")

def load_train(path):
    df = pd.read_csv(path)
    text_col, label_col = infer_text_label_columns(df, is_test=False)
    X = df[text_col].astype(str).fillna("")
    y = normalize_labels(df[label_col].values)
    return X, y

def load_test(path):
    df = pd.read_csv(path)
    text_col, _ = infer_text_label_columns(df, is_test=True)
    X = df[text_col].astype(str).fillna("")
    return X

def build_features():
    word = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.99
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.99
    )
    return FeatureUnion([("word", word), ("char", char)])

def candidate_models():
    return {
        "logreg": LogisticRegression(max_iter=5000, solver="saga",
                                     class_weight="balanced", random_state=RANDOM_STATE),
        "linsvm": LinearSVC(dual=False, class_weight="balanced", random_state=RANDOM_STATE),
        "cnb":    ComplementNB(),
        "mnb":    MultinomialNB(),
    }

def grids():
    base = {"feats__word__min_df": [2, 5], "feats__char__min_df": [2, 5]}
    return {
        "logreg": {**base, "clf__C": [0.5, 1.0, 2.0]},
        "linsvm": {**base, "clf__C": [0.5, 1.0, 2.0]},
        "cnb":    {**base, "clf__alpha": [0.5, 1.0, 2.0]},
        "mnb":    {**base, "clf__alpha": [0.5, 1.0, 2.0]},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train1", default="/mnt/data/spam_train1.csv")
    ap.add_argument("--train2", default="/mnt/data/spam_train2.csv")
    ap.add_argument("--test",   default="/mnt/data/spam_test.csv")
    ap.add_argument("--lastname", required=True)
    args = ap.parse_args()

    os.makedirs("spam/results", exist_ok=True)

    X1, y1 = load_train(args.train1)
    X2, y2 = load_train(args.train2)
    X = pd.concat([X1, X2], axis=0).reset_index(drop=True)
    y = np.concatenate([y1, y2])

    feats = build_features()
    models = candidate_models()
    param_grids = grids()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    best_name, best_estimator, best_score = None, None, -np.inf
    for name, clf in models.items():
        pipe = Pipeline([("feats", feats), ("clf", clf)])
        grid = GridSearchCV(pipe, param_grids[name], cv=skf, scoring="f1",
                            n_jobs=-1, refit=True, verbose=0)
        grid.fit(X, y)
        # quick sanity check on full train
        yhat = grid.predict(X)
        p, r, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        print(f"[{name}] best_params={grid.best_params_}  CV_F1={grid.best_score_:.3f}  TrainApprox_F1={f1:.3f}")
        if grid.best_score_ > best_score:
            best_score, best_name, best_estimator = grid.best_score_, name, grid.best_estimator_

    print(f"Selected model: {best_name}  CV_F1={best_score:.4f}")
    Xte = load_test(args.test)
    preds = best_estimator.predict(Xte)

    out_path = os.path.join("spam", "results", f"{args.lastname}Spam.txt")
    with open(out_path, "w") as f:
        for v in preds:
            f.write(str(int(v)) + "\n")
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
