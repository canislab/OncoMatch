"""Train one classifier per oncology feature on annotated TCGA pathology reports.

Each of the seven annotated features (sex, cancer_type, treatment, tumor,
tumor_location, metastasis, metastatic_site) is its own classification task:
a TF-IDF + RandomForest pipeline trained on the report free text.

Usage:
    python src/train.py                 # trains on data/processed/key4.csv
    python src/train.py --data <csv>    # any CSV with 'text' + the feature columns

Artifacts land in models/extractor/: one <feature>.joblib pipeline per feature
plus metrics.json with held-out scores.
"""
import argparse
import json
import re
import warnings
from datetime import date
from pathlib import Path

# sklearn-internal noise from RandomForest(n_jobs=-1) on some
# sklearn/joblib version combinations; repeated once per worker task
warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning
)

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "sex",
    "cancer_type",
    "treatment",
    "tumor",
    "tumor_location",
    "metastasis",
    "metastatic_site",
]


def normalize_label(value) -> str:
    """Collapse annotation variants: trim, squeeze whitespace, lowercase.

    The team sheets contain e.g. 'Lymph nodes ' vs 'lymph nodes' as distinct
    strings; missing cells become 'unk'.
    """
    if pd.isna(value):
        return "unk"
    label = re.sub(r"\s+", " ", str(value)).strip().lower()
    return label if label else "unk"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "key4.csv",
        help="CSV with a 'text' column and the seven feature columns",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "The annotated TCGA dataset is not distributed with this repository "
            "(see data/README.md).\nTo run on the bundled synthetic demo set:\n"
            "  python src/train.py --data data/sample_annotations.csv --test-size 0.34"
        )

    df = pd.read_csv(args.data, encoding="latin1")
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df = df.reset_index(drop=True)
    for feature in FEATURES:
        df[feature] = df[feature].map(normalize_label)
    print(f"Loaded {len(df)} annotated reports from {args.data.name}")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=42
    )
    print(f"Split: {len(train_df)} train / {len(test_df)} test\n")

    out_dir = ROOT / "models" / "extractor"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    header = f"{'feature':<16} {'classes':>7} {'baseline':>9} {'accuracy':>9} {'f1_w':>7}"
    print(header)
    print("-" * len(header))
    for feature in FEATURES:
        y_train = train_df[feature]
        y_test = test_df[feature]

        pipeline = build_pipeline()
        pipeline.fit(train_df["text"], y_train)
        y_pred = pipeline.predict(test_df["text"])

        # majority-class baseline: the score a model must beat to be useful
        baseline = (y_test == y_train.value_counts().idxmax()).mean()
        acc = accuracy_score(y_test, y_pred)
        f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        n_classes = y_train.nunique()
        print(
            f"{feature:<16} {n_classes:>7} {baseline:>9.3f} {acc:>9.3f} {f1w:>7.3f}"
        )

        joblib.dump(pipeline, out_dir / f"{feature}.joblib")
        metrics[feature] = {
            "n_classes_train": int(n_classes),
            "majority_baseline": round(float(baseline), 4),
            "accuracy": round(float(acc), 4),
            "f1_weighted": round(float(f1w), 4),
        }

    meta = {
        "trained_on": args.data.name,
        "n_reports": len(df),
        "test_size": args.test_size,
        "date": date.today().isoformat(),
        "features": FEATURES,
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {len(FEATURES)} pipelines + metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
