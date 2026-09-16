"""Extract structured oncology features from a pathology report's free text.

Loads the per-feature pipelines produced by src/train.py and prints a JSON
object with a prediction and confidence for each feature.

Usage:
    python src/predict.py --text "FINAL DIAGNOSIS: ..."
    python src/predict.py --file path/to/report.txt
    type report.txt | python src/predict.py
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

# sklearn-internal noise from RandomForest(n_jobs=-1) on some
# sklearn/joblib version combinations; repeated once per worker task
warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning
)

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "extractor"


def load_report_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="latin1")
    text = sys.stdin.read()
    if not text.strip():
        sys.exit("No input: pass --text, --file, or pipe report text on stdin.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Report text passed inline")
    parser.add_argument("--file", help="Path to a file containing the report text")
    args = parser.parse_args()

    if not MODEL_DIR.exists():
        sys.exit(f"No trained models in {MODEL_DIR} - run src/train.py first.")

    text = load_report_text(args)

    meta = json.loads((MODEL_DIR / "metrics.json").read_text())
    result = {}
    for feature in meta["features"]:
        pipeline = joblib.load(MODEL_DIR / f"{feature}.joblib")
        probs = pipeline.predict_proba([text])[0]
        best = probs.argmax()
        result[feature] = {
            "prediction": str(pipeline.classes_[best]),
            "confidence": round(float(probs[best]), 3),
        }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
