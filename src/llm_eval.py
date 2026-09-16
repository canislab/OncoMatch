"""Evaluate LLM-based extraction against the team annotations.

Replays the exact held-out split used by src/train.py (same seed and test
size) so the numbers are directly comparable with models/extractor/metrics.json,
then runs src/llm_extract.py's Claude extraction over every held-out report
and scores it per feature. The training-split label values are passed to the
model as an annotation codebook; held-out labels are never shown to it.

Usage:
    python src/llm_eval.py --limit 3     # smoke test (3 reports)
    python src/llm_eval.py               # full held-out set

Writes per-report predictions, metrics, and token usage to
outputs/llm_eval.json.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from llm_extract import FEATURES, MODEL, extract, make_client
from train import normalize_label

ROOT = Path(__file__).resolve().parents[1]

# Claude Opus 5 first-party rates, $/1M tokens
PRICE_IN, PRICE_OUT = 5.00, 25.00


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "processed" / "key4.csv"
    )
    parser.add_argument("--limit", type=int, help="Only evaluate the first N reports")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="latin1")
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df = df.reset_index(drop=True)
    for feature in FEATURES:
        df[feature] = df[feature].map(normalize_label)

    # identical split to src/train.py -> scores are directly comparable
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    if args.limit:
        test_df = test_df.head(args.limit)
    print(f"Evaluating {len(test_df)} held-out reports with {MODEL}")

    vocab = {f: sorted(train_df[f].unique()) for f in FEATURES}
    client = make_client()

    def run_one(row) -> dict:
        features, usage = extract(client, row["text"], vocab)
        return {
            "patient_id": row.get("patient_id", ""),
            "predicted": features.model_dump(),
            "actual": {f: row[f] for f in FEATURES},
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            },
        }

    rows = [row for _, row in test_df.iterrows()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, rows))

    # score per feature, next to the RF pipeline's stored metrics
    rf_metrics = {}
    rf_path = ROOT / "models" / "extractor" / "metrics.json"
    if rf_path.exists():
        rf_metrics = json.loads(rf_path.read_text())["metrics"]

    metrics = {}
    header = f"{'feature':<16} {'llm_acc':>8} {'llm_f1w':>8} {'rf_acc':>7} {'rf_f1w':>7}"
    print("\n" + header)
    print("-" * len(header))
    for feature in FEATURES:
        y_true = [r["actual"][feature] for r in results]
        y_pred = [normalize_label(r["predicted"][feature]) for r in results]
        acc = accuracy_score(y_true, y_pred)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        rf = rf_metrics.get(feature, {})
        rf_acc = rf.get("accuracy", float("nan"))
        rf_f1 = rf.get("f1_weighted", float("nan"))
        print(f"{feature:<16} {acc:>8.3f} {f1w:>8.3f} {rf_acc:>7.3f} {rf_f1:>7.3f}")
        metrics[feature] = {
            "accuracy": round(float(acc), 4),
            "f1_weighted": round(float(f1w), 4),
        }

    tokens_in = sum(r["usage"]["input_tokens"] for r in results)
    tokens_out = sum(r["usage"]["output_tokens"] for r in results)
    cost = tokens_in / 1e6 * PRICE_IN + tokens_out / 1e6 * PRICE_OUT
    print(f"\nTokens: {tokens_in} in / {tokens_out} out  (~${cost:.2f} at Opus 5 rates)")

    out_path = ROOT / "outputs" / "llm_eval.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "date": date.today().isoformat(),
                "n_reports": len(results),
                "metrics": metrics,
                "tokens": {"input": tokens_in, "output": tokens_out},
                "est_cost_usd": round(cost, 4),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
