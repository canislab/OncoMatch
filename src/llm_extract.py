"""LLM-based oncology feature extraction from pathology report free text.

Uses Claude with structured outputs to extract the seven annotated features.
Unlike src/train.py this needs no training data at all - the key4 annotations
serve as the *evaluation* set instead (see src/llm_eval.py).

Auth: set ANTHROPIC_API_KEY (or log in with `ant auth login`).

Usage:
    python src/llm_extract.py --text "FINAL DIAGNOSIS: ..."
    python src/llm_extract.py --file path/to/report.txt
    type report.txt | python src/llm_extract.py
"""
import argparse
import json
import sys
from pathlib import Path

import anthropic
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-5"

FEATURES = [
    "sex",
    "cancer_type",
    "treatment",
    "tumor",
    "tumor_location",
    "metastasis",
    "metastatic_site",
]


class OncologyFeatures(BaseModel):
    """Structured oncology features extracted from one pathology report."""

    sex: str = Field(description="Patient sex: 'male', 'female', or 'unk' if not stated")
    cancer_type: str = Field(
        description="Cancer diagnosis, e.g. a TCGA study code or histology name; 'unk' if unclear"
    )
    treatment: str = Field(
        description="Treatment evidenced in the report (e.g. 'resection', 'chemo'); 'unk' if not stated"
    )
    tumor: str = Field(description="Is a tumor present: 'yes', 'no', or 'unk'")
    tumor_location: str = Field(
        description="Anatomical site of the primary tumor; 'unk' if not stated"
    )
    metastasis: str = Field(description="Evidence of metastasis: 'yes', 'no', or 'unk'")
    metastatic_site: str = Field(
        description="Site(s) of metastasis; 'n/a' if no metastasis, 'unk' if unclear"
    )


def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


SYSTEM_PROMPT = """You extract structured oncology features from pathology report free text.

Read the report and fill in every field. Use only what the report supports:
answer 'unk' when the report does not state or imply a value, rather than
guessing. Reports are TCGA pathology reports and may contain OCR noise and
encoding artifacts - read past them.

Answer in lowercase throughout.
"""


def vocab_hint(vocab: dict[str, list[str]] | None) -> str:
    """Optional annotation codebook: known label values per feature.

    Passing the vocabulary observed in the training annotations maps the
    model's answers onto the annotation scheme (e.g. TCGA study codes like
    'stad' rather than free-text histology names).
    """
    if not vocab:
        return ""
    lines = ["\nKnown annotation values per feature - prefer the closest match:"]
    for feature, values in vocab.items():
        lines.append(f"- {feature}: {', '.join(sorted(values))}")
    return "\n".join(lines)


def extract(
    client: anthropic.Anthropic,
    report_text: str,
    vocab: dict[str, list[str]] | None = None,
) -> tuple[OncologyFeatures, object]:
    """Extract features from one report. Returns (features, usage)."""
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT + vocab_hint(vocab),
                    # shared across every report in an eval run -> cache it
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": f"Pathology report:\n\n{report_text}"}],
            output_format=OncologyFeatures,
        )
    except TypeError as e:
        if "authentication" in str(e).lower():
            sys.exit(
                "No Anthropic credentials found. Set ANTHROPIC_API_KEY "
                "(https://console.anthropic.com/settings/keys) or run `ant auth login`."
            )
        raise
    return response.parsed_output, response.usage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Report text passed inline")
    parser.add_argument("--file", help="Path to a file containing the report text")
    args = parser.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = Path(args.file).read_text(encoding="latin1")
    else:
        text = sys.stdin.read()
        if not text.strip():
            sys.exit("No input: pass --text, --file, or pipe report text on stdin.")

    client = make_client()
    features, usage = extract(client, text)
    print(features.model_dump_json(indent=2))
    print(
        f"\n[tokens: {usage.input_tokens} in / {usage.output_tokens} out]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
