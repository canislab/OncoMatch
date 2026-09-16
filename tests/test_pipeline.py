"""Tests for the extraction pipelines. No network access and no API key needed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_extract import FEATURES as LLM_FEATURES  # noqa: E402
from llm_extract import OncologyFeatures, vocab_hint  # noqa: E402
from train import FEATURES, build_pipeline, normalize_label  # noqa: E402


class TestNormalizeLabel:
    def test_missing_becomes_unk(self):
        assert normalize_label(np.nan) == "unk"
        assert normalize_label(None) == "unk"
        assert normalize_label("") == "unk"
        assert normalize_label("   ") == "unk"

    def test_case_and_whitespace_collapse(self):
        assert normalize_label(" Lymph  nodes ") == "lymph nodes"
        assert normalize_label("YES") == "yes"
        assert normalize_label("N/A") == "n/a"

    def test_annotation_variants_collapse_to_same_label(self):
        assert normalize_label("Lymph nodes ") == normalize_label("lymph  NODES")


class TestSchemas:
    def test_llm_schema_covers_exactly_the_seven_features(self):
        assert list(OncologyFeatures.model_fields) == FEATURES == LLM_FEATURES

    def test_vocab_hint_empty_without_vocab(self):
        assert vocab_hint(None) == ""
        assert vocab_hint({}) == ""

    def test_vocab_hint_lists_values_per_feature(self):
        hint = vocab_hint({"sex": ["male", "female", "unk"]})
        assert "sex: female, male, unk" in hint


class TestClassicalPipeline:
    @pytest.fixture(scope="class")
    def sample(self):
        df = pd.read_csv(ROOT / "data" / "sample_annotations.csv")
        for feature in FEATURES:
            df[feature] = df[feature].map(normalize_label)
        return df

    def test_pipeline_fits_and_predicts_known_labels(self, sample):
        pipeline = build_pipeline()
        pipeline.fit(sample["text"], sample["cancer_type"])
        predictions = pipeline.predict(sample["text"])
        assert set(predictions) <= set(sample["cancer_type"])
        # the synthetic classes are trivially separable by vocabulary
        assert (predictions == sample["cancer_type"]).mean() >= 0.8

    def test_pipeline_is_deterministic(self, sample):
        runs = []
        for _ in range(2):
            pipeline = build_pipeline()
            pipeline.fit(sample["text"], sample["metastasis"])
            runs.append(list(pipeline.predict(sample["text"])))
        assert runs[0] == runs[1]

    def test_predict_proba_available_for_confidence_scores(self, sample):
        pipeline = build_pipeline()
        pipeline.fit(sample["text"], sample["sex"])
        probs = pipeline.predict_proba([sample["text"][0]])[0]
        assert probs.sum() == pytest.approx(1.0)
