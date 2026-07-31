"""End-to-end local smoke test for StrokeGuard."""

from __future__ import annotations

from pathlib import Path

import joblib

from app_core import (
    CREDIT_NAME,
    DEFAULT_INPUTS,
    DEMO_INPUTS,
    FEATURE_NAMES,
    run_assessment,
)
from pdf_report import generate_pdf_report


BASE_DIR = Path(__file__).resolve().parent


def main() -> int:
    engine = joblib.load(BASE_DIR / "strokeguard_engine.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")

    assert getattr(engine, "n_features_in_", None) == len(FEATURE_NAMES)
    assert getattr(scaler, "n_features_in_", None) == len(FEATURE_NAMES)

    default_result = run_assessment(engine, scaler, DEFAULT_INPUTS)
    demo_result = run_assessment(engine, scaler, DEMO_INPUTS)

    for result in (default_result, demo_result):
        assert 0 <= result["score"] <= 100
        assert result["prediction"] in (0, 1)
        assert result["band"]["label"]
        assert result["guidance"]

    assert demo_result["sensitivity"]
    pdf_bytes = generate_pdf_report(demo_result)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 4_000

    print("StrokeGuard smoke test: PASS")
    print(f"Product credit: {CREDIT_NAME}")
    print(f"Default score: {default_result['score']:.2f}%")
    print(f"Demo score: {demo_result['score']:.2f}%")
    print(f"PDF bytes: {len(pdf_bytes):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
