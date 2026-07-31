"""Core prediction and presentation logic for StrokeGuard."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
import warnings

import numpy as np
import pandas as pd


APP_NAME = "StrokeGuard"
APP_SUBTITLE = "Clinical Decision Support System"
CREDIT_NAME = "Estiuk Arafat Arnob"
REFERENCE_AGE = 39

FEATURE_NAMES = (
    "Age",
    "Gender",
    "Chest_Pain",
    "Hypertension",
    "Irregular_Heartbeat",
    "Shortness_of_Breath",
    "Fatigue_Weakness",
    "Dizziness",
    "Edema",
    "Neck_Jaw_Pain",
    "Excessive_Sweating",
    "Persistent_Cough",
    "Nausea_Vomiting",
    "Chest_Discomfort",
    "Cold_Hands_Feet",
    "Sleep_Apnea",
    "Anxiety",
)

SYMPTOM_FIELDS = (
    ("Chest_Pain", "Chest pain (বুকের ব্যথা)"),
    ("Hypertension", "Hypertension (উচ্চ রক্তচাপ)"),
    ("Irregular_Heartbeat", "Irregular heartbeat (অনিয়মিত হৃদস্পন্দন)"),
    ("Shortness_of_Breath", "Shortness of breath (শ্বাসকষ্ট)"),
    ("Fatigue_Weakness", "Fatigue or weakness (ক্লান্তি বা দুর্বলতা)"),
    ("Dizziness", "Dizziness (মাথা ঘোরা)"),
    ("Edema", "Edema or swelling (শরীরে পানি আসা বা ফোলা)"),
    ("Neck_Jaw_Pain", "Neck or jaw pain (ঘাড় বা চোয়ালের ব্যথা)"),
    ("Excessive_Sweating", "Excessive sweating (অতিরিক্ত ঘাম)"),
    ("Persistent_Cough", "Persistent cough (অবিরাম কাশি)"),
    ("Nausea_Vomiting", "Nausea or vomiting (বমি ভাব বা বমি)"),
    ("Chest_Discomfort", "Chest discomfort (বুকে অস্বস্তি)"),
    ("Cold_Hands_Feet", "Cold hands or feet (হাত-পা ঠাণ্ডা হয়ে আসা)"),
    ("Sleep_Apnea", "Sleep apnea (ঘুমের মধ্যে শ্বাসকষ্ট)"),
    ("Anxiety", "Anxiety (উদ্বেগ)"),
)

SYMPTOM_LABELS = dict(SYMPTOM_FIELDS)

SYMPTOM_GROUPS = {
    "Cardiovascular": (
        "Chest_Pain",
        "Hypertension",
        "Irregular_Heartbeat",
        "Shortness_of_Breath",
        "Chest_Discomfort",
    ),
    "General": (
        "Fatigue_Weakness",
        "Dizziness",
        "Edema",
        "Neck_Jaw_Pain",
        "Excessive_Sweating",
    ),
    "Other indicators": (
        "Persistent_Cough",
        "Nausea_Vomiting",
        "Cold_Hands_Feet",
        "Sleep_Apnea",
        "Anxiety",
    ),
}

DEFAULT_INPUTS: dict[str, Any] = {
    "Age": 39,
    "Gender": "Female",
    **{name: False for name, _ in SYMPTOM_FIELDS},
}

DEMO_INPUTS: dict[str, Any] = {
    "Age": 43,
    "Gender": "Female",
    **{name: False for name, _ in SYMPTOM_FIELDS},
    "Chest_Pain": True,
    "Fatigue_Weakness": True,
    "Anxiety": True,
}

@dataclass(frozen=True)
class RiskBand:
    label: str
    short_label: str
    color: str
    soft_color: str
    description: str


@dataclass(frozen=True)
class SensitivityItem:
    feature: str
    effect_points: float
    comparison: str


def get_risk_band(score: float) -> RiskBand:
    """Map an estimated score to the application's display bands."""
    if score < 30:
        return RiskBand(
            "Lower indicator range",
            "LOWER",
            "#18a978",
            "rgba(24, 169, 120, 0.12)",
            "The information entered produced a lower symptom-based indicator.",
        )
    if score < 60:
        return RiskBand(
            "Moderate indicator range",
            "MODERATE",
            "#d99516",
            "rgba(217, 149, 22, 0.12)",
            "The information entered produced a moderate indicator worth discussing with a doctor.",
        )
    if score < 80:
        return RiskBand(
            "Higher indicator range",
            "HIGHER",
            "#ea6a33",
            "rgba(234, 106, 51, 0.12)",
            "The selected indicators produced an elevated symptom pattern.",
        )
    return RiskBand(
        "Very high indicator range",
        "VERY HIGH",
        "#e3485b",
        "rgba(227, 72, 91, 0.12)",
        "The selected indicators produced a very high symptom-based estimate. Prompt professional review is important.",
    )


def encode_inputs(values: dict[str, Any]) -> pd.DataFrame:
    """Encode UI values in the exact feature order used during training."""
    encoded = [
        int(values["Age"]),
        1 if values["Gender"] == "Male" else 0,
    ]
    encoded.extend(int(bool(values[name])) for name, _ in SYMPTOM_FIELDS)
    return pd.DataFrame(
        np.asarray([encoded], dtype=float),
        columns=FEATURE_NAMES,
    )


def estimate_probability(engine: Any, scaler: Any, sample: pd.DataFrame) -> float:
    """Return positive-class probability for one encoded sample."""
    scaled = scaler.transform(sample)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier.*",
            category=UserWarning,
        )
        return float(engine.predict_proba(scaled)[0, 1])


def compute_local_sensitivity(
    engine: Any,
    scaler: Any,
    values: dict[str, Any],
    current_probability: float,
) -> list[SensitivityItem]:
    """
    Estimate patient-specific indicator sensitivity with controlled perturbations.

    This is deliberately described as sensitivity rather than SHAP or causal
    attribution. Each active symptom is switched off once, while age is
    compared with the training-set reference age.
    """
    items: list[SensitivityItem] = []

    age = int(values["Age"])
    if age != REFERENCE_AGE:
        comparison_values = dict(values)
        comparison_values["Age"] = REFERENCE_AGE
        comparison_probability = estimate_probability(
            engine, scaler, encode_inputs(comparison_values)
        )
        items.append(
            SensitivityItem(
                "Age",
                (current_probability - comparison_probability) * 100,
                f"Compared with reference age {REFERENCE_AGE}",
            )
        )

    for name, label in SYMPTOM_FIELDS:
        if not bool(values[name]):
            continue
        comparison_values = dict(values)
        comparison_values[name] = False
        comparison_probability = estimate_probability(
            engine, scaler, encode_inputs(comparison_values)
        )
        items.append(
            SensitivityItem(
                label,
                (current_probability - comparison_probability) * 100,
                f"Compared with {label.lower()} set to absent",
            )
        )

    return sorted(items, key=lambda item: abs(item.effect_points), reverse=True)


def active_symptoms(values: dict[str, Any]) -> list[str]:
    """Return display labels for selected symptoms."""
    return [label for name, label in SYMPTOM_FIELDS if bool(values[name])]


def build_guidance(values: dict[str, Any], score: float) -> list[dict[str, str]]:
    """Build cautious, input-aware health discussion prompts."""
    guidance: list[dict[str, str]] = []

    if score >= 60:
        guidance.append(
            {
                "title": "Arrange a professional review",
                "text": (
                    "Consider discussing this assessment and your actual medical "
                    "history with a qualified doctor. This estimate should not be "
                    "used on its own for any medical decision."
                ),
                "tone": "priority",
            }
        )
    if values["Hypertension"]:
        guidance.append(
            {
                "title": "Review blood-pressure control",
                "text": (
                    "Keep an accurate blood-pressure record and review persistent "
                    "or uncontrolled readings with a healthcare professional."
                ),
                "tone": "standard",
            }
        )
    if values["Irregular_Heartbeat"]:
        guidance.append(
            {
                "title": "Discuss the irregular heartbeat",
                "text": (
                    "A clinician can decide whether pulse assessment or an ECG is "
                    "appropriate based on symptoms and medical history."
                ),
                "tone": "standard",
            }
        )
    if (
        values["Chest_Pain"]
        or values["Chest_Discomfort"]
        or values["Shortness_of_Breath"]
    ):
        guidance.append(
            {
                "title": "Do not ignore new or severe symptoms",
                "text": (
                    "New, severe, persistent, or worsening chest or breathing "
                    "symptoms require prompt professional assessment."
                ),
                "tone": "priority",
            }
        )
    if values["Sleep_Apnea"]:
        guidance.append(
            {
                "title": "Review sleep quality",
                "text": (
                    "Discuss loud snoring, breathing pauses, or daytime sleepiness "
                    "with a clinician who can determine whether sleep evaluation is needed."
                ),
                "tone": "standard",
            }
        )
    if values["Anxiety"]:
        guidance.append(
            {
                "title": "Separate anxiety from physical warning signs",
                "text": (
                    "Anxiety can overlap with physical symptoms, but sudden or severe "
                    "neurological symptoms should still receive urgent medical attention."
                ),
                "tone": "standard",
            }
        )

    guidance.extend(
        [
            {
                "title": "Know the BE FAST warning signs",
                "text": (
                    "Sudden balance loss, vision change, facial droop, arm weakness, "
                    "or speech difficulty requires emergency action."
                ),
                "tone": "emergency",
            },
            {
                "title": "Support long-term vascular health",
                "text": (
                    "Follow clinician-approved plans for activity, sleep, diet, "
                    "smoking avoidance, and management of blood pressure, glucose, and lipids."
                ),
                "tone": "standard",
            },
        ]
    )
    return guidance


def build_clinical_discussions(
    values: dict[str, Any], score: float
) -> list[dict[str, str]]:
    """Return tests or topics that may be discussed with a clinician."""
    discussions: list[dict[str, str]] = []

    if values["Hypertension"]:
        discussions.append(
            {
                "name": "Blood-pressure review",
                "reason": "Confirm readings and assess control over time.",
            }
        )
    if values["Irregular_Heartbeat"]:
        discussions.append(
            {
                "name": "Pulse assessment or ECG",
                "reason": "A clinician can evaluate rhythm-related concerns.",
            }
        )
    if values["Sleep_Apnea"]:
        discussions.append(
            {
                "name": "Sleep assessment",
                "reason": "Consider only if symptoms and clinical history support it.",
            }
        )
    if values["Chest_Pain"] or values["Chest_Discomfort"]:
        discussions.append(
            {
                "name": "Cardiovascular assessment",
                "reason": "The appropriate evaluation depends on onset, severity, and history.",
            }
        )
    if score >= 30 or int(values["Age"]) >= 50:
        discussions.extend(
            [
                {
                    "name": "Glucose review",
                    "reason": "Discuss screening frequency based on individual risk.",
                },
                {
                    "name": "Lipid profile",
                    "reason": "Review cholesterol and broader vascular risk when appropriate.",
                },
            ]
        )
    if score >= 60 or values["Dizziness"] or values["Fatigue_Weakness"]:
        discussions.append(
            {
                "name": "Focused clinical examination",
                "reason": "A professional examination provides essential context beyond the entered information.",
            }
        )

    if not discussions:
        discussions.append(
            {
                "name": "Routine preventive review",
                "reason": "Use age, history, and clinician guidance to decide screening needs.",
            }
        )

    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in discussions:
        if item["name"] in seen:
            continue
        seen.add(item["name"])
        unique.append(item)
    return unique


def run_assessment(
    engine: Any,
    scaler: Any,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Run prediction and return a serializable assessment result."""
    sample = encode_inputs(values)
    probability = estimate_probability(engine, scaler, sample)
    score = probability * 100
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier.*",
            category=UserWarning,
        )
        prediction = int(engine.predict(scaler.transform(sample))[0])
    band = get_risk_band(score)
    sensitivity = compute_local_sensitivity(
        engine, scaler, values, current_probability=probability
    )

    return {
        "score": score,
        "probability": probability,
        "prediction": prediction,
        "binary_label": "Elevated pattern identified" if prediction == 1 else "Lower pattern identified",
        "decision_confidence": max(probability, 1 - probability) * 100,
        "band": asdict(band),
        "active_symptoms": active_symptoms(values),
        "sensitivity": [asdict(item) for item in sensitivity],
        "guidance": build_guidance(values, score),
        "clinical_discussions": build_clinical_discussions(values, score),
        "assessed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": dict(values),
    }
