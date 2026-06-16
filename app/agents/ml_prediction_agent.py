"""
ML Research Prediction Agent for Kaggle KTAS phase.

Loads trained KTAS research models from data/models/registry.json and returns:
  - predicted KTAS_expert class estimate (1-5)
  - emergency probability estimate where KTAS 1-3 = emergency

These are not Manchester triage labels and are not clinical decisions.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.config import settings
from app.schemas.internal import TriageTimeInput
from app.schemas.workflow import MLPredictionResult
from ml_training.feature_engineering import FEATURE_NAMES, extract_features_from_row


def _load_registry() -> Optional[dict]:
    path = settings.model_registry_path
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _resolve_model_path(raw_path: str) -> Path:
    p = Path(raw_path)
    return p if p.is_absolute() else settings.models_dir / p


def _triage_input_to_row(t: TriageTimeInput) -> dict:
    return {
        "age": t.age,
        "gender": t.gender,
        "group_code": t.group_code,
        "patients_per_hour": t.patients_per_hour,
        "arrival_transport": t.arrival_transport,
        "arrival_mode_code": t.arrival_mode_code,
        "injury_code": t.injury_code,
        "mental_code": t.mental_code,
        "chiefcomplaint": t.chiefcomplaint,
        "temperature": t.temperature,
        "temperature_unit": t.temperature_unit,
        "heartrate": t.heartrate,
        "resprate": t.resprate,
        "o2sat": t.o2sat,
        "sbp": t.sbp,
        "dbp": t.dbp,
        "pain": t.pain,
        "pain_present": t.pain_present,
        "nrs_pain": t.nrs_pain,
    }


def _predict_proba_safe(model, X):
    try:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
    except Exception:
        return None
    return None


def _class_probability_dict(model, probabilities) -> dict[str, float]:
    if probabilities is None:
        return {}
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = list(range(probabilities.shape[1]))
    return {str(int(cls)): float(probabilities[0, i]) for i, cls in enumerate(classes)}


def _prob_for_class(model, probabilities, wanted_class: int) -> Optional[float]:
    if probabilities is None:
        return None
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    for i, cls in enumerate(classes):
        if int(cls) == int(wanted_class):
            return float(probabilities[0, i])
    return None


def run_ml_prediction(triage_input: TriageTimeInput) -> MLPredictionResult:
    registry = _load_registry()
    if not registry:
        return MLPredictionResult(
            model_version="not_loaded",
            model_name="no_registry",
            prediction_available=False,
            model_note=(
                "No trained KTAS model registry found. Run: python scripts/run_ktas_pipeline.py. "
                "No ML estimate is being shown."
            ),
        )

    ktas_info = registry.get("best_ktas_model") or registry.get("best_model")
    if not ktas_info:
        return MLPredictionResult(model_name="no_ktas_model", prediction_available=False)

    ktas_path = _resolve_model_path(ktas_info.get("path", ""))
    if not ktas_path.exists():
        return MLPredictionResult(
            model_version=ktas_info.get("version", "unknown"),
            model_name=ktas_info.get("name", "unknown"),
            prediction_available=False,
            model_note=f"KTAS model file not found: {ktas_path}",
        )

    try:
        row = _triage_input_to_row(triage_input)
        features = extract_features_from_row(row)
        X = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)

        ktas_model = joblib.load(ktas_path)
        predicted_ktas = int(ktas_model.predict(X)[0])
        ktas_proba = _predict_proba_safe(ktas_model, X)
        ktas_prob_dict = _class_probability_dict(ktas_model, ktas_proba)
        top_conf = max(ktas_prob_dict.values()) if ktas_prob_dict else None

        emergency_probability = None
        emergency_info = registry.get("best_emergency_model")
        if emergency_info:
            em_path = _resolve_model_path(emergency_info.get("path", ""))
            if em_path.exists():
                em_model = joblib.load(em_path)
                em_proba = _predict_proba_safe(em_model, X)
                emergency_probability = _prob_for_class(em_model, em_proba, 1)

        if emergency_probability is None and ktas_prob_dict:
            emergency_probability = sum(
                prob for cls, prob in ktas_prob_dict.items() if int(cls) <= 3
            )

        return MLPredictionResult(
            model_version=ktas_info.get("version", registry.get("version", "unknown")),
            model_name=ktas_info.get("name", "unknown"),
            prediction_available=True,
            predicted_ktas_class=predicted_ktas,
            ktas_class_probabilities=ktas_prob_dict,
            emergency_research_estimate=emergency_probability,
            non_emergency_research_estimate=(None if emergency_probability is None else 1.0 - emergency_probability),
            high_acuity_research_estimate=emergency_probability,
            admission_risk_estimate=None,
            top_class_confidence=top_conf,
            model_note=(
                "KTAS research estimate only. It predicts KTAS_expert from public Kaggle data. "
                "KTAS is not Manchester triage. No clinical action may be taken from this output."
            ),
        )
    except Exception as exc:
        return MLPredictionResult(
            model_version=ktas_info.get("version", "unknown"),
            model_name=ktas_info.get("name", "unknown"),
            prediction_available=False,
            model_note=f"ML prediction failed safely: {type(exc).__name__}: {exc}",
        )
