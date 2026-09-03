from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MIN_F1_FOR_TRANSLATION = 0.20
STRONG_F1 = 0.50


@dataclass(frozen=True, slots=True)
class LabelQuality:
    label: str
    f1_score: float | None
    precision: float | None
    recall: float | None
    support: float | None
    status: str

    @property
    def usable(self) -> bool:
        return self.status in {"usable", "strong", "unknown"}


@dataclass(frozen=True, slots=True)
class ModelQualityProfile:
    path: Path | None
    minimum_f1_for_translation: float
    labels: dict[str, LabelQuality]

    @property
    def loaded(self) -> bool:
        return self.path is not None

    def for_label(self, label: str) -> LabelQuality:
        if not self.loaded:
            return LabelQuality(label, None, None, None, None, "unknown")
        return self.labels.get(label, LabelQuality(label, None, None, None, None, "missing"))


def quality_status(f1_score: float | None, *, minimum_f1: float = DEFAULT_MIN_F1_FOR_TRANSLATION) -> str:
    """quality status."""
    if f1_score is None:
        return "missing"
    if f1_score >= STRONG_F1:
        return "strong"
    if f1_score >= minimum_f1:
        return "usable"
    return "weak"


def load_model_quality_profile(path: str | Path | None) -> ModelQualityProfile:
    """load model quality profile."""
    if path is None:
        return ModelQualityProfile(None, DEFAULT_MIN_F1_FOR_TRANSLATION, {})

    profile_path = Path(path)
    if not profile_path.exists():
        return ModelQualityProfile(None, DEFAULT_MIN_F1_FOR_TRANSLATION, {})

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    minimum_f1 = float(data.get("minimum_f1_for_translation", DEFAULT_MIN_F1_FOR_TRANSLATION))
    raw_labels = data.get("labels", {})
    labels: dict[str, LabelQuality] = {}

    if isinstance(raw_labels, dict):
        for label, raw_value in raw_labels.items():
            if not isinstance(raw_value, dict):
                continue
            f1_score = _optional_float(raw_value.get("f1_score"))
            labels[str(label)] = LabelQuality(
                label=str(label),
                f1_score=f1_score,
                precision=_optional_float(raw_value.get("precision")),
                recall=_optional_float(raw_value.get("recall")),
                support=_optional_float(raw_value.get("support")),
                status=str(raw_value.get("status") or quality_status(f1_score, minimum_f1=minimum_f1)),
            )

    return ModelQualityProfile(profile_path, minimum_f1, labels)


    """build quality profile from metrics."""
def build_quality_profile_from_metrics(
    metrics: dict[str, Any],
    *,
    minimum_f1_for_translation: float = DEFAULT_MIN_F1_FOR_TRANSLATION,
) -> dict[str, Any]:
    report = metrics.get("classification_report", {})
    labels = metrics.get("labels", [])
    profile_labels: dict[str, dict[str, float | str | None]] = {}

    for label in labels:
        raw_value = report.get(label, {}) if isinstance(report, dict) else {}
        if not isinstance(raw_value, dict):
            raw_value = {}
        f1_score = _optional_float(raw_value.get("f1-score"))
        profile_labels[str(label)] = {
            "precision": _optional_float(raw_value.get("precision")),
            "recall": _optional_float(raw_value.get("recall")),
            "f1_score": f1_score,
            "support": _optional_float(raw_value.get("support")),
            "status": quality_status(f1_score, minimum_f1=minimum_f1_for_translation),
        }

    return {
        "source": "training_metrics",
        "minimum_f1_for_translation": minimum_f1_for_translation,
        "labels": profile_labels,
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None