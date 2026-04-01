"""High-confidence ML signal extraction from project text."""

from __future__ import annotations

import re
from typing import Any

_PROJECT_ML_FRAMEWORK_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TensorFlow", ("tensorflow", "tensor flow")),
    ("PyTorch", ("pytorch", "py torch")),
    ("scikit-learn", ("scikit-learn", "scikit learn", "sklearn")),
    ("NumPy", ("numpy",)),
    ("Pandas", ("pandas",)),
)

_PROJECT_ML_CONCEPT_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anomaly detection", ("anomaly detection",)),
    ("time-series forecasting", ("time-series forecasting", "time series forecasting")),
    ("machine learning models", ("machine learning models", "ml models", "ml model")),
    ("machine learning", ("machine learning", "machine-learning")),
    ("deep learning", ("deep learning", "deep-learning", "neural network", "neural networks")),
)


def _project_text(resume_json: dict[str, Any]) -> str:
    parts: list[str] = []
    for proj in resume_json.get("projects", []):
        parts.append(str(proj.get("name", "")))
        for bullet in proj.get("bullets", []):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            parts.append(str(text))
    return " ".join(parts).lower()


def _contains_alias(text: str, alias: str) -> bool:
    if alias in text:
        return True
    compact_text = re.sub(r"[^a-z0-9]+", "", text)
    compact_alias = re.sub(r"[^a-z0-9]+", "", alias.lower())
    return bool(compact_alias) and compact_alias in compact_text


def _matched_signals(
    text: str,
    alias_groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> list[str]:
    matched: list[str] = []
    for label, aliases in alias_groups:
        if any(_contains_alias(text, alias) for alias in aliases):
            matched.append(label)
    return matched


def extract_high_confidence_project_ml_signals(resume_json: dict[str, Any]) -> dict[str, list[str]]:
    """Return project-backed ML frameworks and concepts with strict evidence gating."""
    text = _project_text(resume_json)
    frameworks = _matched_signals(text, _PROJECT_ML_FRAMEWORK_ALIASES)
    concepts = _matched_signals(text, _PROJECT_ML_CONCEPT_ALIASES)
    return {
        "frameworks": frameworks,
        "concepts": concepts,
        "all": frameworks + concepts,
    }
