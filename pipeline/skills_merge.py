"""Deterministic skills enrichment: grounded JD keywords only (evidence in non-skills text)."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from pipeline.project_signals import extract_high_confidence_project_ml_signals
from pipeline.scorer import _alnum_compact, jd_keyword_matches_resume

_PRIORITY_ML_SKILL_TERMS = {
    "machine learning",
    "deep learning",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "statistics",
    "algorithms",
    "model optimization",
    "machine learning models",
    "anomaly detection",
    "time-series forecasting",
    "numpy",
    "pandas",
}
_SKILL_CANONICAL_CASE = {
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "machine learning models": "Machine Learning Models",
    "anomaly detection": "Anomaly Detection",
    "time-series forecasting": "Time-Series Forecasting",
    "time series forecasting": "Time-Series Forecasting",
    "model optimization": "Model Optimization",
    "statistics": "Statistics",
    "algorithms": "Algorithms",
}
_LANGUAGE_MARKERS = {
    "python", "java", "c++", "c#", "golang", "go", "rust", "ruby", "php",
    "sql", "javascript", "typescript", "kotlin", "swift", "scala", "html", "css",
}
_ML_MARKERS = {
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "sklearn",
    "numpy", "pandas", "statistics", "algorithms", "model optimization", "machine learning models",
    "anomaly detection", "time-series forecasting",
}
_FRAMEWORK_MARKERS = {
    "fastapi", "flask", "django", "react", "grpc", "graphql", "spark", "kafka",
    "airflow", "docker", "kubernetes", "langchain", "huggingface", "faiss",
    "microservices", "rest", "restful", "api", "apis", "postgres", "mysql",
}
_CLOUD_MARKERS = {
    "aws", "azure", "gcp", "s3", "terraform", "kubernetes", "docker", "cloud",
    "devops", "cicd", "ci/cd",
}


def evidence_text_without_skills(resume: dict[str, Any]) -> str:
    """Experience, education, projects, summary — excludes skills section values."""
    parts: list[str] = []
    parts.append(str(resume.get("summary", "")))
    parts.append(str(resume.get("name", "")))
    for exp in resume.get("experience", []):
        parts.append(str(exp.get("role", "")))
        parts.append(str(exp.get("company", "")))
        for bullet in exp.get("bullets", []):
            t = bullet if isinstance(bullet, str) else bullet.get("text", "")
            parts.append(t)
    for edu in resume.get("education", []):
        parts.append(str(edu.get("degree", "")))
        parts.append(str(edu.get("institution", "")))
        parts.append(str(edu.get("dates", "")))
    for proj in resume.get("projects", []):
        parts.append(str(proj.get("name", "")))
        for bullet in proj.get("bullets", []):
            t = bullet if isinstance(bullet, str) else bullet.get("text", "")
            parts.append(t)
    return " ".join(parts)


def skills_blob_text(resume: dict[str, Any]) -> str:
    parts: list[str] = []
    for cat, values in resume.get("skills", {}).items():
        parts.append(cat)
        if isinstance(values, dict):
            parts.extend(values.get("items", []))
        elif isinstance(values, list):
            parts.extend(values)
    return " ".join(parts).lower()


def _normalize_skills_dict(skills: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cat, vals in skills.items():
        if isinstance(vals, dict) and "items" in vals:
            out[cat] = list(vals["items"])
        elif isinstance(vals, list):
            out[cat] = list(vals)
    return out


def _canonicalize_skill_item(item: str) -> str:
    raw = re.sub(r"\s+", " ", item.strip()).rstrip(".,;:")
    if not raw:
        return raw
    key = raw.lower()
    return _SKILL_CANONICAL_CASE.get(key, raw)


def _dedupe_preserve(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        norm = item.lower().strip()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item)
    return out


def _is_skill_safe_keyword(kw: str) -> bool:
    """Allow only real technologies / compact ML concepts into skills."""
    kl = kw.lower().strip()
    if not kl or len(kl) > 48:
        return False
    if kl in _PRIORITY_ML_SKILL_TERMS:
        return True
    words = set(re.findall(r"\b[a-z][a-z0-9.+#/-]*\b", kl))
    if any(w in _LANGUAGE_MARKERS for w in words):
        return True
    if any(w in _FRAMEWORK_MARKERS for w in words):
        return True
    if any(w in _CLOUD_MARKERS for w in words):
        return True
    # Reject long JD-style action phrases in skills.
    if len(kl.split()) > 4:
        return False
    return False


def _bucket_for_skill(item: str) -> str:
    kl = item.lower()
    if kl in {"c/c++", "c++", "c#"}:
        return "Languages"
    words = set(re.findall(r"\b[a-z][a-z0-9.+#/-]*\b", kl))
    if kl in _PRIORITY_ML_SKILL_TERMS or any(m in kl for m in _ML_MARKERS):
        return "Machine Learning"
    if any(w in _LANGUAGE_MARKERS for w in words):
        return "Languages"
    if any(w in _CLOUD_MARKERS for w in words):
        return "Cloud / DevOps"
    if any(w in _FRAMEWORK_MARKERS for w in words):
        return "Frameworks / Platforms"
    return "Frameworks / Platforms"


def _target_labels_for_count(count: int) -> list[str]:
    if count <= 1:
        return ["Skills"]
    if count == 2:
        return ["Machine Learning", "Core Technologies"]
    if count == 3:
        return ["Machine Learning", "Languages", "Frameworks / Platforms"]
    return ["Machine Learning", "Languages", "Frameworks / Platforms", "Cloud / DevOps"][:count]


def _project_skill_promotions(resume_json: dict[str, Any]) -> list[str]:
    """
    Promote strong project signals into skills:
    - frameworks/tools always
    - compact ML concepts to avoid a weak section with only NumPy/Pandas
    """
    signals = extract_high_confidence_project_ml_signals(resume_json)
    ordered: list[str] = []
    for item in signals.get("frameworks", []):
        ordered.append(_canonicalize_skill_item(item))
    for item in signals.get("concepts", [])[:3]:
        canon = _canonicalize_skill_item(item)
        if canon not in ordered:
            ordered.append(canon)
    return ordered


def _build_grouped_skill_slots(
    slot_count: int,
    items: list[str],
    *,
    preferred_ml_items: list[str] | None = None,
) -> list[tuple[str, list[str]]]:
    """
    Build recruiter-friendly grouped skill lines while preserving the number of lines
    already available in the resume.
    """
    cleaned = _dedupe_preserve([_canonicalize_skill_item(x) for x in items if x.strip()])
    if not cleaned:
        return []

    buckets: dict[str, list[str]] = {
        "Machine Learning": [],
        "Languages": [],
        "Frameworks / Platforms": [],
        "Cloud / DevOps": [],
    }
    for item in cleaned:
        buckets[_bucket_for_skill(item)].append(item)

    preferred_ml = [_canonicalize_skill_item(item) for item in (preferred_ml_items or [])]
    if preferred_ml:
        prioritized_ml = [item for item in preferred_ml if item in buckets["Machine Learning"]]
        buckets["Machine Learning"] = _dedupe_preserve(
            prioritized_ml
            + [item for item in buckets["Machine Learning"] if item not in prioritized_ml]
        )

    # Avoid a weak ML line that contains only support libraries like NumPy/Pandas.
    if buckets["Machine Learning"] and all(
        item in {"NumPy", "Pandas"} for item in buckets["Machine Learning"]
    ):
        buckets["Frameworks / Platforms"] = _dedupe_preserve(
            buckets["Machine Learning"] + buckets["Frameworks / Platforms"]
        )
        buckets["Machine Learning"] = []

    ordered_labels = _target_labels_for_count(slot_count)
    if slot_count == 2:
        core = buckets["Languages"] + buckets["Frameworks / Platforms"] + buckets["Cloud / DevOps"]
        grouped = [
            ("Machine Learning", _dedupe_preserve(buckets["Machine Learning"])),
            ("Core Technologies", _dedupe_preserve(core)),
        ]
    elif slot_count == 3:
        grouped = [
            ("Machine Learning", _dedupe_preserve(buckets["Machine Learning"])),
            ("Languages", _dedupe_preserve(buckets["Languages"])),
            (
                "Frameworks / Platforms",
                _dedupe_preserve(
                    buckets["Frameworks / Platforms"] + buckets["Cloud / DevOps"]
                ),
            ),
        ]
    else:
        grouped = [(label, _dedupe_preserve(buckets.get(label, []))) for label in ordered_labels]

    non_empty = [(label, vals) for label, vals in grouped if vals]
    if not non_empty:
        return [("Skills", cleaned[:])]

    # If we have fewer non-empty groups than slots, split the largest bucket conservatively.
    while len(non_empty) < slot_count:
        split_idx = max(range(len(non_empty)), key=lambda i: len(non_empty[i][1]))
        label, vals = non_empty[split_idx]
        if len(vals) <= 1:
            break
        moved = vals[-1]
        non_empty[split_idx] = (label, vals[:-1])
        filler_label = ordered_labels[min(len(non_empty), len(ordered_labels) - 1)]
        non_empty.append((filler_label, [moved]))

    if len(non_empty) > slot_count:
        head = non_empty[: slot_count - 1]
        tail_items: list[str] = []
        for _label, vals in non_empty[slot_count - 1 :]:
            tail_items.extend(vals)
        head.append((non_empty[slot_count - 1][0], _dedupe_preserve(tail_items)))
        non_empty = head

    return non_empty


def skills_display_lines(
    skills: dict[str, Any],
    doc_lines: list[str] | None = None,
) -> list[str]:
    """Category lines as they appear in the DOCX (when *doc_lines* provided)."""
    normalized = _normalize_skills_dict(skills)
    cats = list(normalized.keys())
    return [
        _original_line_for_category(cat, normalized[cat], i, doc_lines)
        for i, cat in enumerate(cats)
    ]


def _original_line_for_category(
    cat: str,
    items: list[str],
    cat_index: int,
    doc_lines: list[str] | None,
) -> str:
    if doc_lines:
        for line in doc_lines:
            if ":" in line:
                lbl = line.split(":", 1)[0].strip().lower()
                if lbl == cat.lower():
                    return line
        if cat_index < len(doc_lines):
            return doc_lines[cat_index]
    return f"{cat}: {', '.join(items)}"


def _pick_category_index(kw: str, categories: list[str]) -> int:
    kl = kw.lower()
    for i, cat in enumerate(categories):
        if "language" in cat.lower():
            lang_markers = (
                "python", "java", "c++", "c#", "golang", "rust", "ruby", "php",
                "sql", "javascript", "typescript", "kotlin", "swift", "scala", "html", "css",
            )
            if any(m in kl for m in lang_markers):
                return i
    fw_markers = (
        "torch", "tensor", "keras", "scikit", "sklearn", "spark", "kubernetes",
        "docker", "react", "django", "flask", "fastapi", "langchain", "hugging",
        "numpy", "pandas", "grpc", "graphql", "kafka", "airflow",
    )
    if any(m in kl for m in fw_markers):
        for i, cat in enumerate(categories):
            cl = cat.lower()
            if any(x in cl for x in ("developer", "debug", "framework", "technolog", "tool")):
                return i
    # default: category with most items (spread load)
    return 0


def deterministic_skills_append(
    resume_json: dict[str, Any],
    missing_keywords: list[str],
    doc_skill_lines: list[str] | None = None,
    max_append: int = 36,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Append missing JD terms to skills **only** when they are safe, compact skill-like
    terms already supported by experience/education/projects/summary. Then regroup the
    skills section into cleaner ATS-friendly labels using the same number of lines.
    """
    rewrites: list[dict[str, Any]] = []
    skills = resume_json.get("skills") or {}
    normalized = _normalize_skills_dict(skills)
    if not normalized:
        return resume_json, rewrites

    ev_lower = evidence_text_without_skills(resume_json).lower()
    ev_compact = _alnum_compact(ev_lower)

    cats = list(normalized.keys())
    working: dict[str, list[str]] = {c: list(normalized[c]) for c in cats}
    project_promotions = _project_skill_promotions(resume_json)

    appended = 0
    for kw in missing_keywords:
        if appended >= max_append:
            break
        if not _is_skill_safe_keyword(kw):
            continue
        if len(kw) > 48 or len(kw.split()) > 10:
            continue
        if jd_keyword_matches_resume(kw, resume_json):
            continue
        core = _alnum_compact(kw)
        if len(core) < 3:
            continue
        if core not in ev_compact:
            continue

        sk_now = skills_blob_text({"skills": {c: working[c] for c in cats}})
        if kw.lower() in sk_now:
            continue

        idx = _pick_category_index(kw, cats)
        cat = cats[idx]
        items = working[cat]
        if any(kw.lower() == x.lower() for x in items):
            continue
        items.append(_canonicalize_skill_item(kw))
        working[cat] = items
        appended += 1

    all_items: list[str] = []
    for cat in cats:
        all_items.extend(working[cat])
    for item in project_promotions:
        if item.lower() not in {existing.lower() for existing in all_items}:
            all_items.append(item)
    grouped = _build_grouped_skill_slots(
        len(cats),
        all_items,
        preferred_ml_items=project_promotions,
    )
    if not grouped:
        return resume_json, rewrites

    for i, cat in enumerate(cats):
        old_items = normalized[cat]
        if i < len(grouped):
            new_label, new_items = grouped[i]
        else:
            new_label, new_items = cat, old_items
        new_items = _dedupe_preserve(new_items)
        if new_label == cat and new_items == old_items:
            continue
        orig_line = _original_line_for_category(cat, old_items, i, doc_skill_lines)
        new_line = f"{new_label}: {', '.join(new_items)}"
        if orig_line.strip() == new_line.strip():
            continue
        added = [k for k in new_items if k not in old_items]
        rewrites.append({
            "original": orig_line,
            "rewritten": new_line,
            "injected_keywords": added,
        })

    if not rewrites:
        return resume_json, rewrites

    rebuilt_skills: dict[str, Any] = {}
    for i, cat in enumerate(cats):
        if i < len(grouped):
            new_label, new_items = grouped[i]
        else:
            new_label, new_items = cat, normalized[cat]
        if new_label in rebuilt_skills:
            rebuilt_skills[new_label] = _dedupe_preserve(rebuilt_skills[new_label] + new_items)
        else:
            rebuilt_skills[new_label] = _dedupe_preserve(new_items)

    out_resume = deepcopy(resume_json)
    out_resume["skills"] = rebuilt_skills
    return out_resume, rewrites
