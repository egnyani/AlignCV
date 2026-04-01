"""Narrative planning and summary generation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, REWRITE_MODEL
from pipeline.project_signals import extract_high_confidence_project_ml_signals


def _openai_key_available() -> bool:
    return bool((os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY or "").strip())


@dataclass
class BulletPlan:
    exp_idx: int
    bullet_idx: int
    original: str
    action: str
    emphasis: str
    rephraseable_kws: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class NarrativePlan:
    engineering_identity: str
    resume_arc: str
    bullet_plans: list[BulletPlan] = field(default_factory=list)
    uncoverable: list[str] = field(default_factory=list)


def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
    )


def _parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _resume_text(resume_json: dict[str, Any]) -> str:
    parts: list[str] = [resume_json.get("name", ""), resume_json.get("summary", "")]
    for exp in resume_json.get("experience", []):
        parts.append(exp.get("role", ""))
        parts.append(exp.get("company", ""))
        for bullet in exp.get("bullets", []):
            parts.append(bullet if isinstance(bullet, str) else bullet.get("text", ""))
    for cat, vals in resume_json.get("skills", {}).items():
        if isinstance(vals, list):
            parts.extend(vals)
    for proj in resume_json.get("projects", []):
        parts.append(proj.get("name", ""))
        for bullet in proj.get("bullets", []):
            parts.append(bullet if isinstance(bullet, str) else bullet.get("text", ""))
    return " ".join(parts)


def _quick_classify_verbatim(keywords: list[str], resume_text: str) -> list[str]:
    lower = resume_text.lower()
    return [kw for kw in keywords if kw.lower() in lower]


_PRIORITY_ML_TERMS = [
    "machine learning",
    "deep learning",
    "TensorFlow",
    "PyTorch",
    "scikit-learn",
    "statistics",
    "algorithms",
    "model optimization",
    "machine learning models",
]
_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "machine learning models": ("machine learning", "model", "models", "ml model", "ml models"),
    "model optimization": ("optimiz", "model", "performance"),
    "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "TensorFlow": ("tensorflow", "tensor flow"),
    "PyTorch": ("pytorch", "py torch"),
}


def _resume_supports_ml_term(resume_json: dict[str, Any], term: str) -> bool:
    text = _resume_text(resume_json).lower()
    if term.lower() in text:
        return True
    aliases = _TERM_ALIASES.get(term, ())
    return bool(aliases) and all(alias in text for alias in aliases)


def _supported_priority_ml_terms(resume_json: dict[str, Any], jd_text: str, limit: int = 4) -> list[str]:
    jd_lower = jd_text.lower()
    supported: list[str] = []
    for term in _PRIORITY_ML_TERMS:
        if term.lower() not in jd_lower:
            continue
        if _resume_supports_ml_term(resume_json, term):
            supported.append(term)
        if len(supported) >= limit:
            break
    return supported


def _prioritize_terms_for_jd(terms: list[str], jd_text: str, limit: int) -> list[str]:
    jd_lower = jd_text.lower()
    jd_hits = [term for term in terms if term.lower() in jd_lower]
    ordered = jd_hits + [term for term in terms if term not in jd_hits]
    return ordered[:limit]


def _summary_signal_guidance(resume_json: dict[str, Any], jd_text: str) -> dict[str, list[str]]:
    project_signals = extract_high_confidence_project_ml_signals(resume_json)
    preferred_frameworks = _prioritize_terms_for_jd(
        project_signals.get("frameworks", []),
        jd_text,
        limit=1,
    )
    preferred_concepts = _prioritize_terms_for_jd(
        project_signals.get("concepts", []),
        jd_text,
        limit=1,
    )
    fallback_terms = [
        term
        for term in _supported_priority_ml_terms(resume_json, jd_text, limit=4)
        if term not in preferred_frameworks and term not in preferred_concepts
    ][:2]
    return {
        "frameworks": preferred_frameworks,
        "concepts": preferred_concepts,
        "fallback_terms": fallback_terms,
        "all": preferred_frameworks + preferred_concepts + fallback_terms,
    }


def _heuristic_plan(
    resume_json: dict[str, Any],
    keywords: list[str],
    narrative_intent: dict[str, Any],
) -> NarrativePlan:
    resume_text = _resume_text(resume_json)
    verbatim = set(_quick_classify_verbatim(keywords, resume_text))
    uncoverable = [kw for kw in keywords if kw not in verbatim]

    bullet_plans: list[BulletPlan] = []
    for exp_idx, exp in enumerate(resume_json.get("experience", [])):
        for b_idx, bullet in enumerate(exp.get("bullets", [])):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            bullet_plans.append(
                BulletPlan(
                    exp_idx=exp_idx,
                    bullet_idx=b_idx,
                    original=text,
                    action="keep",
                    emphasis=narrative_intent.get("engineering_identity", "relevant engineering work"),
                    rephraseable_kws=[],
                    rationale="Heuristic mode — no rewrite attempted.",
                )
            )

    return NarrativePlan(
        engineering_identity=narrative_intent.get("engineering_identity", "software engineer"),
        resume_arc="Candidate with relevant technical experience.",
        bullet_plans=bullet_plans,
        uncoverable=uncoverable,
    )


def classify_and_plan(
    resume_json: dict[str, Any],
    keywords: list[str],
    narrative_intent: dict[str, Any],
    evidence_map: dict[str, Any] | None = None,
) -> NarrativePlan:
    if not _openai_key_available():
        return _heuristic_plan(resume_json, keywords, narrative_intent)

    resume_compact = _build_compact_resume(resume_json)
    kw_list = "\n".join(f"  - {kw}" for kw in keywords)
    themes = ", ".join(narrative_intent.get("dominant_themes", []))
    identity = narrative_intent.get("engineering_identity", "software engineer")
    arc_hint = narrative_intent.get("arc_description", "")
    evidence_lines: list[str] = []
    for item in (evidence_map or {}).get("keyword_support", []):
        keyword = item.get("keyword", "")
        support_level = item.get("support_level", "unsupported")
        evidence = item.get("evidence", [])[:2]
        placements = item.get("placements", [])[:2]
        evidence_lines.append(
            f'- {keyword}: {support_level}; evidence={"; ".join(evidence) or "none"}; placements={", ".join(placements) or "none"}'
        )
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "  (no evidence map provided)"

    prompt = f"""You are a senior technical career editor preparing a resume for a specific role.

TARGET ROLE ENGINEERING IDENTITY: {identity}
DOMINANT THEMES THE ROLE CARES ABOUT: {themes}
IDEAL CANDIDATE ARC: {arc_hint}

JD KEYWORDS TO ADDRESS:
{kw_list}

SUPPORTED EVIDENCE MAP:
{evidence_block}

CANDIDATE'S RESUME:
{resume_compact}

Your task is to produce a narrative plan as JSON.

1. For each JD keyword, classify it as:
   - "verbatim"
   - "rephraseable"
   - "absent"

2. For each bullet in the experience section, decide:
   - "action": "keep" | "reframe" | "rewrite"
   - "emphasis"
   - "rephraseable_kws"
   - "rationale"

3. Write "resume_arc": 1-2 sentences describing the overall story the resume should tell.

Rules:
- Never turn an unsupported keyword into a rephraseable one.
- Prefer grounded software engineering and platform themes when they are supported.
- Return JSON only.

Return JSON in this shape:
{{
  "resume_arc": "...",
  "keyword_classification": {{"<keyword>": "verbatim" | "rephraseable" | "absent"}},
  "bullet_plans": [
    {{
      "exp_idx": 0,
      "bullet_idx": 0,
      "action": "keep" | "reframe" | "rewrite",
      "emphasis": "...",
      "rephraseable_kws": ["..."],
      "rationale": "..."
    }}
  ]
}}"""

    try:
        response = _client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=3000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        payload = _parse_json((response.choices[0].message.content or "").strip())
    except Exception as exc:
        raise RuntimeError(f"classify_and_plan LLM call failed: {exc}") from exc

    kw_classification: dict[str, str] = payload.get("keyword_classification", {})
    uncoverable = [kw for kw, status in kw_classification.items() if status == "absent"]

    bullet_plans: list[BulletPlan] = []
    for bp in payload.get("bullet_plans", []):
        exp_idx = int(bp.get("exp_idx", 0))
        bullet_idx = int(bp.get("bullet_idx", 0))
        try:
            bullet_raw = resume_json["experience"][exp_idx]["bullets"][bullet_idx]
            original = bullet_raw if isinstance(bullet_raw, str) else bullet_raw.get("text", "")
        except (IndexError, KeyError):
            original = ""
        bullet_plans.append(
            BulletPlan(
                exp_idx=exp_idx,
                bullet_idx=bullet_idx,
                original=original,
                action=str(bp.get("action", "keep")),
                emphasis=str(bp.get("emphasis", "")),
                rephraseable_kws=[str(k) for k in bp.get("rephraseable_kws", [])],
                rationale=str(bp.get("rationale", "")),
            )
        )

    return NarrativePlan(
        engineering_identity=identity,
        resume_arc=str(payload.get("resume_arc", "")),
        bullet_plans=bullet_plans,
        uncoverable=uncoverable,
    )


def generate_narrative_summary(
    resume_json: dict[str, Any],
    narrative_plan: NarrativePlan,
) -> str:
    if not _openai_key_available():
        return ""

    resume_compact = _build_compact_resume(resume_json)
    source_summary = str(resume_json.get("summary", "")).strip()
    summary_signals = _summary_signal_guidance(resume_json, resume_compact)
    framework_line = ", ".join(summary_signals["frameworks"]) if summary_signals["frameworks"] else "(none)"
    concept_line = ", ".join(summary_signals["concepts"]) if summary_signals["concepts"] else "(none)"
    fallback_line = ", ".join(summary_signals["fallback_terms"]) if summary_signals["fallback_terms"] else "(none)"

    prompt = f"""Write a professional summary for a resume targeting this role.

TARGET ROLE: {narrative_plan.engineering_identity}
RESUME ARC: {narrative_plan.resume_arc}

CURRENT SOURCE SUMMARY (if any):
{source_summary or "(none)"}

HIGH-CONFIDENCE PROJECT ML FRAMEWORKS (prefer at most one if natural):
{framework_line}

HIGH-CONFIDENCE PROJECT ML CONCEPTS (prefer at most one if natural):
{concept_line}

OTHER SUPPORTED ML TERMS YOU MAY NATURALLY SURFACE IF ALREADY GROUNDED:
{fallback_line}

CANDIDATE'S EXPERIENCE:
{resume_compact}

RULES:
1. Open with the engineering identity.
2. Connect the candidate's actual experience to the role's core themes.
3. Prefer project-backed ML signals when they exist: at most 1 framework mention and at most 1 concept mention total.
4. Maximum 2 sentences total and maximum 2 ML keyword insertions total.
5. Do NOT use generic phrases like "passionate about", "team player", or "results-driven".
6. Do NOT invent experience not in the resume.
7. Do NOT mention degrees unless already present in the current source summary.
8. Do NOT mention mentoring, research, or leadership unless explicitly present in the resume text.
9. Return only the summary text."""

    try:
        response = _client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=200,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError(f"generate_narrative_summary failed: {exc}") from exc


def generate_grounded_summary(
    resume_json: dict[str, Any],
    target_role: str,
    evidence_map: dict[str, Any],
) -> str:
    supported = [
        item["keyword"]
        for item in evidence_map.get("keyword_support", [])
        if item.get("support_level") in {"direct", "indirect"}
    ][:10]
    summary_signals = _summary_signal_guidance(resume_json, " ".join(supported))

    if not _openai_key_available():
        top_terms = ", ".join((summary_signals["all"] or supported)[:3])
        return (
            f"Software Engineer targeting {target_role} with experience building backend systems "
            f"and data-intensive platforms using {top_terms}."
        )[:320]

    resume_compact = _build_compact_resume(resume_json)
    framework_line = ", ".join(summary_signals["frameworks"]) if summary_signals["frameworks"] else "(none)"
    concept_line = ", ".join(summary_signals["concepts"]) if summary_signals["concepts"] else "(none)"
    fallback_line = ", ".join(summary_signals["fallback_terms"]) if summary_signals["fallback_terms"] else "(none)"

    prompt = f"""Rewrite the summary for a software engineering role.

TARGET ROLE: {target_role}
SUPPORTED JD THEMES: {", ".join(supported)}

HIGH-CONFIDENCE PROJECT ML FRAMEWORKS (prefer at most one if natural):
{framework_line}

HIGH-CONFIDENCE PROJECT ML CONCEPTS (prefer at most one if natural):
{concept_line}

OTHER SUPPORTED ML TERMS:
{fallback_line}

RESUME EVIDENCE:
{resume_compact}

RULES:
1. Lead with software engineer or backend engineer identity, not AI specialist.
2. Mention AI only as one part of broader engineering work.
3. Prefer project-backed ML signals when they are present, but use at most 2 ML insertions total.
4. Maximum 2 sentences total.
5. Make it sound human and credible, not like a keyword list.
6. Do not invent ownership, domains, or unsupported technologies.
7. Return only the summary text."""

    try:
        response = _client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=180,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError(f"generate_grounded_summary failed: {exc}") from exc


def _build_compact_resume(resume_json: dict[str, Any]) -> str:
    lines: list[str] = []
    for exp_idx, exp in enumerate(resume_json.get("experience", [])):
        role = exp.get("role", "")
        company = exp.get("company", "")
        dates = exp.get("dates", "")
        lines.append(f"[Role {exp_idx}] {role} at {company} ({dates})")
        for bullet_idx, bullet in enumerate(exp.get("bullets", [])):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            lines.append(f"  Bullet {bullet_idx}: {text}")
    for proj_idx, proj in enumerate(resume_json.get("projects", [])):
        lines.append(f"[Project {proj_idx}] {proj.get('name', 'Project')}")
        for bullet_idx, bullet in enumerate(proj.get("bullets", [])):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            lines.append(f"  Bullet {bullet_idx}: {text}")
    skills_flat: list[str] = []
    for _cat, vals in resume_json.get("skills", {}).items():
        if isinstance(vals, list):
            skills_flat.extend(vals)
    if skills_flat:
        lines.append(f"Skills: {', '.join(skills_flat)}")
    return "\n".join(lines)
