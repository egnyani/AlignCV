"""Narrative planning layer.

Before any bullet is rewritten, this module:
1. Reads the full resume + JD keywords + narrative intent together
2. Classifies every JD keyword as VERBATIM / REPHRASEABLE / ABSENT
3. Assigns each bullet a story emphasis and a list of rephraseable keywords
4. Produces a NarrativePlan that the rewriter uses instead of brute-force injection

Two public entrypoints:
  - classify_and_plan()        → NarrativePlan
  - generate_narrative_summary() → str (professional summary for the resume header)
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, REWRITE_MODEL


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class BulletPlan:
    exp_idx: int
    bullet_idx: int
    original: str
    action: str                        # "keep" | "reframe" | "rewrite"
    emphasis: str                      # what story angle this bullet reinforces
    rephraseable_kws: list[str] = field(default_factory=list)
    rationale: str = ""                # why this bullet matters for the role


@dataclass
class NarrativePlan:
    engineering_identity: str
    resume_arc: str                    # 1-2 sentence overall story
    bullet_plans: list[BulletPlan] = field(default_factory=list)
    uncoverable: list[str] = field(default_factory=list)  # genuinely absent — never fabricate


# ── helpers ──────────────────────────────────────────────────────────────────

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
    """Flatten the resume to a single string for quick keyword scanning."""
    parts: list[str] = [resume_json.get("name", ""), resume_json.get("summary", "")]
    for exp in resume_json.get("experience", []):
        parts.append(exp.get("role", ""))
        parts.append(exp.get("company", ""))
        for b in exp.get("bullets", []):
            parts.append(b if isinstance(b, str) else b.get("text", ""))
    for cat, vals in resume_json.get("skills", {}).items():
        parts.extend(vals if isinstance(vals, list) else [])
    for proj in resume_json.get("projects", []):
        for b in proj.get("bullets", []):
            parts.append(b if isinstance(b, str) else b.get("text", ""))
    return " ".join(parts)


def _quick_classify_verbatim(keywords: list[str], resume_text: str) -> list[str]:
    """Fast local check: which keywords appear verbatim (case-insensitive)?"""
    lower = resume_text.lower()
    return [kw for kw in keywords if kw.lower() in lower]


# ── heuristic fallback (no API key) ─────────────────────────────────────────

def _heuristic_plan(
    resume_json: dict[str, Any],
    keywords: list[str],
    narrative_intent: dict[str, Any],
) -> NarrativePlan:
    """Build a minimal NarrativePlan without an API call."""
    resume_text = _resume_text(resume_json)
    verbatim = set(_quick_classify_verbatim(keywords, resume_text))
    uncoverable = [kw for kw in keywords if kw not in verbatim]

    bullet_plans: list[BulletPlan] = []
    for exp_idx, exp in enumerate(resume_json.get("experience", [])):
        for b_idx, bullet in enumerate(exp.get("bullets", [])):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            bullet_plans.append(BulletPlan(
                exp_idx=exp_idx,
                bullet_idx=b_idx,
                original=text,
                action="keep",
                emphasis=narrative_intent.get("engineering_identity", "relevant engineering work"),
                rephraseable_kws=[],
                rationale="Heuristic mode — no rewrite attempted.",
            ))

    return NarrativePlan(
        engineering_identity=narrative_intent.get("engineering_identity", "software engineer"),
        resume_arc="Candidate with relevant technical experience.",
        bullet_plans=bullet_plans,
        uncoverable=uncoverable,
    )


# ── main planning call ────────────────────────────────────────────────────────

def classify_and_plan(
    resume_json: dict[str, Any],
    keywords: list[str],
    narrative_intent: dict[str, Any],
    evidence_map: dict[str, Any] | None = None,
) -> NarrativePlan:
    """
    Core planning function. One LLM call that:
      - classifies every keyword as VERBATIM / REPHRASEABLE / ABSENT
      - assigns each bullet an emphasis angle + rephraseable keywords
      - produces the overall resume arc

    Returns a NarrativePlan ready for the rewriter.
    """
    if not OPENAI_API_KEY:
        return _heuristic_plan(resume_json, keywords, narrative_intent)

    # Build a compact resume representation for the prompt
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

Your task is to produce a narrative plan as JSON. Do the following:

1. For each JD keyword, classify it as one of:
   - "verbatim": the keyword already appears in the resume text — no action needed
   - "rephraseable": the underlying concept or work is present in the resume but uses different language — can be reframed to include this keyword naturally
   - "absent": no evidence of this concept anywhere in the resume — do NOT attempt to add it

Additional constraints:
- Prefer software engineering, backend systems, distributed systems, reliability, observability, cloud, monitoring, coding, debugging, and feature-delivery themes when they are supported.
- If the target role is broader software engineering, reduce emphasis on AI/LLM/RAG language unless it is central to the candidate's strongest supported evidence.
- Never turn an unsupported keyword into a rephraseable one.
- Every rewrite should still sound believable to a tech lead reading the resume.

2. For each bullet in the experience section, decide:
   - "action": "keep" (bullet already strongly supports the role), "reframe" (good content, wrong emphasis), or "rewrite" (weak or misaligned for this role)
   - "emphasis": a short phrase describing what story angle this bullet should reinforce for the target role
   - "rephraseable_kws": from the rephraseable keywords above, which ones could naturally fit into this bullet's rewrite
   - "rationale": one sentence explaining why this bullet matters (or doesn't) for the role

3. Write a "resume_arc": 1-2 sentences describing the overall story the resume should tell for this role — the through-line that makes the candidate feel like a natural fit.

Return JSON only in this exact structure:
{{
  "resume_arc": "...",
  "keyword_classification": {{
    "<keyword>": "verbatim" | "rephraseable" | "absent",
    ...
  }},
  "bullet_plans": [
    {{
      "exp_idx": 0,
      "bullet_idx": 0,
      "action": "keep" | "reframe" | "rewrite",
      "emphasis": "...",
      "rephraseable_kws": ["..."],
      "rationale": "..."
    }},
    ...
  ]
}}

Include ALL bullets from ALL experience entries. exp_idx is 0-based (order in resume), bullet_idx is 0-based within each role."""

    try:
        response = _client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=3000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (response.choices[0].message.content or "").strip()
        payload = _parse_json(raw)
    except Exception as exc:
        raise RuntimeError(f"classify_and_plan LLM call failed: {exc}") from exc

    # ── parse keyword classification ─────────────────────────────────────────
    kw_classification: dict[str, str] = payload.get("keyword_classification", {})
    uncoverable = [kw for kw, status in kw_classification.items() if status == "absent"]

    # ── parse bullet plans ───────────────────────────────────────────────────
    bullet_plans: list[BulletPlan] = []
    for bp in payload.get("bullet_plans", []):
        exp_idx = int(bp.get("exp_idx", 0))
        b_idx = int(bp.get("bullet_idx", 0))
        # Fetch original bullet text from resume
        try:
            bullet_raw = resume_json["experience"][exp_idx]["bullets"][b_idx]
            original = bullet_raw if isinstance(bullet_raw, str) else bullet_raw.get("text", "")
        except (IndexError, KeyError):
            original = ""
        bullet_plans.append(BulletPlan(
            exp_idx=exp_idx,
            bullet_idx=b_idx,
            original=original,
            action=str(bp.get("action", "keep")),
            emphasis=str(bp.get("emphasis", "")),
            rephraseable_kws=[str(k) for k in bp.get("rephraseable_kws", [])],
            rationale=str(bp.get("rationale", "")),
        ))

    return NarrativePlan(
        engineering_identity=identity,
        resume_arc=str(payload.get("resume_arc", "")),
        bullet_plans=bullet_plans,
        uncoverable=uncoverable,
    )


# ── summary generation ────────────────────────────────────────────────────────

def generate_narrative_summary(
    resume_json: dict[str, Any],
    narrative_plan: NarrativePlan,
) -> str:
    """
    Generate a 2-3 sentence professional summary that frames the candidate's
    arc for the specific role — not a generic summary, but one that sets up
    the story the rest of the resume tells.
    """
    if not OPENAI_API_KEY:
        return ""

    resume_compact = _build_compact_resume(resume_json)

    prompt = f"""Write a 2-3 sentence professional summary for a resume targeting this role.

TARGET ROLE: {narrative_plan.engineering_identity}
RESUME ARC: {narrative_plan.resume_arc}

CANDIDATE'S EXPERIENCE:
{resume_compact}

RULES:
1. Open with the engineering identity (e.g. "Backend systems engineer with X years...").
2. Connect the candidate's actual experience to the role's core themes — no fluff.
3. The summary should make a hiring manager think "this person has been building toward this role".
4. Do NOT use generic phrases like "passionate about", "team player", "results-driven".
5. Do NOT invent experience not in the resume.
6. 2-3 sentences max. Return only the summary text."""

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
    """
    Generate a summary that leads with software engineering identity and only
    uses supported evidence from the resume.
    """
    supported = [
        item["keyword"] for item in evidence_map.get("keyword_support", [])
        if item.get("support_level") in {"direct", "indirect"}
    ][:10]

    if not OPENAI_API_KEY:
        top_terms = ", ".join(supported[:5])
        return (
            f"Software Engineer targeting {target_role} with experience building backend systems "
            f"and data-intensive platforms using {top_terms}."
        )[:320]

    resume_compact = _build_compact_resume(resume_json)
    prompt = f"""Rewrite the summary for a software engineering role.

TARGET ROLE: {target_role}
SUPPORTED JD THEMES: {", ".join(supported)}

RESUME EVIDENCE:
{resume_compact}

Rules:
1. Lead with software engineer or backend engineer identity, not AI specialist.
2. Mention AI only as one part of broader engineering work.
3. Include supported JD themes such as distributed systems, low latency, reliability, observability, cloud, architecture, and data intensive systems only when grounded in the resume.
4. Keep it to 3 lines maximum.
5. Make it sound human and credible, not like a keyword list.
6. Do not invent ownership, domains, or unsupported technologies.

Return only the summary text."""

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


# ── compact resume builder for prompts ───────────────────────────────────────

def _build_compact_resume(resume_json: dict[str, Any]) -> str:
    """Build a compact but complete resume representation for LLM prompts."""
    lines: list[str] = []
    for exp_idx, exp in enumerate(resume_json.get("experience", [])):
        role = exp.get("role", "")
        company = exp.get("company", "")
        dates = exp.get("dates", "")
        lines.append(f"[Role {exp_idx}] {role} at {company} ({dates})")
        for b_idx, bullet in enumerate(exp.get("bullets", [])):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            lines.append(f"  Bullet {b_idx}: {text}")
    skills_flat: list[str] = []
    for cat, vals in resume_json.get("skills", {}).items():
        if isinstance(vals, list):
            skills_flat.extend(vals)
    if skills_flat:
        lines.append(f"Skills: {', '.join(skills_flat)}")
    return "\n".join(lines)
