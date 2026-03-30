"""
Keyword-driven bullet rewriter.

Strategy
--------
1. Vector-search: for every JD keyword, find the resume bullet whose embedding
   is closest (cosine similarity).  Assign the keyword to that bullet.
2. Load-balance: no bullet gets more than MAX_KEYWORDS_PER_BULLET keywords,
   so overflow keywords fall to the next-best bullet.
3. Rewrite: call GPT for each bullet that has ≥1 keyword assigned.  The prompt
   demands all assigned keywords appear VERBATIM.
4. Verify & fix: a second GPT pass re-injects any keywords that GPT dropped.
5. Return: updated resume JSON, list of rewrites (with injected_keywords field),
   and the assignment map so the caller can report coverage stats.
"""

from __future__ import annotations

from copy import deepcopy
import os
import re
from typing import Any

import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, REWRITE_MODEL


# ── tunables ────────────────────────────────────────────────────────────────

MIN_ASSIGNMENT_SIM: float = 0.28   # below this similarity, skip keyword (don't fabricate)
MAX_KEYWORDS_PER_BULLET: int = 6   # max ATS terms injected per bullet


# ── helpers ─────────────────────────────────────────────────────────────────

TECH_TERMS = {
    "airflow", "aws", "azure", "bigquery", "databricks", "docker", "dbt",
    "etl", "elt", "fastapi", "flink", "gcp", "git", "java", "grpc",
    "graphql", "kafka", "kubernetes", "looker", "mysql", "postgres", "python",
    "redshift", "s3", "snowflake", "spark", "sql", "tableau", "terraform",
    "restful", "rest", "api", "microservices", "c#", "kotlin", "go", "rust",
}
_STOP = {"a", "an", "and", "by", "for", "from", "in", "of", "on", "the", "to", "with"}


def _client() -> OpenAI:
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), base_url=OPENAI_BASE_URL)


def _clean_bullet(text: str) -> str:
    text = re.sub(r"^[\-\*\u2022]\s*", "", text.replace("\n", " ").strip().strip('"'))
    # Remove spaces before punctuation (e.g. "behavior ," → "behavior,")
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    # Ensure bullet ends with a period
    if text and text[-1] not in ".!?":
        text += "."
    return text


def cosine_sim(a: list[float], b: list[float]) -> float:
    a_np, b_np = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    return float(np.dot(a_np, b_np) / denom) if denom > 0 else 0.0


def _extract_factual_anchors(bullet: str, context: dict[str, Any]) -> list[str]:
    """Extract numbers, tech names, and proper nouns to preserve through rewriting."""
    anchors: list[str] = []
    # Metrics / percentages / numbers
    for m in re.findall(r"\b\d+(?:\.\d+)?(?:[%x]|[kKmMbB])?(?:\+)?(?:\s+[A-Za-z]+)?", bullet):
        c = m.strip()
        if c and c not in anchors:
            anchors.append(c)
    # Tech tokens
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9.+#/-]*\b", bullet):
        lower = token.lower()
        if lower in _STOP:
            continue
        if lower in TECH_TERMS or token.isupper() or any(c.isupper() for c in token[1:]):
            if token not in anchors:
                anchors.append(token)
    company = str(context.get("company", "")).strip()
    if company and company.lower() in bullet.lower() and company not in anchors:
        anchors.append(company)
    return anchors


def _missing(text: str, items: list[str]) -> list[str]:
    lower = text.lower()
    return [item for item in items if item.lower() not in lower]


# ── core assignment ──────────────────────────────────────────────────────────

def _flat_bullets(resume_with_embeddings: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a flat list of bullet dicts with position + embedding."""
    result: list[dict[str, Any]] = []
    for exp_idx, exp in enumerate(resume_with_embeddings.get("experience", [])):
        for b_idx, bullet in enumerate(exp.get("bullets", [])):
            if isinstance(bullet, dict) and bullet.get("embedding"):
                result.append({
                    "exp_idx": exp_idx,
                    "bullet_idx": b_idx,
                    "text": bullet["text"],
                    "embedding": bullet["embedding"],
                    "context": exp,
                })
    return result


def assign_keywords_to_bullets(
    keywords_with_emb: list[dict[str, Any]],
    bullets: list[dict[str, Any]],
) -> dict[tuple[int, int], list[str]]:
    """
    For each JD keyword: vector-search → best-matching bullet.
    Load-balances so no bullet exceeds MAX_KEYWORDS_PER_BULLET.
    Keywords whose best similarity < MIN_ASSIGNMENT_SIM are skipped
    (we won't fabricate skills that don't belong anywhere in the resume).

    Returns: {(exp_idx, bullet_idx): [keyword, ...]}
    """
    load: dict[tuple[int, int], int] = {
        (b["exp_idx"], b["bullet_idx"]): 0 for b in bullets
    }
    assignment: dict[tuple[int, int], list[str]] = {}

    for kw_item in keywords_with_emb:
        kw = kw_item["text"]
        kw_emb = kw_item["embedding"]

        # Rank all bullets by similarity to this keyword
        ranked = sorted(
            bullets,
            key=lambda b: cosine_sim(kw_emb, b["embedding"]),
            reverse=True,
        )

        for b in ranked:
            key = (b["exp_idx"], b["bullet_idx"])
            sim = cosine_sim(kw_emb, b["embedding"])
            if sim < MIN_ASSIGNMENT_SIM:
                break   # rest are too dissimilar — skip keyword
            if load[key] < MAX_KEYWORDS_PER_BULLET:
                assignment.setdefault(key, []).append(kw)
                load[key] += 1
                break   # assigned — next keyword

    return assignment


# ── GPT rewriting ────────────────────────────────────────────────────────────

def _rewrite_bullet(
    bullet: str,
    must_include: list[str],
    context: dict[str, Any],
) -> str:
    """
    Ask GPT to rewrite the bullet with ALL `must_include` keywords verbatim.
    A second pass fixes any keywords that GPT dropped.
    """
    anchors = _extract_factual_anchors(bullet, context)
    kw_block = "\n".join(f'  • "{kw}"' for kw in must_include)
    anchor_note = (
        ", ".join(anchors)
        if anchors
        else "all numbers, percentages, and technology names"
    )

    prompt = (
        "You are rewriting a resume bullet to pass ATS keyword screening.\n\n"
        f"ORIGINAL BULLET:\n{bullet}\n\n"
        f"YOU MUST INCLUDE ALL OF THESE EXACT TERMS VERBATIM IN YOUR REWRITE:\n{kw_block}\n\n"
        f"FACTUAL ANCHORS — preserve these exactly (do not remove or alter):\n  {anchor_note}\n\n"
        "RULES:\n"
        "1. Every term in the MUST INCLUDE list must appear word-for-word in your output.\n"
        "2. Keep all factual anchors (numbers, %, tool names, company names).\n"
        "3. Do NOT invent tools, metrics, or experiences not in the original.\n"
        "4. Start with a strong action verb.\n"
        "5. Keep under 35 words.\n"
        "6. Return ONLY the bullet text — no quotes, no preamble, no punctuation at start."
    )

    result = _clean_bullet(
        _client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=160,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content or ""
    )

    # ── Verification pass ────────────────────────────────────────────────────
    missing_kw = _missing(result, must_include)
    missing_anchors = _missing(result, anchors)

    if missing_kw or missing_anchors:
        fix_items = [f'"{k}"' for k in missing_kw] + missing_anchors
        fix_block = "\n".join(f"  • {item}" for item in fix_items)
        fix_prompt = (
            "The rewritten bullet below is missing some required items. Add them back.\n\n"
            f"CURRENT REWRITE:\n{result}\n\n"
            f"MISSING — add these (verbatim):\n{fix_block}\n\n"
            f"ORIGINAL BULLET (for context):\n{bullet}\n\n"
            "RULES: Preserve all other keywords already present. Under 35 words. "
            "Return ONLY the bullet text."
        )
        fixed = _clean_bullet(
            _client().chat.completions.create(
                model=REWRITE_MODEL,
                max_tokens=160,
                temperature=0.1,
                messages=[{"role": "user", "content": fix_prompt}],
            ).choices[0].message.content or ""
        )
        # Accept the fix if it recovered at least some missing keywords
        if len(_missing(fixed, must_include)) < len(missing_kw):
            result = fixed

    return result


# ── public entrypoint ────────────────────────────────────────────────────────

def keyword_driven_rewrite(
    resume_json: dict[str, Any],
    resume_with_embeddings: dict[str, Any],
    keywords_with_emb: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[int, int], list[str]]]:
    """
    Main entrypoint for the new keyword-driven rewriting pipeline.

    Parameters
    ----------
    resume_json              : parsed resume (no embeddings — source of truth)
    resume_with_embeddings   : same resume with bullet embeddings attached
    keywords_with_emb        : [{text: str, embedding: list[float]}, ...]

    Returns
    -------
    updated_resume   : resume_json with rewritten bullet strings
    rewrites         : [{original, rewritten, injected_keywords}, ...]
    assignment       : {(exp_idx, bullet_idx): [keywords]} — for coverage stats
    """
    updated = deepcopy(resume_json)
    bullets = _flat_bullets(resume_with_embeddings)
    assignment = assign_keywords_to_bullets(keywords_with_emb, bullets)

    rewrites: list[dict[str, Any]] = []

    for b in bullets:
        key = (b["exp_idx"], b["bullet_idx"])
        keywords_to_inject = assignment.get(key, [])
        if not keywords_to_inject:
            continue

        original = b["text"]
        context = b["context"]
        rewritten = _rewrite_bullet(original, keywords_to_inject, context)

        # Patch the resume in-place
        updated["experience"][b["exp_idx"]]["bullets"][b["bullet_idx"]] = rewritten

        # Report only keywords that actually landed in the rewritten text,
        # not the planned list — GPT sometimes drops terms even after the
        # verification pass.
        rewritten_lower = rewritten.lower()
        actually_injected = [
            kw for kw in keywords_to_inject
            if kw.lower() in rewritten_lower
        ]

        rewrites.append({
            "original": original,
            "rewritten": rewritten,
            "injected_keywords": actually_injected,
        })

    return updated, rewrites, assignment


# ── narrative-driven rewrite (new pipeline) ───────────────────────────────────

def _narrative_rewrite_bullet(
    bullet: str,
    emphasis: str,
    resume_arc: str,
    engineering_identity: str,
    rephraseable_kws: list[str],
    context: dict[str, Any],
) -> str:
    """
    Rewrite a single bullet using story context rather than verbatim injection.
    Keywords are suggestions — included only if they fit naturally.
    """
    anchors = _extract_factual_anchors(bullet, context)
    anchor_note = ", ".join(anchors) if anchors else "all numbers, percentages, and technology names"
    kw_block = (
        "\n".join(f'  • "{kw}"' for kw in rephraseable_kws)
        if rephraseable_kws
        else "  (none — focus purely on the story angle)"
    )

    prompt = (
        "You are a senior technical career editor rewriting a resume bullet.\n\n"
        f"ROLE THIS RESUME IS TARGETING: {engineering_identity}\n"
        f"OVERALL RESUME STORY: {resume_arc}\n\n"
        f"THIS BULLET'S JOB IN THE STORY: {emphasis}\n\n"
        f"ORIGINAL BULLET:\n{bullet}\n\n"
        f"FACTUAL ANCHORS — preserve these exactly (do not remove or alter):\n  {anchor_note}\n\n"
        f"KEYWORDS TO WEAVE IN NATURALLY (include only if they genuinely fit — do not force):\n{kw_block}\n\n"
        "RULES:\n"
        "1. Reframe the bullet to reinforce the story angle above — adjust emphasis, not facts.\n"
        "2. If a keyword from the list fits naturally in context, use it. If it doesn't fit, skip it.\n"
        "3. Do NOT invent tools, metrics, or experiences not in the original bullet.\n"
        "4. Preserve all factual anchors exactly.\n"
        "5. Prefer concrete software engineering language over generic platform or AI hype.\n"
        "6. Use a believable structure when possible: action + system/product + technologies + purpose + outcome.\n"
        "7. Avoid buzzwords such as leveraged, spearheaded, ecosystem, cutting-edge, transformative.\n"
        "8. Start with a strong action verb. Keep under 35 words.\n"
        "9. Return ONLY the bullet text — no quotes, no preamble."
    )

    result = _clean_bullet(
        _client().chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=160,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content or ""
    )

    # Verify factual anchors are preserved — run a fix pass if any are missing
    missing_anchors = _missing(result, anchors)
    if missing_anchors:
        fix_block = "\n".join(f"  • {a}" for a in missing_anchors)
        fix_prompt = (
            "The rewritten bullet below is missing some required factual anchors. Add them back.\n\n"
            f"CURRENT REWRITE:\n{result}\n\n"
            f"MISSING ANCHORS (add these verbatim):\n{fix_block}\n\n"
            f"ORIGINAL BULLET (for context):\n{bullet}\n\n"
            "RULES: Preserve everything else. Under 35 words. Return ONLY the bullet text."
        )
        fixed = _clean_bullet(
            _client().chat.completions.create(
                model=REWRITE_MODEL,
                max_tokens=160,
                temperature=0.1,
                messages=[{"role": "user", "content": fix_prompt}],
            ).choices[0].message.content or ""
        )
        if len(_missing(fixed, anchors)) < len(missing_anchors):
            result = fixed

    return result


def narrative_driven_rewrite(
    resume_json: dict[str, Any],
    narrative_plan: "NarrativePlan",  # type: ignore[name-defined]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Narrative-aware rewriting pipeline.

    Uses a NarrativePlan (from narrative_planner.classify_and_plan) to rewrite
    bullets with story coherence rather than verbatim keyword injection.

    Parameters
    ----------
    resume_json     : parsed resume (source of truth)
    narrative_plan  : plan produced by narrative_planner.classify_and_plan()

    Returns
    -------
    updated_resume  : resume_json with rewritten bullet strings
    rewrites        : [{original, rewritten, injected_keywords, emphasis}, ...]
    """
    updated = deepcopy(resume_json)
    rewrites: list[dict[str, Any]] = []

    for bp in narrative_plan.bullet_plans:
        # "keep" bullets are left untouched
        if bp.action == "keep" or not bp.original:
            continue

        context = {}
        try:
            context = updated["experience"][bp.exp_idx]
        except (IndexError, KeyError):
            pass

        rewritten = _narrative_rewrite_bullet(
            bullet=bp.original,
            emphasis=bp.emphasis,
            resume_arc=narrative_plan.resume_arc,
            engineering_identity=narrative_plan.engineering_identity,
            rephraseable_kws=bp.rephraseable_kws,
            context=context,
        )

        # Patch the resume in-place
        updated["experience"][bp.exp_idx]["bullets"][bp.bullet_idx] = rewritten

        # Track which rephraseable keywords actually landed in the rewrite
        rewritten_lower = rewritten.lower()
        actually_incorporated = [
            kw for kw in bp.rephraseable_kws
            if kw.lower() in rewritten_lower
        ]

        rewrites.append({
            "original": bp.original,
            "rewritten": rewritten,
            "injected_keywords": actually_incorporated,
            "emphasis": bp.emphasis,
            "action": bp.action,
        })

    return updated, rewrites
