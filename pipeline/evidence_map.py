"""Grounded evidence mapping for JD-to-resume tailoring."""

from __future__ import annotations

import re
from typing import Any


_CONCEPT_ALIASES: dict[str, list[str]] = {
    "distributed systems": ["distributed systems", "distributed system", "microservice", "microservices"],
    "low latency": ["low latency", "latency", "real-time", "real time"],
    "high availability": ["high availability", "uptime", "resilient", "resilience", "fault tolerant"],
    "reliability": ["reliability", "reliable", "uptime", "resilient", "stability"],
    "observability": ["observability", "monitoring", "telemetry", "logging", "metrics", "tracing"],
    "monitoring": ["monitoring", "telemetry", "metrics", "alerting", "logging"],
    "operations at scale": ["production workflows", "multi million", "millions of", "scalable", "scale"],
    "architecture": ["architecture", "architected", "system design", "design", "microservice platforms"],
    "coding": ["python", "java", "javascript", "c#", "c++", "go", "fastapi", "backend"],
    "debugging": ["debugging", "troubleshooting", "root cause", "issue resolution", "bug fixes"],
    "engineering standards": ["best practices", "testing", "ci/cd", "code quality", "standards"],
    "design documents": ["design document", "design docs", "technical design", "documentation"],
    "development practices": ["ci/cd", "testing", "review", "best practices", "standards"],
    "feature design": ["feature", "designed", "design", "product", "platform"],
    "estimation": ["estimate", "estimation", "planning", "delivery"],
    "azure cloud": ["azure", "azure sql", "aks", "azure cloud"],
    "spark": ["spark", "apache spark"],
    "hadoop": ["hadoop"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "eks", "aks"],
    "aws": ["aws", "eks", "kinesis"],
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript"],
    "c#": ["c#"],
    "sql": ["sql", "postgresql", "mysql", "azure sql"],
    "kafka": ["kafka"],
}

_INDIRECT_HINTS: dict[str, list[str]] = {
    "engineering standards": ["testing", "test", "ci/cd", "quality", "standards"],
    "design documents": ["documentation", "technical design", "design", "architecture"],
    "debugging": ["incident", "defect", "issue", "bug", "troubleshoot"],
    "operations at scale": ["production", "scalable", "throughput", "latency", "monitoring"],
}

_ENGINEERING_PRIORITY = {
    "distributed systems",
    "low latency",
    "high availability",
    "reliability",
    "observability",
    "monitoring",
    "operations at scale",
    "architecture",
    "coding",
    "debugging",
    "engineering standards",
    "design documents",
    "development practices",
    "feature design",
    "azure cloud",
    "aws",
    "docker",
    "kubernetes",
    "python",
    "java",
    "javascript",
    "c#",
    "sql",
    "spark",
    "kafka",
}

_AI_HEAVY_TERMS = {
    "ai",
    "llm",
    "llms",
    "rag",
    "embeddings",
    "semantic search",
    "prompt engineering",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.-]+", _normalize(text))


def _iter_resume_segments(resume_json: dict[str, Any]) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []

    summary = (resume_json.get("summary") or "").strip()
    if summary:
        segments.append({"placement": "summary", "text": summary})

    for exp in resume_json.get("experience", []):
        role = exp.get("role", "").strip()
        company = exp.get("company", "").strip()
        placement = f"{company} {role}".strip() or "experience"
        for bullet in exp.get("bullets", []):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            text = (text or "").strip()
            if text:
                segments.append({"placement": placement, "text": text})

    for proj in resume_json.get("projects", []):
        placement = proj.get("name", "").strip() or "projects"
        for bullet in proj.get("bullets", []):
            text = bullet if isinstance(bullet, str) else bullet.get("text", "")
            text = (text or "").strip()
            if text:
                segments.append({"placement": placement, "text": text})

    for category, values in resume_json.get("skills", {}).items():
        if isinstance(values, list):
            items = values
        elif isinstance(values, dict):
            items = values.get("items", [])
        else:
            items = []
        text = ", ".join(str(item).strip() for item in items if str(item).strip())
        if text:
            segments.append({"placement": "skills", "text": f"{category}: {text}"})

    return segments


def _match_aliases(keyword: str, text: str) -> bool:
    normalized = _normalize(text)
    aliases = _CONCEPT_ALIASES.get(keyword.lower(), []) or [keyword]
    return any(alias.lower() in normalized for alias in aliases)


def _classify_support(keyword: str, matching_segments: list[dict[str, str]], all_segments: list[dict[str, str]]) -> str:
    if matching_segments:
        return "direct"

    aliases = _INDIRECT_HINTS.get(keyword.lower(), [])
    if aliases:
        for segment in all_segments:
            lower = _normalize(segment["text"])
            if any(alias in lower for alias in aliases):
                return "indirect"

    kw_tokens = [tok for tok in _tokenize(keyword) if len(tok) > 2]
    if not kw_tokens:
        return "unsupported"
    for segment in all_segments:
        seg_tokens = set(_tokenize(segment["text"]))
        overlap = sum(1 for tok in kw_tokens if tok in seg_tokens)
        if overlap >= max(1, len(kw_tokens) - 1):
            return "indirect"
    return "unsupported"


def _evidence_for_keyword(keyword: str, segments: list[dict[str, str]]) -> tuple[list[str], list[str], str]:
    matches = [segment for segment in segments if _match_aliases(keyword, segment["text"])]
    support_level = _classify_support(keyword, matches, segments)

    if support_level == "direct":
        evidence_segments = matches[:2]
    elif support_level == "indirect":
        hinted = []
        aliases = _INDIRECT_HINTS.get(keyword.lower(), [])
        for segment in segments:
            lower = _normalize(segment["text"])
            if any(alias in lower for alias in aliases):
                hinted.append(segment)
        evidence_segments = hinted[:2]
    else:
        evidence_segments = []

    evidence = [segment["text"] for segment in evidence_segments]
    placements = [segment["placement"] for segment in evidence_segments]
    return evidence, placements, support_level


def build_evidence_map(
    resume_json: dict[str, Any],
    jd_text: str,
    keywords: list[str],
    target_role: str = "",
) -> dict[str, Any]:
    """Build a direct/indirect/unsupported evidence map for JD terms."""
    del jd_text  # reserved for future prompt-based evidence extraction
    segments = _iter_resume_segments(resume_json)

    deduped_keywords: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        cleaned = re.sub(r"\s+", " ", str(keyword).strip())
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            deduped_keywords.append(cleaned)

    keyword_support: list[dict[str, Any]] = []
    for keyword in deduped_keywords:
        evidence, placements, support_level = _evidence_for_keyword(keyword, segments)
        keyword_support.append(
            {
                "keyword": keyword,
                "support_level": support_level,
                "evidence": evidence,
                "placements": placements,
            }
        )

    return {
        "target_role": target_role or "Target role",
        "must_cover_keywords": deduped_keywords,
        "keyword_support": keyword_support,
    }


def prioritized_supported_keywords(
    evidence_map: dict[str, Any],
    target_role: str = "",
) -> list[str]:
    """Order supported keywords so engineering themes win over generic/AI-heavy ones."""
    target_role_lower = _normalize(target_role)
    broader_swe_role = any(
        token in target_role_lower
        for token in ("software engineer", "backend engineer", "engineer ii", "senior software", "platform engineer")
    )

    supported = [
        item["keyword"]
        for item in evidence_map.get("keyword_support", [])
        if item.get("support_level") in {"direct", "indirect"}
    ]

    def sort_key(keyword: str) -> tuple[int, int, str]:
        lower = keyword.lower()
        engineering_rank = 0 if lower in _ENGINEERING_PRIORITY else 1
        ai_penalty = 1 if broader_swe_role and lower in _AI_HEAVY_TERMS else 0
        return (engineering_rank, ai_penalty, lower)

    return sorted(supported, key=sort_key)


def omitted_unsupported_keywords(evidence_map: dict[str, Any]) -> list[str]:
    return [
        item["keyword"]
        for item in evidence_map.get("keyword_support", [])
        if item.get("support_level") == "unsupported"
    ]
