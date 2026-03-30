"""Job description extraction utilities."""
# narrative intent extraction added — extract_jd_narrative_intent()

from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_BASE_URL, REWRITE_MODEL


def heuristic_extract_keywords(jd_text: str, limit: int = 40) -> list[str]:
    """Heuristic keyword extraction when no API key is available."""
    tech_pattern = re.compile(
        r"\b(gRPC|GraphQL|RESTful?|Java|Python|C#|C\+\+|Go|Rust|Kotlin|Swift|"
        r"microservices?|low.latency|high.scale|resilient|production.grade|"
        r"concurrency|multi.threading|performance.tuning|observability|"
        r"data.modeling|on.call|incident.review|API.design|OO.programming|"
        r"Docker|Kubernetes|AWS|Azure|GCP|Terraform|CI/CD|"
        r"Spark|Kafka|Airflow|Snowflake|dbt|PostgreSQL|Redis|"
        r"machine.learning|deep.learning|LLM|RAG|NLP)\b",
        re.IGNORECASE,
    )
    found: list[str] = []
    seen: set[str] = set()
    for m in tech_pattern.finditer(jd_text):
        kw = m.group(0)
        if kw.lower() not in seen:
            found.append(kw)
            seen.add(kw.lower())
    # Also grab bullet lines
    for line in jd_text.splitlines():
        line = line.strip(" -•\t")
        if 2 <= len(line.split()) <= 5 and any(c.isupper() for c in line):
            if line.lower() not in seen:
                found.append(line)
                seen.add(line.lower())
    return found[:limit]


def heuristic_extract_requirements(jd_text: str, limit: int = 8) -> list[str]:
    """Extract short requirement-like phrases without an API call."""
    lines = [line.strip(" -•\t") for line in jd_text.splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(
            token in lowered
            for token in (
                "experience", "build", "design", "develop", "python", "sql",
                "spark", "airflow", "aws", "warehouse", "pipelines", "data",
                "communication", "stakeholder",
            )
        ):
            cleaned = re.sub(r"^(requirements|responsibilities|preferred qualifications)[:\-]?\s*", "", line, flags=re.I)
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
            if 4 <= len(cleaned.split()) <= 15:
                candidates.append(cleaned)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        key = item.lower()
        if key not in seen:
            deduped.append(item)
            seen.add(key)
        if len(deduped) >= limit:
            break
    return deduped or ["Build scalable data pipelines", "Model reliable warehouse datasets", "Partner with stakeholders on analytics needs"]


def _parse_json_content(text: str) -> dict[str, Any]:
    """Parse JSON content, tolerating fenced responses."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def extract_jd_requirements(jd_text: str, dry_run: bool = False) -> list[str]:
    """Extract 5-12 atomic requirements from a job description."""
    if dry_run or not OPENAI_API_KEY:
        return heuristic_extract_requirements(jd_text)

    prompt = (
        "Extract 5 to 12 atomic resume-matching requirements from this job description.\n"
        "Return JSON only in the form {\"requirements\": [\"...\"]}.\n"
        "Each item must be 8 to 15 words, discrete, terse, and specific.\n"
        "Avoid duplicates and soft filler.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        payload = _parse_json_content(text)
        requirements = payload.get("requirements", [])
        if not isinstance(requirements, list) or not requirements:
            raise ValueError("OpenAI returned an empty requirements list.")
        return [str(item).strip() for item in requirements if str(item).strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to extract JD requirements with OpenAI: {exc}") from exc


def extract_jd_keywords(jd_text: str, dry_run: bool = False) -> list[str]:
    """Extract 25-50 ATS-critical keywords and phrases from a job description.

    Each keyword is a 1-5 word term that should appear verbatim in the resume
    to pass ATS screening. Vector-search will assign each keyword to the best
    matching resume bullet for injection.
    """
    if dry_run or not OPENAI_API_KEY:
        return heuristic_extract_keywords(jd_text)

    prompt = (
        "You are an ATS (Applicant Tracking System) expert.\n"
        "Extract every important keyword and phrase from this job description "
        "that a resume MUST contain to pass ATS screening.\n\n"
        "Return JSON only: {\"keywords\": [...]}\n\n"
        "INCLUDE:\n"
        "- All technical tools, languages, frameworks, platforms (exact names as written)\n"
        "- Skill phrases (1–4 words): e.g. 'data modeling', 'on-call rotation', 'low-latency'\n"
        "- Domain terms and methodologies\n"
        "- Action+object pairs: e.g. 'lead incident reviews', 'translate business requirements'\n"
        "- Important qualifiers: e.g. 'production environment', 'high-scale', 'resilient'\n"
        "- OO concepts if mentioned: 'concurrency', 'multi-threading', 'performance tuning'\n\n"
        "EXCLUDE:\n"
        "- Company name, location, salary, benefits\n"
        "- Pure soft-skill filler: 'team player', 'self-starter', 'we offer'\n"
        "- Overly generic standalone verbs: 'work', 'help', 'support'\n\n"
        "Return 25 to 50 keywords/phrases. Each must be 1–5 words.\n"
        "Use the EXACT phrasing from the JD — do not paraphrase.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=700,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        payload = _parse_json_content(text)
        keywords = payload.get("keywords", [])
        if not isinstance(keywords, list) or not keywords:
            raise ValueError("Empty keywords list returned.")
        return [str(k).strip() for k in keywords if str(k).strip()]
    except Exception as exc:
        raise RuntimeError(f"Failed to extract JD keywords: {exc}") from exc


def extract_jd_narrative_intent(jd_text: str, dry_run: bool = False) -> dict[str, Any]:
    """Extract the engineering identity and story arc the JD is looking for.

    Returns a dict with:
      - engineering_identity: short label for the type of engineer wanted
      - dominant_themes: list of 3-5 core themes
      - arc_description: what a believable candidate arc looks like
    """
    if dry_run or not OPENAI_API_KEY:
        # Heuristic fallback: pull the most repeated meaningful nouns
        words = re.findall(r"\b[a-z][a-z\-]+\b", jd_text.lower())
        stopwords = {"and", "the", "with", "for", "our", "you", "your", "this",
                     "that", "will", "have", "from", "are", "such", "ability",
                     "experience", "including", "qualifications", "preferred"}
        freq: dict[str, int] = {}
        for w in words:
            if w not in stopwords and len(w) > 4:
                freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=lambda w: freq[w], reverse=True)[:5]
        return {
            "engineering_identity": "software engineer",
            "dominant_themes": top,
            "arc_description": "Candidate with relevant technical experience.",
        }

    prompt = (
        "Read this job description and extract the engineering identity it is looking for.\n\n"
        "Return JSON only with these exact keys:\n"
        "{\n"
        '  "engineering_identity": "A short label (5-8 words) for the type of engineer this role wants. E.g. reliability-focused distributed systems engineer",\n'
        '  "dominant_themes": ["3 to 5 core themes the role cares most about, in order of importance"],\n'
        '  "arc_description": "1-2 sentences: what a believable candidate arc looks like for this role — what pattern of past work would feel like a natural progression into this job"\n'
        "}\n\n"
        "Focus on engineering substance, not soft skills or culture.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )

    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=400,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        payload = _parse_json_content(text)
        return {
            "engineering_identity": str(payload.get("engineering_identity", "software engineer")).strip(),
            "dominant_themes": [str(t).strip() for t in payload.get("dominant_themes", [])],
            "arc_description": str(payload.get("arc_description", "")).strip(),
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to extract narrative intent: {exc}") from exc


def extract_company_role(jd_text: str, dry_run: bool = False) -> dict[str, str]:
    """Extract company and role names from a job description."""
    if dry_run or not OPENAI_API_KEY:
        role = ""
        company = ""
        first_lines = [line.strip() for line in jd_text.splitlines() if line.strip()][:8]
        for line in first_lines:
            if not role and any(token in line.lower() for token in ("engineer", "manager", "analyst", "scientist", "developer")):
                role = line
            if not company and re.search(r"\bat\b", line, flags=re.I):
                maybe_company = re.split(r"\bat\b", line, flags=re.I)[-1].strip(" -:")
                if maybe_company:
                    company = maybe_company
        if not company:
            company = "Company"
        if not role:
            role = "Role"
        return {"company": company, "role": role}

    prompt = (
        "Extract the company and role title from this job description.\n"
        "Return JSON only in the form {\"company\": \"...\", \"role\": \"...\"}.\n\n"
        f"JOB DESCRIPTION:\n{jd_text}"
    )
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY), base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=REWRITE_MODEL,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
        payload = _parse_json_content(text)
        return {
            "company": str(payload.get("company", "Company")).strip() or "Company",
            "role": str(payload.get("role", "Role")).strip() or "Role",
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to extract company/role from OpenAI: {exc}") from exc
