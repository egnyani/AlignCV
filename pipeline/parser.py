"""DOCX parsing utilities for resumes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.document import Document as DocumentType
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

from build_log_utils import log_file_event
from pipeline.builder import json_to_docx as build_docx_from_json


SECTION_ALIASES = {
    "experience": "experience",
    "professional experience": "experience",
    "work experience": "experience",
    "education": "education",
    "technical skills": "skills",
    "skills": "skills",
    "projects": "projects",
    "certifications": "certifications",
    "summary": "summary",
}


def iter_block_items(parent: DocumentType | _Cell):
    """Yield paragraphs and tables in document order."""
    parent_elm = parent.element.body if isinstance(parent, DocumentType) else parent._tc
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        else:
            yield Table(child, parent)


def normalize_text(text: str) -> str:
    """Normalize paragraph text for parsing."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def is_bold_heading(paragraph: Paragraph) -> bool:
    """Return True when a paragraph is likely a section heading."""
    text = normalize_text(paragraph.text)
    if not text or "\t" in paragraph.text:
        return False
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    if style_name.startswith("heading"):
        return True
    runs = [run for run in paragraph.runs if normalize_text(run.text)]
    if not runs:
        return False
    return all(bool(run.bold) for run in runs)


def is_bullet_paragraph(paragraph: Paragraph) -> bool:
    """Return True for list-like paragraphs."""
    style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
    text = paragraph.text.lstrip()
    return style_name.startswith("list") or text.startswith(("-", "*", "•"))


def is_section_heading(paragraph: Paragraph) -> bool:
    """Return True when the paragraph is a section heading, not a tabbed role line."""
    return is_bold_heading(paragraph) and "\t" not in paragraph.text


def clean_bullet_text(text: str) -> str:
    """Remove bullet markers and normalize whitespace."""
    text = normalize_text(text)
    return re.sub(r"^[•*\-]\s*", "", text).strip()


def looks_like_contact_line(text: str) -> bool:
    """Return True for the contact row under the candidate name."""
    lower = text.lower()
    return (
        "linkedin" in lower
        or "github" in lower
        or "github.com" in lower
        or "@" in text
        or bool(re.search(r"\+?\d[\d\s().-]{7,}", text))
    )


def parse_inline_skills(paragraphs: list[Paragraph]) -> dict[str, list[str]]:
    """Parse inline bold-label skills into a structured dict."""
    runs: list[tuple[str, bool]] = []
    for paragraph in paragraphs:
        for run in paragraph.runs:
            text = run.text.replace("\n", " ")
            if text:
                runs.append((text, bool(run.bold)))

    categories: dict[str, list[str]] = {}
    current_label: str | None = None
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if current_label is None:
            buffer = ""
            return
        values = [item.strip() for item in re.split(r",|;", buffer) if item.strip()]
        categories[current_label] = values
        buffer = ""

    for text, is_bold in runs:
        normalized = text.strip()
        if not normalized:
            continue
        if is_bold and ":" in normalized:
            flush_buffer()
            label = normalized.split(":", 1)[0].strip()
            current_label = label
            remainder = normalized.split(":", 1)[1].strip()
            buffer = remainder
        elif current_label is not None:
            buffer += (" " if buffer else "") + normalized

    flush_buffer()
    return categories


def extract_name(document: DocumentType, paragraphs: list[Paragraph]) -> str:
    """Extract the resume owner's name."""
    for paragraph in paragraphs:
        style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
        text = normalize_text(paragraph.text)
        if text and style_name == "title":
            return text
    for paragraph in paragraphs:
        text = normalize_text(paragraph.text)
        if text:
            return text
    if getattr(document.core_properties, "author", None):
        author = normalize_text(document.core_properties.author)
        if author:
            return author
    return "Unknown Candidate"


def parse_resume(docx_path: str | Path) -> dict[str, Any]:
    """Parse a resume DOCX into structured JSON."""
    document = Document(str(docx_path))
    blocks = [block for block in iter_block_items(document) if isinstance(block, Paragraph)]
    paragraphs = [paragraph for paragraph in blocks if normalize_text(paragraph.text)]

    name = extract_name(document, paragraphs)
    summary = ""
    skills: dict[str, list[str]] = {}
    experience: list[dict[str, Any]] = []
    education: list[dict[str, str]] = []
    projects: list[dict[str, Any]] = []
    certifications: list[str] = []

    current_section: str | None = None
    current_experience: dict[str, Any] | None = None
    current_project: dict[str, Any] | None = None
    last_bullet_container: tuple[str, dict[str, Any] | None] | None = None
    pending_experience_company: str = ""
    pending_experience_location: str = ""

    first_non_name_seen = False
    skills_paragraphs: list[Paragraph] = []

    idx = 0
    while idx < len(paragraphs):
        paragraph = paragraphs[idx]
        raw_text = paragraph.text
        text = normalize_text(raw_text)
        lower = text.lower()

        if text == name and not first_non_name_seen:
            idx += 1
            continue

        section_key = SECTION_ALIASES.get(lower) if is_section_heading(paragraph) else None
        if section_key:
            if current_section == "skills" and skills_paragraphs:
                skills = parse_inline_skills(skills_paragraphs)
                skills_paragraphs = []
            current_section = section_key
            current_experience = None
            current_project = None
            idx += 1
            continue

        if not first_non_name_seen:
            if current_section is None and not is_section_heading(paragraph):
                # The template's second line is a protected contact row, not the summary.
                if looks_like_contact_line(text):
                    idx += 1
                    continue
            first_non_name_seen = True

        if current_section == "summary":
            if not summary:
                summary = text
            idx += 1
            continue

        if current_section == "skills":
            skills_paragraphs.append(paragraph)
            idx += 1
            continue

        if current_section == "experience":
            if "\t" in raw_text:
                left, right = [normalize_text(part) for part in raw_text.split("\t", 1)]
                next_paragraph = paragraphs[idx + 1] if idx + 1 < len(paragraphs) else None
                next_has_tab = bool(next_paragraph and "\t" in next_paragraph.text)
                if next_has_tab and not is_bullet_paragraph(paragraph):
                    pending_experience_company = left
                    pending_experience_location = right
                    idx += 1
                    continue

                role_company_match = re.match(r"^(.*?)\s*[|@-]\s*(.*)$", left)
                if role_company_match and not pending_experience_company:
                    role = role_company_match.group(1).strip()
                    company = role_company_match.group(2).strip()
                else:
                    role = left
                    company = pending_experience_company
                current_experience = {
                    "role": role,
                    "company": company,
                    "dates": right,
                    "location": pending_experience_location,
                    "bullets": [],
                }
                experience.append(current_experience)
                pending_experience_company = ""
                pending_experience_location = ""
                last_bullet_container = None
                idx += 1
                continue
            if is_bullet_paragraph(paragraph) and current_experience is not None:
                bullet = clean_bullet_text(text)
                current_experience["bullets"].append(bullet)
                last_bullet_container = ("experience", current_experience)
                idx += 1
                continue
            if (
                last_bullet_container
                and last_bullet_container[0] == "experience"
                and current_experience is not None
                and not is_section_heading(paragraph)
                and "\t" not in raw_text
                and not is_bullet_paragraph(paragraph)
            ):
                current_experience["bullets"][-1] = normalize_text(
                    f'{current_experience["bullets"][-1]} {text}'
                )
                idx += 1
                continue

        if current_section == "education":
            if "\t" in raw_text:
                left, right = [normalize_text(part) for part in raw_text.split("\t", 1)]
                degree, institution = left, ""
                if " at " in left.lower():
                    degree, institution = re.split(r"\sat\s", left, flags=re.IGNORECASE, maxsplit=1)
                elif " | " in left:
                    degree, institution = [part.strip() for part in left.split("|", 1)]
                education.append(
                    {"degree": degree.strip(), "institution": institution.strip(), "dates": right}
                )
            elif text:
                education.append({"degree": text, "institution": "", "dates": ""})
            idx += 1
            continue

        if current_section == "projects":
            if (
                last_bullet_container
                and last_bullet_container[0] == "project"
                and current_project is not None
                and not is_section_heading(paragraph)
                and "\t" not in raw_text
                and not is_bullet_paragraph(paragraph)
            ):
                current_project["bullets"][-1] = normalize_text(
                    f'{current_project["bullets"][-1]} {text}'
                )
                idx += 1
                continue
            if not is_bullet_paragraph(paragraph):
                project_name = normalize_text(raw_text.split("\t", 1)[0]) if "\t" in raw_text else text
                current_project = {"name": project_name, "bullets": []}
                projects.append(current_project)
                idx += 1
                continue
            if current_project is not None:
                current_project["bullets"].append(clean_bullet_text(text))
                last_bullet_container = ("project", current_project)
                idx += 1
                continue

        if current_section == "certifications":
            certifications.append(clean_bullet_text(text))
            idx += 1
            continue

        idx += 1

    if current_section == "skills" and skills_paragraphs:
        skills = parse_inline_skills(skills_paragraphs)

    return {
        "name": name,
        "summary": summary,
        "skills": skills,
        "experience": experience,
        "education": education,
        "projects": projects,
        "certifications": certifications,
    }


def save_resume_json(data: dict[str, Any], output_path: str | Path) -> None:
    """Save structured resume JSON to disk."""
    target = Path(output_path)
    existed = target.exists()
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log_file_event("MODIFIED" if existed else "CREATED", target, "Saved structured resume JSON")


def json_to_docx(data: dict[str, Any], output_path: str | Path) -> tuple[str, list[str]]:
    """Rebuild a resume DOCX from structured data."""
    return build_docx_from_json(data, output_path)
