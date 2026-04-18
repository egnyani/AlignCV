"""HTML + Puppeteer PDF export (ResumeForge-style) for tailored resume JSON."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _split_dates(dates: str) -> tuple[str, str]:
    """Split a date range like '2023 - Present' into start/end."""
    if not dates or not dates.strip():
        return "", ""
    text = dates.strip()
    for sep in (" – ", " — ", " - ", "–", "—"):
        if sep in text:
            parts = text.split(sep, 1)
            return parts[0].strip(), parts[1].strip()
    return text, ""


def parse_contact_from_docx(docx_path: str | Path) -> dict[str, str]:
    """Extract phone, email, linkedin, github from the contact line in a resume DOCX."""
    from docx import Document

    from pipeline.parser import looks_like_contact_line, normalize_text

    path = Path(docx_path)
    if not path.exists():
        return _empty_contact()

    document = Document(str(path))
    for paragraph in document.paragraphs:
        text = normalize_text(paragraph.text)
        if text and looks_like_contact_line(text):
            merged = _parse_contact_line(text)
            for key, val in _contact_fields_from_hyperlinks(path).items():
                if val:
                    merged[key] = val
            return merged
    return _empty_contact()


def _empty_contact() -> dict[str, str]:
    return {"phone": "", "email": "", "linkedin": "", "github": ""}


_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
_REL_OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_HYPERLINK_REL_TYPE = f"{_REL_OFFICE}/hyperlink"


def _load_doc_hyperlink_targets(docx_path: Path) -> dict[str, str]:
    """Map relationship Id -> hyperlink Target URL (document.xml.rels)."""
    with zipfile.ZipFile(docx_path) as zf:
        rels_xml = zf.read("word/_rels/document.xml.rels")
    root = ET.fromstring(rels_xml)
    out: dict[str, str] = {}
    for rel in root.findall(f"{{{_REL_PKG}}}Relationship"):
        if rel.get("Type") != _HYPERLINK_REL_TYPE:
            continue
        rid = rel.get("Id")
        target = (rel.get("Target") or "").strip()
        if rid and target:
            out[rid] = target
    return out


def _body_paragraph_elements(docx_path: Path) -> list[ET.Element]:
    with zipfile.ZipFile(docx_path) as zf:
        doc_xml = zf.read("word/document.xml")
    root = ET.fromstring(doc_xml)
    body = root.find(f".//{{{_W_NS}}}body")
    if body is None:
        return []
    return [c for c in body if c.tag == f"{{{_W_NS}}}p"]


def _paragraph_plain_text(p: ET.Element) -> str:
    return "".join(t.text or "" for t in p.iter(f"{{{_W_NS}}}t"))


def _hyperlink_rid(h: ET.Element) -> str | None:
    key_id = f"{{{_REL_OFFICE}}}id"
    rid = h.get(key_id)
    if rid:
        return rid
    for attr, val in h.attrib.items():
        if attr.endswith("}id"):
            return val
    return None


def _contact_fields_from_hyperlinks(docx_path: Path) -> dict[str, str]:
    """
    Read GitHub / LinkedIn / mailto targets from Word hyperlinks on the contact row.

    Display text is often \"LinkedIn\" / \"Github\" without the URL in runs; the real
    URL only appears in document.xml.rels.
    """
    path = Path(docx_path)
    if not path.is_file():
        return _empty_contact()

    from pipeline.parser import looks_like_contact_line, normalize_text

    rid_to_target = _load_doc_hyperlink_targets(path)
    out = _empty_contact()
    for p in _body_paragraph_elements(path):
        plain = normalize_text(_paragraph_plain_text(p))
        if not plain or not looks_like_contact_line(plain):
            continue
        for h in p.findall(f".//{{{_W_NS}}}hyperlink"):
            rid = _hyperlink_rid(h)
            if not rid:
                continue
            raw = (rid_to_target.get(rid) or "").strip()
            if not raw:
                continue
            low = raw.lower()
            if low.startswith("mailto:"):
                addr = raw.split(":", 1)[1].strip()
                if addr:
                    out["email"] = addr
            elif "linkedin.com" in low:
                out["linkedin"] = raw.replace("https://", "").replace("http://", "")
            elif "github.com" in low:
                out["github"] = raw.replace("https://", "").replace("http://", "")
        break
    return out


def _parse_contact_line(text: str) -> dict[str, str]:
    out = _empty_contact()
    email_m = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    if email_m:
        out["email"] = email_m.group(0).strip()
    phone_m = re.search(r"\+?\d[\d\s().-]{8,}\d", text)
    if phone_m:
        out["phone"] = re.sub(r"\s+", " ", phone_m.group(0).strip())

    lower = text.lower()
    for pattern in (
        r"(https?://(?:www\.)?linkedin\.com/[^\s|]+)",
        r"((?:www\.)?linkedin\.com/[^\s|]+)",
    ):
        m = re.search(pattern, text, re.I)
        if m:
            link = m.group(1).strip().rstrip(".,;")
            out["linkedin"] = link.replace("https://", "").replace("http://", "")
            break

    for pattern in (
        r"(https?://(?:www\.)?github\.com/[^\s|]+)",
        r"((?:www\.)?github\.com/[^\s|]+)",
    ):
        m = re.search(pattern, text, re.I)
        if m:
            link = m.group(1).strip().rstrip(".,;")
            out["github"] = link.replace("https://", "").replace("http://", "")
            break

    if "linkedin.com" in lower and not out["linkedin"]:
        out["linkedin"] = "linkedin.com/in/"
    return out


def semantic_resume_to_resume_data(
    resume: dict[str, Any],
    contact: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Map semantic-profile-optimizer resume JSON to ResumeForge-compatible export shape.
    """
    contact = contact or _empty_contact()
    name = (resume.get("name") or "Candidate").strip()

    raw_summary = resume.get("summary") or ""
    if isinstance(raw_summary, list):
        summary_list = [str(s).strip() for s in raw_summary if str(s).strip()]
    else:
        s = str(raw_summary).strip()
        if not s:
            summary_list = []
        else:
            parts = re.split(r"(?<=[.!?])\s+", s)
            summary_list = [p.strip() for p in parts if p.strip()] or [s]

    experience_out: list[dict[str, Any]] = []
    for i, exp in enumerate(resume.get("experience") or []):
        if not isinstance(exp, dict):
            continue
        dates = str(exp.get("dates") or "")
        start, end = _split_dates(dates)
        experience_out.append(
            {
                "id": f"exp-{i}",
                "company": str(exp.get("company") or "").strip(),
                "title": str(exp.get("role") or "").strip(),
                "location": str(exp.get("location") or "").strip(),
                "start": start,
                "end": end or "Present",
                "bullets": [str(b).strip() for b in (exp.get("bullets") or []) if str(b).strip()],
            }
        )

    education_out: list[dict[str, Any]] = []
    for edu in resume.get("education") or []:
        if not isinstance(edu, dict):
            continue
        dates = str(edu.get("dates") or "")
        start, end = _split_dates(dates)
        education_out.append(
            {
                "degree": str(edu.get("degree") or "").strip(),
                "school": str(edu.get("institution") or "").strip(),
                "location": "",
                "start": start,
                "end": end,
            }
        )

    projects_out: list[dict[str, Any]] = []
    for i, proj in enumerate(resume.get("projects") or []):
        if not isinstance(proj, dict):
            continue
        projects_out.append(
            {
                "id": f"proj-{i}",
                "name": str(proj.get("name") or "").strip(),
                "date": "",
                "bullets": [str(b).strip() for b in (proj.get("bullets") or []) if str(b).strip()],
                "tags": [],
            }
        )

    skills = resume.get("skills") or {}
    if not isinstance(skills, dict):
        skills = {}
    skills_clean: dict[str, list[str]] = {}
    for key, vals in skills.items():
        if isinstance(vals, list):
            skills_clean[str(key)] = [str(v).strip() for v in vals if str(v).strip()]
        else:
            skills_clean[str(key)] = []

    return {
        "contact": {
            "name": name,
            "phone": contact.get("phone") or "",
            "email": contact.get("email") or "",
            "linkedin": contact.get("linkedin") or "linkedin.com",
            "github": (contact.get("github") or "").strip() or None,
        },
        "summary": summary_list,
        "experience": experience_out,
        "education": education_out,
        "skills": skills_clean,
        "projects": projects_out,
    }


def generate_html_resume_pdf(
    resume_data: dict[str, Any],
    output_pdf: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Write resume JSON to a temp file, run Node/tsx Puppeteer exporter, return path to PDF.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    output_pdf = Path(output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    npm_cmd = "npm.cmd" if sys.platform.startswith("win") else "npm"
    fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix="resume_export_")
    os.close(fd)
    tmp_input = Path(tmp_name)
    try:
        tmp_input.write_text(json.dumps(resume_data, ensure_ascii=True, indent=2), encoding="utf-8")
        try:
            rel_out = str(output_pdf.relative_to(root.resolve()))
        except ValueError:
            rel_out = str(output_pdf)

        cmd = [
            npm_cmd,
            "run",
            "generate:resume-pdf",
            "--",
            "--input",
            str(tmp_input.resolve()),
            "--output",
            rel_out,
        ]
        result = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or result.stdout or "npm generate:resume-pdf failed",
            )
        if not output_pdf.exists():
            raise RuntimeError(f"PDF not written: {output_pdf}")
        return output_pdf
    finally:
        if tmp_input.exists():
            try:
                tmp_input.unlink()
            except OSError:
                pass


def try_generate_html_resume_pdf(
    resume: dict[str, Any],
    docx_for_contact: str | Path,
    output_pdf: str | Path,
    repo_root: str | Path | None = None,
) -> Path | None:
    """
    Best-effort HTML PDF. Returns None on failure (caller may fall back to DOCX PDF).
    """
    try:
        contact = parse_contact_from_docx(docx_for_contact)
        data = semantic_resume_to_resume_data(resume, contact)
        return generate_html_resume_pdf(data, output_pdf, repo_root=repo_root)
    except Exception as exc:
        logger.warning("HTML resume PDF export failed: %s", exc)
        return None
