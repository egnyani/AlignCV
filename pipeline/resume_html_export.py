"""HTML + Puppeteer PDF export (ResumeForge-style) for tailored resume JSON."""

from __future__ import annotations

import html
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

from playwright.sync_api import sync_playwright

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


def _ensure_href(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return ""
    if re.match(r"^https?://", u, re.I):
        return u
    return f"https://{u}"


def _inline_bold_html(text: str) -> str:
    parts = re.split(r"\*\*(.+?)\*\*", text)
    if len(parts) == 1:
        return html.escape(text)
    chunks: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            chunks.append(f"<strong>{html.escape(part)}</strong>")
        else:
            chunks.append(html.escape(part))
    return "".join(chunks)


def _resume_data_to_html(data: dict[str, Any]) -> str:
    """
    Static HTML mirroring resume_export/ResumeTemplate.tsx + resumeLayout styles.
    Used for PDF generation where Node/npm is unavailable (e.g. Vercel Python).
    """
    c = data.get("contact") or {}
    name = html.escape(str(c.get("name") or "Candidate"))
    phone = html.escape(str(c.get("phone") or ""))
    email = html.escape(str(c.get("email") or ""))
    li_href = html.escape(_ensure_href(str(c.get("linkedin") or "linkedin.com")))
    gh = (c.get("github") or "").strip()
    gh_href = html.escape(_ensure_href(gh)) if gh else ""

    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>",
        "<style>",
        "*{margin:0;padding:0;box-sizing:border-box;}body{background:#fff;}",
        ".page{width:100%;max-width:750px;margin:0 auto;background:#fff;padding:32px;",
        "font-family:'Times New Roman',Times,Georgia,serif;font-size:11pt;line-height:1.35;color:#000;}",
        "h1{font-size:18pt;font-weight:700;text-align:center;margin:0 0 3px;letter-spacing:.5px;}",
        ".contact{font-size:10pt;color:#457885;text-align:center;margin:0 0 6px;letter-spacing:.1px;}",
        "a.contact-link{color:#457885;text-decoration:underline;}",
        "h2{font-size:10.5pt;font-weight:700;font-variant:small-caps;letter-spacing:.5px;",
        "border-bottom:1px solid #000;padding-bottom:1px;margin:7px 0 3px;text-transform:uppercase;}",
        "ul.summary{list-style:disc;margin-left:20px;margin-top:0;padding:0;}",
        "ul.summary li{font-size:11pt;line-height:1.45;margin-bottom:2px;text-align:justify;}",
        ".row{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0;}",
        ".company{font-size:10.5pt;font-weight:700;}",
        ".date{font-size:11pt;white-space:nowrap;font-weight:400;}",
        ".title{font-size:11pt;font-style:italic;margin-bottom:1px;}",
        ".loc{font-size:10pt;font-style:normal;margin-bottom:1px;}",
        "ul.bullets{list-style:disc;margin-left:20px;padding:0;margin-top:1px;}",
        "ul.bullets li{font-size:11pt;line-height:1.45;margin-bottom:2px;text-align:justify;}",
        ".skill{font-size:11pt;line-height:1.55;margin-bottom:1px;}",
        ".skill b{font-weight:700;}",
        "</style></head><body>",
        f'<div class="page"><h1>{name}</h1><p class="contact">',
        f"{phone}&nbsp;&nbsp;|&nbsp;&nbsp;",
        f'<a class="contact-link" href="mailto:{email}">{email}</a>',
        "&nbsp;&nbsp;|&nbsp;&nbsp;",
        f'<a class="contact-link" href="{li_href}">LinkedIn</a>',
    ]
    if gh:
        parts.append("&nbsp;&nbsp;|&nbsp;&nbsp;")
        parts.append(f'<a class="contact-link" href="{gh_href}">Github</a>')
    parts.append("</p>")

    parts.append("<section><h2>Summary</h2><ul class=\"summary\">")
    for pt in data.get("summary") or []:
        parts.append(f"<li>{_inline_bold_html(str(pt))}</li>")
    parts.append("</ul></section>")

    parts.append("<section><h2>Experience</h2>")
    for role in data.get("experience") or []:
        if not isinstance(role, dict):
            continue
        parts.append("<div style=\"margin-bottom:6px;\">")
        parts.append("<div class=\"row\">")
        parts.append(f"<span class=\"company\">{html.escape(str(role.get('company') or ''))}</span>")
        parts.append(
            f"<span class=\"date\">{html.escape(str(role.get('start') or ''))} - "
            f"{html.escape(str(role.get('end') or ''))}</span></div>"
        )
        parts.append(f"<div class=\"title\">{html.escape(str(role.get('title') or ''))}</div>")
        loc = (role.get("location") or "").strip()
        if loc:
            parts.append(f"<div class=\"loc\">{html.escape(loc)}</div>")
        parts.append("<ul class=\"bullets\">")
        for bullet in role.get("bullets") or []:
            parts.append(f"<li>{_inline_bold_html(str(bullet))}</li>")
        parts.append("</ul></div>")
    parts.append("</section>")

    parts.append("<section><h2>Technical Skills</h2>")
    for cat, vals in (data.get("skills") or {}).items():
        if not isinstance(vals, list):
            continue
        joined = html.escape(", ".join(str(v) for v in vals if str(v).strip()))
        parts.append(
            f'<div class="skill"><b>{html.escape(str(cat))}:</b> {joined}</div>'
        )
    parts.append("</section>")

    parts.append("<section><h2>Education</h2>")
    for edu in data.get("education") or []:
        if not isinstance(edu, dict):
            continue
        parts.append("<div style=\"margin-bottom:4px;\">")
        parts.append("<div class=\"row\">")
        parts.append(f"<span class=\"company\">{html.escape(str(edu.get('school') or ''))}</span>")
        parts.append(
            f"<span class=\"date\">{html.escape(str(edu.get('start') or ''))} - "
            f"{html.escape(str(edu.get('end') or ''))}</span></div>"
        )
        parts.append("<div class=\"row\">")
        parts.append(f"<span class=\"title\" style=\"font-style:normal\">{html.escape(str(edu.get('degree') or ''))}</span>")
        parts.append(
            f"<span style=\"font-size:11pt;color:#555;white-space:nowrap\">"
            f"{html.escape(str(edu.get('location') or ''))}</span></div></div>"
        )
    parts.append("</section>")

    projects = data.get("projects") or []
    if projects:
        parts.append("<section><h2>Projects</h2>")
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            parts.append("<div style=\"margin-bottom:6px;\">")
            parts.append("<div class=\"row\">")
            parts.append(f"<span class=\"company\">{html.escape(str(proj.get('name') or ''))}</span>")
            parts.append(f"<span class=\"date\">{html.escape(str(proj.get('date') or ''))}</span></div>")
            parts.append("<ul class=\"bullets\">")
            for bullet in proj.get("bullets") or []:
                parts.append(f"<li>{_inline_bold_html(str(bullet))}</li>")
            parts.append("</ul></div>")
        parts.append("</section>")

    parts.append("</div></body></html>")
    return "".join(parts)


def _generate_html_resume_pdf_playwright(resume_data: dict[str, Any], output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    html_doc = _resume_data_to_html(resume_data)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.set_viewport_size({"width": 816, "height": 600})
            page.set_content(html_doc, wait_until="load", timeout=60_000)
            pdf_height = page.evaluate(
                """() => {
                    const c = document.body.firstElementChild;
                    if (!c) return Math.ceil(document.body.scrollHeight);
                    return Math.ceil(c.getBoundingClientRect().height);
                }"""
            )
            page.pdf(
                path=str(output_pdf),
                width="8.5in",
                height=f"{pdf_height}px",
                print_background=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
            )
        finally:
            browser.close()


def _generate_html_resume_pdf_npm(
    resume_data: dict[str, Any],
    output_pdf: Path,
    repo_root: Path,
) -> None:
    """Local dev fallback: Node + Puppeteer (matches resume_export/exportPdf.ts)."""
    npm_cmd = "npm.cmd" if sys.platform.startswith("win") else "npm"
    fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix="resume_export_")
    os.close(fd)
    tmp_input = Path(tmp_name)
    try:
        tmp_input.write_text(json.dumps(resume_data, ensure_ascii=True, indent=2), encoding="utf-8")
        try:
            rel_out = str(output_pdf.relative_to(repo_root.resolve()))
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
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr or result.stdout or "npm generate:resume-pdf failed",
            )
    finally:
        if tmp_input.exists():
            try:
                tmp_input.unlink()
            except OSError:
                pass


def generate_html_resume_pdf(
    resume_data: dict[str, Any],
    output_pdf: str | Path,
    repo_root: str | Path | None = None,
) -> Path:
    """
    Render resume JSON to PDF: Playwright (Chromium) first — works on Vercel where
    ``npm`` is not bundled with the Python function. Falls back to ``npm run
    generate:resume-pdf`` when Chromium is missing locally.
    """
    output_pdf = Path(output_pdf).resolve()
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]

    try:
        _generate_html_resume_pdf_playwright(resume_data, output_pdf)
    except Exception as exc:
        logger.warning("Playwright HTML resume PDF failed (%s); trying npm fallback", exc)
        _generate_html_resume_pdf_npm(resume_data, output_pdf, root)

    if not output_pdf.exists() or output_pdf.stat().st_size == 0:
        raise RuntimeError(f"PDF not written: {output_pdf}")
    return output_pdf


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
