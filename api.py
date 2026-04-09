"""
FastAPI backend for the Resume-Matcher tool.

New pipeline (keyword-coverage strategy)
-----------------------------------------
1. extract_jd_keywords()          → 25-50 ATS keyword phrases
2. embed_resume()                  → bullet embeddings
3. embed_keywords()                → keyword embeddings
4. compute_keyword_coverage(before)
5. keyword_driven_rewrite()        → vector-search assign → GPT inject verbatim
6. compute_keyword_coverage(after)
7. xml_patch_docx()                → formatting-preserving DOCX

Endpoints
---------
POST /api/generate   – run full pipeline, return coverage stats + changes
GET  /api/download/{filename} – download generated DOCX
GET  /api/health     – liveness check
"""

from __future__ import annotations

import base64
import os
import re
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()
load_dotenv(".env.local", override=True)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel

from pipeline.embedder import embed_keywords, embed_resume
from pipeline.evidence_map import build_evidence_map, omitted_unsupported_keywords, prioritized_supported_keywords
from pipeline.fit_controller import patch_with_fit_control
from pipeline.jd_extractor import extract_company_role, extract_jd_keywords, extract_jd_narrative_intent
from pipeline.layout_validator import layout_validation_failures, layout_validation_passed
from pipeline.parser import parse_resume
from pipeline.rewriter import keyword_driven_rewrite, narrative_driven_rewrite
from pipeline.narrative_planner import classify_and_plan, generate_grounded_summary, generate_narrative_summary
from pipeline.scorer import compute_keyword_coverage, score_tailored_resume
from pipeline.resume_html_export import try_generate_html_resume_pdf
from pipeline.xml_builder import collect_modified_sections

# ── constants ──────────────────────────────────────────────────────────────

# Vercel's filesystem is read-only everywhere except /tmp.
# When running on Vercel (VERCEL env var is set to "1"), write generated
# DOCX files to /tmp/outputs so the download endpoint can serve them.
# Locally, the existing "outputs/" directory is used as before.
_ON_VERCEL = bool(os.environ.get("VERCEL"))
# DOCX used for layout (styles/margins), parsing, skills lines, and xml_patch source.
RESUME_LAYOUT_DOCX = Path("data/Gnyani_resume_layout.docx")
OUTPUTS_DIR = Path("/tmp/outputs" if _ON_VERCEL else "outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ── app ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Resume Matcher API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],   # "null" covers file:// origin in browsers
    allow_credentials=False,       # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:40]


def _html_to_text(html: str) -> str:
    """Convert raw HTML into readable text."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _fetch_url_text_static(url: str) -> str:
    """Fetch raw text from a URL with a plain HTTP request."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    return _html_to_text(html)


def _fetch_url_text_browser(url: str) -> str:
    """Fetch rendered page text from a JavaScript-heavy URL."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0 Safari/537.36"
            )
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except PlaywrightTimeoutError:
                pass
            text = page.locator("body").inner_text(timeout=10000).strip()
            text = re.sub(r"\s{3,}", "\n\n", text)
            return text
        finally:
            context.close()
            browser.close()


def _fetch_url_text(url: str) -> str:
    """Fetch job-description text from a URL, with JS-rendered fallback."""
    JS_SPA_HINTS = [
        "explore.jobs.netflix", "jobs.lever.co", "boards.greenhouse.io",
        "jobs.ashbyhq.com", "careers.smartrecruiters.com", "workday.com",
        "icims.com", "taleo.net", "successfactors.com",
    ]
    is_known_spa = any(h in url for h in JS_SPA_HINTS)

    try:
        text = _fetch_url_text_static(url)
    except Exception as exc:
        text = ""
        static_error = exc
    else:
        static_error = None

    if len(text.split()) >= 150:
        return text

    try:
        rendered_text = _fetch_url_text_browser(url)
        if len(rendered_text.split()) >= 150:
            return rendered_text
    except Exception as exc:
        browser_error = exc
    else:
        browser_error = None

    if static_error:
        detail = f"Could not fetch URL: {static_error}"
    elif browser_error:
        detail = (
            "Could not extract meaningful text from the URL, even after "
            f"rendering it in a browser: {browser_error}"
        )
    else:
        detail = (
            "Could not extract enough job-description text from the URL after "
            "trying both static and browser-rendered fetches."
        )

    raise HTTPException(status_code=422, detail=detail)
    return text


def _enrich_rewrites(
    rewrites: list[dict[str, Any]],
    resume_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add 'section' label to each rewrite by cross-referencing the parsed resume."""
    enriched: list[dict[str, Any]] = []
    for rw in rewrites:
        section = ""
        norm = rw["original"].strip().lower()[:80]
        for exp in resume_json.get("experience", []):
            for bullet in exp.get("bullets", []):
                text = bullet if isinstance(bullet, str) else bullet.get("text", "")
                if text.strip().lower()[:80] == norm:
                    company = exp.get("company", "")
                    role = exp.get("role", "")
                    section = f"{company} — {role}".strip(" —")
                    break
            if section:
                break
        enriched.append({**rw, "section": section})
    return enriched


def _validation_report(layout_validation: dict[str, Any], tailoring_score: dict[str, Any], modified_sections: list[str], evidence_map: dict[str, Any]) -> dict[str, Any]:
    single_summary_in_place = all(
        layout_validation.get(key) is True
        for key in (
            "summary_heading_count_same",
            "summary_heading_position_same",
            "top_paragraphs_before_summary_same",
            "summary_body_in_place",
        )
    )
    line_fit_ratio = float(layout_validation.get("content_fill_ratio", 0) or 0)
    line_locked_fit_score = max(0.0, min(100.0, 100.0 - (abs(1.0 - line_fit_ratio) * 250.0)))
    return {
        "formatting_preserved": layout_validation_passed(layout_validation),
        "contact_line_preserved": layout_validation.get("contact_header_structure_same") is True,
        "single_summary_in_place": single_summary_in_place,
        "one_page": layout_validation.get("one_page") is True,
        "line_locked_fit_score": round(line_locked_fit_score, 1),
        "supported_keyword_coverage": tailoring_score.get("supported_keyword_coverage", 0.0),
        "omitted_unsupported_keywords": omitted_unsupported_keywords(evidence_map),
        "sections_modified": modified_sections,
    }


# ── request / response ─────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    jd_text: str = ""
    jd_url: str = ""


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Serve the browser UI.

    FileResponse reads frontend.html from the project root, which is bundled
    into the Vercel deployment package (the file is committed to git and not
    gitignored).  Falls back to a JSON message if the file is somehow absent.
    """
    html_path = Path("frontend.html")
    if html_path.exists():
        return FileResponse(
            str(html_path),
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
    return {"message": "Resume Matcher API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict[str, Any]:
    """Run the full keyword-coverage pipeline and return results."""

    # 1. Resolve JD text
    jd_text = req.jd_text.strip()
    if not jd_text and req.jd_url.strip():
        jd_text = _fetch_url_text(req.jd_url.strip())
    if not jd_text:
        raise HTTPException(status_code=400, detail="Provide jd_text or jd_url.")

    if not RESUME_LAYOUT_DOCX.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Resume layout DOCX not found at {RESUME_LAYOUT_DOCX}",
        )

    # 2. Extract JD keywords (25-50 ATS terms) + narrative intent
    keywords = extract_jd_keywords(jd_text)
    narrative_intent = extract_jd_narrative_intent(jd_text)
    identity = extract_company_role(jd_text)
    target_role = identity.get("role", "Role")

    # 3. Parse resume (same DOCX as layout / xml_patch source)
    resume_json = parse_resume(RESUME_LAYOUT_DOCX)

    # 4. Stage 1: build an evidence map so downstream rewriting only uses
    # grounded JD terms that the resume actually supports.
    evidence_map = build_evidence_map(resume_json, jd_text, keywords, target_role=target_role)
    supported_keywords = prioritized_supported_keywords(evidence_map, target_role=target_role)
    rewrite_keywords = supported_keywords or keywords

    # 5. Embed bullets + supported keywords (still used for coverage scoring)
    resume_with_emb = embed_resume(resume_json)
    keywords_with_emb = embed_keywords(rewrite_keywords)

    # 6. Keyword coverage BEFORE rewriting (supported terms only)
    before_cov = compute_keyword_coverage(resume_json, rewrite_keywords)

    # 7. Build narrative plan: classify supported keywords + assign per-bullet emphasis
    narrative_plan = classify_and_plan(
        resume_json,
        rewrite_keywords,
        narrative_intent,
        evidence_map=evidence_map,
    )

    # 8. Narrative-driven rewrite: story coherence + natural keyword incorporation
    updated_resume, rewrites = narrative_driven_rewrite(resume_json, narrative_plan)

    # 9. Separate grounded summary pass, with a safe fallback to the narrative summary
    summary = generate_grounded_summary(resume_json, target_role, evidence_map)
    if not summary:
        summary = generate_narrative_summary(resume_json, narrative_plan)
    if summary:
        updated_resume["summary"] = summary

    # 10. Weighted score AFTER rewriting
    after_cov = compute_keyword_coverage(updated_resume, rewrite_keywords)
    tailoring_score = score_tailored_resume(updated_resume, evidence_map, jd_text)

    # Enrich rewrites before patching so validation logs can name exact sections.
    enriched = _enrich_rewrites(rewrites, resume_json)
    modified_sections = collect_modified_sections(enriched, include_summary=bool(summary))

    # 11. Build output DOCX (XML-patch — preserves formatting)
    company = _sanitize(identity.get("company", "Company"))
    role = _sanitize(identity.get("role", "Role"))
    docx_filename = f"Gnyani_{company}_{role}.docx"
    output_path = OUTPUTS_DIR / docx_filename
    try:
        output_path, pdf_path, layout_validation, fitted_rewrites = patch_with_fit_control(
            RESUME_LAYOUT_DOCX,
            output_path,
            enriched,
            summary_text=summary or None,
            original_summary=(resume_json.get("summary") or None),
            modified_sections=modified_sections,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "PDF_SERVICE_URL" in msg:
            raise HTTPException(
                status_code=503,
                detail=(
                    "PDF conversion is not configured for this deployment. "
                    "Set the PDF_SERVICE_URL environment variable to your DOCX→PDF service URL "
                    "(see pipeline/pdf_converter.py). Underlying error: "
                    + msg
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=msg) from exc
    if not layout_validation_passed(layout_validation):
        failed = layout_validation_failures(layout_validation)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Template preservation failed",
                "failed_checks": failed,
                "layout_validation": layout_validation,
            },
        )
    docx_bytes = output_path.read_bytes()
    if not docx_bytes:
        raise HTTPException(status_code=500, detail="Generated DOCX is empty.")

    # 12. Primary user-facing PDF: HTML + Puppeteer (same mechanism as ResumeForge).
    # The DOCX pipeline still produced pdf_path for layout validation inside patch_with_fit_control.
    docx_pdf_bytes = pdf_path.read_bytes()
    html_pdf = try_generate_html_resume_pdf(
        updated_resume,
        docx_for_contact=output_path,
        output_pdf=OUTPUTS_DIR / f"{Path(docx_filename).stem}_html.pdf",
        repo_root=Path(__file__).resolve().parent,
    )
    if html_pdf is not None:
        download_path = html_pdf
        filename = html_pdf.name
        pdf_bytes = download_path.read_bytes()
        pdf_export_source = "html"
    else:
        download_path = pdf_path
        filename = pdf_path.name
        pdf_bytes = docx_pdf_bytes
        pdf_export_source = "docx_fallback"
    print("sending docx to pdf converter, size:", len(docx_bytes))
    print("pdf bytes (download):", len(pdf_bytes), "source:", pdf_export_source)
    print("docx-pipeline pdf bytes (validation only):", len(docx_pdf_bytes))

    # 13. Enrich rewrites with section labels
    print("modified sections:", ", ".join(modified_sections) or "None")
    print("layout validation:", layout_validation)
    validation_report = _validation_report(layout_validation, tailoring_score, modified_sections, evidence_map)

    # 14. Build per-keyword before/after diff
    before_map = {r["keyword"]: r["matched"] for r in before_cov["keywords"]}
    after_map  = {r["keyword"]: r["matched"] for r in after_cov["keywords"]}

    # Map each keyword → the change card index where it was injected (for UI scroll-to)
    kw_to_change_idx: dict[str, int] = {}
    for i, change in enumerate(fitted_rewrites):
        for kw in change.get("injected_keywords", []):
            kw_to_change_idx[kw] = i

    keyword_diff = [
        {
            "keyword": kw,
            "before":     before_map.get(kw, False),
            "after":      after_map.get(kw, False),
            "gained":     (not before_map.get(kw, False)) and after_map.get(kw, False),
            "change_idx": kw_to_change_idx.get(kw, None),  # None if already present / still missing
        }
        for kw in rewrite_keywords
    ]

    # Encode file as base64 so the frontend can download without a second request
    # (avoids Vercel cold-start "File not found" on the separate download endpoint)
    final_bytes = download_path.read_bytes()
    if not final_bytes:
        raise HTTPException(status_code=500, detail=f"Generated output file is empty: {filename}")
    file_b64 = base64.b64encode(final_bytes).decode("ascii")
    print("final filename:", filename)
    print("docx bytes:", len(docx_bytes) if docx_bytes else 0)
    print("pdf bytes:", len(pdf_bytes) if pdf_bytes else 0)
    print("file_b64 length:", len(file_b64) if file_b64 else 0)

    return {
        "filename": filename,
        "file_b64": file_b64,
        "pdf_export_source": pdf_export_source,
        "target_role": target_role,
        "evidence_map": evidence_map,
        "tailoring_score": tailoring_score,
        "layout_validation": layout_validation,
        "validation_report": validation_report,
        # Primary metric: keyword coverage
        "keywords_total":   len(rewrite_keywords),
        "keywords_before":  before_cov["matched"],
        "keywords_after":   after_cov["matched"],
        "coverage_before":  before_cov["pct"],
        "coverage_after":   after_cov["pct"],
        "coverage_gain":    round(after_cov["pct"] - before_cov["pct"], 1),
        # Per-keyword breakdown
        "keyword_diff": keyword_diff,
        # Changes (bullets rewritten)
        "rewrites_count": len(fitted_rewrites),
        "changes": fitted_rewrites,
    }


@app.get("/api/download/{filename}")
def download(filename: str):
    """Download the generated resume (PDF or DOCX)."""
    safe_name = Path(filename).name
    path = OUTPUTS_DIR / safe_name
    if not path.exists() or path.suffix not in (".pdf", ".docx"):
        raise HTTPException(status_code=404, detail="File not found.")
    media_type = (
        "application/pdf"
        if path.suffix == ".pdf"
        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    return FileResponse(path=str(path), filename=safe_name, media_type=media_type)
