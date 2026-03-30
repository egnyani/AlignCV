"""
XML-patching DOCX builder.

Instead of rebuilding from scratch (builder.py), this module patches the
original resume DOCX in-place: opens the ZIP, rewrites only the matching
bullet paragraph XML runs, and saves a new ZIP.  All fonts, colours, bullet
styles, margins, headers, and footers are preserved exactly.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from lxml import etree


_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_NS}}}"
_XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def _normalize(text: str) -> str:
    """Lowercase, strip bullets and extra whitespace for fuzzy matching."""
    text = re.sub(r"^[\-\*\u2022\u2013]\s*", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _para_text(p: etree._Element) -> str:
    """Return concatenated run text of a paragraph element."""
    return "".join(
        "".join(t.text or "" for t in r.findall(f"{_W}t"))
        for r in p.findall(f".//{_W}r")
    )


def _body_paragraphs(body: etree._Element) -> list[etree._Element]:
    """Return only top-level body paragraphs in document order."""
    return [child for child in body if child.tag == f"{_W}p"]


def _nonempty_body_paragraphs(body: etree._Element) -> list[tuple[int, etree._Element, str]]:
    items: list[tuple[int, etree._Element, str]] = []
    for idx, paragraph in enumerate(_body_paragraphs(body)):
        raw = _para_text(paragraph)
        if raw.strip():
            items.append((idx, paragraph, raw))
    return items


def _find_heading_index(items: list[tuple[int, etree._Element, str]], heading: str) -> int:
    target = _normalize(heading)
    for idx, _paragraph, raw in items:
        if _normalize(raw) == target:
            return idx
    raise RuntimeError(f"Template anchor not found for heading: {heading}")


def _next_nonempty_index(items: list[tuple[int, etree._Element, str]], after_idx: int) -> int:
    for idx, _paragraph, raw in items:
        if idx > after_idx and raw.strip():
            return idx
    raise RuntimeError(f"Expected paragraph after template anchor index {after_idx}")


def _template_anchors(body: etree._Element) -> dict[str, Any]:
    """Locate exact locked-template paragraph anchors."""
    items = _nonempty_body_paragraphs(body)
    if len(items) < 4:
        raise RuntimeError("Template is missing required top-of-document paragraphs.")

    summary_heading_idx = _find_heading_index(items, "SUMMARY")
    summary_body_idx = _next_nonempty_index(items, summary_heading_idx)

    return {
        "name_idx": items[0][0],
        "contact_idx": items[1][0],
        "summary_heading_idx": summary_heading_idx,
        "summary_body_idx": summary_body_idx,
        "work_heading_idx": _find_heading_index(items, "WORK EXPERIENCE"),
        "education_heading_idx": _find_heading_index(items, "EDUCATION"),
        "project_heading_idx": _find_heading_index(items, "PROJECT"),
        "nonempty_items": items,
        "protected_idxs": {
            items[0][0],
            items[1][0],
            summary_heading_idx,
            _find_heading_index(items, "WORK EXPERIENCE"),
            _find_heading_index(items, "EDUCATION"),
            _find_heading_index(items, "PROJECT"),
        },
    }


def _replace_para_text(p: etree._Element, new_text: str) -> None:
    """Replace paragraph text while preserving existing runs and formatting."""
    text_nodes = p.findall(f".//{_W}t")
    if not text_nodes:
        return  # can't patch a run-less paragraph

    first = text_nodes[0]
    first.text = new_text
    first.set(_XML_SPACE, "preserve")
    for node in text_nodes[1:]:
        node.text = ""


def collect_modified_sections(
    rewrites: list[dict[str, str]],
    include_summary: bool = False,
) -> list[str]:
    """Collect a stable list of human-readable sections touched by the patch."""
    modified: list[str] = []
    seen: set[str] = set()
    if include_summary:
        seen.add("Summary")
        modified.append("Summary")
    for rw in rewrites:
        section = str(rw.get("section") or rw.get("company") or "Experience").strip()
        if section and section not in seen:
            seen.add(section)
            modified.append(section)
    return modified


def xml_patch_docx(
    source_path: str | Path,
    rewrites: list[dict[str, str]],
    output_path: str | Path,
    summary_text: str | None = None,
    original_summary: str | None = None,
) -> Path:
    """
    Copy *source_path* DOCX and patch bullet text via XML rewriting.

    Parameters
    ----------
    source_path:
        Original resume DOCX (preserved as template).
    rewrites:
        List of ``{"original": str, "rewritten": str}`` dicts from the pipeline.
    output_path:
        Where to save the tailored DOCX.

    Returns
    -------
    Path of the written output DOCX.
    """
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build normalised-key → new-text lookup for bullet paragraphs only.
    rewrite_map: dict[str, str] = {}
    for rw in rewrites:
        orig = rw.get("original", "")
        new = rw.get("rewritten", "")
        if orig and new:
            key = _normalize(orig)
            rewrite_map[key] = new

    # Read all files from the source ZIP
    all_files: dict[str, bytes] = {}
    with zipfile.ZipFile(source_path, "r") as zin:
        for name in zin.namelist():
            all_files[name] = zin.read(name)

    # Parse and patch document.xml
    xml_bytes = all_files["word/document.xml"]
    root = etree.fromstring(xml_bytes)
    body = root.find(f"{_W}body")
    if body is None:
        raise RuntimeError("Malformed DOCX: no <w:body> found")

    anchors = _template_anchors(body)
    body_paragraphs = _body_paragraphs(body)
    patched = 0
    protected_idxs: set[int] = anchors["protected_idxs"]

    # Replace only the original summary body paragraph immediately under SUMMARY.
    if summary_text:
        summary_idx = anchors["summary_body_idx"]
        summary_para = body_paragraphs[summary_idx]
        _replace_para_text(summary_para, summary_text)
        patched += 1

    # Replace only existing experience/project bullet paragraphs; never touch the
    # sacred top-of-document paragraphs or section heading paragraphs.
    for idx, p in enumerate(body_paragraphs):
        if idx in protected_idxs or idx == anchors["summary_body_idx"]:
            continue
        raw = _para_text(p)
        if not raw.strip():
            continue

        norm = _normalize(raw)
        new_text = rewrite_map.get(norm)
        if new_text:
            _replace_para_text(p, new_text)
            patched += 1

    new_xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    all_files["word/document.xml"] = new_xml

    # Write new DOCX
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name, data in all_files.items():
            zout.writestr(name, data)

    return output_path
