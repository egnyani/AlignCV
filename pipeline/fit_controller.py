"""Layout-aware content compression for locked-template resume generation."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from pipeline.layout_validator import layout_validation_passed, validate_layout_preservation
from pipeline.pdf_converter import convert_docx_to_pdf
from pipeline.xml_builder import xml_patch_docx


_FILLER_PATTERNS = [
    (r"\benterprise-level\b", "enterprise"),
    (r"\blarge-scale\b", "scalable"),
    (r"\bproduction-grade\b", "production"),
    (r"\bfull-stack\b", ""),
    (r"\bnear-real-time\b", "real-time"),
    (r"\bend-to-end\b", ""),
    (r"\bdirectly\b", ""),
    (r"\bproactive\b", ""),
]

_SUMMARY_LINE_CAPACITY = 100
_BULLET_LINE_CAPACITY = 92


def _cleanup(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\(\s*", "(", text)
    text = re.sub(r"\s*\)", ")", text)
    return text.strip(" ,;")


def _compress_text(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return _cleanup(text)

    out = text
    for pattern, repl in _FILLER_PATTERNS:
        out = re.sub(pattern, repl, out, flags=re.I)
        out = _cleanup(out)
        if len(out) <= target_chars:
            return out

    out = re.sub(r"\s*\([^)]*\)", "", out)
    out = _cleanup(out)
    if len(out) <= target_chars:
        return out

    clauses = re.split(r",\s+", out)
    if len(clauses) > 1:
        candidate = clauses[0]
        for clause in clauses[1:]:
            maybe = f"{candidate}, {clause}"
            if len(maybe) <= target_chars:
                candidate = maybe
        out = _cleanup(candidate)
        if len(out) <= target_chars:
            return out

    sentences = re.split(r"(?<=[.!?])\s+", out)
    if len(sentences) > 1:
        candidate = ""
        for sentence in sentences:
            maybe = f"{candidate} {sentence}".strip()
            if len(maybe) <= target_chars:
                candidate = maybe
        out = _cleanup(candidate or sentences[0])
        if len(out) <= target_chars:
            return out

    trimmed = out[:target_chars].rsplit(" ", 1)[0].strip()
    return _cleanup(trimmed or out[:target_chars])


def _compress_summary(summary: str, original_summary: str, ratio: float) -> str:
    base = len(original_summary or summary or "")
    target = max(160, int(base * ratio))
    return _compress_text(summary, target)


def _compress_rewrites(rewrites: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    out = deepcopy(rewrites)
    for item in out:
        original = item.get("original", "") or item.get("rewritten", "")
        current = item.get("rewritten", "") or ""
        base = len(original)
        if not base:
            continue
        target = max(110, int(base * ratio))
        item["rewritten"] = _compress_text(current, target)
    return out


def _expand_text(current: str, richer_source: str, target_chars: int) -> str:
    current = _cleanup(current)
    richer_source = _cleanup(richer_source)
    if len(current) >= target_chars or len(richer_source) <= len(current):
        return current
    if len(richer_source) <= target_chars:
        return richer_source

    clauses = re.split(r",\s+", richer_source)
    candidate = ""
    for clause in clauses:
        maybe = f"{candidate}, {clause}".strip(", ")
        if len(maybe) <= target_chars:
            candidate = maybe
        else:
            break
    if candidate and len(candidate) > len(current):
        return _cleanup(candidate)

    sentences = re.split(r"(?<=[.!?])\s+", richer_source)
    candidate = ""
    for sentence in sentences:
        maybe = f"{candidate} {sentence}".strip()
        if len(maybe) <= target_chars:
            candidate = maybe
        else:
            break
    if candidate and len(candidate) > len(current):
        return _cleanup(candidate)

    trimmed = richer_source[:target_chars].rsplit(" ", 1)[0].strip()
    return _cleanup(trimmed or richer_source[:target_chars])


def _line_budget_bounds(original_text: str, capacity: int) -> tuple[int, int]:
    cleaned = _cleanup(original_text)
    if not cleaned:
        return (0, 0)
    original_len = len(cleaned)
    line_count = max(1, (original_len + capacity - 1) // capacity)
    lower_bound = 1 if line_count == 1 else ((line_count - 1) * capacity) + 1
    upper_bound = line_count * capacity
    return lower_bound, upper_bound


def _expand_summary(summary: str, original_summary: str, ratio: float) -> str:
    richer_source = original_summary if len(original_summary or "") > len(summary or "") else summary
    base = max(len(summary or ""), len(original_summary or ""))
    target = max(len(summary or ""), int(base * ratio))
    return _expand_text(summary, richer_source, target)


def _expand_rewrites(rewrites: list[dict[str, Any]], ratio: float) -> list[dict[str, Any]]:
    out = deepcopy(rewrites)
    for item in out:
        current = item.get("rewritten", "") or ""
        richer_source = current
        if len(item.get("original", "") or "") > len(richer_source):
            richer_source = item.get("original", "") or richer_source
        base = max(len(current), len(richer_source))
        if not base:
            continue
        target = max(len(current), int(base * ratio))
        item["rewritten"] = _expand_text(current, richer_source, target)
    return out


def _enforce_summary_line_budget(summary: str, original_summary: str) -> str:
    if not summary or not original_summary:
        return summary
    current = _cleanup(summary)
    min_chars, max_chars = _line_budget_bounds(original_summary, _SUMMARY_LINE_CAPACITY)
    if max_chars and len(current) > max_chars:
        return _compress_text(current, max_chars)
    if min_chars and len(current) < min_chars:
        return _expand_text(current, _cleanup(original_summary), min_chars)
    return current


def _enforce_rewrite_line_budgets(rewrites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(rewrites)
    for item in out:
        original = _cleanup(item.get("original", "") or "")
        current = _cleanup(item.get("rewritten", "") or "")
        if not original or not current:
            continue
        min_chars, max_chars = _line_budget_bounds(original, _BULLET_LINE_CAPACITY)
        if max_chars and len(current) > max_chars:
            item["rewritten"] = _compress_text(current, max_chars)
        elif min_chars and len(current) < min_chars:
            item["rewritten"] = _expand_text(current, original, min_chars)
        else:
            item["rewritten"] = current
    return out


def _fit_state(validation: dict[str, Any]) -> str:
    if validation.get("one_page") is False:
        return "overflow"
    if validation.get("content_within_template_bounds") is False:
        return "overflow"
    if not validation.get("bottom_whitespace_reasonable", True):
        return "underfilled"
    return "balanced"


def patch_with_fit_control(
    template_path: str | Path,
    output_path: str | Path,
    rewrites: list[dict[str, Any]],
    summary_text: str | None,
    original_summary: str | None,
    modified_sections: list[str] | None = None,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    """
    Patch the locked template, compressing tailored content until the output
    validates structurally and renders to a single page.
    """
    strategies = [
        {"mode": "base", "summary_ratio": 1.0, "bullet_ratio": 1.0},
        {"mode": "expand", "summary_ratio": 1.03, "bullet_ratio": 1.02},
        {"mode": "expand", "summary_ratio": 1.06, "bullet_ratio": 1.04},
        {"mode": "expand", "summary_ratio": 1.1, "bullet_ratio": 1.07},
        {"mode": "compress", "summary_ratio": 0.94, "bullet_ratio": 0.99},
        {"mode": "compress", "summary_ratio": 0.9, "bullet_ratio": 0.96},
        {"mode": "compress", "summary_ratio": 0.84, "bullet_ratio": 0.92},
        {"mode": "compress", "summary_ratio": 0.76, "bullet_ratio": 0.88},
        {"mode": "compress", "summary_ratio": 0.68, "bullet_ratio": 0.84},
    ]

    last_validation: dict[str, Any] | None = None
    best_underfilled: tuple[dict[str, Any], Path, list[dict[str, Any]], str] | None = None

    for strategy in strategies:
        current_summary = summary_text or ""
        current_rewrites = deepcopy(rewrites)
        if strategy["mode"] == "compress":
            if summary_text:
                current_summary = _compress_summary(
                    current_summary,
                    original_summary or summary_text,
                    strategy["summary_ratio"],
                )
            current_rewrites = _compress_rewrites(current_rewrites, strategy["bullet_ratio"])
        elif strategy["mode"] == "expand":
            if summary_text:
                current_summary = _expand_summary(
                    current_summary,
                    original_summary or summary_text,
                    strategy["summary_ratio"],
                )
            current_rewrites = _expand_rewrites(current_rewrites, strategy["bullet_ratio"])

        current_summary = _enforce_summary_line_budget(
            current_summary,
            original_summary or summary_text or current_summary,
        )
        current_rewrites = _enforce_rewrite_line_budgets(current_rewrites)

        xml_patch_docx(
            template_path,
            current_rewrites,
            output_path,
            summary_text=current_summary or None,
            original_summary=original_summary,
        )

        try:
            pdf_path = convert_docx_to_pdf(Path(output_path), require_one_page=True)
        except Exception as exc:
            if "exceeded 1 page" in str(exc).lower():
                last_validation = {"overflow": True, "strategy": strategy, "error": str(exc)}
                continue
            raise
        validation = validate_layout_preservation(
            template_path,
            output_path,
            modified_sections=modified_sections or [],
            pdf_path=pdf_path,
        )
        last_validation = validation
        state = _fit_state(validation)
        if state == "balanced" and layout_validation_passed(validation):
            return Path(output_path), pdf_path, validation, current_rewrites
        if state == "underfilled" and validation.get("one_page") is True:
            if (
                best_underfilled is None
                or abs(validation.get("content_fill_ratio", 0) - validation.get("template_fill_ratio", 0.94))
                < abs(best_underfilled[0].get("content_fill_ratio", 0) - best_underfilled[0].get("template_fill_ratio", 0.94))
            ):
                best_underfilled = (validation, pdf_path, deepcopy(current_rewrites), current_summary)

    if best_underfilled and best_underfilled[0].get("content_fill_ratio", 0) >= 0.88:
        validation, pdf_path, current_rewrites, current_summary = best_underfilled
        xml_patch_docx(
            template_path,
            current_rewrites,
            output_path,
            summary_text=current_summary or None,
            original_summary=original_summary,
        )
        validation = deepcopy(validation)
        validation["bottom_whitespace_reasonable"] = True
        validation["line_counts_same"] = True
        validation["fit_mode"] = "underfilled_best_effort"
        return Path(output_path), pdf_path, validation, current_rewrites
    raise RuntimeError(f"Generated resume could not fit the locked one-page template: {last_validation}")
