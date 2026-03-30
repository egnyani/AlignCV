"""Reporting utilities for resume match runs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from build_log_utils import log_file_event


def build_report_markdown(
    before_score: dict[str, Any],
    after_score: dict[str, Any],
    rewrites: list[dict[str, str]],
) -> str:
    """Build a markdown report summarizing the run."""
    lines = [
        "# Resume Match Report",
        "",
        "## Overall Score",
        f'Before: {before_score.get("overall_score", 0) * 100:.1f}% -> After: {after_score.get("overall_score", 0) * 100:.1f}%',
        "",
        "## JD Requirement Coverage",
        "| Requirement | Best Match | Score |",
        "| --- | --- | --- |",
    ]
    for item in after_score.get("requirement_coverage", []):
        lines.append(
            f'| {item["requirement"]} | {item["best_match_bullet"]} | {item["score"]:.3f} |'
        )

    lines.extend(["", "## Rewrites Applied"])
    if rewrites:
        for item in rewrites:
            lines.append(f'- [{item["original"]}] -> [{item["rewritten"]}]')
    else:
        lines.append("- No rewrites applied.")
    return "\n".join(lines)


def print_and_save_report(
    before_score: dict[str, Any],
    after_score: dict[str, Any],
    rewrites: list[dict[str, str]],
    output_dir: str | Path = "outputs",
) -> str:
    """Print and save the markdown report."""
    markdown = build_report_markdown(before_score, after_score, rewrites)
    print(markdown)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"report_{timestamp}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existed = output_path.exists()
    output_path.write_text(markdown, encoding="utf-8")
    log_file_event("MODIFIED" if existed else "CREATED", output_path, "Saved markdown resume match report")
    return str(output_path)
