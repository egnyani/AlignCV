"""CLI entrypoint for the smart resume tailoring pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from build_log_utils import log_file_event, log_run_event
from pipeline.builder import make_output_filename
from pipeline.embedder import embed_jd_requirements, embed_resume
from pipeline.fit_controller import patch_with_fit_control
from pipeline.resume_html_export import try_generate_html_resume_pdf
from pipeline.jd_extractor import extract_company_role, extract_jd_requirements
from pipeline.layout_validator import layout_validation_passed
from pipeline.parser import parse_resume, save_resume_json
from pipeline.reporter import print_and_save_report
from pipeline.rewriter import rewrite_weak_bullets
from pipeline.scorer import score_resume_against_jd
from pipeline.xml_builder import collect_modified_sections


def read_jd_input(jd_arg: str) -> str:
    """Read JD text from a file path or raw string."""
    path = Path(jd_arg)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return jd_arg


def dump_intermediate(data: dict[str, Any], path: str | Path) -> None:
    """Pretty-print an intermediate JSON file."""
    target = Path(path)
    existed = target.exists()
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log_file_event("MODIFIED" if existed else "CREATED", target, "Saved intermediate pipeline JSON")


def _command_string(args: argparse.Namespace) -> str:
    """Build a readable command string for logging."""
    parts = [f"--resume {args.resume}", f"--jd {args.jd}"]
    if args.output:
        parts.append(f"--output {args.output}")
    if args.company:
        parts.append(f"--company {args.company}")
    if args.role:
        parts.append(f"--role {args.role}")
    if args.dry_run:
        parts.append("--dry-run")
    if args.verbose:
        parts.append("--verbose")
    return "main.py " + " ".join(parts)


def print_verbose_scores(score_data: dict[str, Any]) -> None:
    """Print per-requirement and weak-bullet scores."""
    print("⏳ Verbose score details")
    for item in score_data.get("requirement_coverage", []):
        print(
            f'  combined={item["score"]:.3f} '
            f'kw={item.get("keyword_score", 0):.2f} '
            f'sem={item.get("semantic_score", 0):.3f}  '
            f'"{item["requirement"]}"'
        )
    for item in score_data.get("weak_bullets", []):
        print(
            f'  weak_bullet section="{item["section"]}" '
            f'index={item["index"]} best_score={item["best_score"]:.3f} '
            f'text="{item["bullet"]}"'
        )


def resolve_output_path(args: argparse.Namespace, jd_text: str) -> Path:
    """Resolve the output path from args or a JD-derived filename."""
    if args.output:
        return Path(args.output)
    identity = extract_company_role(jd_text, dry_run=args.dry_run)
    company = args.company or identity["company"]
    role = args.role or identity["role"]
    return Path("outputs") / make_output_filename(company, role)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Smart resume tailoring pipeline")
    parser.add_argument("--resume", required=True, help="Path to the source resume DOCX")
    parser.add_argument("--jd", required=True, help="Path to a JD .txt file or raw JD text")
    parser.add_argument("--output", help="Output DOCX path")
    parser.add_argument("--company", help="Optional company name override")
    parser.add_argument("--role", help="Optional role name override")
    parser.add_argument("--dry-run", action="store_true", help="Skip rewrite and DOCX build")
    parser.add_argument("--verbose", action="store_true", help="Print verbose scoring details")
    return parser


def main() -> int:
    """Run the full resume tailoring pipeline."""
    args = build_parser().parse_args()
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    try:
        log_run_event(_command_string(args), "Pipeline started")
        print("⏳ Parsing resume")
        resume_json = parse_resume(args.resume)
        save_resume_json(resume_json, output_dir / "parsed_resume.json")
        print("✓ Parsed resume")

        print("⏳ Reading job description")
        jd_text = read_jd_input(args.jd)
        requirements = extract_jd_requirements(jd_text, dry_run=args.dry_run)
        dump_intermediate({"requirements": requirements}, output_dir / "jd_requirements.json")
        print(f"✓ Extracted {len(requirements)} job requirements")

        print("⏳ Embedding resume and job requirements")
        embedded_resume = embed_resume(resume_json, dry_run=args.dry_run)
        embedded_jd = embed_jd_requirements(requirements, dry_run=args.dry_run)
        print("✓ Embeddings created")

        print("⏳ Scoring resume against job description")
        before_score = score_resume_against_jd(embedded_resume, embedded_jd)
        dump_intermediate(before_score, output_dir / "score_before.json")
        print(f'✓ Initial overall score: {before_score["overall_score"] * 100:.1f}%')
        if args.verbose:
            print_verbose_scores(before_score)

        rewrites: list[dict[str, str]] = []
        after_score = before_score
        updated_resume = resume_json

        if args.dry_run:
            print("✓ Dry run enabled: skipping rewrite and DOCX build")
        else:
            print("⏳ Rewriting weak bullets (coverage-aware)")
            updated_resume, rewrites = rewrite_weak_bullets(
                resume_json,
                before_score["weak_bullets"],
                requirements,
                jd_embeddings=embedded_jd,
                requirement_scores=before_score["requirement_coverage"],
            )
            save_resume_json(updated_resume, output_dir / "rewritten_resume.json")
            print(f"✓ Rewrote {len(rewrites)} bullets")

            print("⏳ Re-embedding and rescoring updated resume")
            embedded_updated_resume = embed_resume(updated_resume)
            after_score = score_resume_against_jd(embedded_updated_resume, embedded_jd)
            dump_intermediate(after_score, output_dir / "score_after.json")
            print(f'✓ Updated overall score: {after_score["overall_score"] * 100:.1f}%')
            if args.verbose:
                print_verbose_scores(after_score)

            print("⏳ Building tailored DOCX from the fixed template")
            output_path = resolve_output_path(args, jd_text)
            output_existed = output_path.exists()
            final_docx, final_pdf, layout_validation, fitted_rewrites = patch_with_fit_control(
                "data/Gnyani_resume_layout.docx",
                output_path=output_path,
                rewrites=rewrites,
                summary_text=None,
                original_summary=None,
                modified_sections=collect_modified_sections(rewrites),
            )
            warnings: list[str] = []
            for warning in warnings:
                print(f"⏳ Warning: {warning}")
                log_run_event(_command_string(args), warning)
            log_file_event(
                "MODIFIED" if output_existed else "CREATED",
                final_docx,
                "Generated tailored resume DOCX from fixed template",
            )
            modified_sections = collect_modified_sections(fitted_rewrites)
            print(f"⏳ Modified sections: {', '.join(modified_sections) or 'None'}")
            print(
                "⏳ Layout validation: "
                f"page_size_same={layout_validation['page_size_same']} "
                f"paragraph_count_same={layout_validation['paragraph_count_same']} "
                f"major_styles_same={layout_validation['major_styles_same']} "
                f"section_order_same={layout_validation['section_order_same']} "
                f"margins_same={layout_validation['margins_same']} "
                f"one_page={layout_validation['one_page']}"
            )
            if not layout_validation_passed(layout_validation):
                raise RuntimeError(f"Template preservation failed: {layout_validation}")
            log_run_event(_command_string(args), f"Modified sections: {', '.join(modified_sections) or 'None'}")
            log_run_event(
                _command_string(args),
                "Layout validation "
                f"page_size_same={layout_validation['page_size_same']} "
                f"paragraph_count_same={layout_validation['paragraph_count_same']} "
                f"major_styles_same={layout_validation['major_styles_same']} "
                f"section_order_same={layout_validation['section_order_same']} "
                f"margins_same={layout_validation['margins_same']} "
                f"one_page={layout_validation['one_page']}",
            )
            print(f"✓ Wrote tailored resume to {final_docx}")

            html_pdf = try_generate_html_resume_pdf(
                updated_resume,
                docx_for_contact=final_docx,
                output_pdf=final_docx.with_name(f"{final_docx.stem}_html.pdf"),
                repo_root=Path(__file__).resolve().parent,
            )
            if html_pdf:
                print(f"✓ Wrote HTML-rendered PDF (Puppeteer): {html_pdf}")
            else:
                print("⚠ HTML PDF export skipped (see logs); use DOCX pipeline PDF from earlier step")

        print("⏳ Writing report")
        report_path = print_and_save_report(before_score, after_score, rewrites, output_dir=output_dir)
        log_run_event(_command_string(args), f"Pipeline completed successfully; report saved to {report_path}")
        print(f"✓ Saved report to {report_path}")
        return 0
    except Exception as exc:
        log_run_event(_command_string(args), f"Pipeline failed: {exc}")
        print(f"✗ Pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
