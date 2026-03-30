"""
Convert a DOCX file to PDF.

Local (macOS with Microsoft Word installed):
    Uses docx2pdf, which drives Word via AppleScript → pixel-perfect output
    that matches the DOCX exactly. No font substitution or spacing hacks needed.

Production (Vercel / no Word available):
    Falls back to the LibreOffice microservice on Render.com. All DOCX
    pre-processing (font substitution + line-spacing adjustment) happens before
    the file is sent, so Render is purely a dumb converter.

Set env var PDF_SERVICE_URL to the Render service URL for the Render path.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

# ── Font substitution (LibreOffice path only) ──────────────────────────────
_FONT_MAP = {
    "Calibri": "Carlito",
    "Arial": "Liberation Sans",
    "Times New Roman": "Liberation Serif",
}
_FONT_XML_FILES = {
    "word/document.xml",
    "word/styles.xml",
    "word/settings.xml",
    "word/theme/theme1.xml",
    "word/fontTable.xml",
}

# ── Line-spacing scales to try (LibreOffice path only) ────────────────────
_SCALES = [1.11, 1.08, 1.05, 1.0]


def _build_docx_for_libreoffice(docx_path: Path, line_scale: float) -> bytes:
    """Patch fonts, scale line spacing, and normalise column layout for LibreOffice."""

    def _scale_spacing(xml: str) -> str:
        def _replace(m: re.Match) -> str:
            tag = m.group(0)
            if 'w:lineRule="exact"' in tag:
                return tag
            def _scale_val(vm: re.Match) -> str:
                return f'w:line="{int(int(vm.group(1)) * line_scale)}"'
            return re.sub(r'w:line="(\d+)"', _scale_val, tag)
        return re.sub(r'<w:spacing\b[^/]*/>', _replace, xml)

    def _fix_columns(xml: str) -> str:
        return re.sub(
            r'<w:cols\b[^>]*(?:/>|>.*?</w:cols>)',
            '<w:cols w:space="720"/>',
            xml,
            flags=re.S,
        )

    out = io.BytesIO()
    with zipfile.ZipFile(docx_path, "r") as zin, \
         zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename in _FONT_XML_FILES:
                text = data.decode("utf-8")
                for src, dst in _FONT_MAP.items():
                    text = text.replace(src, dst)
                if item.filename in ("word/document.xml", "word/styles.xml"):
                    text = _scale_spacing(text)
                    text = _fix_columns(text)
                data = text.encode("utf-8")
            zout.writestr(item, data)
    return out.getvalue()


def _convert_with_word(docx_path: Path) -> Path:
    """Convert using Microsoft Word via AppleScript (macOS only). Returns PDF path."""
    import subprocess
    pdf_path = docx_path.with_suffix(".pdf")
    # Resolve absolute paths so AppleScript POSIX file references work correctly
    docx_abs = str(docx_path.resolve())
    pdf_abs  = str(pdf_path.resolve())
    script = f"""
tell application "Microsoft Word"
    activate
    open POSIX file "{docx_abs}"
    delay 3
    set theDoc to active document
    save as theDoc file name "{pdf_abs}" file format format PDF
    close theDoc saving no
end tell
"""
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Word AppleScript failed: {result.stderr.strip()}")
    if not pdf_path.exists():
        raise RuntimeError("Word did not produce a PDF file.")
    return pdf_path


def _post_to_render(docx_bytes: bytes, filename: str, base_url: str) -> bytes | None:
    """POST DOCX bytes to Render. Returns PDF bytes, or None if PDF > 1 page (422)."""
    boundary = "PdfServiceBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document\r\n\r\n"
    ).encode() + docx_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base_url}/convert",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            pdf_bytes = resp.read()
            if not pdf_bytes:
                raise RuntimeError("PDF service returned an empty response.")
            return pdf_bytes
    except urllib.error.HTTPError as e:
        if e.code == 422:
            return None  # PDF overflowed — caller will retry with smaller scale
        raise RuntimeError(f"PDF service error {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def _word_available() -> bool:
    """Return True if Microsoft Word can be driven via AppleScript on this machine."""
    import subprocess
    try:
        r = subprocess.run(
            ["osascript", "-e", 'id of app "Microsoft Word"'],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0 and "com.microsoft.Word" in r.stdout
    except Exception:
        return False


def pdf_page_count(pdf_path: Path) -> int:
    """Best-effort PDF page count without depending on a specific viewer."""
    try:
        from pypdf import PdfReader  # type: ignore
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        data = pdf_path.read_bytes()
        matches = re.findall(rb"/Type\s*/Page\b", data)
        return len(matches)


def convert_docx_to_pdf(docx_path: Path, require_one_page: bool = False) -> Path:
    """Convert DOCX to PDF. Uses Word locally; falls back to Render on Vercel."""

    # ── Local path: use Microsoft Word for pixel-perfect output ──────────────
    if not os.environ.get("VERCEL") and _word_available():
        pdf_path = _convert_with_word(docx_path)
        if require_one_page and pdf_page_count(pdf_path) > 1:
            raise RuntimeError("PDF exceeded 1 page")
        return pdf_path

    # ── Production path: LibreOffice via Render microservice ─────────────────
    base_url = os.environ.get("PDF_SERVICE_URL", "").strip().rstrip("/")
    if not base_url:
        raise RuntimeError("PDF_SERVICE_URL env var is not set and docx2pdf is unavailable.")

    for scale in _SCALES:
        docx_bytes = _build_docx_for_libreoffice(docx_path, scale)
        pdf_bytes = _post_to_render(docx_bytes, docx_path.name, base_url)
        if pdf_bytes is not None:
            pdf_path = docx_path.with_suffix(".pdf")
            pdf_path.write_bytes(pdf_bytes)
            if require_one_page and pdf_page_count(pdf_path) > 1:
                raise RuntimeError("PDF exceeded 1 page")
            return pdf_path

    # All scales overflowed — last resort: send unscaled
    docx_bytes = _build_docx_for_libreoffice(docx_path, 1.0)
    pdf_bytes = _post_to_render(docx_bytes, docx_path.name, base_url) or b""
    pdf_path = docx_path.with_suffix(".pdf")
    pdf_path.write_bytes(pdf_bytes)
    if require_one_page and pdf_page_count(pdf_path) > 1:
        raise RuntimeError("PDF exceeded 1 page")
    return pdf_path
