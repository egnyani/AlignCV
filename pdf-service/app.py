import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from flask import Flask, request, send_file, jsonify
from pypdf import PdfReader

app = Flask(__name__)


def _page_count(pdf_path: Path) -> int:
    return len(PdfReader(str(pdf_path)).pages)


def _libreoffice_convert(docx_path: Path, out_dir: str) -> Path:
    result = subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", out_dir, str(docx_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    pdf_path = docx_path.with_suffix(".pdf")
    if not pdf_path.exists():
        raise RuntimeError("PDF not generated")
    return pdf_path


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/convert")
def convert():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".docx"):
        return jsonify({"error": "Only .docx files accepted"}), 400

    with tempfile.TemporaryDirectory() as tmp:
        docx_path = Path(tmp) / f.filename
        f.save(str(docx_path))

        try:
            pdf_path = _libreoffice_convert(docx_path, tmp)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

        if _page_count(pdf_path) > 1:
            return jsonify({"error": "PDF exceeded 1 page"}), 422

        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes:
            return jsonify({"error": "Generated PDF was empty"}), 500

        return send_file(
            BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=pdf_path.name,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
