#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Resume Matcher – launcher
# Usage: ./start.sh
# ─────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .env with OPENAI_API_KEY
if [ ! -f ".env" ]; then
  echo "⚠️  No .env file found. Create one with:"
  echo "    echo 'OPENAI_API_KEY=sk-...' > .env"
  exit 1
fi

# Install / verify deps
pip install -r requirements.txt --break-system-packages -q
if command -v npm >/dev/null 2>&1; then
  npm install --silent
else
  echo "⚠️  npm not found — HTML→PDF export will fall back to the DOCX pipeline PDF."
fi

echo ""
echo "✅  Starting Resume Matcher API on http://localhost:8000"
echo "   Open frontend.html in your browser after the server starts."
echo ""

# Start uvicorn (Ctrl+C to stop)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
