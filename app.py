# app.py – Vercel ASGI entrypoint
#
# Vercel's Python runtime scans for `app = FastAPI()` in a fixed list of
# filenames: app.py, main.py, index.py, server.py, wsgi.py, asgi.py (and the
# same inside api/, src/, app/ subdirectories).  Our FastAPI app lives in
# api.py, which is NOT on that list, hence the "No fastapi entrypoint found"
# error.  This one-line re-export is the minimal fix.
#
# Nothing else changes — all routes, middleware, and pipeline logic stay in
# api.py exactly as they are.

from api import app  # noqa: F401  – re-exported for Vercel detection
