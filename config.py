"""Configuration for the resume matcher pipeline."""

from __future__ import annotations

import os

OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL: str = "https://api.openai.com/v1"

SIMILARITY_THRESHOLD: float = 0.68       # rewrite all bullets scoring below 70%
EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI — better semantic resolution
REWRITE_MODEL: str = "gpt-4o"            # stronger model for better targeted rewrites
MAX_BULLETS_TO_REWRITE: int = 20
EMBEDDING_BATCH_SIZE: int = 100
