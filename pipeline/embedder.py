"""Embedding utilities using OpenAI text-embedding-3-small."""

from __future__ import annotations

from copy import deepcopy
import os
from typing import Any

from openai import OpenAI

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, OPENAI_API_KEY


def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str], dry_run: bool = False) -> list[list[float]]:
    """Embed a list of texts using OpenAI embeddings."""
    if not texts:
        return []
    client = _client()
    embeddings: list[list[float]] = []
    try:
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings
    except Exception as exc:
        raise RuntimeError(f"Failed to create embeddings: {exc}") from exc


def embed_resume(resume_json: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Attach embeddings to resume bullets and skills."""
    enriched = deepcopy(resume_json)
    texts: list[str] = []
    locations: list[tuple[str, int, int | None]] = []

    for exp_index, item in enumerate(enriched.get("experience", [])):
        for bullet_index, bullet in enumerate(item.get("bullets", [])):
            texts.append(bullet)
            locations.append(("experience", exp_index, bullet_index))

    skills = enriched.get("skills", {})
    for skill_index, (category, values) in enumerate(skills.items()):
        text = f"{category}: {', '.join(values)}"
        texts.append(text)
        locations.append(("skills", skill_index, None))

    vectors = embed_texts(texts, dry_run=dry_run)
    skill_keys = list(skills.keys())

    for vector, location in zip(vectors, locations):
        section, first_index, second_index = location
        if section == "experience":
            bullet = enriched["experience"][first_index]["bullets"][second_index]
            enriched["experience"][first_index]["bullets"][second_index] = {
                "text": bullet,
                "embedding": vector,
            }
        else:
            category = skill_keys[first_index]
            enriched["skills"][category] = {
                "items": skills[category],
                "embedding": vector,
            }

    return enriched


def embed_jd_requirements(requirements: list[str], dry_run: bool = False) -> list[dict[str, Any]]:
    """Embed atomic JD requirements."""
    vectors = embed_texts(requirements, dry_run=dry_run)
    return [{"text": text, "embedding": vector} for text, vector in zip(requirements, vectors)]


def embed_keywords(keywords: list[str], dry_run: bool = False) -> list[dict[str, Any]]:
    """Embed individual JD keyword strings for vector-search assignment.

    Each keyword is embedded as its own unit so cosine similarity against
    resume bullet embeddings reflects conceptual closeness, not sentence-level
    match.  Returns same format as embed_jd_requirements.
    """
    if not keywords:
        return []
    # Wrap bare keywords in a short sentence for better embedding quality
    # e.g. "gRPC" → "Experience with gRPC" so the embedding space is richer
    wrapped = [f"Experience with {kw}" if len(kw.split()) <= 2 else kw for kw in keywords]
    vectors = embed_texts(wrapped, dry_run=dry_run)
    return [{"text": original, "embedding": vector} for original, vector in zip(keywords, vectors)]
