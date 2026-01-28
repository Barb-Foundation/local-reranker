# -*- coding: utf-8 -*-
"""Jina prompt formatting utilities for reranking tasks."""

import re
from typing import List

SANITIZE_PATTERN = re.compile(r"<\|(?:startoftext|endoftext)\|>")
QUERY_EMBED_TOKEN = "<|embed_query|>"
PASSAGE_EMBED_TOKEN = "<|embed_passage|>"


def _sanitize_input(text: str) -> str:
    """Remove special tokens from text.

    Args:
        text: The input text to sanitize.

    Returns:
        Text with special tokens removed.
    """
    return SANITIZE_PATTERN.sub("", text)


def _format_jina_prompt(query: str, documents: List[str]) -> str:
    """Format query and documents into a Jina reranker prompt.

    The prompt structure follows Jina's reranker format with:
    - A fixed system message for reranking tasks
    - Query with embed token immediately after
    - Documents wrapped in passage tags with embed tokens after each

    Args:
        query: The search query
        documents: List of document texts to rerank

    Returns:
        Formatted prompt string
    """
    sanitized_query = _sanitize_input(query)

    system_message = (
        "You are a helpful assistant. Your task is to rerank the following passages "
        "based on their relevance to the query. You should assign a higher score to "
        "passages that directly answer or are relevant to the query."
    )

    prompt_parts = [system_message, "\n"]

    prompt_parts.append(f"Query: {sanitized_query}{QUERY_EMBED_TOKEN}\n\n")

    for i, doc in enumerate(documents):
        sanitized_doc = _sanitize_input(doc)
        prompt_parts.append(
            f'<passage id="{i}">{sanitized_doc}{PASSAGE_EMBED_TOKEN}</passage>\n'
        )

    return "".join(prompt_parts)
