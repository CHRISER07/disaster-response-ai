"""
core/answer_formatter.py

Deterministic plain-text answer formatting for disaster response outputs.

Design goals
------------
1. No markdown emphasis markers (single or double asterisks) in ANY answer.
2. Stable operational template:
       Status / Evidence / Recommendation / Alert Generated
3. No trailing whitespace, and at most one blank line between blocks.
4. Pure function: identical input always produces identical output.
   No LLM calls, no randomness, no network access.

This module is the single source of truth for answer presentation. Both the
agent runtime and the evaluation harness must call format_answer() before an
answer is stored, scored, or displayed.
"""
from __future__ import annotations

import re

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_UNDERSCORE_BOLD = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", re.DOTALL)
_HEADER = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*")
_BULLET_STAR = re.compile(r"(?m)^(\s*)\*[ \t]+")
_STRAY_ASTERISK = re.compile(r"\*")

TEMPLATE_SECTIONS = ("Status:", "Evidence:", "Recommendation:", "Alert Generated:")


def strip_markdown(text: str) -> str:
    """Remove markdown emphasis/heading/bullet markers without changing wording."""
    if text is None:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")
    s = _BOLD.sub(r"\1", s)
    s = _UNDERSCORE_BOLD.sub(r"\1", s)
    s = _ITALIC.sub(r"\1", s)
    s = _HEADER.sub("", s)
    s = _BULLET_STAR.sub(r"\1- ", s)
    s = _STRAY_ASTERISK.sub("", s)
    return s


def format_answer(text: str) -> str:
    """
    Normalize an answer into the clean, asterisk-free presentation template.

    Returns a single trailing newline; empty input returns an empty string.
    """
    s = strip_markdown(text)
    lines = [line.rstrip() for line in s.split("\n")]

    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        if not line.strip():
            if not previous_blank:
                normalized.append("")
            previous_blank = True
        else:
            normalized.append(line)
            previous_blank = False

    while normalized and not normalized[0].strip():
        normalized.pop(0)
    while normalized and not normalized[-1].strip():
        normalized.pop()

    if not normalized:
        return ""
    return "\n".join(normalized) + "\n"


def template_sections_present(text: str) -> dict[str, bool]:
    """Report which operational-template sections an answer contains (audit only)."""
    s = text or ""
    return {section: section in s for section in TEMPLATE_SECTIONS}
