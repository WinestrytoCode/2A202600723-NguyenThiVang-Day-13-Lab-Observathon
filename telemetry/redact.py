"""PII redaction (Track 1).

Regex-based detection of common PII so it never reaches logs/traces. Zero extra
deps (regex only), so the mock path runs offline. For production-grade detection
of names/addresses, swap in Microsoft Presidio (see README). Vietnamese IDs
(CCCD = 12 digits, VN mobile) are included; tune for your locale.

Golden rule (Day 13 Section 13): redact at the point of origin, before the data
enters the logging/tracing pipeline -- not after an audit.
"""
from __future__ import annotations
import os
import re

# Order matters: match longer/structured patterns (card, CCCD) before phone.
_PATTERNS: dict[str, re.Pattern] = {
    "EMAIL":       re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "CREDIT_CARD": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "CCCD":        re.compile(r"\b\d{12}\b"),            # VN citizen ID
    "PHONE_VN":    re.compile(r"\b(?:\+84|0)\d{9}\b"),   # VN mobile
    "IP":          re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
}


def redact(text, mask: str = "[REDACTED:{}]"):
    """Return (redacted_text, num_redactions). Non-strings pass through unchanged."""
    if not isinstance(text, str):
        return text, 0
    # Optimization 1: Pre-filter (heuristics)
    # Most PII patterns here require numbers or an '@' symbol.
    if not any(c.isdigit() or c == '@' for c in text):
        return text, 0
    
    # Optimization 2: Length limit to prevent ReDoS on massive inputs
    max_len = 4000
    suffix = ""
    if len(text) > max_len:
        suffix = text[max_len:]
        text = text[:max_len]

    n = 0
    for label, pat in _PATTERNS.items():
        text, k = pat.subn(mask.format(label), text)
        n += k
    return text + suffix, n


def redact_value(value):
    """Recursively redact strings inside str/dict/list (for structured log payloads)."""
    if isinstance(value, str):
        return redact(value)[0]
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def enabled() -> bool:
    """REDACT_PII env toggle (default ON). Set REDACT_PII=0 to disable for debugging."""
    return os.getenv("REDACT_PII", "1").lower() not in ("0", "false", "no", "")
