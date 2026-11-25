"""Gemini LLM judge for evaluating meaning preservation between comments."""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
import toxicity

SYSTEM_PROMPT = (
    "You are reviewing a rewrite for meaning preservation. "
    "NEVER quote, repeat, or spell out offensive words in your answer; "
    'refer to them generically (e.g., "a slur", "profanity", "an insult"). '
    "Judge whether the core message, complaint, or intent of the original survives "
    "in the candidate rewrite, and flag if the rewrite flips the meaning, drops the "
    "main point, or adds new claims.\n\n"
    "Respond ONLY with a valid JSON object using this exact schema:\n"
    "{\n"
    '  "meaning_preserved": "yes" | "partial" | "no",\n'
    '  "what_was_lost": "<short generic summary of lost or altered meaning, or none>",\n'
    '  "reason": "<brief explanation without quoting offensive terms>"\n'
    "}"
)

SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
]


def _make_cache_key(original: str, candidate: str) -> str:
    """Create a deterministic SHA-256 key from original and candidate texts."""
    payload = f"{original}\n<<<PAIR_SEP>>>\n{candidate}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_cache(cache_path: Path = config.JUDGE_CACHE_PATH) -> dict[str, dict[str, Any]]:
    """Load cached judge responses from disk."""
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, dict[str, Any]], cache_path: Path = config.JUDGE_CACHE_PATH) -> None:
    """Persist judge cache dictionary to disk."""
    try:
        cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _strip_code_fences(raw_text: str) -> str:
    """Remove Markdown code fences around JSON responses."""
    cleaned = raw_text.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return cleaned


def _parse_judge_json(raw_text: str) -> Optional[dict[str, str]]:
    """Safely parse the JSON object returned by Gemini."""
    if not raw_text or not raw_text.strip():
        return None

    cleaned = _strip_code_fences(raw_text)
    parsed: Any = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not brace_match:
            return None
        try:
            parsed = json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed, dict):
        return None

    meaning = str(parsed.get("meaning_preserved", "")).strip().lower()
    if meaning not in {"yes", "partial", "no"}:
        return None

    what_was_lost = str(parsed.get("what_was_lost", "")).strip() or "None reported."
    reason = str(parsed.get("reason", "")).strip() or "No reason provided."
    return {
        "meaning_preserved": meaning,
        "what_was_lost": toxicity.mask(what_was_lost),
        "reason": toxicity.mask(reason),
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if an exception indicates API rate limiting (HTTP 429)."""
    if isinstance(exc, genai_errors.APIError) and getattr(exc, "code", None) == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate" in msg or "resource_exhausted" in msg or "quota" in msg


def _unavailable_result(reason: str) -> dict[str, Any]:
    """Build a safe fallback response when the Gemini judge cannot respond."""
    return {
        "meaning_preserved": "partial",
        "what_was_lost": "Judge unavailable.",
        "reason": toxicity.mask(reason),
        "available": False,
    }


def evaluate(original: str, candidate: str) -> dict[str, Any]:
    """Evaluate whether a candidate rewrite preserves the meaning of the original comment.

    Args:
        original: Original toxic comment.
        candidate: Candidate neutral rewrite.

    Returns:
        Dict with keys 'meaning_preserved', 'what_was_lost', 'reason', and 'available'.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "").strip()

    if not original or not original.strip() or not candidate or not candidate.strip():
        return {
            "meaning_preserved": "no",
            "what_was_lost": "Empty original or candidate text.",
            "reason": "Both original comment and candidate rewrite must be non-empty.",
            "available": True,
        }

    cache_key = _make_cache_key(original, candidate)
    cache = _load_cache()
    if cache_key in cache:
        cached = dict(cache[cache_key])
        cached["what_was_lost"] = toxicity.mask(str(cached.get("what_was_lost", "")))
        cached["reason"] = toxicity.mask(str(cached.get("reason", "")))
        cached["available"] = True
        return cached

    if not api_key or not model_name:
        return _unavailable_result(
            "Gemini judge unavailable: GEMINI_API_KEY or GEMINI_MODEL is not set."
        )

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    prompt = (
        "Compare the original comment and the candidate rewrite.\n"
        f"Original comment: {original}\n"
        f"Candidate rewrite: {candidate}\n"
    )
    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=config.GEMINI_TEMPERATURE,
        response_mime_type="application/json",
        safety_settings=SAFETY_SETTINGS,
    )

    last_error = "Empty or blocked response from Gemini."
    for attempt in range(config.GEMINI_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )
            raw_text = getattr(response, "text", None) or ""
            parsed = _parse_judge_json(raw_text)
            if parsed is not None:
                result = {
                    **parsed,
                    "available": True,
                }
                cache[cache_key] = parsed
                _save_cache(cache)
                return result
            last_error = "Gemini returned an empty, blocked, or malformed JSON response."
            break
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc) and attempt < config.GEMINI_MAX_RETRIES - 1:
                delay = config.GEMINI_BASE_BACKOFF_SECONDS * (2**attempt)
                time.sleep(delay)
                continue
            last_error = f"Gemini API error: {type(exc).__name__}"
            break

    return _unavailable_result(last_error)


SUGGEST_SYSTEM_PROMPT = (
    "You are helping improve a public comment rewrite. "
    "Given an original comment and an optional candidate rewrite, provide 3 respectful, "
    "non-toxic alternative rewrites that preserve the original core message, complaint, or intent. "
    "NEVER quote, repeat, or include profanity, slurs, or insults. "
    'Respond ONLY with a valid JSON object: {"suggestions": ["...", "...", "..."]}'
)


def suggest_rewrites(original: str, candidate: str = "", count: int = 3) -> dict[str, Any]:
    """Ask Gemini for non-toxic, meaning-preserving rewrite suggestions.

    Args:
        original: Original comment text.
        candidate: Optional current candidate rewrite.
        count: Number of suggestions to request (default 3).

    Returns:
        Dict with 'suggestions' (list[str]), 'available' (bool), and 'note' (str).
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = os.getenv("GEMINI_MODEL", "").strip()

    if not original or not original.strip():
        return {"suggestions": [], "available": False, "note": "Original comment is empty."}

    if not api_key or not model_name:
        return {
            "suggestions": [],
            "available": False,
            "note": "Gemini unavailable: GEMINI_API_KEY or GEMINI_MODEL is not set.",
        }

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1beta"))
    prompt = f"Provide {count} non-toxic, meaning-preserving rewrites.\nOriginal comment: {original}\n"
    if candidate and candidate.strip():
        prompt += f"Current candidate rewrite: {candidate}\n"

    gen_config = types.GenerateContentConfig(
        system_instruction=SUGGEST_SYSTEM_PROMPT,
        temperature=0.3,
        response_mime_type="application/json",
        safety_settings=SAFETY_SETTINGS,
    )

    last_note = "Empty or blocked response from Gemini."
    for attempt in range(config.GEMINI_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=gen_config,
            )
            raw_text = getattr(response, "text", None) or ""
            cleaned = _strip_code_fences(raw_text)
            parsed = json.loads(cleaned)
            raw_items = parsed.get("suggestions", []) if isinstance(parsed, dict) else []
            suggestions = [
                toxicity.mask(str(item).strip())
                for item in raw_items
                if str(item).strip()
            ][:count]
            if suggestions:
                return {"suggestions": suggestions, "available": True, "note": ""}
            last_note = "Gemini returned no suggestions."
            break
        except Exception as exc:  # noqa: BLE001
            if _is_rate_limit_error(exc) and attempt < config.GEMINI_MAX_RETRIES - 1:
                time.sleep(config.GEMINI_BASE_BACKOFF_SECONDS * (2**attempt))
                continue
            last_note = f"Gemini API error: {type(exc).__name__}"
            break

    return {"suggestions": [], "available": False, "note": toxicity.mask(last_note)}


if __name__ == "__main__":
    sample_json = '```json\n{"meaning_preserved": "yes", "what_was_lost": "Profanity removed", "reason": "Core criticism kept."}\n```'
    print("Parsed fenced JSON:", _parse_judge_json(sample_json))
    print("Parsed malformed JSON:", _parse_judge_json("not json at all"))
    result = evaluate("You are an idiot for breaking the build.", "Please be careful not to break the build.")
    print("Evaluate result:", result)

