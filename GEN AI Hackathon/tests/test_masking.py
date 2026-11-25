"""Safety tests verifying no offensive words leak unmasked into outputs or reports."""

import json
import re
import sys
from pathlib import Path

# Ensure project root is importable when running pytest from any directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import report
import toxicity

KNOWN_OFFENSIVE_WORDS = [
    "fuck",
    "shit",
    "bitch",
    "asshole",
    "idiot",
    "stupid",
    "moron",
    "crap",
]


def _contains_unmasked_word(text: str, words: list[str]) -> list[str]:
    """Return any offensive words found unmasked as whole words in text."""
    found = []
    for word in words:
        if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
            found.append(word)
    return found


def test_mask_preserves_first_letter_and_hides_rest() -> None:
    """Offensive words from both better-profanity and Toxic-BERT are masked cleanly."""
    raw = "You stupid idiot, stop talking shit and fuck off!"
    masked = toxicity.mask(raw)

    assert "f***" in masked.lower()
    assert "s***" in masked.lower()
    assert "s*****" in masked.lower()
    assert "i****" in masked.lower()
    assert _contains_unmasked_word(masked, KNOWN_OFFENSIVE_WORDS) == []


def test_mask_leaves_clean_text_intact() -> None:
    """Clean neutral sentences should remain unchanged."""
    clean = "Please update the documentation before merging this pull request."
    assert toxicity.mask(clean) == clean


def test_report_contains_no_unmasked_offensive_words(monkeypatch) -> None:
    """Generated reports must never contain any unmasked offensive words in any field."""
    # Simulate a worst-case judge response that ignores instructions and repeats profanity
    monkeypatch.setattr(
        "judge.evaluate",
        lambda orig, cand: {
            "meaning_preserved": "yes",
            "what_was_lost": "Removed words like asshole and bitch.",
            "reason": "The original called someone a stupid idiot and moron.",
            "available": True,
        },
    )

    toxic_original = "You are a stupid asshole and a total moron, fuck this shit!"
    toxic_candidate = "Stop being an idiot and fix this crap, bitch."

    generated = report.build(toxic_original, toxic_candidate)
    serialized = json.dumps(generated)

    leaked = _contains_unmasked_word(serialized, KNOWN_OFFENSIVE_WORDS)
    assert leaked == [], f"Unmasked offensive words leaked into report: {leaked}"
    assert "original" not in generated
    assert "candidate" not in generated

