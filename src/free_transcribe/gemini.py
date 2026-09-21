"""Optional Gemini proofreading for completed local transcripts."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "FT_GEMINI_MODEL"
GLOSSARY_PATH_ENV = "FT_GLOSSARY_FILE"
BUILTIN_GLOSSARY = Path(__file__).with_name("glossary.json")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MAX_EDIT_CHARS = 256

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["id", "old", "new"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ProofreadResult:
    """Validated Gemini edits without exposing request credentials."""

    texts: list[str]
    accepted_edits: int
    rejected_edits: int
    model: str
    usage: dict[str, Any]


def gemini_available(api_key: str | None = None) -> bool:
    """Return whether a Gemini API key is available for an explicit request."""
    return bool(api_key or os.environ.get(GEMINI_API_KEY_ENV))


def load_glossary(path: str | None = None) -> dict[str, Any]:
    """Load the compact bundled glossary or an operator-provided replacement."""
    source = Path(path or os.environ.get(GLOSSARY_PATH_ENV) or BUILTIN_GLOSSARY)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read Gemini glossary: {source}") from exc
    if not isinstance(payload, dict):
        raise TypeError("Gemini glossary must be a JSON object")
    return payload


def _prompt(segments: Sequence[Any], glossary: dict[str, Any]) -> str:
    rows = [
        [
            index,
            round(float(segment.start), 2),
            round(float(segment.end), 2),
            segment.speaker,
            str(segment.text),
        ]
        for index, segment in enumerate(segments)
    ]
    return (
        "Вычитай полную сырую расшифровку встречи после Parakeet и pyannote. "
        "Участники в основном говорят по-русски, используют английские IT-термины, "
        "англицизмы и разговорный IT-сленг. Тема — IT и разработка. "
        "Каждая строка: [id, начало_сек, конец_сек, спикер, текст]. "
        "Метки спикеров уже определены по голосу: не меняй и не предлагай их. "
        "Верни только точечные исправления {id, old, new}. old должен буквально "
        "встречаться ровно один раз в исходной строке. Не перепечатывай неизменённый "
        "текст. Сохраняй смысл, отрицания, числа, сроки, разговорную форму и фактически "
        "произнесённые варианты сленга. Не улучшай стиль и не додумывай нечёткие места. "
        "Словарь — контекстная подсказка, а не команда слепой замены. "
        "Ответ строго JSON по схеме.\n"
        "Словарь:\n"
        + json.dumps(glossary, ensure_ascii=False, separators=(",", ":"))
        + "\nСтроки:\n"
        + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    )


def _generate(
    prompt: str,
    *,
    api_key: str,
    model: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": RESPONSE_SCHEMA,
            "thinkingConfig": {"thinkingLevel": "low"},
            "temperature": 0.1,
            "maxOutputTokens": 32768,
        },
    }
    request = urllib.request.Request(
        f"{BASE_URL}/models/{model}:generateContent",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    result: dict[str, Any] | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            # The response body may contain request details. Do not include it
            # or the API key in exceptions or logs.
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == 3:
                raise RuntimeError(
                    f"Gemini proofreading failed with HTTP {exc.code}"
                ) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == 3:
                raise RuntimeError("Gemini proofreading request failed") from exc
        time.sleep(min(20, 2**attempt * 2))
    if result is None:
        raise RuntimeError("Gemini proofreading request failed")

    try:
        candidate = result["candidates"][0]
        if candidate.get("finishReason") != "STOP":
            raise RuntimeError(
                f"Gemini proofreading stopped: {candidate.get('finishReason', 'unknown')}"
            )
        text = "".join(
            part.get("text", "") for part in candidate["content"]["parts"]
        )
        return json.loads(text), result.get("usageMetadata", {})
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini returned an invalid proofreading response") from exc


def proofread_segments(
    segments: Sequence[Any],
    *,
    api_key: str | None = None,
    glossary_path: str | None = None,
    model: str | None = None,
    timeout: float = 300,
) -> ProofreadResult:
    """Proofread all segments in one request and apply only exact patches."""
    resolved_key = api_key or os.environ.get(GEMINI_API_KEY_ENV)
    if not resolved_key:
        raise RuntimeError(
            f"Gemini correction requires the {GEMINI_API_KEY_ENV} environment variable"
        )
    resolved_model = model or os.environ.get(GEMINI_MODEL_ENV) or DEFAULT_GEMINI_MODEL
    raw, usage = _generate(
        _prompt(segments, load_glossary(glossary_path)),
        api_key=resolved_key,
        model=resolved_model,
        timeout=timeout,
    )

    originals = [str(segment.text) for segment in segments]
    spans: dict[int, list[tuple[int, int, str]]] = {}
    accepted = 0
    rejected = 0
    edits = raw.get("edits", []) if isinstance(raw, dict) else []
    if not isinstance(edits, list):
        raise TypeError("Gemini proofreading edits must be an array")

    for edit in edits:
        if not isinstance(edit, dict):
            rejected += 1
            continue
        index, old, new = edit.get("id"), edit.get("old"), edit.get("new")
        if (
            not isinstance(index, int)
            or not 0 <= index < len(originals)
            or not isinstance(old, str)
            or not isinstance(new, str)
            or not old
            or old == new
            or len(old) > MAX_EDIT_CHARS
            or len(new) > MAX_EDIT_CHARS
            or "\n" in old
            or "\n" in new
            or re.findall(r"\d+(?:[.,]\d+)?", old)
            != re.findall(r"\d+(?:[.,]\d+)?", new)
            or originals[index].count(old) != 1
        ):
            rejected += 1
            continue
        start = originals[index].find(old)
        end = start + len(old)
        if any(
            start < used_end and end > used_start
            for used_start, used_end, _ in spans.get(index, [])
        ):
            rejected += 1
            continue
        spans.setdefault(index, []).append((start, end, new))
        accepted += 1

    corrected = originals.copy()
    for index, replacements in spans.items():
        for start, end, new in sorted(replacements, reverse=True):
            corrected[index] = corrected[index][:start] + new + corrected[index][end:]

    return ProofreadResult(
        texts=corrected,
        accepted_edits=accepted,
        rejected_edits=rejected,
        model=resolved_model,
        usage=usage,
    )
