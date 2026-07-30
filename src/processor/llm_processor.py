"""LLM-assisted transcript cleaning and structuring."""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.models.schemas import KeyPoint, OutputDoc
from src.processor.prompt import (
    CLEAN_SYSTEM_PROMPT,
    CLEAN_USER_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
)

logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 8_000
CONTEXT_CHARS = 500
MIN_CLEAN_RATIO = 0.55
MAX_CLEAN_RATIO = 1.35


def process(
    api_key: str,
    title: str,
    podcast_name: str,
    pub_date: str,
    show_notes: str,
    transcript_text: str,
    *,
    base_url: str,
    provider: str,
    model: str,
    work_dir: Path | None = None,
) -> tuple[OutputDoc, int, int]:
    """Clean a transcript in chunks, then generate compact reading aids.

    Returns:
        (OutputDoc, input_tokens, output_tokens)
    """
    client = OpenAI(api_key=api_key, base_url=base_url)
    chunks = split_transcript(transcript_text)
    cleaned_chunks: list[str] = []
    input_tokens = 0
    output_tokens = 0

    logger.info("Cleaning transcript with %s/%s in %d chunks", provider, model, len(chunks))
    for index, chunk in enumerate(chunks):
        chunk_id = index + 1
        cached = _load_cached_chunk(work_dir, chunk_id, chunk, provider, model)
        if cached is not None:
            logger.info("Using cached cleaned chunk %d/%d", chunk_id, len(chunks))
            cleaned_chunks.append(cached)
            continue

        previous_context = chunks[index - 1][-CONTEXT_CHARS:] if index else "（无）"
        next_context = chunks[index + 1][:CONTEXT_CHARS] if index + 1 < len(chunks) else "（无）"
        prompt = CLEAN_USER_PROMPT.format(
            show_notes=(show_notes or "（无）")[:6_000],
            previous_context=previous_context,
            chunk_id=chunk_id,
            chunk_text=chunk,
            next_context=next_context,
        )
        cleaned = ""
        for validation_attempt in range(2):
            data, inp, out = _request_json(client, model, CLEAN_SYSTEM_PROMPT, prompt, max_tokens=16_384)
            input_tokens += inp
            output_tokens += out
            try:
                cleaned = _validate_cleaned_chunk(data, chunk_id, chunk)
                break
            except RuntimeError as exc:
                if validation_attempt == 0:
                    logger.warning("Cleaned chunk %d failed validation; retrying once: %s", chunk_id, exc)
                else:
                    raise
        _save_cached_chunk(work_dir, chunk_id, chunk, cleaned, provider, model)
        cleaned_chunks.append(cleaned)

    cleaned_transcript = "\n\n".join(cleaned_chunks)
    summary_prompt = SUMMARY_USER_PROMPT.format(
        title=title,
        podcast_name=podcast_name,
        show_notes=show_notes or "（无）",
        transcript_text=cleaned_transcript,
    )
    summary, inp, out = _request_json(client, model, SUMMARY_SYSTEM_PROMPT, summary_prompt, max_tokens=4_096)
    input_tokens += inp
    output_tokens += out
    doc = _build_output_doc(summary, title, podcast_name, pub_date, show_notes, cleaned_transcript)
    return doc, input_tokens, output_tokens


def split_transcript(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Split text at existing line boundaries without losing content."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    lines = text.splitlines(keepends=True)
    if not lines:
        return [""]

    chunks: list[str] = []
    current = ""
    for line in lines:
        while len(line) > max_chars:
            if current:
                chunks.append(current.rstrip())
                current = ""
            chunks.append(line[:max_chars].rstrip())
            line = line[max_chars:]
        if current and len(current) + len(line) > max_chars:
            chunks.append(current.rstrip())
            current = ""
        current += line
    if current or not chunks:
        chunks.append(current.rstrip())
    return chunks


def _request_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
) -> tuple[dict, int, int]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            if choice.finish_reason != "stop":
                raise RuntimeError(f"LLM response was incomplete: finish_reason={choice.finish_reason}")
            content = choice.message.content or ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                raise RuntimeError("LLM returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise RuntimeError("LLM JSON response must be an object")
            usage = response.usage
            return (
                data,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                logger.warning("LLM request failed validation; retrying once: %s", exc)
    assert last_error is not None
    raise last_error


def _validate_cleaned_chunk(data: dict, expected_id: int, source: str) -> str:
    if data.get("chunk_id") != expected_id:
        raise RuntimeError(f"LLM returned the wrong chunk_id: expected {expected_id}")
    cleaned = data.get("cleaned_text")
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise RuntimeError(f"LLM returned an empty cleaned_text for chunk {expected_id}")
    ratio = len(cleaned) / max(len(source), 1)
    if not MIN_CLEAN_RATIO <= ratio <= MAX_CLEAN_RATIO:
        raise RuntimeError(
            f"Cleaned chunk {expected_id} has suspicious length ratio {ratio:.2f} "
            f"(expected {MIN_CLEAN_RATIO:.2f}-{MAX_CLEAN_RATIO:.2f})"
        )
    return cleaned.strip()


def _chunk_cache_path(work_dir: Path | None, chunk_id: int) -> Path | None:
    return work_dir / f"chunk_{chunk_id:03d}.json" if work_dir else None


def _load_cached_chunk(
    work_dir: Path | None,
    chunk_id: int,
    source: str,
    provider: str,
    model: str,
) -> str | None:
    path = _chunk_cache_path(work_dir, chunk_id)
    if not path or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if (
        data.get("source_sha256") == expected_hash
        and data.get("provider") == provider
        and data.get("model") == model
        and isinstance(data.get("cleaned_text"), str)
    ):
        return data["cleaned_text"]
    return None


def _save_cached_chunk(
    work_dir: Path | None,
    chunk_id: int,
    source: str,
    cleaned: str,
    provider: str,
    model: str,
) -> None:
    path = _chunk_cache_path(work_dir, chunk_id)
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "provider": provider,
        "model": model,
        "cleaned_text": cleaned,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_output_doc(
    data: dict[str, Any],
    title: str,
    podcast_name: str,
    pub_date: str,
    show_notes: str,
    full_text: str,
) -> OutputDoc:
    key_points = []
    for item in data.get("key_points", []):
        if isinstance(item, dict) and item.get("point"):
            key_points.append(
                KeyPoint(point=str(item["point"]).strip(), evidence=str(item.get("evidence", "")).strip())
            )
    quotes = [str(item).strip() for item in data.get("highlight_quotes", []) if str(item).strip()]
    speaker_intro = data.get("speaker_intro", "")
    speaker_mapping = data.get("speaker_mapping", {})
    if isinstance(speaker_mapping, dict):
        for speaker, name in speaker_mapping.items():
            if re.fullmatch(r"SPEAKER_\d+", str(speaker)) and isinstance(name, str) and name.strip():
                full_text = full_text.replace(f"[{speaker}]", f"{name.strip()}：")
    return OutputDoc(
        title=title,
        podcast_name=podcast_name,
        pub_date=pub_date,
        show_notes=show_notes,
        key_points=key_points,
        highlight_quotes=quotes,
        full_text=full_text,
        speaker_intro=speaker_intro if isinstance(speaker_intro, str) else "",
    )


def _clean_evidence(evidence: str, point_title: str) -> str:
    """Remove duplicated point title from the start of evidence text."""
    text = evidence.strip()
    # Remove opening punctuation like ：:、,，
    text = text.lstrip("：:、,， ")
    # If evidence starts with the point title, remove it
    if point_title and text.startswith(point_title):
        text = text[len(point_title):]
    # Clean up stray markdown bold markers at the start
    text = text.lstrip("*").lstrip("：:、,， ")
    return text.strip()


def _parse_output(
    content: str,
    title: str,
    podcast_name: str,
    pub_date: str,
    show_notes: str,
) -> OutputDoc:
    """Parse LLM output into OutputDoc structure.

    The LLM output follows the template format:
      # Title
      ## 内容提要
      - **point**: evidence
      ## 闪光语句
      - quote
      ## 人物介绍
      > **主持人**：...
      ## 全文转录
      ...
    """
    key_points: list[KeyPoint] = []
    highlight_quotes: list[str] = []

    # --- Parse 内容提要 section ---
    m_toc = re.search(r"##\s*内容提要\s*\n(.*?)(?=##\s*(?:闪光语句|全文转录)|$)", content, re.DOTALL)
    if m_toc:
        toc_section = m_toc.group(1)
        for line in toc_section.strip().split("\n"):
            line = line.strip()
            if line.startswith("- ") or line.startswith("* "):
                # Extract bold text as point title
                bold_m = re.search(r"\*\*(.*?)\*\*", line)
                if bold_m:
                    point_title = bold_m.group(1).strip()
                    # Everything after the bold closing marker
                    after_bold = line[bold_m.end():].strip()
                    evidence = _clean_evidence(after_bold, point_title)
                    key_points.append(KeyPoint(point=point_title, evidence=evidence))

    # --- Parse 闪光语句 section ---
    m_quotes = re.search(r"##\s*闪光语句\s*\n(.*?)(?=##\s*(?:人物介绍|人物简介|全文转录)|$)", content, re.DOTALL)
    if m_quotes:
        quotes_section = m_quotes.group(1)
        for line in quotes_section.strip().split("\n"):
            raw = line.strip()
            # Remove leading list marker (- or *)
            if raw.startswith("- "):
                raw = raw[2:]
            elif raw.startswith("* "):
                raw = raw[2:]
            raw = raw.strip("\"'").strip("“”").strip()
            if raw and len(raw) >= 5:  # skip empty or too-short lines
                highlight_quotes.append(raw)

    # --- Parse 人物介绍/人物简介 section ---
    speaker_intro = ""
    m_intro = re.search(r"##\s*人物(?:介绍|简介)\s*\n(.*?)(?=##\s*全文转录|$)", content, re.DOTALL)
    if m_intro:
        speaker_intro = m_intro.group(1).strip()

    # --- Parse 全文转录 section ---
    m_full = re.search(r"##\s*全文转录\s*\n(.*)", content, re.DOTALL)
    full_text = m_full.group(1).strip() if m_full else content.strip()

    return OutputDoc(
        title=title,
        podcast_name=podcast_name,
        pub_date=pub_date,
        show_notes=show_notes,
        key_points=key_points,
        highlight_quotes=highlight_quotes,
        full_text=full_text,
        speaker_intro=speaker_intro,
    )
