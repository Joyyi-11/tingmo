"""Run a faithful-cleaning A/B test on one Whisper transcript slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

BENCHMARK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCHMARK_DIR / "results"
DEFAULT_AUDIO = PROJECT_DIR / "benchmark/transcribe_cpp/assets/audio/e236_36m_56m.wav"
DEFAULT_RAW = RESULTS_DIR / "e236_36m_56m_whisper_medium_raw.json"
MODEL_NAMES = {
    "qwen": "qwen3.7-plus",
    "deepseek": "deepseek-v4-flash",
}
CHUNK_MAX_CHARS = 2_000


def extract_show_notes(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    try:
        return text.split("# Show Notes", 1)[1].split("## 内容提要", 1)[0].strip()
    except IndexError as exc:
        raise RuntimeError(f"Could not extract Show Notes from {path}") from exc


def transcribe(audio: Path, output: Path) -> None:
    from src.transcriber.local import LocalTranscriber

    started = time.perf_counter()
    result = LocalTranscriber(model_size="medium").transcribe(audio, duration_sec=1_200)
    payload = {
        "model": "faster-whisper-medium",
        "compute_type": "int8",
        "language": "zh",
        "beam_size": 5,
        "audio": audio.name,
        "audio_seconds": result.duration_sec,
        "elapsed_seconds": time.perf_counter() - started,
        "text": result.raw_text,
        "segments": result.segments,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "chars": len(result.raw_text),
                "segments": len(result.segments),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


def clean(provider: str, raw_path: Path, source_output: Path) -> None:
    from src.processor.llm_processor import (
        CONTEXT_CHARS,
        _load_cached_chunk,
        _request_json,
        _save_cached_chunk,
        _validate_cleaned_chunk,
        split_transcript,
    )
    from src.processor.prompt import CLEAN_SYSTEM_PROMPT, CLEAN_USER_PROMPT

    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set in the process environment")
    base_url = os.getenv("LLM_BASE_URL", "https://voltapi.ai/v1").rstrip("/")
    model = MODEL_NAMES[provider]
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_text = raw_payload["text"]
    show_notes = extract_show_notes(source_output)
    chunks = split_transcript(raw_text, max_chars=CHUNK_MAX_CHARS)
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache_dir = RESULTS_DIR / "cache" / provider

    started = time.perf_counter()
    cleaned_chunks = []
    input_tokens = 0
    output_tokens = 0
    for index, chunk in enumerate(chunks):
        chunk_id = index + 1
        cached = _load_cached_chunk(cache_dir, chunk_id, chunk, provider, model)
        if cached is not None:
            cleaned_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_chars": len(chunk),
                    "cleaned_chars": len(cached),
                    "cleaned_text": cached,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached": True,
                }
            )
            continue
        previous_context = chunks[index - 1][-CONTEXT_CHARS:] if index else "（无）"
        next_context = chunks[index + 1][:CONTEXT_CHARS] if index + 1 < len(chunks) else "（无）"
        prompt = CLEAN_USER_PROMPT.format(
            show_notes=show_notes[:6_000],
            previous_context=previous_context,
            chunk_id=chunk_id,
            chunk_text=chunk,
            next_context=next_context,
        )
        data, prompt_tokens, completion_tokens = _request_json(
            client,
            model,
            CLEAN_SYSTEM_PROMPT,
            prompt,
            max_tokens=8_192,
        )
        cleaned = _validate_cleaned_chunk(data, chunk_id, chunk)
        _save_cached_chunk(cache_dir, chunk_id, chunk, cleaned, provider, model)
        cleaned_chunks.append(
            {
                "chunk_id": chunk_id,
                "source_chars": len(chunk),
                "cleaned_chars": len(cleaned),
                "cleaned_text": cleaned,
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            }
        )
        input_tokens += prompt_tokens
        output_tokens += completion_tokens

    payload = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "source": raw_path.name,
        "source_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "context_chars": CONTEXT_CHARS,
        "elapsed_seconds": time.perf_counter() - started,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "text": "\n\n".join(chunk["cleaned_text"] for chunk in cleaned_chunks),
        "chunks": cleaned_chunks,
    }
    output = RESULTS_DIR / f"e236_36m_56m_{provider}_{model}_cleaned.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "chunks": len(chunks),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe_parser = subparsers.add_parser("transcribe")
    transcribe_parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    transcribe_parser.add_argument("--output", type=Path, default=DEFAULT_RAW)

    clean_parser = subparsers.add_parser("clean")
    clean_parser.add_argument("provider", choices=sorted(MODEL_NAMES))
    clean_parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    clean_parser.add_argument("--source-output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "transcribe":
        transcribe(args.audio, args.output)
    else:
        clean(args.provider, args.raw, args.source_output)


if __name__ == "__main__":
    main()
