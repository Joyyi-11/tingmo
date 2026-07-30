"""Reproduce the deletion-aware E236 benchmark metrics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BENCHMARK_DIR.parent.parent
QWEN_06_RESULT = "e236_36m_56m_qwen3_asr_0.6b_q8_chunk30.json"
QWEN_17_RESULT = "e236_46m30s_51m30s_qwen3_asr_1.7b_q5km_chunk30.json"

TERM_SAMPLE = (
    "Jack", "Claude", "OpenAI", "Codex", "coding agent", "Variant",
    "Typeless", "Vibe Coding", "Dify", "Alfred", "Chatbot",
    "Long Horizon Agent", "Gemini", "Raycast", "unstructure to structure",
    "OpenClaw", "agent orchestration layer", "Manus", "deep research",
    "designing", "Claude Code", "Kolento", "Claude Cowork", "Suno",
    "Startup", "Landing Page", "Dynamic View", "Web Dev", "Preview",
)


def normalize(text: str) -> str:
    text = re.sub(r"^(?:主播|主持人|嘉宾)\s*[^：:\n]{0,20}[：:]", "", text.strip())
    text = re.sub(r"[*_`\\]", "", text)
    return "".join(char.lower() for char in text if char.isalnum())


def ngram_coverage(reference: str, candidate: str, size: int = 3) -> float:
    reference_ngrams = Counter(
        reference[index:index + size]
        for index in range(max(0, len(reference) - size + 1))
    )
    candidate_ngrams = Counter(
        candidate[index:index + size]
        for index in range(max(0, len(candidate) - size + 1))
    )
    total = sum(reference_ngrams.values())
    return sum((reference_ngrams & candidate_ngrams).values()) / total if total else 0.0


def transcript_paragraphs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").split("## 播客摘录", 1)[1]
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
        and not paragraph.lstrip().startswith(("![", ">", "#"))
    ]


def json_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_term(text: str, term: str) -> bool:
    return normalize(term) in normalize(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of readable text")
    parser.add_argument("--published", type=Path, required=True, help="Human-edited reference Markdown")
    parser.add_argument("--current", type=Path, required=True, help="Current pipeline output Markdown")
    args = parser.parse_args()

    published = args.published
    current = args.current
    results_dir = BENCHMARK_DIR / "results"
    qwen_06 = json_result(results_dir / QWEN_06_RESULT)
    qwen_17 = json_result(results_dir / QWEN_17_RESULT)

    paragraphs = transcript_paragraphs(published)
    current_lines = current.read_text(encoding="utf-8").splitlines()

    # The current output provides time anchors. Keep only published paragraphs
    # that closely overlap its 36:00-56:00 slice, so editorial deletions do not
    # count as ASR omissions.
    current_20m = "\n".join(current_lines[250:367])
    normalized_current_20m = normalize(current_20m)
    selected = [
        paragraph for paragraph in paragraphs
        if ngram_coverage(normalize(paragraph), normalized_current_20m) >= 0.45
    ]
    reference_20m = "".join(normalize(paragraph) for paragraph in selected)

    full_metrics = {
        "reference_paragraphs": len(selected),
        "reference_normalized_chars": len(reference_20m),
        "current_output_3gram_coverage": ngram_coverage(
            reference_20m, normalized_current_20m
        ),
        "qwen_0.6b_3gram_coverage": ngram_coverage(
            reference_20m, normalize(qwen_06["text"])
        ),
    }

    # A fixed, terminology-dense five-minute interval used for the 1.7B run.
    reference_5m = "".join(normalize(paragraph) for paragraph in paragraphs[61:73])
    current_5m = "\n".join(current_lines[304:335])
    qwen_06_5m = "\n".join(chunk["text"] for chunk in qwen_06["chunks"][21:31])
    qwen_17_5m = qwen_17["text"]
    candidates = {
        "current_output": current_5m,
        "qwen_0.6b": qwen_06_5m,
        "qwen_1.7b": qwen_17_5m,
    }
    focused_metrics = {
        name: {
            "3gram_coverage": ngram_coverage(reference_5m, normalize(text)),
            "term_hits": sum(contains_term(text, term) for term in TERM_SAMPLE),
            "term_total": len(TERM_SAMPLE),
            "missing_terms": [term for term in TERM_SAMPLE if not contains_term(text, term)],
        }
        for name, text in candidates.items()
    }

    payload = {"full_20m": full_metrics, "focused_5m": focused_metrics}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("E236 36:00-56:00 deletion-aware reference")
    print(f"  paragraphs: {full_metrics['reference_paragraphs']}")
    print(f"  normalized chars: {full_metrics['reference_normalized_chars']}")
    print(f"  current output: {full_metrics['current_output_3gram_coverage']:.1%}")
    print(f"  Qwen3-ASR 0.6B: {full_metrics['qwen_0.6b_3gram_coverage']:.1%}")
    print("\nE236 46:30-51:30 terminology-dense interval")
    for name, metrics in focused_metrics.items():
        print(
            f"  {name}: coverage={metrics['3gram_coverage']:.1%}, "
            f"terms={metrics['term_hits']}/{metrics['term_total']}"
        )


if __name__ == "__main__":
    main()
