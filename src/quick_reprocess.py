r"""Quick re-run: transcribe + LLM process from existing audio.

Usage:
    python -m src.quick_reprocess <xiaoyuzhou-url>
"""

import argparse
import logging
import sys
from pathlib import Path

# Windows terminal encoding fix
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("quick-reprocess")

OUTPUT_DIR = Path(__file__).parent.parent / "output"
AUDIO_WAV = OUTPUT_DIR / "podcast_audio.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description="使用已有音频重新转录和整理")
    parser.add_argument("url", help="小宇宙播客单集链接")
    parser.add_argument("--audio", type=Path, default=AUDIO_WAV, help="已有 WAV 音频路径")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--llm-provider", default="qwen", choices=["qwen", "deepseek"])
    parser.add_argument("--llm-model", help="覆盖提供方的默认模型")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"错误：找不到音频文件 {args.audio}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Transcribe (skip diarization to avoid OOM) ---
    logger.info("转录中...")
    from src.audio import get_duration_seconds
    from src.transcriber.local import LocalTranscriber

    duration_sec = get_duration_seconds(args.audio)
    transcriber = LocalTranscriber(model_size=args.model)
    transcript = transcriber.transcribe(args.audio, duration_sec)
    print(f"  → 转录完成：{len(transcript.raw_text)} 字")

    # Use raw transcript directly (no diarization)
    transcript_text = transcript.raw_text

    # --- Step 2: LLM Process ---
    logger.info("DeepSeek 后处理中...")
    from src.config import get_llm_config
    from src.processor import llm_processor

    llm_config = get_llm_config(args.llm_provider, args.llm_model)
    if not llm_config.api_key:
        print("错误：LLM_API_KEY 未设置", file=sys.stderr)
        sys.exit(1)

    # Scrape episode metadata for show_notes
    from src.scraper.xiaoyuzhou import scrape_episode

    episode = scrape_episode(args.url)

    doc, inp_tok, out_tok = llm_processor.process(
        llm_config.api_key,
        episode.title,
        episode.podcast_name,
        episode.pub_date,
        episode.show_notes,
        transcript_text,
        base_url=llm_config.base_url,
        provider=llm_config.provider,
        model=llm_config.model,
        work_dir=OUTPUT_DIR / ".work" / "quick_reprocess",
    )

    # --- Step 3: Write output ---
    from src.main import build_output_markdown
    from src.utils import safe_filename

    md_content = build_output_markdown(doc)
    safe_name = safe_filename(episode.title)
    output_path = OUTPUT_DIR / f"{safe_name}.md"
    output_path.write_text(md_content, encoding="utf-8")

    print(f"\n{'='*50}")
    print("[OK] 完成！")
    print(f"  输出文件: {output_path}")
    print(f"  输入 token: {inp_tok:,}")
    print(f"  输出 token: {out_tok:,}")
    print(f"  要点数: {len(doc.key_points)}")
    print(f"  闪光语句: {len(doc.highlight_quotes)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
