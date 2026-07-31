"""撷声 CLI.

Usage:
    xiesheng https://www.xiaoyuzhoufm.com/episode/xxx
    xiesheng https://www.xiaoyuzhoufm.com/episode/xxx -o output/
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

from src.audio import convert_to_wav, download_audio, get_duration_seconds
from src.config import DEFAULT_LLM_PROVIDER, LLM_MODELS, get_llm_config
from src.models.schemas import OutputDoc
from src.processor import llm_processor
from src.scraper.xiaoyuzhou import scrape_episode
from src.utils import CostTracker, Timer, fmt_time, safe_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xiesheng")


def build_output_markdown(doc: OutputDoc) -> str:
    """Build the final markdown document."""
    lines = [
        f"# {doc.title}",
        "",
        f"> 来源：{doc.podcast_name}  |  {doc.pub_date}",
    ]
    if doc.show_notes:
        lines += ["", "# Show Notes", "", doc.show_notes]

    lines += ["", "## 内容提要", ""]
    for kp in doc.key_points:
        evidence = f"：{kp.evidence}" if kp.evidence else ""
        lines.append(f"- **{kp.point}**{evidence}")

    if doc.highlight_quotes:
        lines += ["", "## 闪光语句", ""]
        for q in doc.highlight_quotes:
            lines.append(f"- {q}")

    if doc.speaker_intro:
        lines += ["", "## 人物简介", "", doc.speaker_intro]

    lines += ["", "## 全文转录", "", doc.full_text]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="撷声 - 输入小宇宙播客单集链接，输出结构化 Markdown 文稿"
    )
    parser.add_argument("url", help="小宇宙播客单集链接")
    parser.add_argument("-o", "--output", default="output", help="输出目录 (默认: output/)")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper 模型大小 (默认: medium)")
    parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=sorted(LLM_MODELS),
                        help="LLM 后处理提供方 (默认: qwen)")
    parser.add_argument("--llm-model", help="覆盖提供方的默认模型")
    parser.add_argument("--no-llm", action="store_true", help="仅转录，不进行 LLM 后处理")
    parser.add_argument("--no-diarization", action="store_true", help="跳过说话人日志，不区分说话人")
    parser.add_argument("--speakers", type=int, help="已知说话人数；默认自动检测")
    args = parser.parse_args()

    llm_config = get_llm_config(args.llm_provider, args.llm_model)
    if not args.no_llm and not llm_config.api_key:
        print(
            "错误：LLM_API_KEY 未设置\n"
            "请设置 Windows 用户环境变量 LLM_API_KEY",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = CostTracker()
    timers = {}

    try:
        # --- Step 1: Scrape ---
        with Timer() as t:
            logger.info("Step 1/6: 爬取节目信息...")
            episode = scrape_episode(args.url)
            print(f"  → {episode.title}")
            print(f"  播客: {episode.podcast_name}")
        timers["scrape"] = t.elapsed

        # --- Step 2: Download audio ---
        with Timer() as t:
            logger.info("Step 2/6: 下载音频...")
            audio_file = download_audio(episode.audio_url, output_dir)
            wav_file = convert_to_wav(audio_file, output_dir)
            duration_sec = get_duration_seconds(wav_file)
            print(f"  → 音频时长: {fmt_time(duration_sec)}")
        timers["download"] = t.elapsed

        # --- Step 3: Transcribe ---
        with Timer() as t:
            from src.transcriber.local import LocalTranscriber

            logger.info("Step 3/6: 本地转录中（faster-whisper %s, CPU）...", args.model)
            transcriber = LocalTranscriber(model_size=args.model)
            transcript = transcriber.transcribe(wav_file, duration_sec)
            tracker.add_transcription(transcript.cost_yuan)
        timers["transcribe"] = t.elapsed
        char_count = len(transcript.raw_text)
        rtf = t.elapsed / duration_sec if duration_sec else 0
        est_2h = int(7200 * rtf)
        print(f"  → 转录完成：{char_count} 字, {fmt_time(t.elapsed)}, RTF={rtf:.2f}")
        print(f"  → [估算] 2小时节目约需 {fmt_time(est_2h)}（当前模型: {args.model}）")

        safe_name = safe_filename(episode.title)
        raw_path = output_dir / f"{safe_name}_raw.txt"
        raw_path.write_text(transcript.raw_text, encoding="utf-8")
        logger.info("原始转录已保存到 %s", raw_path)

        if args.no_llm:
            print(f"\n原始转录已保存到: {raw_path}")
        else:
            # --- Step 4: Speaker Diarization ---
            transcript_text = transcript.raw_text
            if not args.no_diarization:
                with Timer() as t:
                    from src.diarization.speaker_diarization import assign_speakers, run_diarization

                    logger.info("Step 4/6: 说话人日志（Speaker Diarization）...")
                    try:
                        diarization_segments = run_diarization(wav_file, num_speakers=args.speakers)
                        transcript_text = assign_speakers(transcript.segments, diarization_segments)
                    except Exception as e:
                        logger.warning("Diarization 失败，跳过: %s", e)
                        transcript_text = transcript.raw_text
                timers["diarization"] = t.elapsed

            # --- Step 5: LLM Process ---
            with Timer() as t:
                logger.info("Step 5/6: %s/%s 分块校订与整理中...", llm_config.provider, llm_config.model)
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
                    work_dir=output_dir / ".work" / safe_name,
                )
                tracker.add_llm_usage(inp_tok, out_tok)
            timers["process"] = t.elapsed

            # --- Step 6: Write output ---
            with Timer() as t:
                logger.info("Step 6/6: 写入 Markdown...")
                doc.costs = {
                    "transcription": tracker.transcription_yuan,
                    "llm": tracker.llm_cost_yuan if tracker.llm_cost_known else None,
                }
                doc.timings = timers
                md_content = build_output_markdown(doc)
                output_path = output_dir / f"{safe_name}.md"
                output_path.write_text(md_content, encoding="utf-8")
            timers["write"] = t.elapsed

            total_time = sum(timers.values())
            print(f"\n{'='*50}")
            print("[OK] 完成！")
            print(f"  输出文件: {output_path}")
            print(f"  总耗时: {fmt_time(total_time)}")
            print(f"  费用: {tracker.summary()}")
            print(f"  要点数: {len(doc.key_points)}")
            print(f"  闪光语句: {len(doc.highlight_quotes)}")
            print(f"{'='*50}")

    except Exception as e:
        logger.exception("处理失败")
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
