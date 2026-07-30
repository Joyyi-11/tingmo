"""Local transcription using faster-whisper (CPU) with long-audio chunking."""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Limit MKL/OpenBLAS threads to reduce memory usage on constrained systems
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from faster_whisper import WhisperModel

from src.models.schemas import TranscriptResult
from src.transcriber.base import Transcriber

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "medium"
CHUNK_SECONDS = 600  # 10 min chunks to avoid OOM on long audio
MODEL_CACHE = str(Path.home() / ".cache" / "faster_whisper")


class LocalTranscriber(Transcriber):
    """Transcribe locally using faster-whisper on CPU."""

    def __init__(self, model_size: str = DEFAULT_MODEL):
        logger.info("Loading faster-whisper model: %s (CPU, int8)...", model_size)
        t0 = time.time()

        # Try local cache first to avoid HF network issues
        cache_path = Path(MODEL_CACHE) / ("models--Systran--faster-whisper-" + model_size)
        snapshots = cache_path / "snapshots"
        if snapshots.exists():
            snapshot_dirs = sorted(
                snapshots.iterdir(),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshot_dirs:
                local_model_path = snapshot_dirs[0]
                if (local_model_path / "model.bin").exists():
                    logger.info("Found cached model at %s, loading locally...", local_model_path)
                    self.model = WhisperModel(
                        str(local_model_path),
                        device="cpu",
                        compute_type="int8",
                    )
                    elapsed = time.time() - t0
                    logger.info("Model loaded in %.1f sec", elapsed)
                    return

        # Download via Hub
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=MODEL_CACHE,
        )
        elapsed = time.time() - t0
        logger.info("Model loaded in %.1f sec", elapsed)

    def transcribe(self, audio_path: Path, duration_sec: float | None = None) -> TranscriptResult:
        t0 = time.time()

        # Chunk long audio to avoid out-of-memory in feature extractor
        if (duration_sec or 0) > CHUNK_SECONDS:
            chunk_paths = self._split_audio(audio_path, duration_sec)
        else:
            chunk_paths = [audio_path]

        all_text_parts = []
        all_segments = []
        total_duration = 0.0

        try:
            for i, chunk in enumerate(chunk_paths):
                logger.info("Transcribing chunk %d/%d: %s...", i + 1, len(chunk_paths), chunk.name)
                segments, info = self.model.transcribe(
                    str(chunk),
                    language="zh",
                    beam_size=5,
                    word_timestamps=False,
                )
                for seg in segments:
                    all_text_parts.append(seg.text.strip())
                    all_segments.append({
                        "text": seg.text.strip(),
                        "start_time": seg.start + i * CHUNK_SECONDS,
                        "end_time": seg.end + i * CHUNK_SECONDS,
                    })
                total_duration += info.duration if info and info.duration else 0
        finally:
            temp_dirs = {chunk.parent for chunk in chunk_paths if chunk != audio_path}
            for chunk in chunk_paths:
                if chunk != audio_path:
                    chunk.unlink(missing_ok=True)
            for temp_dir in temp_dirs:
                try:
                    temp_dir.rmdir()
                except OSError:
                    logger.warning("Could not remove temporary chunk directory: %s", temp_dir)

        full_text = "\n".join(all_text_parts)
        actual_duration = total_duration or duration_sec or 0
        elapsed = time.time() - t0

        logger.info(
            "Transcription done: %d chars in %.1f sec (RTF=%.2f)",
            len(full_text), elapsed, elapsed / actual_duration if actual_duration else 0,
        )

        return TranscriptResult(
            raw_text=full_text,
            segments=all_segments,
            duration_sec=actual_duration,
            cost_yuan=0.0,
        )

    def _split_audio(self, audio_path: Path, duration_sec: float | None) -> list[Path]:
        """Split audio into CHUNK_SECONDS chunks using ffmpeg."""
        if duration_sec is None:
            # probe duration
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                check=True, capture_output=True, text=True, timeout=30,
            )
            duration_sec = float(result.stdout.strip())

        n_chunks = int(duration_sec) // CHUNK_SECONDS + 1
        logger.info("Splitting %.1f sec audio into %d chunks (%d sec each)...",
                     duration_sec, n_chunks, CHUNK_SECONDS)

        tmp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))
        out_pattern = str(tmp_dir / "chunk_%03d.wav")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-f", "segment",
                "-segment_time", str(CHUNK_SECONDS),
                "-c", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                out_pattern,
            ],
            check=True, capture_output=True, timeout=3600,
        )

        chunk_files = sorted(tmp_dir.glob("chunk_*.wav"))
        logger.info("Split into %d chunks", len(chunk_files))
        return chunk_files
