"""Audio download and conversion utilities."""

import logging
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)


def download_audio(url: str, output_dir: Path | None = None) -> Path:
    """Download audio file from URL. Returns path to downloaded file."""
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "tingmo-audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = _guess_extension(url)
    dest = output_dir / f"podcast_audio{ext}"

    logger.info("Downloading audio from %s ...", url)
    urlretrieve(url, str(dest))
    logger.info("Downloaded to %s (%.1f MB)", dest, dest.stat().st_size / 1_000_000)
    return dest


def convert_to_wav(input_path: Path, output_dir: Path | None = None) -> Path:
    """Convert audio to WAV 16kHz mono using ffmpeg."""
    if output_dir is None:
        output_dir = input_path.parent
    output_path = output_dir / "podcast_audio.wav"

    logger.info("Converting to WAV 16kHz mono...")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_path),
        ],
        check=True, capture_output=True, timeout=600,
    )
    logger.info("Converted to %s", output_path)
    return output_path


def get_duration_seconds(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True, capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip())


def _guess_extension(url: str) -> str:
    """Guess file extension from URL."""
    path = url.split("?")[0].rstrip("/")
    for ext in (".mp3", ".m4a", ".wav", ".ogg", ".aac"):
        if path.endswith(ext):
            return ext
    return ".mp3"  # default
