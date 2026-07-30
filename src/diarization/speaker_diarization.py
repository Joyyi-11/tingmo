"""Speaker diarization — assign speaker labels to transcribed segments."""

import logging
from pathlib import Path

from diarize import diarize

logger = logging.getLogger(__name__)

SPEAKER_GAP = 0.5  # seconds — merge whisper segments with smaller gaps


def run_diarization(audio_path: Path, num_speakers: int | None = None) -> list[dict]:
    """Run speaker diarization on an audio file.

    Returns list of dicts with start, end, speaker keys.
    """
    logger.info("Running speaker diarization on %s...", audio_path.name)
    result = diarize(str(audio_path), num_speakers=num_speakers)
    segments = [
        {"start": seg.start, "end": seg.end, "speaker": seg.speaker}
        for seg in result.segments
    ]
    logger.info(
        "Diarization done: %d speakers, %d segments",
        result.num_speakers,
        len(segments),
    )
    return segments


def assign_speakers(
    whisper_segments: list[dict],
    diarization_segments: list[dict],
) -> str:
    """Merge diarization speaker labels with Whisper text segments.

    Each whisper_segment: {"text": ..., "start_time": ..., "end_time": ...}
    Each diarization_segment: {"start": ..., "end": ..., "speaker": ...}

    Returns formatted transcript with [SPEAKER_XX] labels per utterance group.
    """
    # Assign best-overlap speaker to each whisper segment
    labeled = []
    for wseg in whisper_segments:
        ws, we = wseg["start_time"], wseg["end_time"]
        best_overlap, best_speaker = 0.0, "UNKNOWN"
        for dseg in diarization_segments:
            overlap = min(we, dseg["end"]) - max(ws, dseg["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = dseg["speaker"]
        labeled.append({"speaker": best_speaker, "text": wseg["text"]})

    # Merge consecutive same-speaker segments into utterances
    merged = []
    for seg in labeled:
        if merged and merged[-1]["speaker"] == seg["speaker"]:
            merged[-1]["text"] += " " + seg["text"]
        else:
            merged.append({"speaker": seg["speaker"], "text": seg["text"]})

    return "\n".join(f"[{seg['speaker']}] {seg['text']}" for seg in merged)
