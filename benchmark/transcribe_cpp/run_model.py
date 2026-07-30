"""Run one transcribe.cpp model against mono 16 kHz float32 PCM."""

import argparse
import json
import time
from array import array
from pathlib import Path

import transcribe_cpp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pcm", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--n-ctx", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    chunk_results = []
    audio_seconds = 0.0
    with transcribe_cpp.Model(args.model, backend=args.backend) as model:
        loaded = time.perf_counter()
        for pcm_path in args.pcm:
            pcm = array("f")
            with pcm_path.open("rb") as stream:
                pcm.fromfile(stream, pcm_path.stat().st_size // pcm.itemsize)
            with model.session(n_ctx=args.n_ctx) as session:
                result = session.run(pcm, timestamps="auto")
            seconds = len(pcm) / 16_000
            audio_seconds += seconds
            chunk_results.append({
                "file": pcm_path.name,
                "audio_seconds": seconds,
                "language": result.language,
                "timestamp_kind": result.timestamp_kind,
                "text": result.text,
                "segments": [segment.__dict__ for segment in result.segments],
            })
    finished = time.perf_counter()

    payload = {
        "library_version": transcribe_cpp.__version__,
        "native_version": transcribe_cpp.native_version(),
        "provider": transcribe_cpp.native_provider(),
        "backends": [device.__dict__ for device in transcribe_cpp.backends()],
        "model": args.model.name,
        "backend": args.backend,
        "n_ctx": args.n_ctx,
        "audio_seconds": audio_seconds,
        "model_load_seconds": loaded - started,
        "transcribe_seconds": finished - loaded,
        "language": ",".join(dict.fromkeys(chunk["language"] for chunk in chunk_results)),
        "timestamp_kind": chunk_results[0]["timestamp_kind"],
        "text": "\n".join(chunk["text"] for chunk in chunk_results),
        "chunks": chunk_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "model", "backend", "audio_seconds", "model_load_seconds",
        "transcribe_seconds", "language", "timestamp_kind",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
