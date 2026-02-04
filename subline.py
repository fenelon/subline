#!/usr/bin/env python3
"""Generate SRT/VTT subtitles from video files using faster-whisper.

Usage:
    subline [options] <path...>

Examples:
    subline video.mp4
    subline --language es --model turbo --audio-track 2 video.mp4
    subline --skip-existing /path/to/videos/
    subline --format vtt --output-dir ./subs video1.mp4 video2.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wav", ".flac", ".mp3"}


def check_ffmpeg():
    """Exit if ffmpeg or ffprobe are not on PATH."""
    for cmd in ("ffmpeg", "ffprobe"):
        if shutil.which(cmd) is None:
            print(f"Error: '{cmd}' not found on PATH. Install ffmpeg first.")
            sys.exit(1)


def detect_device():
    """Return best available device. CTranslate2 supports cuda and cpu only."""
    try:
        import ctranslate2

        if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
            return "cuda"
    except (ImportError, RuntimeError):
        pass
    return "cpu"


def get_audio_tracks(video_path):
    """Return list of (stream_index, language) for audio tracks."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type:stream_tags=language",
            "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True,
    )
    tracks = []
    for line in result.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) >= 2 and parts[1] == "audio":
            idx = int(parts[0])
            lang = parts[2] if len(parts) > 2 else "und"
            tracks.append((idx, lang))
    return tracks


def extract_audio(video_path, stream_index, output_path):
    """Extract audio track to 16kHz mono WAV."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-map", f"0:{stream_index}",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(output_path),
        ],
        capture_output=True,
    )


def get_audio_duration(wav_path):
    """Get duration in seconds from an audio file."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0


def format_timestamp(seconds, fmt):
    """Convert seconds to SRT or VTT timestamp format."""
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    ts = f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    if fmt == "srt":
        return ts.replace(".", ",")
    return ts  # vtt uses dot


def transcribe_file(wav_path, out_path, model, language, fmt):
    """Transcribe WAV to SRT/VTT with progress indication."""
    duration = get_audio_duration(wav_path)
    print(f"  Transcribing {duration / 60:.0f} min of audio...", flush=True)

    start_time = time.time()
    kwargs = {"language": language} if language else {}
    segments, info = model.transcribe(str(wav_path), **kwargs)

    if not language:
        print(f"  Detected language: {info.language} ({info.language_probability:.0%})", flush=True)

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        if fmt == "vtt":
            f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            if fmt == "srt":
                f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start, fmt)} --> {format_timestamp(seg.end, fmt)}\n")
            f.write(f"{seg.text.strip()}\n\n")
            f.flush()
            count = i

            if i % 50 == 0:
                pct = min(seg.end / duration * 100, 100) if duration else 0
                elapsed = time.time() - start_time
                if pct > 0:
                    eta = elapsed / pct * (100 - pct)
                    eta_m, eta_s = divmod(int(eta), 60)
                    eta_str = f", ETA {eta_m}m{eta_s:02d}s"
                else:
                    eta_str = ""
                print(f"  [{pct:5.1f}%] {i} segments, {elapsed:.0f}s elapsed{eta_str}", flush=True)

    elapsed = time.time() - start_time
    em, es = divmod(int(elapsed), 60)
    print(f"  Done: {count} segments in {em}m{es:02d}s -> {out_path}", flush=True)


def find_videos(paths):
    """Find video/audio files from a list of paths."""
    found = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS))
        else:
            print(f"Warning: skipping '{p}' (not a file or directory)")
    return found


def pick_audio_track(video, manual_track):
    """Select the audio stream index to use."""
    if manual_track is not None:
        print(f"  Using audio track {manual_track} (manual)")
        return manual_track

    tracks = get_audio_tracks(video)
    if not tracks:
        print("  No audio tracks found, skipping")
        return None
    if len(tracks) == 1:
        print(f"  Using audio track {tracks[0][0]} ({tracks[0][1]})")
        return tracks[0][0]

    # Multiple tracks: use the first one, hint about --audio-track
    print(f"  Audio tracks: {[(i, l) for i, l in tracks]}")
    stream_idx = tracks[0][0]
    print(f"  Using first track {stream_idx} ({tracks[0][1]}). "
          f"Override with --audio-track N")
    return stream_idx


def main():
    print("\n\033[1m"
          "░▄▀▀░█▒█░██▄░█▒░░█░█▄░█▒██▀\n"
          "▒▄██░▀▄█▒█▄█▒█▄▄░█░█▒▀█░█▄▄\n"
          "\033[0m")

    parser = argparse.ArgumentParser(
        prog="subline",
        description="Generate SRT/VTT subtitles from video files using faster-whisper",
    )
    parser.add_argument("--model", default="turbo",
                        help="Whisper model (tiny/base/small/medium/turbo, default: turbo)")
    parser.add_argument("--language", default=None,
                        help="Language code (auto-detect if omitted)")
    parser.add_argument("--audio-track", type=int, default=None,
                        help="Audio stream index (auto-detect if omitted)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have a subtitle file")
    parser.add_argument("--format", choices=["srt", "vtt"], default="srt",
                        help="Output format (default: srt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write subtitle files (default: next to source)")
    parser.add_argument("--device", default=None,
                        help="Compute device: cuda or cpu (default: auto-detect)")
    parser.add_argument("path", nargs="+",
                        help="Video/audio files or directories to transcribe")
    args = parser.parse_args()

    check_ffmpeg()

    videos = find_videos(args.path)
    if not videos:
        print(f"No media files found in: {' '.join(args.path)}")
        sys.exit(1)

    device = args.device or detect_device()
    compute_type = "float16" if device == "cuda" else "int8"

    lang_str = args.language or "auto-detect"
    print(f"Found {len(videos)} file(s) | model={args.model} | "
          f"language={lang_str} | device={device}\n")

    # Load model once
    from faster_whisper import WhisperModel

    print(f"Loading model '{args.model}' on {device}...", flush=True)
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    print()

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for vi, video in enumerate(videos, 1):
        print(f"[{vi}/{len(videos)}] {video.name}")

        # Determine output path
        if args.output_dir:
            out_path = Path(args.output_dir) / video.with_suffix(f".{args.format}").name
        else:
            out_path = video.with_suffix(f".{args.format}")

        if args.skip_existing and out_path.exists():
            print("  Skipping (subtitle file exists)", flush=True)
            continue

        stream_idx = pick_audio_track(video, args.audio_track)
        if stream_idx is None:
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            print(f"  Extracting audio (track {stream_idx})...", flush=True)
            extract_audio(video, stream_idx, wav_path)
            transcribe_file(wav_path, out_path, model, args.language, args.format)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

        print()

    print("All done.\n")


if __name__ == "__main__":
    main()
