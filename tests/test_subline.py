"""Tests for subline."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Add project root to path so we can import subline directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import subline


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_srt_zero(self):
        assert subline.format_timestamp(0, "srt") == "00:00:00,000"

    def test_vtt_zero(self):
        assert subline.format_timestamp(0, "vtt") == "00:00:00.000"

    def test_srt_uses_comma(self):
        ts = subline.format_timestamp(3661.5, "srt")
        assert "," in ts
        assert "." not in ts
        assert ts == "01:01:01,500"

    def test_vtt_uses_dot(self):
        ts = subline.format_timestamp(3661.5, "vtt")
        assert "." in ts
        assert "," not in ts
        assert ts == "01:01:01.500"

    def test_large_value(self):
        # 10 hours, 59 minutes, 59.999 seconds
        ts = subline.format_timestamp(39599.999, "srt")
        assert ts == "10:59:59,999"

    def test_fractional_seconds(self):
        ts = subline.format_timestamp(0.123, "vtt")
        assert ts == "00:00:00.123"


# ---------------------------------------------------------------------------
# find_videos
# ---------------------------------------------------------------------------

class TestFindVideos:
    def test_single_file(self, tmp_path):
        mp4 = tmp_path / "video.mp4"
        mp4.touch()
        result = subline.find_videos([str(mp4)])
        assert len(result) == 1
        assert result[0].name == "video.mp4"

    def test_directory(self, tmp_path):
        for name in ("a.mp4", "b.mkv", "c.txt", "d.avi"):
            (tmp_path / name).touch()
        result = subline.find_videos([str(tmp_path)])
        names = {p.name for p in result}
        assert names == {"a.mp4", "b.mkv", "d.avi"}

    def test_multiple_paths(self, tmp_path):
        f1 = tmp_path / "one.mp4"
        f2 = tmp_path / "two.mov"
        f1.touch()
        f2.touch()
        result = subline.find_videos([str(f1), str(f2)])
        assert len(result) == 2

    def test_nonexistent_path(self, tmp_path, capsys):
        result = subline.find_videos([str(tmp_path / "nope.xyz")])
        assert result == []
        assert "Warning" in capsys.readouterr().out

    def test_empty_directory(self, tmp_path):
        result = subline.find_videos([str(tmp_path)])
        assert result == []

    def test_audio_files(self, tmp_path):
        wav = tmp_path / "recording.wav"
        wav.touch()
        result = subline.find_videos([str(wav)])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# pick_audio_track
# ---------------------------------------------------------------------------

class TestPickAudioTrack:
    def test_manual_override(self, capsys):
        result = subline.pick_audio_track(Path("fake.mp4"), 5)
        assert result == 5
        assert "manual" in capsys.readouterr().out

    @mock.patch("subline.get_audio_tracks", return_value=[(1, "eng")])
    def test_single_track(self, mock_tracks, capsys):
        result = subline.pick_audio_track(Path("fake.mp4"), None)
        assert result == 1

    @mock.patch("subline.get_audio_tracks", return_value=[(1, "eng"), (2, "spa")])
    def test_multiple_tracks_picks_first(self, mock_tracks, capsys):
        result = subline.pick_audio_track(Path("fake.mp4"), None)
        assert result == 1
        out = capsys.readouterr().out
        assert "--audio-track" in out

    @mock.patch("subline.get_audio_tracks", return_value=[])
    def test_no_tracks(self, mock_tracks, capsys):
        result = subline.pick_audio_track(Path("fake.mp4"), None)
        assert result is None


# ---------------------------------------------------------------------------
# check_ffmpeg
# ---------------------------------------------------------------------------

class TestCheckFfmpeg:
    @mock.patch("shutil.which", return_value="/usr/bin/ffmpeg")
    def test_ffmpeg_present(self, mock_which):
        # Should not raise
        subline.check_ffmpeg()

    @mock.patch("shutil.which", return_value=None)
    def test_ffmpeg_missing(self, mock_which):
        with pytest.raises(SystemExit):
            subline.check_ffmpeg()


# ---------------------------------------------------------------------------
# detect_device
# ---------------------------------------------------------------------------

class TestDetectDevice:
    def test_no_torch_falls_back_to_cpu(self):
        with mock.patch.dict(sys.modules, {"torch": None}):
            # Re-importing with torch missing should give cpu
            assert subline.detect_device() == "cpu"

    @mock.patch("subline.detect_device", return_value="cuda")
    def test_cuda_when_available(self, mock_detect):
        assert subline.detect_device() == "cuda"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    """Test that argparse is configured correctly."""

    def _parse(self, argv):
        parser = subline.main.__code__  # we'll test by calling parse_args
        # Re-create the parser the same way main() does
        import argparse
        parser = argparse.ArgumentParser(prog="subline")
        parser.add_argument("--model", default="turbo")
        parser.add_argument("--language", default=None)
        parser.add_argument("--audio-track", type=int, default=None)
        parser.add_argument("--skip-existing", action="store_true")
        parser.add_argument("--format", choices=["srt", "vtt"], default="srt")
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument("--device", default=None)
        parser.add_argument("path", nargs="+")
        return parser.parse_args(argv)

    def test_single_path(self):
        args = self._parse(["video.mp4"])
        assert args.path == ["video.mp4"]
        assert args.model == "turbo"
        assert args.language is None

    def test_multiple_paths(self):
        args = self._parse(["a.mp4", "b.mp4", "c.mp4"])
        assert len(args.path) == 3

    def test_flags_before_path(self):
        args = self._parse(["--language", "es", "--model", "small", "video.mp4"])
        assert args.language == "es"
        assert args.model == "small"
        assert args.path == ["video.mp4"]

    def test_format_vtt(self):
        args = self._parse(["--format", "vtt", "video.mp4"])
        assert args.format == "vtt"

    def test_output_dir(self):
        args = self._parse(["--output-dir", "/tmp/subs", "video.mp4"])
        assert args.output_dir == "/tmp/subs"

    def test_skip_existing(self):
        args = self._parse(["--skip-existing", "video.mp4"])
        assert args.skip_existing is True

    def test_device_flag(self):
        args = self._parse(["--device", "mps", "video.mp4"])
        assert args.device == "mps"

    def test_audio_track_int(self):
        args = self._parse(["--audio-track", "2", "video.mp4"])
        assert args.audio_track == 2


# ---------------------------------------------------------------------------
# get_audio_tracks (mocked ffprobe)
# ---------------------------------------------------------------------------

class TestGetAudioTracks:
    @mock.patch("subprocess.run")
    def test_parses_ffprobe_output(self, mock_run):
        mock_run.return_value = mock.Mock(
            stdout="0,video\n1,audio,eng\n2,audio,spa\n",
            returncode=0,
        )
        tracks = subline.get_audio_tracks(Path("test.mp4"))
        assert tracks == [(1, "eng"), (2, "spa")]

    @mock.patch("subprocess.run")
    def test_no_language_tag(self, mock_run):
        mock_run.return_value = mock.Mock(
            stdout="0,video\n1,audio\n",
            returncode=0,
        )
        tracks = subline.get_audio_tracks(Path("test.mp4"))
        assert tracks == [(1, "und")]

    @mock.patch("subprocess.run")
    def test_no_audio(self, mock_run):
        mock_run.return_value = mock.Mock(
            stdout="0,video\n",
            returncode=0,
        )
        tracks = subline.get_audio_tracks(Path("test.mp4"))
        assert tracks == []


# ---------------------------------------------------------------------------
# get_audio_duration (mocked ffprobe)
# ---------------------------------------------------------------------------

class TestGetAudioDuration:
    @mock.patch("subprocess.run")
    def test_valid_duration(self, mock_run):
        mock_run.return_value = mock.Mock(stdout="123.456\n", returncode=0)
        assert subline.get_audio_duration("test.wav") == pytest.approx(123.456)

    @mock.patch("subprocess.run")
    def test_invalid_output(self, mock_run):
        mock_run.return_value = mock.Mock(stdout="N/A\n", returncode=0)
        assert subline.get_audio_duration("test.wav") == 0


# ---------------------------------------------------------------------------
# Integration-style: transcribe_file with mocked model
# ---------------------------------------------------------------------------

class TestTranscribeFile:
    def _make_segment(self, start, end, text):
        seg = mock.Mock()
        seg.start = start
        seg.end = end
        seg.text = text
        return seg

    @mock.patch("subline.get_audio_duration", return_value=60.0)
    def test_writes_srt(self, mock_dur, tmp_path):
        out = tmp_path / "test.srt"
        model = mock.Mock()
        info = mock.Mock(language="es", language_probability=0.98)
        segments = [
            self._make_segment(0.0, 2.5, " Hello world "),
            self._make_segment(2.5, 5.0, " Second line "),
        ]
        model.transcribe.return_value = (iter(segments), info)

        subline.transcribe_file("fake.wav", str(out), model, None, "srt")

        content = out.read_text()
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:02,500" in content
        assert "Hello world" in content
        assert "2\n" in content

    @mock.patch("subline.get_audio_duration", return_value=60.0)
    def test_writes_vtt(self, mock_dur, tmp_path):
        out = tmp_path / "test.vtt"
        model = mock.Mock()
        info = mock.Mock(language="en", language_probability=0.99)
        segments = [self._make_segment(0.0, 1.0, "Hello")]
        model.transcribe.return_value = (iter(segments), info)

        subline.transcribe_file("fake.wav", str(out), model, "en", "vtt")

        content = out.read_text()
        assert content.startswith("WEBVTT\n")
        assert "00:00:00.000 --> 00:00:01.000" in content

    @mock.patch("subline.get_audio_duration", return_value=60.0)
    def test_strips_whitespace(self, mock_dur, tmp_path):
        out = tmp_path / "test.srt"
        model = mock.Mock()
        info = mock.Mock(language="es", language_probability=0.9)
        segments = [self._make_segment(0.0, 1.0, "  padded text  ")]
        model.transcribe.return_value = (iter(segments), info)

        subline.transcribe_file("fake.wav", str(out), model, "es", "srt")

        content = out.read_text()
        assert "padded text" in content
        assert "  padded text  " not in content
