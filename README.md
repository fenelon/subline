```
            _     _ _
  ___ _   _| |__ | (_)_ __   ___
 / __| | | | '_ \| | | '_ \ / _ \
 \__ \ |_| | |_) | | | | | |  __/
 |___/\__,_|_.__/|_|_|_| |_|\___|

  video in -> subtitles out
```

# subline

Dead-simple subtitle generation for video files.
Point it at a file (or a folder full of them) and get `.srt` or `.vtt` subtitles -- powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

```
subline movie.mp4          # auto-detect language, writes movie.srt
```

```
  .-------------------.
  |  movie.mp4        |
  '-------------------'
          |
          v
  +-----------------+       +-----------------+
  |  ffmpeg extract |  -->  | faster-whisper  |
  |  audio track    |       | transcribe      |
  +-----------------+       +-----------------+
                                    |
                                    v
                            .-------------------.
                            |  movie.srt        |
                            |                   |
                            |  1                |
                            |  00:00:01,000 --> |
                            |  00:00:03,500     |
                            |  Hello world.     |
                            '-------------------'
```

## Features

- **Auto language detection** -- or pin a language with `--language es`
- **Batch mode** -- pass multiple files or a whole directory
- **SRT & VTT** output formats
- **GPU auto-detect** -- uses CUDA / MPS / CPU, whichever is available
- **Multi-track aware** -- picks the first audio track and tells you about the rest
- **Skip existing** -- resume interrupted batch runs with `--skip-existing`

## Requirements

- Python 3.10+
- **ffmpeg** and **ffprobe** on your PATH
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu / Debian
  sudo apt install ffmpeg
  # Windows (scoop)
  scoop install ffmpeg
  ```

## Install

```bash
# From PyPI (once published)
pipx install subline

# Or straight from GitHub
pipx install git+https://github.com/fenelon/subline.git

# Or in a venv
pip install git+https://github.com/fenelon/subline.git
```

## Usage

```
subline [options] <path...>
```

| Flag | Default | Description |
|------|---------|-------------|
| `--language CODE` | auto-detect | ISO 639-1 language code (`es`, `en`, `fr`, ...) |
| `--model NAME` | `turbo` | Whisper model: `tiny`, `base`, `small`, `medium`, `turbo` |
| `--audio-track N` | auto (first track) | Stream index of the audio track to use |
| `--format srt\|vtt` | `srt` | Output subtitle format |
| `--output-dir DIR` | next to source | Where to write subtitle files |
| `--device DEVICE` | auto | Force `cuda`, `mps`, or `cpu` |
| `--skip-existing` | off | Don't re-transcribe if subtitle already exists |

### Examples

```bash
# Single file, auto-detect language
subline interview.mp4

# Spanish audio, specific model
subline --language es --model medium lecture.mp4

# Whole directory, skip already-done files
subline --skip-existing ~/Videos/series/

# Multiple files, VTT output in a separate folder
subline --format vtt --output-dir ./subs ep01.mp4 ep02.mp4 ep03.mp4

# Force second audio track (0-indexed stream id from ffprobe)
subline --audio-track 2 foreign_film.mkv
```

### Output

```
Found 3 file(s) | model=turbo | language=auto-detect | device=mps

Loading model 'turbo' on mps...

[1/3] ep01.mp4
  Using audio track 1 (spa)
  Extracting audio (track 1)...
  Transcribing 45 min of audio...
  Detected language: es (98%)
  [ 52.3%] 350 segments, 124s elapsed, ETA 1m53s
  Done: 671 segments in 3m58s -> ep01.srt

[2/3] ep02.mp4
  ...
```

## Development

```bash
git clone https://github.com/fenelon/subline.git
cd subline
pip install -e ".[dev]"
pytest
```

## License

[MIT](LICENSE)
