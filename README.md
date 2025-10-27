# Video Captioner - Compact Version

A streamlined video captioning tool that uses OpenAI's Whisper model to generate accurate subtitles from video files.

## Features

- 🚀 Fast transcription with faster-whisper implementation
- 📝 Generates VTT and SRT subtitle formats
- 🎯 Smart subtitle segmentation to avoid single-word segments
- 💻 Both CLI and GUI interfaces
- 📦 Batch processing support
- 🔧 Customizable subtitle formatting (line length, segment size)

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### CLI Mode

Process a single video:
```bash
python captioner_compact.py input.mp4 -o output_folder
```

With custom settings:
```bash
python captioner_compact.py input.mp4 -o output_folder -m large-v3 --max-chars 50
```

### GUI Mode

Launch the graphical interface:
```bash
python captioner_compact.py --gui
```

### Batch Processing

Use the included batch script:
```bash
./batch_process_videos.sh
```

## Options

- `-m, --model`: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
- `--max-chars`: Maximum characters per subtitle line (default: 42)
- `--max-segment-chars`: Maximum characters per subtitle segment (default: 84)
- `--no-timestamps`: Generate transcript only without timestamps
- `-v, --verbose`: Enable verbose output

## Files in this Workspace

- `captioner_compact.py` - Main application with CLI and GUI support
- `batch_process_videos.sh` - Batch processing script for multiple videos
- `CLI_USAGE.md` - Detailed command-line usage documentation
- `venv/` - Python virtual environment (not tracked in git)
- `OLD_FILES_TO_DELETE/` - Archived test files and old implementations

## Requirements

- FFmpeg must be installed and accessible in your system PATH
- Python packages: faster-whisper, PyQt5, torch, librosa

## Performance

The tool processes videos at approximately 7-10x realtime speed on modern hardware, depending on the model size and system specifications.

## Output Formats

- `.txt` - Plain text transcript
- `.vtt` - WebVTT subtitle format with positioning
- `.srt` - SubRip subtitle format

## License

This project is for internal use. Please refer to your organization's licensing policies.
