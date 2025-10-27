# Video Captioner - Command Line Interface

The Video Captioner now supports command-line usage for batch processing and automation workflows.

## Installation

Ensure you have Python 3.8+ and all dependencies installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Process a single video with default settings:
```bash
python captioner.py input.mp4 -o output_folder
```

This will:
- Extract audio from the video
- Transcribe using the Whisper AI model (default: small)
- Generate three files in the output folder:
  - `input.txt` - Full transcript
  - `input.vtt` - WebVTT subtitle file (42 chars/line max)
  - `input.srt` - SRT subtitle file (42 chars/line max)

### Launch GUI mode (default if no arguments):
```bash
python captioner.py
# or explicitly:
python captioner.py --gui
```

## Command Line Options

```
usage: captioner.py [-h] [-o OUTPUT] [-m MODEL] [--max-chars MAX_CHARS] 
                    [--max-segment-chars MAX_SEGMENT_CHARS] [--no-timestamps] 
                    [--gui] [-v] [input]

High Performance Video Captioner - Generate subtitles from video files

positional arguments:
  input                 Input MP4 file path

options:
  -h, --help            Show help message and exit
  -o, --output OUTPUT   Output folder for generated files
  -m, --model {tiny,base,small,medium,large,large-v2,large-v3}
                        Whisper model to use (default: small)
  --max-chars MAX_CHARS
                        Maximum characters per line (default: 42)
  --max-segment-chars MAX_SEGMENT_CHARS
                        Maximum characters per subtitle segment (default: 84)
  --no-timestamps       Generate transcript only without timestamps
  --gui                 Launch GUI mode (default if no input file provided)
  -v, --verbose         Enable verbose output
```

## Examples

### Process with a larger model for better accuracy:
```bash
python captioner.py video.mp4 -o subtitles -m large-v3
```

### Process with custom character limits:
```bash
python captioner.py video.mp4 -o output --max-chars 50 --max-segment-chars 100
```

### Generate transcript only (no VTT/SRT files):
```bash
python captioner.py video.mp4 -o transcripts --no-timestamps
```

### Process with verbose output for debugging:
```bash
python captioner.py video.mp4 -o output --verbose
```

## Model Selection

Available models (from fastest to most accurate):
- `tiny` - Fastest, least accurate (~39 MB)
- `base` - Fast, good for drafts (~74 MB)
- `small` - Balanced speed/accuracy (default) (~244 MB)
- `medium` - More accurate, slower (~769 MB)
- `large` - Very accurate, slow (~1550 MB)
- `large-v2` - Latest large model
- `large-v3` - Best quality, slowest

## Performance

Processing speed depends on:
- Model size (smaller = faster)
- Hardware (GPU > CPU)
- Video duration
- Audio complexity

Typical speeds (on modern hardware):
- `small` model: 5-10x realtime
- `large` model: 1-3x realtime

## Exit Codes

- `0` - Success
- `1` - Error (file not found, processing failed, etc.)

## Integration Examples

### Batch processing with shell script:
```bash
#!/bin/bash
for video in *.mp4; do
    python captioner.py "$video" -o subtitles/
done
```

### Python integration:
```python
import subprocess

result = subprocess.run([
    'python', 'captioner.py',
    'input.mp4',
    '-o', 'output',
    '-m', 'medium'
], capture_output=True)

if result.returncode == 0:
    print("Success!")
else:
    print(f"Error: {result.stderr.decode()}")
```

### Automated workflow with error handling:
```bash
#!/bin/bash
VIDEO="input.mp4"
OUTPUT="subtitles"

if python captioner.py "$VIDEO" -o "$OUTPUT" --verbose; then
    echo "✅ Successfully processed $VIDEO"
    # Additional processing of generated files
else
    echo "❌ Failed to process $VIDEO"
    exit 1
fi
```

## Tips

1. **For best quality**: Use `large-v3` model with default character limits
2. **For fastest processing**: Use `tiny` or `base` model
3. **For automation**: Use exit codes to detect success/failure
4. **For debugging**: Use `--verbose` flag to see detailed progress

## Limitations

- Input must be a valid video/audio file supported by FFmpeg
- Output folder must be writable
- Sufficient disk space needed for temporary audio extraction
- Memory requirements vary by model size

## Troubleshooting

If you encounter errors:

1. **FFmpeg not found**: Ensure FFmpeg is installed and in PATH
2. **Model download fails**: Check internet connection
3. **Out of memory**: Use a smaller model
4. **Permission denied**: Check output folder permissions
