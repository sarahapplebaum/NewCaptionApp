# Video Captioner - Compact Version

A streamlined video captioning tool that uses OpenAI's Whisper model to generate accurate subtitles from video files.

## Features

- 🚀 Fast transcription with faster-whisper implementation
- 📝 Generates VTT and SRT subtitle formats
- 🎯 Smart subtitle segmentation to avoid single-word segments
- 💻 Both CLI and GUI interfaces
- 📦 Batch processing support
- 🔧 Customizable subtitle formatting (line length, segment size)
- 📚 Context prompting for domain-specific vocabulary
- ✏️ Post-processing vocabulary correction with CSV word lists

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

With context prompt for domain-specific vocabulary (e.g., Unity tutorials):
```bash
python captioner_compact.py input.mp4 -o output_folder --prompt "This video covers Unity game engine topics including GameObjects, Rigidbody, and NavMesh"
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
- `--max-segment-chars`: Maximum characers per subtitle segment (default: 84)
- `--no-timestamps`: Generate transcript only without timestamps
- `-p, --prompt`: Context prompt to help faster-whisper recognize domain-specific vocabulary
- `--vocab-csv`: Path to vocabulary CSV file for post-processing correction
- `--vocab-sensitivity`: Fuzzy match sensitivity percentage (70-100, default: 85)
- `--no-vocab-fallback`: Disable title case fallback for unknown terms
- `-v, --verbose`: Enable verbose output

## Context Prompting

The `--prompt` option allows you to provide context to faster-whisper, improving transcription accuracy for domain-specific vocabulary. This is especially useful for technical content like:

- **Unity tutorials**: `--prompt "This video covers Unity game engine, including GameObjects, Prefabs, and C# scripting"`
- **Medical content**: `--prompt "Medical training video discussing cardiovascular procedures"`
- **Legal presentations**: `--prompt "Legal presentation covering contract law and arbitration"`

The context prompt helps faster-whisper:
- Better recognize technical terms and proper nouns
- Maintain consistent capitalization
- Improve accuracy for specialized vocabulary

## Vocabulary Correction (Post-Processing)

For cases where context prompting alone isn't sufficient, vocabulary correction provides a post-processing step that corrects transcription errors using a CSV word list.

### CSV File Format

The vocabulary CSV should have a `Term` column containing the correct spellings:
```csv
Term,Notes
GameObject,Core Unity class
Rigidbody,Physics component
TextMeshPro,Text rendering
Animator Controller,Animation system
```

### CLI Usage

```bash
# With vocabulary correction
python captioner_compact.py input.mp4 -o output_folder --vocab-csv vocabulary.csv

# Adjust sensitivity (70-100%, higher = stricter matching)
python captioner_compact.py input.mp4 -o output_folder --vocab-csv vocabulary.csv --vocab-sensitivity 90

# Combine with context prompting for best results
python captioner_compact.py input.mp4 -o output_folder \
  --prompt "Unity game engine tutorial" \
  --vocab-csv unity_terms.csv
```

### GUI Usage

1. Check "Enable vocabulary correction" in the Vocabulary Correction section
2. Click "Select CSV" to choose your vocabulary file
3. Adjust the fuzzy match sensitivity slider (85% recommended)
4. Optionally enable/disable title case fallback for unknown terms

### Correction Types

The vocabulary corrector applies several types of corrections:

- **Exact match**: Case-insensitive exact matches (e.g., "gameobject" → "GameObject")
- **Fuzzy match**: Similar spellings within the sensitivity threshold (e.g., "rigidbod" → "Rigidbody")
- **Multi-word terms**: Recognizes 2-3 word terms (e.g., "animator controller" → "Animator Controller")
- **Title case fallback**: Capitalizes words that look like class/component names

### When to Use Each Feature

| Scenario | Recommended Approach |
|----------|---------------------|
| First transcription | Start with context prompting only |
| Common misspellings | Add vocabulary correction |
| Highly technical content | Use both together |
| Custom terminology | Create a domain-specific CSV |

## Files in this Workspace

- `captioner_compact.py` - Main application with CLI and GUI support
- `src/` - Modular source code (core, gui, utils)
- `batch_process_videos.sh` - Batch processing script for multiple videos
- `CLI_USAGE.md` - Detailed command-line usage documentation
- `venv/` - Python virtual environment (not tracked in git)

## Requirements

- FFmpeg must be installed and accessible in your system PATH
- Python packages: faster-whisper, PyQt5, torch

## Performance

The tool processes videos at approximately 7-10x realtime speed on modern hardware, depending on the model size and system specifications.

## Output Formats

- `.txt` - Plain text transcript
- `.vtt` - WebVTT subtitle format with positioning
- `.srt` - SubRip subtitle format

## License

This project is for internal use. Please refer to your organization's licensing policies.
