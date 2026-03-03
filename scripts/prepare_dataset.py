"""
prepare_dataset.py — Phase 1: VTT + MP4 → HuggingFace AudioFolder

Converts pairs of Unity tutorial videos (.mp4) and their transcripts (.vtt)
into a HuggingFace-compatible AudioFolder dataset directory ready for upload.

Usage:
    python scripts/prepare_dataset.py \
        --input  ./my_unity_videos \
        --output ./unity_dataset \
        --target-duration 25.0 \
        --split 0.1

Input folder structure:
    my_unity_videos/
    ├── tutorial_01.mp4
    ├── tutorial_01.vtt
    ├── tutorial_02.mp4
    ├── tutorial_02.vtt
    └── ...

Output folder structure (HuggingFace AudioFolder format):
    unity_dataset/
    ├── metadata.csv          ← file_name, transcription columns
    ├── train/
    │   ├── clip_0001.wav
    │   ├── clip_0002.wav
    │   └── ...
    └── test/
        ├── clip_xxxx.wav
        └── ...
"""

import argparse
import csv
import logging
import os
import re
import subprocess
import shutil
import sys
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── VTT Parsing ────────────────────────────────────────────────────────────────

@dataclass
class VttCue:
    start: float   # seconds
    end: float     # seconds
    text: str


def _timestamp_to_seconds(ts: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to float seconds."""
    ts = ts.strip()
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(parts[0])


_CUE_PATTERN = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{2}:\d{2}[.,]\d{1,3})"
    r"\s*-->\s*"
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{2}:\d{2}[.,]\d{1,3})"
)


def parse_vtt(vtt_path: Path) -> List[VttCue]:
    """Parse a .vtt file and return a list of VttCue objects."""
    cues: List[VttCue] = []
    text_path = str(vtt_path)

    with open(text_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into blocks by blank lines
    blocks = re.split(r"\n\s*\n", content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        timing_line = None
        timing_idx = -1

        for i, line in enumerate(lines):
            if _CUE_PATTERN.search(line):
                timing_line = line
                timing_idx = i
                break

        if timing_line is None:
            continue

        match = _CUE_PATTERN.search(timing_line)
        if not match:
            continue

        start = _timestamp_to_seconds(match.group(1))
        end = _timestamp_to_seconds(match.group(2))

        # Text is everything after the timing line
        text_lines = lines[timing_idx + 1:]
        # Strip VTT tags like <c>, </c>, <00:00:00.000>
        text = " ".join(text_lines)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        if text and end > start:
            cues.append(VttCue(start=start, end=end, text=text))

    logger.info(f"  Parsed {len(cues)} cues from {vtt_path.name}")
    return cues


# ── Cue Merging ────────────────────────────────────────────────────────────────

def merge_cues(cues: List[VttCue], target_duration: float = 25.0,
               max_duration: float = 29.5) -> List[VttCue]:
    """
    Merge adjacent short cues into clips approaching target_duration seconds.
    Whisper processes 30-second windows, so we target 20-25s for safety.
    """
    if not cues:
        return []

    merged: List[VttCue] = []
    current_start = cues[0].start
    current_end = cues[0].end
    current_texts = [cues[0].text]

    for cue in cues[1:]:
        proposed_duration = cue.end - current_start
        gap = cue.start - current_end  # silence gap between cues

        # Start a new clip if: exceeds max duration, or gap > 2s (new sentence topic)
        if proposed_duration > max_duration or gap > 2.0:
            merged.append(VttCue(
                start=current_start,
                end=current_end,
                text=" ".join(current_texts),
            ))
            current_start = cue.start
            current_end = cue.end
            current_texts = [cue.text]
        else:
            current_end = cue.end
            current_texts.append(cue.text)

    # Flush last group
    if current_texts:
        merged.append(VttCue(
            start=current_start,
            end=current_end,
            text=" ".join(current_texts),
        ))

    logger.info(f"  Merged into {len(merged)} clips (target ≤{target_duration}s each)")
    return merged


# ── FFmpeg Audio Extraction ────────────────────────────────────────────────────

def find_ffmpeg() -> Optional[str]:
    """Locate the FFmpeg binary."""
    # Check common macOS locations first
    candidates = [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    found = shutil.which("ffmpeg")
    return found


def extract_clip(ffmpeg: str, video_path: Path, start: float, end: float,
                 output_wav: Path) -> bool:
    """
    Extract a mono 16kHz WAV clip from a video using FFmpeg.
    Returns True on success.
    """
    duration = end - start
    cmd = [
        ffmpeg,
        "-y",                         # overwrite output
        "-ss", f"{start:.3f}",        # seek before input for speed
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-vn",                        # no video
        "-acodec", "pcm_s16le",       # 16-bit PCM
        "-ar", "16000",               # 16 kHz (Whisper required)
        "-ac", "1",                   # mono
        "-loglevel", "error",
        str(output_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.warning(f"    ⚠️  FFmpeg error: {result.stderr.strip()}")
        return False
    return True


# ── Dataset Builder ────────────────────────────────────────────────────────────

def build_dataset(
    input_dir: Path,
    output_dir: Path,
    target_duration: float,
    test_split: float,
    seed: int,
) -> Tuple[int, int]:
    """
    Main build function. Returns (train_count, test_count).
    """
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        logger.error("❌ FFmpeg not found. Install via: brew install ffmpeg")
        sys.exit(1)
    logger.info(f"✅ FFmpeg: {ffmpeg}")

    # Discover MP4 + VTT pairs
    video_files = sorted(input_dir.glob("*.mp4")) + sorted(input_dir.glob("*.MP4"))
    pairs: List[Tuple[Path, Path]] = []

    for video in video_files:
        vtt = video.with_suffix(".vtt")
        if vtt.exists():
            pairs.append((video, vtt))
        else:
            logger.warning(f"  ⚠️  No matching VTT for {video.name} — skipping")

    if not pairs:
        logger.error(f"❌ No MP4+VTT pairs found in {input_dir}")
        sys.exit(1)

    logger.info(f"📂 Found {len(pairs)} video+VTT pairs")

    # Collect all merged clips across all videos
    all_clips: List[Tuple[Path, VttCue]] = []  # (video_path, merged_cue)

    for video, vtt in pairs:
        logger.info(f"\n🎬 Processing: {video.name}")
        cues = parse_vtt(vtt)
        if not cues:
            logger.warning(f"  ⚠️  No cues parsed — skipping {vtt.name}")
            continue
        merged = merge_cues(cues, target_duration=target_duration)
        for cue in merged:
            all_clips.append((video, cue))

    logger.info(f"\n📊 Total clips to extract: {len(all_clips)}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_clips)
    split_idx = max(1, int(len(all_clips) * test_split))
    test_clips = all_clips[:split_idx]
    train_clips = all_clips[split_idx:]

    logger.info(f"   Train: {len(train_clips)} clips | Test: {len(test_clips)} clips")

    # Create output directories
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: List[dict] = []
    clip_counter = 0
    failed = 0

    def process_clips(clips: List[Tuple[Path, VttCue]], split_name: str) -> int:
        nonlocal clip_counter, failed
        count = 0
        target_subdir = output_dir / split_name

        for video_path, cue in clips:
            clip_counter += 1
            clip_name = f"clip_{clip_counter:05d}.wav"
            wav_path = target_subdir / clip_name
            rel_path = f"{split_name}/{clip_name}"

            logger.debug(f"  [{split_name}] {clip_name}: {cue.start:.1f}s–{cue.end:.1f}s")

            success = extract_clip(
                ffmpeg, video_path, cue.start, cue.end, wav_path
            )
            if not success:
                failed += 1
                logger.warning(f"  ❌ Failed to extract {clip_name}")
                continue

            # Validate the clip has content
            if wav_path.stat().st_size < 1000:  # < 1KB is likely empty/corrupt
                logger.warning(f"  ⚠️  Skipping tiny clip: {clip_name}")
                wav_path.unlink(missing_ok=True)
                failed += 1
                continue

            metadata_rows.append({
                "file_name": rel_path,
                "transcription": cue.text,
            })
            count += 1

            if count % 50 == 0:
                logger.info(f"  [{split_name}] {count}/{len(clips)} clips extracted...")

        return count

    train_count = process_clips(train_clips, "train")
    test_count = process_clips(test_clips, "test")

    # Write metadata.csv
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "transcription"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info(f"\n✅ Dataset written to: {output_dir}")
    logger.info(f"   Train clips: {train_count}")
    logger.info(f"   Test clips:  {test_count}")
    logger.info(f"   Failed/skipped: {failed}")
    logger.info(f"   metadata.csv: {len(metadata_rows)} entries")

    return train_count, test_count


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a HuggingFace AudioFolder dataset from Unity tutorial videos + VTT transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/prepare_dataset.py --input ./my_unity_videos --output ./unity_dataset

  # Custom target clip duration and 15%% test split
  python scripts/prepare_dataset.py \\
      --input  ./my_unity_videos \\
      --output ./unity_dataset \\
      --target-duration 20.0 \\
      --split 0.15
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Folder containing .mp4 and matching .vtt files"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output folder for the HuggingFace AudioFolder dataset"
    )
    parser.add_argument(
        "--target-duration", type=float, default=25.0,
        help="Target clip duration in seconds when merging VTT cues (default: 25.0)"
    )
    parser.add_argument(
        "--split", type=float, default=0.1,
        help="Fraction of clips to reserve for test split (default: 0.10)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split (default: 42)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        logger.error(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Unity Whisper Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"  Input:           {input_dir}")
    logger.info(f"  Output:          {output_dir}")
    logger.info(f"  Target duration: {args.target_duration}s per clip")
    logger.info(f"  Test split:      {args.split * 100:.0f}%")
    logger.info("=" * 60)

    train_count, test_count = build_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        target_duration=args.target_duration,
        test_split=args.split,
        seed=args.seed,
    )

    total = train_count + test_count
    logger.info(f"\n🎉 Done! {total} clips ready.")
    logger.info(f"\nNext step — upload to HuggingFace:")
    logger.info(f"  python scripts/upload_dataset.py --dataset {output_dir}")


if __name__ == "__main__":
    main()
