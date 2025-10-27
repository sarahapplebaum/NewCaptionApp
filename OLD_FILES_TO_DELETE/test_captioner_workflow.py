#!/usr/bin/env python3
"""Test the actual workflow to debug timing issues"""

import sys
from pathlib import Path
from captioner_compact import SubtitleFormatter, merge_split_compound_words

# Sample word-level output that would come from Whisper
sample_words = [
    {"word": "In", "start": 2.580, "end": 2.700},
    {"word": "this", "start": 2.700, "end": 2.850},
    {"word": "video,", "start": 2.850, "end": 3.100},
    {"word": "we're", "start": 3.100, "end": 3.250},
    {"word": "going", "start": 3.250, "end": 3.400},
    {"word": "to", "start": 3.400, "end": 3.500},
    {"word": "go", "start": 3.500, "end": 3.600},
    {"word": "over", "start": 3.600, "end": 3.800},
    {"word": "screen", "start": 3.800, "end": 4.000},
    {"word": "space", "start": 4.000, "end": 4.200},
    {"word": "reflections,", "start": 4.200, "end": 4.500},
    {"word": "which", "start": 4.500, "end": 4.650},
    {"word": "is", "start": 4.650, "end": 4.750},
    {"word": "a", "start": 4.750, "end": 4.930},
    {"word": "complex", "start": 4.930, "end": 5.280},
    {"word": "topic", "start": 5.280, "end": 5.600},
    {"word": "that", "start": 5.600, "end": 5.800},
    {"word": "deserves", "start": 5.800, "end": 6.200},
    {"word": "a", "start": 6.200, "end": 6.300},
    {"word": "little", "start": 6.300, "end": 6.500},
    {"word": "bit", "start": 6.500, "end": 6.700},
    {"word": "more", "start": 6.700, "end": 6.900},
    {"word": "time", "start": 6.900, "end": 7.200},
    {"word": "for", "start": 7.200, "end": 9.500}
]

print("=== SIMULATING ACTUAL CAPTIONER WORKFLOW ===\n")

# Step 1: Merge compound words (if any)
print("Step 1: Merging compound words...")
merged_words = merge_split_compound_words(sample_words)
print(f"Original words: {len(sample_words)}, Merged words: {len(merged_words)}")

# Step 2: Create optimized segments
print("\nStep 2: Creating optimized segments...")
segments = SubtitleFormatter.create_optimized_segments(merged_words, max_chars=84)

print(f"\nCreated {len(segments)} segments:")
for i, seg in enumerate(segments):
    duration = seg['end'] - seg['start']
    words = len(seg['text'].split())
    print(f"{i}: [{seg['start']:.3f} -> {seg['end']:.3f}] ({duration:.3f}s, {words} words)")
    print(f"   '{seg['text']}'")
    print()

# Step 3: Create VTT content
print("\n=== GENERATED VTT ===")
vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
print(vtt_content)

# Analyze the specific problematic segment
print("\n=== ANALYZING SINGLE-WORD SEGMENT ===")
for i, seg in enumerate(segments):
    if "complex" in seg['text'] and seg['text'].strip() == "complex":
        print(f"Found single-word 'complex' segment:")
        print(f"  Index: {i}")
        print(f"  Start: {seg['start']:.3f}")
        print(f"  End: {seg['end']:.3f}")
        print(f"  Duration: {seg['end'] - seg['start']:.3f}s")
        print(f"  Text: '{seg['text']}'")
