#!/usr/bin/env python3
"""Debug script to understand why single word segments aren't being merged properly"""

import sys
import json
from captioner_compact import SubtitleFormatter

# Test segments that would produce the problematic output
test_segments = [
    {
        "start": 2.580,
        "end": 4.930,
        "text": "In this video, we're going to go over screen space reflections, which is a"
    },
    {
        "start": 4.930,
        "end": 5.280,  # Actual timing from whisper (should be ~0.35s for one word)
        "text": "complex"
    },
    {
        "start": 5.280,
        "end": 9.500,
        "text": "topic that deserves a little bit more time for"
    }
]

print("=== ORIGINAL SEGMENTS ===")
for i, seg in enumerate(test_segments):
    duration = seg['end'] - seg['start']
    words = len(seg['text'].split())
    print(f"{i}: [{seg['start']:.3f} -> {seg['end']:.3f}] ({duration:.3f}s, {words} words)")
    print(f"   '{seg['text']}'")
    print()

# Test the merge_short_segments function
print("\n=== TESTING merge_short_segments ===")
merged = SubtitleFormatter.merge_short_segments(test_segments.copy())
print(f"Original segments: {len(test_segments)}")
print(f"Merged segments: {len(merged)}")
print()

for i, seg in enumerate(merged):
    duration = seg['end'] - seg['start']
    words = len(seg['text'].split())
    print(f"{i}: [{seg['start']:.3f} -> {seg['end']:.3f}] ({duration:.3f}s, {words} words)")
    print(f"   '{seg['text']}'")
    print()

# Test with post_process_segments which applies all timing adjustments
print("\n=== TESTING post_process_segments ===")
processed = SubtitleFormatter.post_process_segments(test_segments.copy())
print(f"Processed segments: {len(processed)}")
print()

for i, seg in enumerate(processed):
    duration = seg['end'] - seg['start']
    words = len(seg['text'].split())
    print(f"{i}: [{seg['start']:.3f} -> {seg['end']:.3f}] ({duration:.3f}s, {words} words)")
    print(f"   '{seg['text']}'")
    print()

# Check character counts that might prevent merging
print("\n=== CHARACTER COUNT ANALYSIS ===")
seg1_text = test_segments[0]['text']
seg2_text = test_segments[1]['text']
seg3_text = test_segments[2]['text']

combined_1_2 = seg1_text + " " + seg2_text
combined_2_3 = seg2_text + " " + seg3_text

print(f"Segment 1: {len(seg1_text)} chars")
print(f"Segment 2: {len(seg2_text)} chars")
print(f"Segment 3: {len(seg3_text)} chars")
print(f"Combined 1+2: {len(combined_1_2)} chars (limit is 84)")
print(f"Combined 2+3: {len(combined_2_3)} chars (limit is 84)")

# Test line formatting
print("\n=== LINE FORMATTING TEST ===")
print("Combined 1+2 lines:")
lines_1_2 = SubtitleFormatter.format_text_lines(combined_1_2, 42, 2)
for line in lines_1_2:
    print(f"  '{line}' ({len(line)} chars)")
print(f"  Total lines: {len(lines_1_2)}")

print("\nCombined 2+3 lines:")
lines_2_3 = SubtitleFormatter.format_text_lines(combined_2_3, 42, 2)
for line in lines_2_3:
    print(f"  '{line}' ({len(line)} chars)")
print(f"  Total lines: {len(lines_2_3)}")
