#!/usr/bin/env python3
"""Test CLI processing to verify timing improvements"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path

# First, let's create a test script that uses the actual CLI
test_video = None

# Find a test video in the output_vtts directory
for file in Path("output_vtts").glob("*.vtt"):
    # Check if corresponding video exists
    video_name = file.stem
    for ext in ['.mp4', '.mov', '.mkv']:
        video_path = Path(f"{video_name}{ext}")
        if video_path.exists():
            test_video = str(video_path)
            break
    if test_video:
        break

if not test_video:
    print("No test video found. Please specify a video file path.")
    sys.exit(1)

print(f"Testing with video: {test_video}")

# Create temp output directory
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Output directory: {temp_dir}")
    
    # Run the captioner
    cmd = [
        sys.executable,
        "captioner_compact.py",
        test_video,
        "-o", temp_dir,
        "-m", "small",
        "--max-chars", "42"
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running captioner: {result.stderr}")
        sys.exit(1)
    
    # Check the generated VTT
    vtt_file = Path(temp_dir) / f"{Path(test_video).stem}.vtt"
    
    if vtt_file.exists():
        print("\n=== GENERATED VTT CONTENT ===")
        with open(vtt_file, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            print(content[:1000])  # First 1000 chars
            
            # Look for problematic timings
            lines = content.strip().split('\n')
            for i, line in enumerate(lines):
                if '-->' in line:
                    next_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j].strip() and '-->' not in lines[j]:
                        next_lines.append(lines[j])
                        j += 1
                    
                    text = ' '.join(next_lines)
                    words = len(text.split())
                    
                    # Parse timing
                    parts = line.split(' --> ')
                    if len(parts) == 2:
                        start_str = parts[0].strip()
                        end_parts = parts[1].split(' ')
                        end_str = end_parts[0].strip()
                        
                        # Convert to seconds
                        def time_to_seconds(time_str):
                            parts = time_str.split(':')
                            if len(parts) == 3:
                                h, m, s = parts
                                s_parts = s.split('.')
                                if len(s_parts) == 2:
                                    return int(h) * 3600 + int(m) * 60 + float(f"{s_parts[0]}.{s_parts[1]}")
                            return 0
                        
                        start = time_to_seconds(start_str)
                        end = time_to_seconds(end_str)
                        duration = end - start
                        
                        if words == 1 and duration > 1.0:
                            print(f"\n⚠️  FOUND SINGLE WORD WITH LONG DURATION:")
                            print(f"   Text: '{text.strip()}'")
                            print(f"   Duration: {duration:.3f}s")
                            print(f"   Timing: {start_str} --> {end_str}")
    else:
        print(f"VTT file not found: {vtt_file}")
