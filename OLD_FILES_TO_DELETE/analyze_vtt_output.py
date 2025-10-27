#!/usr/bin/env python3
"""Analyze VTT output to understand single-word segments"""

import re

def analyze_vtt(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    segments = []
    
    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            timing = lines[i]
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                text_lines.append(lines[i])
                i += 1
            
            text = ' '.join(text_lines)
            
            # Parse timing
            match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', timing)
            if match:
                start_str, end_str = match.groups()
                
                # Convert to seconds
                def time_to_seconds(t):
                    h, m, s = t.split(':')
                    s_parts = s.split('.')
                    return int(h) * 3600 + int(m) * 60 + float(f"{s_parts[0]}.{s_parts[1]}")
                
                start = time_to_seconds(start_str)
                end = time_to_seconds(end_str)
                duration = end - start
                
                segments.append({
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'text': text.strip(),
                    'words': len(text.strip().split()),
                    'chars': len(text.strip())
                })
        else:
            i += 1
    
    # Analyze segments
    print("=== VTT SEGMENT ANALYSIS ===\n")
    
    single_word_segments = []
    short_segments = []
    
    for i, seg in enumerate(segments):
        if seg['words'] == 1:
            single_word_segments.append((i, seg))
        elif seg['words'] <= 2:
            short_segments.append((i, seg))
    
    print(f"Total segments: {len(segments)}")
    print(f"Single-word segments: {len(single_word_segments)}")
    print(f"Two-word segments: {len(short_segments)}")
    
    if single_word_segments:
        print("\n=== SINGLE-WORD SEGMENTS ===")
        for idx, seg in single_word_segments:
            print(f"\nSegment {idx}: '{seg['text']}'")
            print(f"  Duration: {seg['duration']:.3f}s")
            print(f"  Timing: {seg['start']:.3f} --> {seg['end']:.3f}")
            
            # Check neighbors
            if idx > 0:
                prev = segments[idx-1]
                print(f"  Previous: '{prev['text']}' ({prev['words']} words, {prev['chars']} chars)")
                combined_chars = prev['chars'] + 1 + seg['chars']
                print(f"    Combined would be: {combined_chars} chars")
            
            if idx < len(segments) - 1:
                next_seg = segments[idx+1]
                print(f"  Next: '{next_seg['text']}' ({next_seg['words']} words, {next_seg['chars']} chars)")
                combined_chars = seg['chars'] + 1 + next_seg['chars']
                print(f"    Combined would be: {combined_chars} chars")

# Analyze the generated VTT
analyze_vtt('test_improved/M1-11_Screen-Space-Reflection.vtt')
