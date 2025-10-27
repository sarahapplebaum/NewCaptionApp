#!/usr/bin/env python3
"""
Analyze VTT files for short subtitle issues before and after improvement
"""

import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple

def parse_vtt_segments(vtt_path: str) -> List[Dict]:
    """Parse VTT file and extract segments"""
    segments = []
    
    with open(vtt_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Split by double newlines to get blocks
    blocks = content.split('\n\n')
    
    for block in blocks:
        block = block.strip()
        if not block or block.startswith('WEBVTT'):
            continue
        
        lines = block.split('\n')
        if len(lines) >= 2 and '-->' in lines[0]:
            # Parse timestamp
            timestamp_line = lines[0]
            match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})', timestamp_line)
            
            if match:
                start_time = match.group(1)
                end_time = match.group(2)
                
                # Get text (remaining lines)
                text_lines = []
                for line in lines[1:]:
                    # Stop at position info or empty line
                    if 'align:' in line or not line.strip():
                        break
                    text_lines.append(line.strip())
                
                text = ' '.join(text_lines)
                
                if text:
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text,
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'line_count': len(text_lines)
                    })
    
    return segments

def analyze_vtt_file(vtt_path: str, label: str) -> Dict:
    """Analyze a VTT file for short subtitle issues"""
    print(f"\n{'='*60}")
    print(f"📄 {label}: {Path(vtt_path).name}")
    print(f"{'='*60}")
    
    segments = parse_vtt_segments(vtt_path)
    
    if not segments:
        print("❌ No segments found!")
        return {}
    
    # Analyze segments
    short_segments = []  # 1-2 words
    very_short_segments = []  # 1 word
    long_segments = []  # Over 84 chars
    
    total_words = 0
    total_chars = 0
    
    for i, seg in enumerate(segments):
        words = seg['word_count']
        chars = seg['char_count']
        
        total_words += words
        total_chars += chars
        
        if words <= 2:
            short_segments.append((i, seg))
            if words == 1:
                very_short_segments.append((i, seg))
        
        if chars > 84:
            long_segments.append((i, seg))
    
    # Calculate stats
    avg_words = total_words / len(segments) if segments else 0
    avg_chars = total_chars / len(segments) if segments else 0
    
    print(f"📊 Overall Statistics:")
    print(f"   Total segments: {len(segments)}")
    print(f"   Total words: {total_words}")
    print(f"   Average words per segment: {avg_words:.1f}")
    print(f"   Average chars per segment: {avg_chars:.1f}")
    
    print(f"\n📏 Segment Length Distribution:")
    word_dist = {}
    for seg in segments:
        w = seg['word_count']
        word_dist[w] = word_dist.get(w, 0) + 1
    
    # Show distribution for 1-10 words
    for w in range(1, 11):
        count = word_dist.get(w, 0)
        if count > 0:
            bar = '█' * min(count, 40)
            print(f"   {w:2d} words: {bar} ({count})")
    
    # Show 11+ words
    over_10 = sum(count for w, count in word_dist.items() if w > 10)
    if over_10 > 0:
        bar = '█' * min(over_10, 40)
        print(f"   11+ words: {bar} ({over_10})")
    
    print(f"\n⚠️  Issues Found:")
    print(f"   Short segments (1-2 words): {len(short_segments)} ({len(short_segments)/len(segments)*100:.1f}%)")
    print(f"   Very short (1 word): {len(very_short_segments)} ({len(very_short_segments)/len(segments)*100:.1f}%)")
    print(f"   Too long (>84 chars): {len(long_segments)} ({len(long_segments)/len(segments)*100:.1f}%)")
    
    # Show examples of short segments
    if short_segments:
        print(f"\n📝 Examples of short segments:")
        for i, (idx, seg) in enumerate(short_segments[:10]):
            print(f"   #{idx+1}: \"{seg['text']}\" ({seg['word_count']} words)")
            if i >= 9 and len(short_segments) > 10:
                print(f"   ... and {len(short_segments) - 10} more")
                break
    
    return {
        'total_segments': len(segments),
        'short_segments': len(short_segments),
        'very_short_segments': len(very_short_segments),
        'long_segments': len(long_segments),
        'avg_words': avg_words,
        'avg_chars': avg_chars,
        'segments': segments
    }

def compare_improvements(before_path: str, after_path: str):
    """Compare two VTT files to show improvements"""
    print("\n🔄 SHORT SUBTITLE ANALYSIS")
    
    # Analyze both files
    before_stats = analyze_vtt_file(before_path, "BEFORE (Original)")
    after_stats = analyze_vtt_file(after_path, "AFTER (Improved)")
    
    # Show improvement summary
    print(f"\n{'='*60}")
    print(f"📈 IMPROVEMENT SUMMARY")
    print(f"{'='*60}")
    
    if before_stats and after_stats:
        # Calculate improvements
        seg_reduction = before_stats['total_segments'] - after_stats['total_segments']
        short_reduction = before_stats['short_segments'] - after_stats['short_segments']
        very_short_reduction = before_stats['very_short_segments'] - after_stats['very_short_segments']
        
        print(f"✅ Segment count: {before_stats['total_segments']} → {after_stats['total_segments']} "
              f"(reduced by {seg_reduction}, {seg_reduction/before_stats['total_segments']*100:.1f}%)")
        
        print(f"✅ Short segments (1-2 words): {before_stats['short_segments']} → {after_stats['short_segments']} "
              f"(reduced by {short_reduction}, {short_reduction/before_stats['short_segments']*100:.1f}% improvement)")
        
        print(f"✅ Very short segments (1 word): {before_stats['very_short_segments']} → {after_stats['very_short_segments']} "
              f"(reduced by {very_short_reduction}, "
              f"{very_short_reduction/before_stats['very_short_segments']*100:.1f}% improvement)" if before_stats['very_short_segments'] > 0 else "(eliminated!)")
        
        print(f"✅ Average words per segment: {before_stats['avg_words']:.1f} → {after_stats['avg_words']:.1f} "
              f"(+{after_stats['avg_words'] - before_stats['avg_words']:.1f} words)")
        
        print(f"✅ Average chars per segment: {before_stats['avg_chars']:.1f} → {after_stats['avg_chars']:.1f} "
              f"(+{after_stats['avg_chars'] - before_stats['avg_chars']:.1f} chars)")
        
        # Check reading rates
        print(f"\n🕐 Reading Rate Analysis (25 chars/sec max):")
        
        fast_before = 0
        fast_after = 0
        
        # Parse times and calculate rates
        for seg in before_stats.get('segments', []):
            start_parts = seg['start'].split(':')
            end_parts = seg['end'].split(':')
            
            start_secs = float(start_parts[0]) * 3600 + float(start_parts[1]) * 60 + float(start_parts[2])
            end_secs = float(end_parts[0]) * 3600 + float(end_parts[1]) * 60 + float(end_parts[2])
            duration = end_secs - start_secs
            
            if duration > 0:
                rate = seg['char_count'] / duration
                if rate > 25:
                    fast_before += 1
        
        for seg in after_stats.get('segments', []):
            start_parts = seg['start'].split(':')
            end_parts = seg['end'].split(':')
            
            start_secs = float(start_parts[0]) * 3600 + float(start_parts[1]) * 60 + float(start_parts[2])
            end_secs = float(end_parts[0]) * 3600 + float(end_parts[1]) * 60 + float(end_parts[2])
            duration = end_secs - start_secs
            
            if duration > 0:
                rate = seg['char_count'] / duration
                if rate > 25:
                    fast_after += 1
        
        print(f"   Before: {fast_before} segments exceed 25 chars/sec")
        print(f"   After: {fast_after} segments exceed 25 chars/sec")
        
        if fast_after < fast_before:
            print(f"   ✅ Improved by {fast_before - fast_after} segments")
        elif fast_after == 0:
            print(f"   ✅ All segments within reading rate limit!")

def main():
    if len(sys.argv) != 3:
        # Default to comparing original vs improved
        before_path = "/Users/sarah.applebaum/Documents/FinalRenders/HDRP Lighting Fundamentals/Camtasia/mp4/M1-11_Screen-Space-Reflection.vtt"
        after_path = "test_improved/M1-11_Screen-Space-Reflection.vtt"
        
        if not Path(before_path).exists() or not Path(after_path).exists():
            print("Usage: python analyze_short_subtitles.py <before.vtt> <after.vtt>")
            sys.exit(1)
    else:
        before_path = sys.argv[1]
        after_path = sys.argv[2]
    
    if not Path(before_path).exists():
        print(f"❌ File not found: {before_path}")
        sys.exit(1)
    
    if not Path(after_path).exists():
        print(f"❌ File not found: {after_path}")
        sys.exit(1)
    
    compare_improvements(before_path, after_path)

if __name__ == "__main__":
    main()
