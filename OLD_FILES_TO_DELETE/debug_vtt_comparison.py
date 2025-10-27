#!/usr/bin/env python3
"""
Debug script to compare TXT and VTT files word by word
"""

import re
from pathlib import Path

def extract_text_from_vtt(vtt_path):
    """Extract all text from VTT file, ignoring timestamps and formatting"""
    with open(vtt_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    # Split into lines
    lines = content.split('\n')
    
    # Extract only subtitle text (skip WEBVTT header, timestamps, empty lines)
    text_lines = []
    skip_next = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines, header, and timestamps
        if not line or line.startswith('WEBVTT') or '-->' in line:
            continue
        
        # This is subtitle text
        text_lines.append(line)
    
    # Join all text and split into words
    full_text = ' '.join(text_lines)
    return full_text

def compare_files(txt_path, vtt_path):
    """Compare TXT and VTT files word by word"""
    # Read TXT file
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        txt_content = f.read().strip()
    
    # Extract text from VTT
    vtt_content = extract_text_from_vtt(vtt_path)
    
    # Split into words and clean
    txt_words = txt_content.split()
    vtt_words = vtt_content.split()
    
    print(f"📊 Word Count Comparison:")
    print(f"   TXT: {len(txt_words)} words")
    print(f"   VTT: {len(vtt_words)} words")
    print()
    
    # Clean words for comparison (remove punctuation)
    def clean_word(word):
        return word.lower().strip('.,!?";:\'"()[]{}')
    
    txt_words_clean = [clean_word(w) for w in txt_words if clean_word(w)]
    vtt_words_clean = [clean_word(w) for w in vtt_words if clean_word(w)]
    
    # Find missing words
    txt_set = set(txt_words_clean)
    vtt_set = set(vtt_words_clean)
    
    missing_in_vtt = txt_set - vtt_set
    extra_in_vtt = vtt_set - txt_set
    
    print(f"🔍 Unique Word Analysis:")
    print(f"   TXT unique words: {len(txt_set)}")
    print(f"   VTT unique words: {len(vtt_set)}")
    print(f"   Missing from VTT: {len(missing_in_vtt)}")
    print(f"   Extra in VTT: {len(extra_in_vtt)}")
    print()
    
    if missing_in_vtt:
        print(f"❌ Words missing from VTT:")
        for word in sorted(missing_in_vtt)[:20]:
            print(f"   - {word}")
        if len(missing_in_vtt) > 20:
            print(f"   ... and {len(missing_in_vtt) - 20} more")
        print()
    
    # Check word order by comparing sequences
    print(f"🔍 Sequential Comparison (first 100 words):")
    mismatches = []
    for i in range(min(100, len(txt_words), len(vtt_words))):
        if i < len(txt_words) and i < len(vtt_words):
            if clean_word(txt_words[i]) != clean_word(vtt_words[i]):
                mismatches.append({
                    'position': i,
                    'txt': txt_words[i],
                    'vtt': vtt_words[i]
                })
    
    if mismatches:
        print(f"   Found {len(mismatches)} word mismatches in first 100 words:")
        for m in mismatches[:10]:
            print(f"   Position {m['position']}: TXT='{m['txt']}' vs VTT='{m['vtt']}'")
        print()
    else:
        print("   ✅ First 100 words match perfectly")
        print()
    
    # Check for word frequency differences
    print(f"🔍 Word Frequency Analysis:")
    from collections import Counter
    txt_counter = Counter(txt_words_clean)
    vtt_counter = Counter(vtt_words_clean)
    
    freq_differences = []
    for word, txt_count in txt_counter.items():
        vtt_count = vtt_counter.get(word, 0)
        if txt_count != vtt_count:
            freq_differences.append({
                'word': word,
                'txt_count': txt_count,
                'vtt_count': vtt_count,
                'diff': txt_count - vtt_count
            })
    
    # Sort by biggest differences
    freq_differences.sort(key=lambda x: abs(x['diff']), reverse=True)
    
    if freq_differences:
        print(f"   Found {len(freq_differences)} words with different frequencies:")
        for fd in freq_differences[:10]:
            print(f"   '{fd['word']}': TXT={fd['txt_count']}, VTT={fd['vtt_count']} (diff={fd['diff']})")
        print()
    else:
        print("   ✅ All word frequencies match")
        print()
    
    # Final verdict
    if len(txt_words) == len(vtt_words) and not missing_in_vtt and not freq_differences:
        print("✅ VERDICT: VTT contains all words from TXT in correct quantities")
    else:
        print("❌ VERDICT: VTT is missing words or has incorrect word counts")
        if len(txt_words) != len(vtt_words):
            print(f"   - Total word count mismatch: {len(txt_words)} vs {len(vtt_words)}")
        if missing_in_vtt:
            print(f"   - {len(missing_in_vtt)} unique words missing")
        if freq_differences:
            print(f"   - {len(freq_differences)} words have incorrect frequencies")
    
    # Show raw text comparison for debugging
    print("\n" + "="*60)
    print("RAW TEXT COMPARISON (first 500 chars)")
    print("="*60)
    print("TXT:", txt_content[:500])
    print("-"*60)
    print("VTT:", vtt_content[:500])
    print("="*60)

if __name__ == "__main__":
    import sys
    
    # Use command line arguments if provided, otherwise use defaults
    if len(sys.argv) >= 3:
        txt_path = sys.argv[1]
        vtt_path = sys.argv[2]
    else:
        # Default paths - can be changed
        txt_path = "/Users/sarah.applebaum/Documents/FinalRenders/HDRP Lighting Fundamentals/Camtasia/mp4/M1-11_Screen-Space-Reflection.txt"
        vtt_path = "/Users/sarah.applebaum/Documents/FinalRenders/HDRP Lighting Fundamentals/Camtasia/mp4/M1-11_Screen-Space-Reflection.vtt"
    
    print("🔍 Debugging VTT Word Preservation")
    print("="*60)
    print(f"TXT: {txt_path}")
    print(f"VTT: {vtt_path}")
    print("="*60)
    print()
    
    compare_files(txt_path, vtt_path)
