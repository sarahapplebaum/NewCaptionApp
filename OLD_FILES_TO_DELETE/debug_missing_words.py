#!/usr/bin/env python3
"""
Debug script to trace exactly where specific words are being lost
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from captioner import SubtitleFormatter

def debug_word_loss():
    """Debug where words are being lost in the formatting process"""
    
    # Read the TXT file (what should be preserved)
    with open('output_vtts/M1-11_Screen-Space-Reflection.txt', 'r', encoding='utf-8-sig') as f:
        original_text = f.read().strip()
    
    # Read the VTT file (what actually got saved)
    with open('output_vtts/M1-11_Screen-Space-Reflection.vtt', 'r', encoding='utf-8-sig') as f:
        vtt_content = f.read()
    
    print("🔍 DEBUGGING WORD LOSS")
    print("=" * 60)
    
    # Extract words from original text
    original_words = original_text.split()
    original_clean = [w.lower().strip('.,!?";:\'"') for w in original_words if w.strip()]
    
    print(f"📝 Original text: {len(original_words)} words")
    print(f"🎯 Target words to find: 'complex', 'targeting', 'high-end'")
    
    # Check if target words exist in original
    target_words = ['complex', 'targeting', 'high-end']
    for target in target_words:
        target_clean = target.lower()
        if target_clean in [w.lower() for w in original_words]:
            print(f"✅ '{target}' found in original text")
            # Find context
            for i, word in enumerate(original_words):
                if word.lower().strip('.,!?";:\'"') == target_clean:
                    context_start = max(0, i-3)
                    context_end = min(len(original_words), i+4)
                    context = ' '.join(original_words[context_start:context_end])
                    print(f"   Context: ...{context}...")
                    break
        else:
            print(f"❌ '{target}' NOT found in original text")
    
    print()
    
    # Extract text from VTT segments
    vtt_text_parts = []
    lines = vtt_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip until timestamp
        if '-->' not in line:
            i += 1
            continue
        
        # Collect subtitle text lines
        i += 1
        while i < len(lines) and lines[i].strip():
            vtt_text_parts.append(lines[i].strip())
            i += 1
        i += 1
    
    vtt_text = ' '.join(vtt_text_parts)
    vtt_words = vtt_text.split()
    
    print(f"📦 VTT text: {len(vtt_words)} words")
    
    # Check if target words exist in VTT
    for target in target_words:
        target_clean = target.lower()
        if target_clean in [w.lower().strip('.,!?";:\'"') for w in vtt_words]:
            print(f"✅ '{target}' found in VTT text")
        else:
            print(f"❌ '{target}' MISSING from VTT text")
    
    print()
    
    # Look for the problematic segments
    print("🔍 CHECKING VTT SEGMENTS:")
    print("-" * 40)
    
    lines = vtt_content.split('\n')
    segment_num = 0
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip until timestamp
        if '-->' not in line:
            i += 1
            continue
        
        segment_num += 1
        timestamp = line
        
        # Collect subtitle text lines
        i += 1
        segment_lines = []
        while i < len(lines) and lines[i].strip():
            segment_lines.append(lines[i].strip())
            i += 1
        
        segment_text = ' '.join(segment_lines)
        
        # Check if this segment should contain our target words
        for target in target_words:
            if target.lower() in segment_text.lower():
                print(f"✅ Segment {segment_num}: Found '{target}'")
                print(f"   {timestamp}")
                print(f"   Text: {segment_text}")
                print()
            elif target == 'high-end' and 'high' in segment_text.lower() and 'end' in segment_text.lower():
                print(f"🔍 Segment {segment_num}: Contains 'high' and 'end' (compound word issue?)")
                print(f"   {timestamp}")
                print(f"   Text: {segment_text}")
                print()
        
        i += 1
    
    print("🔍 SEARCHING FOR MISSING WORD CONTEXTS:")
    print("-" * 50)
    
    # Search for the contexts where missing words should appear
    contexts_to_find = [
        "which is a topic that deserves",  # should be "complex topic"
        "if you're very",  # should be "targeting very high-end"
    ]
    
    for context in contexts_to_find:
        print(f"Looking for context: '{context}'")
        found_in_original = context.lower() in original_text.lower()
        found_in_vtt = context.lower() in vtt_text.lower()
        
        print(f"  Original: {'✅' if found_in_original else '❌'}")
        print(f"  VTT: {'✅' if found_in_vtt else '❌'}")
        
        if found_in_original:
            # Show the full context from original
            start_idx = original_text.lower().find(context.lower())
            if start_idx >= 0:
                context_start = max(0, start_idx - 50)
                context_end = min(len(original_text), start_idx + len(context) + 50)
                full_context = original_text[context_start:context_end]
                print(f"  Original context: ...{full_context}...")
        
        if found_in_vtt:
            # Show the full context from VTT
            start_idx = vtt_text.lower().find(context.lower())
            if start_idx >= 0:
                context_start = max(0, start_idx - 50)
                context_end = min(len(vtt_text), start_idx + len(context) + 50)
                full_context = vtt_text[context_start:context_end]
                print(f"  VTT context: ...{full_context}...")
        
        print()

if __name__ == "__main__":
    debug_word_loss()
