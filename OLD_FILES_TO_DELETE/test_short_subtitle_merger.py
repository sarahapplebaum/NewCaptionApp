#!/usr/bin/env python3
"""
Test the short subtitle merging functionality
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from captioner import SubtitleFormatter

def test_short_subtitle_merging():
    """Test that short subtitles are properly merged"""
    
    print("🧪 Testing Short Subtitle Merging")
    print("=" * 60)
    
    # Test case 1: Several short segments that should be merged
    test_segments = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 1.5, "text": "there"},
        {"start": 1.5, "end": 3.0, "text": "this is a longer segment"},
        {"start": 3.0, "end": 3.5, "text": "but"},
        {"start": 3.5, "end": 4.0, "text": "this"},
        {"start": 4.0, "end": 4.5, "text": "is"},
        {"start": 4.5, "end": 6.0, "text": "another complete thought"}
    ]
    
    print("\n📝 Original segments:")
    for i, seg in enumerate(test_segments):
        print(f"   {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text']}\" ({len(seg['text'].split())} words)")
    
    # Process segments
    processed = SubtitleFormatter.post_process_segments(test_segments.copy())
    
    print("\n✅ Processed segments (after merging):")
    for i, seg in enumerate(processed):
        words = len(seg['text'].split())
        duration = seg['end'] - seg['start']
        char_rate = len(seg['text']) / duration if duration > 0 else 0
        print(f"   {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text']}\" ")
        print(f"      ({words} words, {char_rate:.1f} chars/sec)")
    
    # Test case 2: Segments that exceed character limits when merged
    print("\n" + "=" * 60)
    print("📝 Test case 2: Character limit enforcement")
    
    test_segments_2 = [
        {"start": 0.0, "end": 1.0, "text": "This"},
        {"start": 1.0, "end": 1.5, "text": "is"},
        {"start": 1.5, "end": 4.0, "text": "a very long segment that already uses most of the available character space"},
        {"start": 4.0, "end": 4.5, "text": "so"},
        {"start": 4.5, "end": 5.0, "text": "these"},
        {"start": 5.0, "end": 6.0, "text": "should stay separate"}
    ]
    
    print("\nOriginal segments:")
    for i, seg in enumerate(test_segments_2):
        print(f"   {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text']}\" ({len(seg['text'])} chars)")
    
    processed_2 = SubtitleFormatter.post_process_segments(test_segments_2.copy())
    
    print("\nProcessed segments:")
    for i, seg in enumerate(processed_2):
        print(f"   {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text']}\" ({len(seg['text'])} chars)")
    
    # Test case 3: Reading rate enforcement
    print("\n" + "=" * 60)
    print("📝 Test case 3: Reading rate enforcement (25 chars/sec max)")
    
    test_segments_3 = [
        {"start": 0.0, "end": 1.0, "text": "This segment has way too many characters for one second duration"},
        {"start": 1.0, "end": 2.0, "text": "This one is also quite long for the time"},
        {"start": 2.0, "end": 5.0, "text": "This one should be fine"}
    ]
    
    print("\nOriginal segments:")
    for i, seg in enumerate(test_segments_3):
        duration = seg['end'] - seg['start']
        char_rate = len(seg['text']) / duration
        print(f"   {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text']}\"")
        print(f"      ({char_rate:.1f} chars/sec - {'❌ TOO FAST' if char_rate > 25 else '✅ OK'})")
    
    processed_3 = SubtitleFormatter.post_process_segments(test_segments_3.copy())
    
    print("\nProcessed segments (with adjusted timing):")
    for i, seg in enumerate(processed_3):
        duration = seg['end'] - seg['start']
        char_rate = len(seg['text']) / duration
        print(f"   {i+1}. [{seg['start']:.1f}-{seg['end']:.1f}s] \"{seg['text']}\"")
        print(f"      ({char_rate:.1f} chars/sec - {'❌ TOO FAST' if char_rate > 25 else '✅ OK'})")
    
    # Generate VTT output to show final result
    print("\n" + "=" * 60)
    print("📝 Final VTT output (with 42 char line limit):")
    print()
    
    vtt_content = SubtitleFormatter.create_vtt(processed, max_chars_per_line=42)
    print(vtt_content)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Original segments: {len(test_segments)}")
    print(f"   After merging: {len(processed)}")
    print(f"   Reduction: {len(test_segments) - len(processed)} segments merged")
    
    # Validate all constraints
    print("\n🔍 Validation:")
    all_valid = True
    
    for seg in processed:
        # Check minimum words (should be 3+ or last segment)
        words = len(seg['text'].split())
        if words < 3 and seg != processed[-1]:
            print(f"   ❌ Segment has only {words} words: \"{seg['text']}\"")
            all_valid = False
        
        # Check reading rate
        duration = seg['end'] - seg['start']
        char_rate = len(seg['text']) / duration if duration > 0 else 0
        if char_rate > 25:
            print(f"   ❌ Reading rate too fast ({char_rate:.1f} chars/sec): \"{seg['text']}\"")
            all_valid = False
        
        # Check character limit (84 chars max for 2 lines)
        if len(seg['text']) > 84:
            print(f"   ❌ Text too long ({len(seg['text'])} chars): \"{seg['text']}\"")
            all_valid = False
    
    if all_valid:
        print("   ✅ All segments meet the requirements!")
    
    return all_valid


if __name__ == "__main__":
    success = test_short_subtitle_merging()
    sys.exit(0 if success else 1)
