#!/usr/bin/env python3
# vtt_reconciler.py
"""
VTT Word Reconciler - Standalone tool to fix missing words in VTT files
Compares transcript with VTT and inserts missing words with line overflow handling
Supports single file or batch folder processing
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict

class VTTSegment:
    """VTT subtitle segment"""
    def __init__(self, start: str, end: str, text: str, position: str = ""):
        self.start = start
        self.end = end
        self.text = text
        self.position = position

def parse_vtt(vtt_path: str) -> tuple:
    """Parse VTT file"""
    with open(vtt_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    
    header_match = re.match(r'(WEBVTT.*?\n\n)', content, re.DOTALL)
    header = header_match.group(1) if header_match else "WEBVTT\n\n"
    
    segments = []
    pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s+-->\s+(\d{2}:\d{2}:\d{2}\.\d{3})\s*([^\n]*)\n'
        r'((?:(?!\d{2}:\d{2}:\d{2}\.\d{3}).+\n?)+)',
        re.MULTILINE
    )
    
    for match in pattern.finditer(content):
        segments.append(VTTSegment(
            match.group(1),
            match.group(2),
            match.group(4).strip(),
            match.group(3).strip()
        ))
    
    return segments, header

def read_transcript(txt_path: str) -> str:
    """Read transcript"""
    with open(txt_path, 'r', encoding='utf-8-sig') as f:
        text = f.read().strip()
    return text

def extract_text(segments: List[VTTSegment]) -> str:
    """Extract text from segments"""
    return ' '.join([seg.text.replace('\n', ' ') for seg in segments])

def find_missing_words(transcript: str, vtt_text: str) -> List[Dict]:
    """Find missing words"""
    def normalize(w):
        return w.lower().strip('.,!?";:\'"') if w else None
    
    trans_words = transcript.split()
    vtt_words = vtt_text.split()
    
    trans_norm = [normalize(w) for w in trans_words]
    vtt_norm = [normalize(w) for w in vtt_words]
    
    missing = []
    vtt_idx = 0
    
    for i, word_norm in enumerate(trans_norm):
        if not word_norm:
            continue
        
        found = False
        
        if vtt_idx < len(vtt_norm):
            if word_norm == vtt_norm[vtt_idx]:
                vtt_idx += 1
                found = True
            else:
                for offset in range(1, min(6, len(vtt_norm) - vtt_idx)):
                    if word_norm == vtt_norm[vtt_idx + offset]:
                        vtt_idx += offset + 1
                        found = True
                        break
        
        if not found:
            missing.append({
                'word': trans_words[i],
                'before': trans_words[i-1] if i > 0 else None,
                'after': trans_words[i+1] if i < len(trans_words)-1 else None
            })
    
    return missing

def insert_words(segments: List[VTTSegment], missing: List[Dict], verbose: bool = True) -> tuple:
    """Insert missing words into segments with improved context matching"""
    if not missing:
        return segments, 0
    
    def normalize(w):
        return w.lower().strip('.,!?";:\'"') if w else None
    
    def find_word_position(words_norm, target_norm):
        """Find position of word in normalized word list"""
        try:
            return words_norm.index(target_norm)
        except ValueError:
            return -1
    
    inserted = set()
    failed_insertions = []
    
    for seg_idx, seg in enumerate(segments):
        text = seg.text
        original_text = text
        prev_seg = segments[seg_idx - 1] if seg_idx > 0 else None
        next_seg = segments[seg_idx + 1] if seg_idx < len(segments) - 1 else None
        
        for miss_idx, miss in enumerate(missing):
            if miss_idx in inserted:
                continue
            
            before = miss['before']
            after = miss['after']
            word = miss['word']
            
            before_norm = normalize(before)
            after_norm = normalize(after)
            
            text_flat = text.replace('\n', ' ')
            words = text_flat.split()
            words_norm = [normalize(w) for w in words]
            
            before_in = before_norm in words_norm if before_norm else False
            after_in = after_norm in words_norm if after_norm else False
            
            # Get positions if present
            before_pos = find_word_position(words_norm, before_norm) if before_norm else -1
            after_pos = find_word_position(words_norm, after_norm) if after_norm else -1
            
            segment_reason = None
            inserted_this_segment = False
            
            # Case 1: Both in same segment AND adjacent or near each other
            if before_in and after_in and before and after:
                # Check if they're reasonably close (within 5 words)
                if before_pos >= 0 and after_pos >= 0 and 0 < (after_pos - before_pos) <= 5:
                    # Insert between them by reconstructing
                    new_words = words[:after_pos] + [word] + words[after_pos:]
                    text = ' '.join(new_words)
                    # Restore line breaks (simple heuristic)
                    if '\n' in original_text:
                        # Keep original line structure roughly
                        words_per_line = len(original_text.split('\n')[0].split())
                        lines = []
                        current_line = []
                        for w in new_words:
                            current_line.append(w)
                            if len(current_line) >= words_per_line:
                                lines.append(' '.join(current_line))
                                current_line = []
                        if current_line:
                            lines.append(' '.join(current_line))
                        text = '\n'.join(lines)
                    
                    inserted.add(miss_idx)
                    inserted_this_segment = True
                else:
                    segment_reason = f"words not adjacent (positions: {before_pos}, {after_pos})"
            
            # Case 2: Only after word present - insert before it
            elif after_in and not before_in and after and not inserted_this_segment:
                if after_pos >= 0:
                    # Insert before the after word
                    new_words = words[:after_pos] + [word] + words[after_pos:]
                    text = ' '.join(new_words)
                    if '\n' in original_text and len(original_text.split('\n')) > 1:
                        # Preserve line breaks roughly
                        mid = len(new_words) // 2
                        text = ' '.join(new_words[:mid]) + '\n' + ' '.join(new_words[mid:])
                    
                    inserted.add(miss_idx)
                    inserted_this_segment = True
                else:
                    segment_reason = f"after '{after}' found but position unknown"
            
            # Case 3: Only before word present - insert after it  
            elif before_in and not after_in and before and not inserted_this_segment:
                if before_pos >= 0:
                    # Insert after the before word
                    insert_pos = before_pos + 1
                    new_words = words[:insert_pos] + [word] + words[insert_pos:]
                    text = ' '.join(new_words)
                    if '\n' in original_text and len(original_text.split('\n')) > 1:
                        mid = len(new_words) // 2
                        text = ' '.join(new_words[:mid]) + '\n' + ' '.join(new_words[mid:])
                    
                    inserted.add(miss_idx)
                    inserted_this_segment = True
                else:
                    segment_reason = f"before '{before}' found but position unknown"
            
            # Case 4: Cross-segment (before at end of this, after at start of next)
            elif before_in and next_seg and not inserted_this_segment:
                if before_pos == len(words_norm) - 1:  # Last word
                    next_words = next_seg.text.replace('\n', ' ').split()
                    next_norm = [normalize(w) for w in next_words]
                    
                    if after_norm and after_norm in next_norm[:3]:
                        text = text.rstrip() + ' ' + word
                        inserted.add(miss_idx)
                        inserted_this_segment = True
                    else:
                        segment_reason = f"cross-seg: after not at next start"
                else:
                    segment_reason = f"cross-seg: before not at end (pos {before_pos}/{len(words_norm)-1})"
            
            # Case 5: Cross-segment (before at end of prev, after at start of this)
            elif after_in and prev_seg and not inserted_this_segment:
                if after_pos <= 2:  # Near start
                    prev_words = prev_seg.text.replace('\n', ' ').split()
                    prev_norm = [normalize(w) for w in prev_words]
                    
                    if before_norm and before_norm in prev_norm[-3:]:
                        text = word + ' ' + text.lstrip()
                        inserted.add(miss_idx)
                        inserted_this_segment = True
                    else:
                        segment_reason = f"cross-seg: before not at prev end"
                else:
                    segment_reason = f"cross-seg: after not at start (pos {after_pos})"
            
            else:
                segment_reason = f"no context (before='{before}' [{before_in}], after='{after}' [{after_in}])"
            
            # Record failure
            if not inserted_this_segment and segment_reason and miss_idx not in inserted:
                if not any(f['word'] == word for f in failed_insertions):
                    failed_insertions.append({
                        'word': word,
                        'before': before,
                        'after': after,
                        'seg_idx': seg_idx,
                        'reason': segment_reason
                    })
        
        # Update segment
        if text != original_text:
            seg.text = text
    
    # Report failures
    not_inserted = [missing[i] for i in range(len(missing)) if i not in inserted]
    
    if not verbose and not_inserted:
        print(f"      ❌ Failed to insert {len(not_inserted)} words:")
        for miss in not_inserted:
            word = miss['word']
            before = miss['before']
            after = miss['after']
            failure = next((f for f in failed_insertions if f['word'] == word), None)
            if failure:
                print(f"         '{word}' (before='{before}', after='{after}')")
                print(f"         Reason: {failure['reason']}")
    
    return segments, len(inserted)





def reflow_segments(segments: List[VTTSegment], max_chars: int, verbose: bool = True) -> tuple:
    """Reflow lines exceeding character limit WITHOUT losing words"""
    reflowed = 0
    
    for seg in segments:
        lines = seg.text.split('\n')
        needs_reflow = any(len(line) > max_chars for line in lines)
        
        if needs_reflow:
            # Get ALL words from the segment
            all_words = seg.text.replace('\n', ' ').split()
            
            # Rebuild with proper line breaks (max 2 lines)
            new_lines = []
            current = ""
            
            for word in all_words:
                test = f"{current} {word}" if current else word
                if len(test) <= max_chars:
                    current = test
                else:
                    # Current line is full
                    if current:
                        new_lines.append(current)
                    
                    # Check if we've reached max lines
                    if len(new_lines) >= 2:
                        # Already have 2 lines, can't add more
                        # Keep the word in current for potential truncation
                        current = word
                    else:
                        # Start new line
                        if len(word) <= max_chars:
                            current = word
                        else:
                            # Word itself is too long
                            current = word[:max_chars-3] + "..."
            
            # Add the last line
            if current:
                if len(new_lines) >= 2:
                    # Already at max, append to last line or truncate
                    new_lines[-1] = new_lines[-1] + "..."  # Indicate truncation
                else:
                    new_lines.append(current)
            
            old_word_count = len(all_words)
            new_word_count = len(' '.join(new_lines).split())
            
            # Only use new text if we didn't lose words
            if new_word_count >= old_word_count:
                seg.text = '\n'.join(new_lines)
                reflowed += 1
            else:
                # Don't reflow if it would lose words
                if verbose:
                    print(f"      ⚠️  Skipped reflow to preserve {old_word_count - new_word_count} words")
    
    if verbose and reflowed > 0:
        print(f"🔄 Reflowed {reflowed} segments")
    
    return segments, reflowed


def write_vtt(segments: List[VTTSegment], header: str, output: str):
    """Write VTT file"""
    lines = [header.rstrip(), ""]
    
    for seg in segments:
        timestamp = f"{seg.start} --> {seg.end}"
        if seg.position:
            timestamp += f" {seg.position}"
        lines.extend([timestamp, seg.text, ""])
    
    with open(output, 'w', encoding='utf-8-sig') as f:
        f.write("\n".join(lines))

def process_single_file(txt_path: Path, vtt_path: Path, output_path: Path, max_chars: int, verbose: bool = True):
    """Process a single TXT/VTT pair"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"📄 {txt_path.name}")
        print(f"{'='*60}")
    
    # Read files
    transcript = read_transcript(str(txt_path))
    segments, header = parse_vtt(str(vtt_path))
    
    if verbose:
        print(f"✅ Transcript: {len(transcript.split())} words")
        print(f"✅ Parsed: {len(segments)} segments")
    
    # Find missing
    vtt_text = extract_text(segments)
    missing = find_missing_words(transcript, vtt_text)
    
    if verbose:
        print(f"📊 VTT words: {len(vtt_text.split())} | Missing: {len(missing)}")
        if missing:
            print(f"🔍 Missing words: {[m['word'] for m in missing[:10]]}")
    
    # Insert
    inserted_count = 0
    if missing:
        if verbose:
            print(f"\n🔧 Inserting...")
        segments, inserted_count = insert_words(segments, missing, verbose)
    
    # Reflow
    if verbose:
        print(f"\n🔄 Checking overflow...")
    segments, reflowed_count = reflow_segments(segments, max_chars, verbose)
    
    # Write
    write_vtt(segments, header, str(output_path))
    
    if verbose:
        print(f"✅ Written: {output_path.name}")
    
    # Verify
    final_text = extract_text(segments)
    trans_words = len(transcript.split())
    final_words = len(final_text.split())
    diff = trans_words - final_words
    
    return {
        'success': True,
        'inserted': inserted_count,
        'reflowed': reflowed_count,
        'perfect_match': diff == 0,
        'word_diff': abs(diff),
        'transcript_words': trans_words,
        'final_words': final_words
    }

def process_folder(folder_path: Path, max_chars: int):
    """Process all TXT/VTT pairs in a folder"""
    print(f"\n🚀 Batch VTT Reconciliation")
    print(f"📁 Folder: {folder_path}")
    print(f"📏 Max chars/line: {max_chars}")
    print(f"\n{'='*60}")
    
    # Find all TXT files
    txt_files = sorted(folder_path.glob("*.txt"))
    
    if not txt_files:
        print("❌ No TXT files found in folder")
        return
    
    print(f"Found {len(txt_files)} TXT files")
    
    # Also check for VTT files
    vtt_files = sorted(folder_path.glob("*.vtt"))
    print(f"Found {len(vtt_files)} VTT files")
    print()
    
    results = {
        'total': 0,
        'processed': 0,
        'skipped': 0,
        'perfect_matches': 0,
        'partial_matches': 0,
        'total_inserted': 0,
        'total_reflowed': 0
    }
    
    for txt_path in txt_files:
        results['total'] += 1
        
        # Find corresponding VTT
        vtt_path = txt_path.with_suffix('.vtt')
        
        print(f"\n📄 Processing: {txt_path.name}")
        print(f"   TXT exists: {txt_path.exists()}")
        print(f"   VTT exists: {vtt_path.exists()}")
        
        if not vtt_path.exists():
            print(f"   ⏭️  SKIPPED: No matching VTT file")
            results['skipped'] += 1
            continue
        
        # Check if already has _corrected version
        corrected_path = txt_path.parent / f"{txt_path.stem}_corrected.vtt"
        
        try:
            # Read and check word counts first
            transcript = read_transcript(str(txt_path))
            segments, header = parse_vtt(str(vtt_path))
            
            # Count words BEFORE any processing
            vtt_text_before = extract_text(segments)
            vtt_words_before = len(vtt_text_before.split())
            segment_count = len(segments)
            
            missing = find_missing_words(transcript, vtt_text_before)
            
            trans_word_count = len(transcript.split())
            
            print(f"   📊 Transcript: {trans_word_count} words")
            print(f"   📊 VTT (before): {vtt_words_before} words in {segment_count} segments")
            print(f"   📊 Missing: {len(missing)} words")
            
            if missing:
                print(f"   🔍 Missing: {[m['word'] for m in missing]}")
            
            # Process
            if missing:
                print(f"   🔧 Inserting...")
                segments, inserted_count = insert_words(segments, missing, verbose=False)
                
                # Count words AFTER insertion
                vtt_text_after = extract_text(segments)
                vtt_words_after = len(vtt_text_after.split())
                
                print(f"   📊 VTT (after insert): {vtt_words_after} words")
                print(f"   ✅ Marked {inserted_count} as inserted")
                print(f"   📊 Actual word gain: {vtt_words_after - vtt_words_before}")
            else:
                inserted_count = 0
                vtt_words_after = vtt_words_before
                print(f"   ✅ No missing words")
            
            # Reflow
            segments, reflowed_count = reflow_segments(segments, max_chars, verbose=False)
            if reflowed_count > 0:
                print(f"   🔄 Reflowed {reflowed_count} segments")
            
            # Write
            write_vtt(segments, header, str(corrected_path))
            print(f"   💾 Written to: {corrected_path.name}")
            
            # Verify by reading the file back
            segments_verify, _ = parse_vtt(str(corrected_path))
            final_text = extract_text(segments_verify)
            final_words = len(final_text.split())
            diff = trans_word_count - final_words
            
            print(f"   📊 Final (verified): {final_words} words")
            print(f"   📊 Difference: {abs(diff)} words")
            
            results['processed'] += 1
            results['total_inserted'] += inserted_count
            results['total_reflowed'] += reflowed_count
            
            if diff == 0:
                results['perfect_matches'] += 1
                print(f"   ✅ PERFECT MATCH")
            else:
                results['partial_matches'] += 1
                print(f"   ⚠️  PARTIAL ({abs(diff)} words still missing)")
                
                # Show which words are still missing
                still_missing = find_missing_words(transcript, final_text)
                if still_missing:
                    print(f"   ❌ Still missing: {[m['word'] for m in still_missing]}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results['skipped'] += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {results['total']}")
    print(f"Processed: {results['processed']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Perfect matches: {results['perfect_matches']}")
    print(f"Partial matches: {results['partial_matches']}")
    print(f"Total marked as inserted: {results['total_inserted']}")
    print(f"Total segments reflowed: {results['total_reflowed']}")
    print(f"\n📁 Corrected files saved as: *_corrected.vtt")



def main():
    parser = argparse.ArgumentParser(
        description="VTT Word Reconciler - Fix missing words in VTT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single file:
    python vtt_reconciler.py transcript.txt subtitles.vtt
    
  Single file with options:
    python vtt_reconciler.py transcript.txt subtitles.vtt --max-chars 50 -o output.vtt
    
  Batch process folder:
    python vtt_reconciler.py --folder /path/to/folder
    
  Interactive folder selection:
    python vtt_reconciler.py --folder
        """
    )
    
    # Positional arguments (for single file mode)
    parser.add_argument("txt_file", nargs='?', help="Transcript TXT file")
    parser.add_argument("vtt_file", nargs='?', help="VTT subtitle file")
    
    # Optional arguments
    parser.add_argument("-o", "--output", help="Output file (default: *_corrected.vtt)")
    parser.add_argument("--max-chars", type=int, default=42, help="Max chars/line (default: 42)")
    parser.add_argument("--folder", nargs='?', const='PROMPT', help="Process all files in folder")
    
    args = parser.parse_args()
    
    # Folder mode
    if args.folder:
        if args.folder == 'PROMPT':
            # Interactive folder selection
            try:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()
                folder_path = filedialog.askdirectory(title="Select folder with TXT and VTT files")
                root.destroy()
                
                if not folder_path:
                    print("❌ No folder selected")
                    sys.exit(1)
                
                folder_path = Path(folder_path)
            except ImportError:
                print("❌ tkinter not available for GUI folder selection")
                print("💡 Use: python vtt_reconciler.py --folder /path/to/folder")
                sys.exit(1)
        else:
            folder_path = Path(args.folder)
        
        if not folder_path.exists():
            print(f"❌ Folder not found: {folder_path}")
            sys.exit(1)
        
        if not folder_path.is_dir():
            print(f"❌ Not a folder: {folder_path}")
            sys.exit(1)
        
        process_folder(folder_path, args.max_chars)
        return
    
    # Single file mode
    if not args.txt_file or not args.vtt_file:
        parser.print_help()
        sys.exit(1)
    
    txt_path = Path(args.txt_file)
    vtt_path = Path(args.vtt_file)
    
    if not txt_path.exists():
        print(f"❌ TXT not found: {txt_path}")
        sys.exit(1)
    
    if not vtt_path.exists():
        print(f"❌ VTT not found: {vtt_path}")
        sys.exit(1)
    
    output = Path(args.output) if args.output else vtt_path.parent / f"{vtt_path.stem}_corrected.vtt"
    
    print(f"🚀 VTT Reconciler (Single File)")
    print(f"📄 TXT: {txt_path}")
    print(f"📄 VTT: {vtt_path}")
    print(f"📄 Out: {output}")
    print(f"📏 Max: {args.max_chars} chars/line")
    
    # Process
    result = process_single_file(txt_path, vtt_path, output, args.max_chars, verbose=True)
    
    # Final summary
    print(f"\n📊 Result:")
    if result['perfect_match']:
        print(f"   ✅ Perfect match! All {result['transcript_words']} words present")
    else:
        print(f"   ⚠️  {result['word_diff']} words difference")
        print(f"   Transcript: {result['transcript_words']} words")
        print(f"   Final VTT: {result['final_words']} words")

if __name__ == "__main__":
    main()
