#!/usr/bin/env python3
"""
Simplified test runner for the captioner GUI
Returns test results in a format suitable for display
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from captioner import (
    SubtitleFormatter, 
    FasterWhisperModelManager, 
    TranscriptionProcessor,
    AudioProcessor,
    ProcessingStats
)


class CaptionerTestRunner:
    """Run tests and return formatted results"""
    
    def __init__(self):
        self.results = []
        
    def run_quick_tests(self, test_video_path: str = None) -> Tuple[bool, str]:
        """
        Run a quick test suite with a video file or test data
        Returns (success, formatted_results)
        """
        output = []
        output.append("🧪 CAPTIONER TEST SUITE")
        output.append("=" * 60)
        
        # If no video provided, use test data
        if not test_video_path or not os.path.exists(test_video_path):
            output.append("📝 Using synthetic test data (no video file provided)")
            success, test_output = self._run_synthetic_tests()
            output.extend(test_output)
        else:
            output.append(f"📹 Testing with video: {os.path.basename(test_video_path)}")
            success, test_output = self._run_video_tests(test_video_path)
            output.extend(test_output)
        
        # Summary
        output.append("")
        output.append("=" * 60)
        if success:
            output.append("✅ ALL TESTS PASSED!")
        else:
            output.append("❌ SOME TESTS FAILED - Please review results above")
        
        return success, "\n".join(output)
    
    def test_existing_files(self, vtt_path: str, txt_path: str = None) -> Tuple[bool, str]:
        """
        Test existing VTT and TXT files without re-transcription
        Returns (success, formatted_results)
        """
        output = []
        output.append("🧪 CAPTIONER FILE VALIDATION")
        output.append("=" * 60)
        output.append(f"📄 Testing VTT: {os.path.basename(vtt_path)}")
        
        if txt_path:
            output.append(f"📄 Testing TXT: {os.path.basename(txt_path)}")
        
        success, test_output = self._run_file_tests(vtt_path, txt_path)
        output.extend(test_output)
        
        # Summary
        output.append("")
        output.append("=" * 60)
        if success:
            output.append("✅ ALL TESTS PASSED!")
        else:
            output.append("❌ SOME TESTS FAILED - Please review results above")
        
        return success, "\n".join(output)
    
    def _run_synthetic_tests(self) -> Tuple[bool, List[str]]:
        """Run tests with synthetic data"""
        output = []
        all_passed = True
        
        # Test 1: VTT Creation
        output.append("\n🔍 Test 1: VTT File Creation")
        try:
            segments = [
                {"start": 0.0, "end": 2.0, "text": "This is a test subtitle"},
                {"start": 2.0, "end": 4.0, "text": "With multiple lines that need to be formatted correctly"},
            ]
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            
            if vtt_content.startswith("WEBVTT") and len(vtt_content) > 50:
                output.append("   ✅ VTT creation successful")
            else:
                output.append("   ❌ VTT creation failed")
                all_passed = False
        except Exception as e:
            output.append(f"   ❌ Error: {e}")
            all_passed = False
        
        # Test 2: Character Limit
        output.append("\n🔍 Test 2: 42 Character Line Limit")
        try:
            test_text = "This is a very long line that definitely exceeds the forty-two character limit and needs to be split"
            segments = [{"start": 0.0, "end": 3.0, "text": test_text}]
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            
            # Check all lines
            lines = vtt_content.split('\n')
            violations = []
            for line in lines:
                if line and not line.startswith('WEBVTT') and '-->' not in line and len(line) > 42:
                    violations.append(f"{line} ({len(line)} chars)")
            
            if not violations:
                output.append("   ✅ All lines within 42 character limit")
            else:
                output.append(f"   ❌ Lines exceeding limit: {violations[0]}")
                all_passed = False
        except Exception as e:
            output.append(f"   ❌ Error: {e}")
            all_passed = False
        
        # Test 3: 2 Line Limit
        output.append("\n🔍 Test 3: 2 Line Subtitle Limit")
        try:
            test_text = "This is a very long subtitle that needs to be broken into multiple lines but should never exceed two lines per subtitle segment"
            segments = [{"start": 0.0, "end": 3.0, "text": test_text}]
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            
            # Count lines per subtitle
            subtitle_line_counts = []
            lines = vtt_content.split('\n')
            i = 0
            while i < len(lines):
                if '-->' in lines[i]:
                    i += 1
                    line_count = 0
                    while i < len(lines) and lines[i].strip():
                        line_count += 1
                        i += 1
                    if line_count > 0:
                        subtitle_line_counts.append(line_count)
                else:
                    i += 1
            
            max_lines = max(subtitle_line_counts) if subtitle_line_counts else 0
            
            if max_lines <= 2:
                output.append(f"   ✅ All subtitles within 2 line limit (max: {max_lines})")
            else:
                output.append(f"   ❌ Subtitles exceed 2 line limit (max: {max_lines})")
                all_passed = False
        except Exception as e:
            output.append(f"   ❌ Error: {e}")
            all_passed = False
        
        # Test 4: Word Preservation
        output.append("\n🔍 Test 4: Word Preservation")
        try:
            original_text = "Testing word preservation with compound words like high-end and URLs"
            words = original_text.split()
            
            # Simulate word-level data
            word_data = []
            time = 0.0
            for word in words:
                word_data.append({
                    'word': word,
                    'start': time,
                    'end': time + 0.5
                })
                time += 0.5
            
            segments = SubtitleFormatter.create_optimized_segments(word_data, max_chars=84)
            
            # Extract text from segments
            segment_text = ' '.join([s.get('text', '') for s in segments])
            segment_words = segment_text.split()
            
            if len(words) == len(segment_words):
                output.append(f"   ✅ All {len(words)} words preserved")
            else:
                output.append(f"   ❌ Word loss: {len(words)} → {len(segment_words)}")
                all_passed = False
        except Exception as e:
            output.append(f"   ❌ Error: {e}")
            all_passed = False
        
        return all_passed, output
    
    def _run_video_tests(self, video_path: str) -> Tuple[bool, List[str]]:
        """Run tests with actual video file"""
        output = []
        all_passed = True
        
        try:
            # Use temporary directory for test output
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process video
                output.append("\n🎬 Processing video...")
                model_manager = FasterWhisperModelManager()
                processor = TranscriptionProcessor()
                stats = ProcessingStats()
                
                # Extract audio
                audio_path, duration = processor.extract_audio_optimized(video_path, stats)
                output.append(f"   ✅ Audio extracted: {duration:.1f}s")
                
                # Transcribe
                with model_manager.model_context("tiny") as model:  # Use tiny for speed
                    text, segments = processor.transcribe_optimized(
                        audio_path, model, True, 84, stats
                    )
                
                output.append(f"   ✅ Transcribed: {len(text)} chars, {len(segments)} segments")
                
                # Save outputs
                txt_path = Path(temp_dir) / "test.txt"
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
                vtt_path = Path(temp_dir) / "test.vtt"
                with open(vtt_path, 'w', encoding='utf-8') as f:
                    f.write(vtt_content)
                
                # Run tests
                output.append("\n🧪 Running validation tests...")
                
                # Test 1: VTT Creation
                if vtt_path.exists() and vtt_path.stat().st_size > 0:
                    output.append("   ✅ Test 1: VTT file created successfully")
                else:
                    output.append("   ❌ Test 1: VTT file creation failed")
                    all_passed = False
                
                # Test 2: Word Preservation
                original_words = text.split()
                vtt_text = self._extract_text_from_vtt(vtt_content)
                vtt_words = vtt_text.split()
                
                if len(original_words) == len(vtt_words):
                    output.append(f"   ✅ Test 2: All {len(original_words)} words preserved")
                else:
                    output.append(f"   ❌ Test 2: Word loss {len(original_words)} → {len(vtt_words)}")
                    all_passed = False
                
                # Test 3: Character Limit
                violations = self._check_character_limit(vtt_content, 42)
                if not violations:
                    output.append("   ✅ Test 3: All lines within 42 character limit")
                else:
                    output.append(f"   ❌ Test 3: {len(violations)} lines exceed limit")
                    all_passed = False
                
                # Test 4: Line Limit
                max_lines = self._check_line_limit(vtt_content)
                if max_lines <= 2:
                    output.append(f"   ✅ Test 4: All subtitles within 2 line limit")
                else:
                    output.append(f"   ❌ Test 4: Max {max_lines} lines per subtitle")
                    all_passed = False
                
                # Cleanup
                processor.cleanup_temp_files()
                
        except Exception as e:
            output.append(f"\n❌ Test error: {e}")
            all_passed = False
        
        return all_passed, output
    
    def _extract_text_from_vtt(self, vtt_content: str) -> str:
        """Extract all text from VTT content"""
        lines = vtt_content.split('\n')
        text_parts = []
        
        i = 0
        while i < len(lines):
            if '-->' in lines[i]:
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_parts.append(lines[i].strip())
                    i += 1
            else:
                i += 1
        
        return ' '.join(text_parts)
    
    def _check_character_limit(self, vtt_content: str, limit: int) -> List[str]:
        """Check for lines exceeding character limit"""
        violations = []
        lines = vtt_content.split('\n')
        
        for line in lines:
            if (line and 
                not line.startswith('WEBVTT') and 
                '-->' not in line and 
                not line.startswith('NOTE') and
                len(line) > limit):
                violations.append(line)
        
        return violations
    
    def _check_line_limit(self, vtt_content: str) -> int:
        """Check maximum lines per subtitle"""
        max_lines = 0
        lines = vtt_content.split('\n')
        
        i = 0
        while i < len(lines):
            if '-->' in lines[i]:
                i += 1
                line_count = 0
                while i < len(lines) and lines[i].strip():
                    line_count += 1
                    i += 1
                max_lines = max(max_lines, line_count)
            else:
                i += 1
        
        return max_lines
    
    def _run_file_tests(self, vtt_path: str, txt_path: str = None) -> Tuple[bool, List[str]]:
        """Test existing VTT and TXT files"""
        output = []
        all_passed = True
        
        try:
            # Read VTT file
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                vtt_content = f.read()
            
            # Read TXT file if provided
            txt_content = None
            if txt_path and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8-sig') as f:
                    txt_content = f.read()
            
            # Test 1: VTT File Validity
            output.append("\n🔍 Test 1: VTT File Validity")
            if vtt_content.startswith("WEBVTT") and len(vtt_content) > 50:
                output.append("   ✅ Valid VTT format")
            else:
                output.append("   ❌ Invalid VTT format")
                all_passed = False
            
            # Test 2: 42 Character Line Limit
            output.append("\n🔍 Test 2: 42 Character Line Limit")
            violations = self._check_character_limit(vtt_content, 42)
            
            if not violations:
                output.append("   ✅ All lines within 42 character limit")
            else:
                output.append(f"   ❌ {len(violations)} lines exceed limit:")
                for v in violations[:3]:  # Show first 3 violations
                    output.append(f"      '{v}' ({len(v)} chars)")
                if len(violations) > 3:
                    output.append(f"      ... and {len(violations) - 3} more")
                all_passed = False
            
            # Test 3: 2 Line Subtitle Limit
            output.append("\n🔍 Test 3: 2 Line Subtitle Limit")
            max_lines = self._check_line_limit(vtt_content)
            
            if max_lines <= 2:
                output.append(f"   ✅ All subtitles within 2 line limit (max: {max_lines})")
            else:
                output.append(f"   ❌ Subtitles exceed 2 line limit (max: {max_lines} lines)")
                all_passed = False
            
            # Test 4: Word Preservation (if TXT provided)
            if txt_content:
                output.append("\n🔍 Test 4: Word Preservation")
                
                # Extract words from TXT
                txt_words = txt_content.strip().split()
                txt_word_count = len(txt_words)
                
                # Extract words from VTT
                vtt_text = self._extract_text_from_vtt(vtt_content)
                vtt_words = vtt_text.split()
                vtt_word_count = len(vtt_words)
                
                if txt_word_count == vtt_word_count:
                    output.append(f"   ✅ All {txt_word_count} words preserved")
                else:
                    output.append(f"   ❌ Word count mismatch: TXT has {txt_word_count}, VTT has {vtt_word_count}")
                    
                    # Find missing words
                    txt_words_clean = [w.lower().strip('.,!?";:\'"') for w in txt_words]
                    vtt_words_clean = [w.lower().strip('.,!?";:\'"') for w in vtt_words]
                    
                    txt_set = set(txt_words_clean)
                    vtt_set = set(vtt_words_clean)
                    missing = txt_set - vtt_set
                    
                    if missing:
                        output.append(f"      Missing words: {list(missing)[:10]}")
                    
                    all_passed = False
            
            # Test 5: Chronological Order (basic check)
            output.append("\n🔍 Test 5: Chronological Order")
            timestamps = []
            lines = vtt_content.split('\n')
            
            for line in lines:
                if '-->' in line:
                    try:
                        start_time_str = line.split(' --> ')[0].strip()
                        # Parse time (HH:MM:SS.mmm)
                        parts = start_time_str.split(':')
                        if len(parts) == 3:
                            h, m, s_ms = parts
                            s, ms = s_ms.split('.')
                            timestamp = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
                            timestamps.append(timestamp)
                    except:
                        pass
            
            # Check if timestamps are in order
            is_ordered = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            
            if is_ordered:
                output.append(f"   ✅ Subtitles in chronological order ({len(timestamps)} entries)")
            else:
                output.append(f"   ❌ Subtitles not in chronological order")
                all_passed = False
            
        except Exception as e:
            output.append(f"\n❌ Error testing files: {e}")
            all_passed = False
        
        return all_passed, output
    
    def test_output_folder(self, folder_path: str) -> Tuple[bool, str]:
        """Test all VTT/TXT pairs in an output folder"""
        output = []
        output.append("🧪 BATCH FILE VALIDATION")
        output.append("=" * 60)
        output.append(f"📂 Testing folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            output.append("❌ Folder does not exist")
            return False, "\n".join(output)
        
        # Find all VTT files
        vtt_files = list(Path(folder_path).glob("*.vtt"))
        
        if not vtt_files:
            output.append("❌ No VTT files found in folder")
            return False, "\n".join(output)
        
        output.append(f"📊 Found {len(vtt_files)} VTT files to test")
        output.append("")
        
        all_passed = True
        passed_count = 0
        
        for vtt_path in vtt_files:
            # Look for corresponding TXT file
            txt_path = vtt_path.with_suffix('.txt')
            
            output.append(f"\n{'='*50}")
            output.append(f"📄 Testing: {vtt_path.name}")
            
            # Test this file pair
            success, test_output = self._run_file_tests(str(vtt_path), str(txt_path) if txt_path.exists() else None)
            
            if success:
                passed_count += 1
                output.append("✅ PASSED")
            else:
                all_passed = False
                output.append("❌ FAILED")
            
            output.extend(test_output)
        
        # Summary
        output.append(f"\n{'='*60}")
        output.append(f"📊 SUMMARY: {passed_count}/{len(vtt_files)} files passed")
        
        if all_passed:
            output.append("✅ ALL FILES PASSED!")
        else:
            output.append("❌ SOME FILES FAILED - Please review results above")
        
        return all_passed, "\n".join(output)


if __name__ == "__main__":
    # Run standalone test
    runner = CaptionerTestRunner()
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test-existing":
        # Test existing files mode
        if len(sys.argv) >= 4:
            txt_path = sys.argv[2]
            vtt_path = sys.argv[3]
            success, output = runner.test_existing_files(vtt_path, txt_path)
        elif len(sys.argv) == 3:
            vtt_path = sys.argv[2]
            success, output = runner.test_existing_files(vtt_path)
        else:
            print("Usage: test_runner.py --test-existing <txt_path> <vtt_path>")
            print("   or: test_runner.py --test-existing <vtt_path>")
            sys.exit(1)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-folder":
        # Test folder mode
        if len(sys.argv) >= 3:
            folder_path = sys.argv[2]
            success, output = runner.test_output_folder(folder_path)
        else:
            print("Usage: test_runner.py --test-folder <folder_path>")
            sys.exit(1)
    else:
        # Regular mode: test with video or synthetic data
        test_video = None
        if len(sys.argv) > 1:
            test_video = sys.argv[1]
        
        success, output = runner.run_quick_tests(test_video)
    
    print(output)
    sys.exit(0 if success else 1)
