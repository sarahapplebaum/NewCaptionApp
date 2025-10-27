#!/usr/bin/env python3
"""
Comprehensive test suite for the captioner script
Tests the 4 key requirements:
1. VTT file generation and saving
2. Word preservation (no missing words)
3. 42 character max per line
4. 2 lines max per subtitle
"""

import os
import sys
import tempfile
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import json

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from captioner import (
    SubtitleFormatter, 
    FasterWhisperModelManager, 
    TranscriptionProcessor,
    AudioProcessor
)

class CaptionerTestSuite:
    """Comprehensive test suite for captioner functionality"""
    
    def __init__(self):
        self.test_results = []
        self.temp_dirs = []
        
    def create_test_audio(self, duration_seconds: float = 10.0) -> str:
        """Create a test audio file using ffmpeg"""
        try:
            ffmpeg_path = AudioProcessor.find_ffmpeg_optimized()
            if not ffmpeg_path:
                raise RuntimeError("FFmpeg not found - cannot create test audio")
            
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            audio_path = os.path.join(temp_dir, "test_audio.wav")
            
            # Generate a sine wave audio file
            cmd = [
                ffmpeg_path,
                '-f', 'lavfi',
                '-i', f'sine=frequency=440:duration={duration_seconds}',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg test audio creation failed: {result.stderr}")
            
            return audio_path
            
        except Exception as e:
            print(f"⚠️ Could not create test audio: {e}")
            return None
    
    def create_mock_transcription_data(self) -> Tuple[str, List[Dict]]:
        """Create mock transcription data for testing"""
        # Test text with known problematic words
        test_text = (
            "Welcome to this comprehensive tutorial about lighting fundamentals. "
            "We'll be covering screen space reflections as an option if you're targeting "
            "very high-end devices. These metallic surfaces reflect light beautifully. "
            "However, transparency cannot be used on materials as those will always use approximation. "
            "This creates more realistic rendering with better performance optimization."
        )
        
        # Create word-level timestamps
        words = []
        current_time = 0.0
        
        for word in test_text.split():
            word_duration = len(word) * 0.08 + 0.2  # Realistic duration
            words.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration
            })
            current_time += word_duration + 0.1  # Small gap between words
        
        return test_text, words
    
    def test_vtt_file_generation(self) -> bool:
        """Test 1: VTT file generation and saving"""
        print("\n🧪 TEST 1: VTT File Generation and Saving")
        print("-" * 50)
        
        try:
            # Create test data
            test_text, words = self.create_mock_transcription_data()
            
            # Create segments
            segments = SubtitleFormatter.create_optimized_segments(words, max_chars=42)
            
            if not segments:
                print("❌ No segments created")
                return False
            
            # Generate VTT content
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            
            if not vtt_content or not vtt_content.startswith("WEBVTT"):
                print("❌ Invalid VTT content generated")
                return False
            
            # Save to output_vtts directory
            output_dir = Path(__file__).parent / "output_vtts"
            output_dir.mkdir(exist_ok=True)
            vtt_path = output_dir / "test_output.vtt"
            
            with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                f.write(vtt_content)
            
            # Verify file exists and has content
            if not os.path.exists(vtt_path):
                print("❌ VTT file was not created")
                return False
            
            file_size = os.path.getsize(vtt_path)
            if file_size == 0:
                print("❌ VTT file is empty")
                return False
            
            # Verify file is readable and contains expected content
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                saved_content = f.read()
            
            if saved_content != vtt_content:
                print("❌ Saved VTT content doesn't match generated content")
                return False
            
            print(f"✅ VTT file successfully created: {vtt_path}")
            print(f"   File size: {file_size} bytes")
            print(f"   Contains {len(segments)} segments")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    def test_word_preservation(self) -> bool:
        """Test 2: All missing words are present in new VTT file"""
        print("\n🧪 TEST 2: Word Preservation (No Missing Words)")
        print("-" * 50)
        
        try:
            # Create test data with known problematic words
            test_text, words = self.create_mock_transcription_data()
            
            # Extract original words for comparison
            original_words = test_text.split()
            original_words_clean = [w.lower().strip('.,!?";:\'"') for w in original_words if w.strip()]
            
            print(f"📝 Original text has {len(original_words)} words")
            print(f"   Key test words: targeting, metallic, as, more")
            
            # Process through the pipeline
            segments = SubtitleFormatter.create_optimized_segments(words, max_chars=42)
            
            if not segments:
                print("❌ No segments created")
                return False
            
            # Extract words from segments
            segment_text = ' '.join([s.get('text', '') for s in segments])
            segment_words = segment_text.split()
            segment_words_clean = [w.lower().strip('.,!?";:\'"') for w in segment_words if w.strip()]
            
            print(f"📦 Processed segments have {len(segment_words)} words")
            
            # Check for missing words
            original_set = set(original_words_clean)
            segment_set = set(segment_words_clean)
            missing_words = original_set - segment_set
            
            if missing_words:
                print(f"❌ Missing words detected: {missing_words}")
                print(f"   Original: {len(original_words_clean)} unique words")
                print(f"   Segments: {len(segment_words_clean)} unique words")
                return False
            
            # Check specific problematic words
            test_words = ['targeting', 'metallic', 'as', 'more', 'comprehensive']
            missing_test_words = []
            
            for test_word in test_words:
                if test_word.lower() in [w.lower() for w in original_words]:
                    if test_word.lower() not in [w.lower() for w in segment_words]:
                        missing_test_words.append(test_word)
            
            if missing_test_words:
                print(f"❌ Critical test words missing: {missing_test_words}")
                return False
            
            # Check word counts (handle repeated words)
            word_count_errors = []
            original_counts = {}
            segment_counts = {}
            
            for word in original_words_clean:
                original_counts[word] = original_counts.get(word, 0) + 1
            
            for word in segment_words_clean:
                segment_counts[word] = segment_counts.get(word, 0) + 1
            
            for word, orig_count in original_counts.items():
                seg_count = segment_counts.get(word, 0)
                if seg_count < orig_count:
                    word_count_errors.append(f"'{word}': {orig_count} → {seg_count}")
            
            if word_count_errors:
                print(f"❌ Word count mismatches: {word_count_errors}")
                return False
            
            print(f"✅ All words preserved successfully")
            print(f"   Original: {len(original_words_clean)} words")
            print(f"   Segments: {len(segment_words_clean)} words")
            print(f"   Test words all present: {test_words}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    def test_42_character_limit(self) -> bool:
        """Test 3: All lines have maximum of 42 characters"""
        print("\n🧪 TEST 3: 42 Character Line Limit")
        print("-" * 50)
        
        try:
            # Create test data including some long sentences
            long_text = (
                "This is a deliberately long sentence that should be broken down into multiple lines "
                "to ensure that no single line exceeds the forty-two character limit that we have established "
                "for optimal readability and display compatibility across different devices and screen sizes."
            )
            
            # Create words with timestamps
            words = []
            current_time = 0.0
            
            for word in long_text.split():
                words.append({
                    "word": word,
                    "start": current_time,
                    "end": current_time + 0.4
                })
                current_time += 0.5
            
            # Create segments
            segments = SubtitleFormatter.create_optimized_segments(words, max_chars=42)
            
            if not segments:
                print("❌ No segments created")
                return False
            
            # Generate VTT
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            
            # Parse VTT and check each line
            lines = vtt_content.split('\n')
            line_violations = []
            subtitle_lines = []
            
            for i, line in enumerate(lines, 1):
                # Skip WEBVTT header, timestamps, and empty lines
                if (not line.strip() or 
                    line.strip().startswith('WEBVTT') or
                    '-->' in line or
                    line.strip().startswith('NOTE')):
                    continue
                
                # This is subtitle text
                subtitle_lines.append((i, line))
                
                if len(line) > 42:
                    line_violations.append(f"Line {i}: '{line}' ({len(line)} chars)")
            
            if line_violations:
                print(f"❌ Lines exceeding 42 characters:")
                for violation in line_violations[:10]:  # Show first 10
                    print(f"   {violation}")
                return False
            
            print(f"✅ All lines within 42 character limit")
            print(f"   Checked {len(subtitle_lines)} subtitle lines")
            print(f"   Max line length: {max((len(line) for _, line in subtitle_lines), default=0)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    def test_2_line_limit(self) -> bool:
        """Test 4: All subtitles have maximum of 2 lines"""
        print("\n🧪 TEST 4: 2 Line Subtitle Limit")
        print("-" * 50)
        
        try:
            # Create test data with varying sentence lengths
            test_sentences = [
                "Short sentence.",
                "This is a medium length sentence that should fit nicely.",
                "This is a very long sentence that might need to be split across multiple lines to ensure proper formatting and readability for viewers watching the content.",
                "Another sentence here.",
                "Final sentence to test the formatting."
            ]
            
            # Create words
            words = []
            current_time = 0.0
            
            for sentence in test_sentences:
                for word in sentence.split():
                    words.append({
                        "word": word,
                        "start": current_time,
                        "end": current_time + 0.4
                    })
                    current_time += 0.5
            
            # Create segments
            segments = SubtitleFormatter.create_optimized_segments(words, max_chars=84)  # Allow longer segments to test line breaking
            
            if not segments:
                print("❌ No segments created")
                return False
            
            # Generate VTT
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            
            # Parse VTT and analyze subtitle blocks
            lines = vtt_content.split('\n')
            subtitle_violations = []
            current_subtitle_lines = 0
            subtitle_count = 0
            line_counts = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip until we find a timestamp
                if '-->' not in line:
                    i += 1
                    continue
                
                # Found timestamp, now count following subtitle lines
                i += 1
                current_subtitle_lines = 0
                subtitle_text_lines = []
                
                # Count non-empty lines until next empty line
                while i < len(lines) and lines[i].strip():
                    current_subtitle_lines += 1
                    subtitle_text_lines.append(lines[i])
                    i += 1
                
                subtitle_count += 1
                line_counts.append(current_subtitle_lines)
                
                if current_subtitle_lines > 2:
                    subtitle_violations.append(
                        f"Subtitle {subtitle_count}: {current_subtitle_lines} lines"
                    )
                    # Show the problematic subtitle
                    for j, text_line in enumerate(subtitle_text_lines[:5]):  # Show first 5 lines
                        subtitle_violations.append(f"  Line {j+1}: '{text_line}'")
                
                i += 1
            
            if subtitle_violations:
                print(f"❌ Subtitles exceeding 2 line limit:")
                for violation in subtitle_violations[:20]:  # Show first 20
                    print(f"   {violation}")
                return False
            
            print(f"✅ All subtitles within 2 line limit")
            print(f"   Checked {subtitle_count} subtitles")
            if line_counts:
                print(f"   Max lines per subtitle: {max(line_counts)}")
                print(f"   Average lines per subtitle: {sum(line_counts)/len(line_counts):.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            return False
    
    def test_full_pipeline_integration(self) -> bool:
        """Integration test: Full pipeline from input to VTT output"""
        print("\n🧪 INTEGRATION TEST: Full Pipeline")
        print("-" * 50)
        
        try:
            # Create test audio (if possible)
            audio_path = self.create_test_audio(5.0)  # 5 second test audio
            
            if audio_path:
                print(f"📁 Created test audio: {audio_path}")
                
                # Test audio extraction (should be no-op for .wav file)
                extracted_path, duration = AudioProcessor.extract_audio_optimized(audio_path)
                
                if not os.path.exists(extracted_path):
                    print("❌ Audio extraction failed")
                    return False
                
                print(f"🎵 Audio extracted: {duration:.2f}s")
            else:
                print("⚠️ Skipping audio test - no FFmpeg available")
            
            # Test with mock data for transcription pipeline
            test_text, words = self.create_mock_transcription_data()
            
            # Full processing pipeline
            segments = SubtitleFormatter.create_optimized_segments(words, max_chars=84)
            vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
            srt_content = SubtitleFormatter.create_srt(segments, max_chars_per_line=42)
            
            # Save outputs to output_vtts directory
            output_dir = Path(__file__).parent / "output_vtts"
            output_dir.mkdir(exist_ok=True)
            
            vtt_path = output_dir / "integration_test.vtt"
            srt_path = output_dir / "integration_test.srt"
            txt_path = output_dir / "integration_test.txt"
            
            with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                f.write(vtt_content)
            
            with open(srt_path, 'w', encoding='utf-8-sig') as f:
                f.write(srt_content)
            
            with open(txt_path, 'w', encoding='utf-8-sig') as f:
                f.write(test_text)
            
            # Verify all files exist
            for file_path, name in [(vtt_path, "VTT"), (srt_path, "SRT"), (txt_path, "TXT")]:
                if not os.path.exists(file_path):
                    print(f"❌ {name} file not created")
                    return False
                
                size = os.path.getsize(file_path)
                if size == 0:
                    print(f"❌ {name} file is empty")
                    return False
                
                print(f"✅ {name} file created: {size} bytes")
            
            print(f"📁 Integration test files in: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print("🚀 CAPTIONER COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        tests = [
            ("VTT File Generation", self.test_vtt_file_generation),
            ("Word Preservation", self.test_word_preservation),
            ("42 Character Line Limit", self.test_42_character_limit),
            ("2 Line Subtitle Limit", self.test_2_line_limit),
            ("Full Pipeline Integration", self.test_full_pipeline_integration)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    self.test_results.append(f"✅ {test_name}: PASSED")
                else:
                    self.test_results.append(f"❌ {test_name}: FAILED")
            except Exception as e:
                results[test_name] = False
                self.test_results.append(f"💥 {test_name}: ERROR - {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        for result_line in self.test_results:
            print(result_line)
        
        print(f"\n🎯 FINAL SCORE: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Captioner is working correctly.")
        else:
            print("⚠️  Some tests failed. Please review the issues above.")
        
        return results
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Could not cleanup {temp_dir}: {e}")


def main():
    """Run the comprehensive test suite"""
    print("🔧 Initializing Captioner Test Suite...")
    
    test_suite = CaptionerTestSuite()
    
    try:
        # Run all tests
        results = test_suite.run_all_tests()
        
        # Check if all required tests passed
        required_tests = [
            "VTT File Generation",
            "Word Preservation", 
            "42 Character Line Limit",
            "2 Line Subtitle Limit"
        ]
        
        all_required_passed = all(results.get(test, False) for test in required_tests)
        
        print(f"\n{'='*60}")
        print("🎯 REQUIREMENT VALIDATION")
        print(f"{'='*60}")
        
        for test in required_tests:
            status = "✅ PASS" if results.get(test, False) else "❌ FAIL"
            print(f"   {test}: {status}")
        
        if all_required_passed:
            print(f"\n🎉 SUCCESS: All 4 core requirements are met!")
            print("   ✅ VTT files are generated and saved")
            print("   ✅ All words are preserved (no missing words)")
            print("   ✅ All lines have maximum 42 characters")
            print("   ✅ All subtitles have maximum 2 lines")
            return_code = 0
        else:
            print(f"\n❌ FAILURE: Some core requirements are not met!")
            print("   Please fix the issues and run tests again.")
            return_code = 1
        
    except Exception as e:
        print(f"💥 Test suite crashed: {e}")
        return_code = 2
    
    finally:
        # Cleanup
        test_suite.cleanup()
    
    return return_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
