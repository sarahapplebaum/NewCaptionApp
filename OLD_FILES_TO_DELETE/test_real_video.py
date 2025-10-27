#!/usr/bin/env python3
"""
Test the captioner script with a real MP4 video file
Process through the full pipeline and validate all 4 requirements
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import time

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from captioner import (
    SubtitleFormatter, 
    FasterWhisperModelManager, 
    TranscriptionProcessor,
    AudioProcessor,
    ProcessingStats
)

class RealVideoTest:
    """Test captioner with real video file"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.output_dir = Path(__file__).parent / "output_vtts"
        self.output_dir.mkdir(exist_ok=True)
        self.test_results = []
        
    def process_video(self) -> Tuple[str, list, ProcessingStats]:
        """Process the video through the full captioner pipeline"""
        print(f"🎬 Processing video: {os.path.basename(self.video_path)}")
        
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Initialize components
        model_manager = FasterWhisperModelManager()
        processor = TranscriptionProcessor()
        stats = ProcessingStats()
        stats.start_time = time.time()
        
        try:
            # Step 1: Extract audio
            print("🎵 Extracting audio...")
            audio_path, duration = processor.extract_audio_optimized(self.video_path, stats)
            stats.audio_duration = duration
            print(f"   ✅ Audio extracted: {duration:.2f}s")
            
            # Step 2: Load model and transcribe
            print("🤖 Loading AI model...")
            model_id = "small"  # Use faster model for testing
            
            with model_manager.model_context(model_id) as model:
                if model is None:
                    raise RuntimeError("Failed to load model")
                
                print("🎙️ Transcribing...")
                text, segments = processor.transcribe_optimized(
                    audio_path, model, True, 84, stats
                )
                
                stats.characters_transcribed = len(text or "")
                stats.end_time = time.time()
                
                print(f"   ✅ Transcription complete: {len(text or '')} chars, {len(segments)} segments")
                
                return text, segments, stats
                
        finally:
            # Cleanup
            processor.cleanup_temp_files()
    
    def save_outputs(self, text: str, segments: list, filename_base: str) -> Dict[str, Path]:
        """Save all output formats"""
        paths = {}
        
        # Save TXT
        txt_path = self.output_dir / f"{filename_base}.txt"
        with open(txt_path, 'w', encoding='utf-8-sig') as f:
            f.write(text)
        paths['txt'] = txt_path
        
        # Save VTT
        vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line=42)
        vtt_path = self.output_dir / f"{filename_base}.vtt"
        with open(vtt_path, 'w', encoding='utf-8-sig') as f:
            f.write(vtt_content)
        paths['vtt'] = vtt_path
        
        # Save SRT
        srt_content = SubtitleFormatter.create_srt(segments, max_chars_per_line=42)
        srt_path = self.output_dir / f"{filename_base}.srt"
        with open(srt_path, 'w', encoding='utf-8-sig') as f:
            f.write(srt_content)
        paths['srt'] = srt_path
        
        return paths
    
    def test_vtt_file_creation(self, vtt_path: Path) -> bool:
        """Test 1: VTT file was created and saved"""
        print("\n🧪 TEST 1: VTT File Creation and Saving")
        print("-" * 50)
        
        try:
            if not vtt_path.exists():
                print("❌ VTT file was not created")
                return False
            
            file_size = vtt_path.stat().st_size
            if file_size == 0:
                print("❌ VTT file is empty")
                return False
            
            # Check if it's valid VTT
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            if not content.startswith("WEBVTT"):
                print("❌ Invalid VTT format")
                return False
            
            print(f"✅ VTT file successfully created: {vtt_path}")
            print(f"   File size: {file_size} bytes")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def test_word_preservation(self, txt_path: Path, vtt_path: Path) -> bool:
        """Test 2: All words preserved in VTT"""
        print("\n🧪 TEST 2: Word Preservation")
        print("-" * 50)
        
        try:
            # Read original text
            with open(txt_path, 'r', encoding='utf-8-sig') as f:
                original_text = f.read().strip()
            
            # Read VTT and extract text
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                vtt_content = f.read()
            
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
            
            # Compare word counts
            original_words = original_text.split()
            vtt_words = vtt_text.split()
            
            original_words_clean = [w.lower().strip('.,!?";:\'"') for w in original_words if w.strip()]
            vtt_words_clean = [w.lower().strip('.,!?";:\'"') for w in vtt_words if w.strip()]
            
            print(f"📝 Original text: {len(original_words)} words")
            print(f"📦 VTT text: {len(vtt_words)} words")
            
            # Check for missing words
            original_set = set(original_words_clean)
            vtt_set = set(vtt_words_clean)
            missing_words = original_set - vtt_set
            
            if missing_words:
                print(f"❌ Missing words detected: {list(missing_words)[:10]}...")
                print(f"   Missing count: {len(missing_words)}")
                return False
            
            # Check word counts
            word_count_errors = []
            original_counts = {}
            vtt_counts = {}
            
            for word in original_words_clean:
                original_counts[word] = original_counts.get(word, 0) + 1
            
            for word in vtt_words_clean:
                vtt_counts[word] = vtt_counts.get(word, 0) + 1
            
            for word, orig_count in original_counts.items():
                vtt_count = vtt_counts.get(word, 0)
                if vtt_count < orig_count:
                    word_count_errors.append(f"'{word}': {orig_count} → {vtt_count}")
            
            if word_count_errors:
                print(f"❌ Word count mismatches:")
                for error in word_count_errors[:10]:
                    print(f"   {error}")
                return False
            
            print(f"✅ All words preserved successfully")
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def test_42_character_limit(self, vtt_path: Path) -> bool:
        """Test 3: 42 character line limit"""
        print("\n🧪 TEST 3: 42 Character Line Limit")
        print("-" * 50)
        
        try:
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            lines = content.split('\n')
            violations = []
            subtitle_lines = []
            
            for i, line in enumerate(lines, 1):
                # Skip headers, timestamps, empty lines
                if (not line.strip() or 
                    line.strip().startswith('WEBVTT') or
                    '-->' in line or
                    line.strip().startswith('NOTE')):
                    continue
                
                subtitle_lines.append((i, line))
                
                if len(line) > 42:
                    violations.append(f"Line {i}: '{line}' ({len(line)} chars)")
            
            if violations:
                print(f"❌ Lines exceeding 42 characters:")
                for violation in violations[:10]:
                    print(f"   {violation}")
                return False
            
            max_length = max((len(line) for _, line in subtitle_lines), default=0)
            print(f"✅ All lines within 42 character limit")
            print(f"   Checked {len(subtitle_lines)} lines, max length: {max_length}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def test_2_line_limit(self, vtt_path: Path) -> bool:
        """Test 4: 2 line subtitle limit"""
        print("\n🧪 TEST 4: 2 Line Subtitle Limit")
        print("-" * 50)
        
        try:
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            lines = content.split('\n')
            violations = []
            subtitle_count = 0
            line_counts = []
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip until timestamp
                if '-->' not in line:
                    i += 1
                    continue
                
                # Count subtitle lines
                i += 1
                current_lines = 0
                subtitle_text = []
                
                while i < len(lines) and lines[i].strip():
                    current_lines += 1
                    subtitle_text.append(lines[i])
                    i += 1
                
                subtitle_count += 1
                line_counts.append(current_lines)
                
                if current_lines > 2:
                    violations.append(f"Subtitle {subtitle_count}: {current_lines} lines")
                    for j, text_line in enumerate(subtitle_text[:3]):
                        violations.append(f"  Line {j+1}: '{text_line}'")
                
                i += 1
            
            if violations:
                print(f"❌ Subtitles exceeding 2 line limit:")
                for violation in violations[:20]:
                    print(f"   {violation}")
                return False
            
            max_lines = max(line_counts) if line_counts else 0
            avg_lines = sum(line_counts) / len(line_counts) if line_counts else 0
            
            print(f"✅ All subtitles within 2 line limit")
            print(f"   Checked {subtitle_count} subtitles")
            print(f"   Max lines: {max_lines}, Average: {avg_lines:.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def test_word_order(self, txt_path: Path, vtt_path: Path) -> bool:
        """Test 5: Word order preservation"""
        print("\n🧪 TEST 5: Word Order Preservation")
        print("-" * 50)
        
        try:
            # Read original text
            with open(txt_path, 'r', encoding='utf-8-sig') as f:
                original_text = f.read().strip()
            
            # Read VTT and extract text in order
            with open(vtt_path, 'r', encoding='utf-8-sig') as f:
                vtt_content = f.read()
            
            # Extract text from VTT segments in chronological order
            vtt_segments = []
            lines = vtt_content.split('\n')
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip until timestamp
                if '-->' not in line:
                    i += 1
                    continue
                
                # Extract timestamp for ordering
                timestamp_line = line
                start_time = timestamp_line.split(' --> ')[0]
                
                # Collect subtitle text lines
                i += 1
                subtitle_text = []
                while i < len(lines) and lines[i].strip():
                    subtitle_text.append(lines[i].strip())
                    i += 1
                
                if subtitle_text:
                    vtt_segments.append({
                        'timestamp': start_time,
                        'text': ' '.join(subtitle_text)
                    })
                
                i += 1
            
            # Sort by timestamp to ensure chronological order
            vtt_segments.sort(key=lambda x: x['timestamp'])
            
            # Extract sequential words from both sources
            original_words = original_text.split()
            vtt_words = []
            
            for segment in vtt_segments:
                vtt_words.extend(segment['text'].split())
            
            # Clean words for comparison (remove punctuation, normalize case)
            def clean_word(word):
                return word.lower().strip('.,!?";:\'"()[]{}')
            
            original_clean = [clean_word(w) for w in original_words if clean_word(w)]
            vtt_clean = [clean_word(w) for w in vtt_words if clean_word(w)]
            
            print(f"📝 Original sequence: {len(original_clean)} words")
            print(f"📦 VTT sequence: {len(vtt_clean)} words")
            
            # Check if we have reasonable word counts to compare
            if len(vtt_clean) == 0:
                print("❌ No words found in VTT")
                return False
            
            # Find the longest common subsequence to identify order issues
            order_errors = []
            vtt_idx = 0
            
            for i, orig_word in enumerate(original_clean):
                # Find this word in the remaining VTT words
                found = False
                search_start = vtt_idx
                
                # Look ahead up to 10 words to handle minor reordering
                search_end = min(len(vtt_clean), vtt_idx + 10)
                
                for j in range(search_start, search_end):
                    if vtt_clean[j] == orig_word:
                        # Found the word, update position
                        if j > vtt_idx:
                            # Words were skipped - they might be out of order
                            skipped_words = vtt_clean[vtt_idx:j]
                            if len(skipped_words) > 0:
                                order_errors.append(f"Skipped words before '{orig_word}': {skipped_words}")
                        
                        vtt_idx = j + 1
                        found = True
                        break
                
                if not found:
                    # Word not found in expected position - major order issue
                    if vtt_idx < len(vtt_clean):
                        expected_next = vtt_clean[vtt_idx:vtt_idx+3]
                        order_errors.append(f"Expected '{orig_word}' but found: {expected_next}")
                    else:
                        order_errors.append(f"Expected '{orig_word}' but reached end of VTT")
                    
                    # Skip this word and continue
                    continue
            
            # Additional check: look for major word sequence disruptions
            if len(original_clean) > 10 and len(vtt_clean) > 10:
                # Define common words that can appear anywhere and should be ignored
                common_words = {'a', 'an', 'the', 'to', 'of', 'and', 'in', 'on', 'at', 'for',
                               'this', 'that', 'is', 'are', 'was', 'were', 'been', 'have', 'has',
                               'will', 'be', 'it', 'you', 'we', 'they', 'all', 'more', 'some',
                               'from', 'with', 'by', 'as', 'or', 'but', 'not', 'can', 'here',
                               'there', 'where', 'when', 'what', 'which', 'who', 'how', 'why',
                               'well', 'just', 'like', 'so', 'very', 'much', 'many', 'such',
                               'only', 'also', 'then', 'now', 'than', 'these', 'those'}
                
                # Sample every 10th word and verify rough order (skip common words)
                sample_size = min(10, len(original_clean) // 10)
                sample_interval = len(original_clean) // sample_size if sample_size > 0 else len(original_clean)
                
                sampled_count = 0
                for i in range(0, len(original_clean), sample_interval):
                    if i >= len(original_clean) or sampled_count >= sample_size:
                        break
                    
                    sample_word = original_clean[i]
                    
                    # Skip common words that can appear anywhere
                    if sample_word in common_words:
                        continue
                    
                    sampled_count += 1
                    
                    # Find this word in VTT
                    try:
                        vtt_position = vtt_clean.index(sample_word)
                        expected_position_ratio = i / len(original_clean)
                        actual_position_ratio = vtt_position / len(vtt_clean)
                        
                        # Allow for some deviation but flag major order issues
                        deviation = abs(expected_position_ratio - actual_position_ratio)
                        if deviation > 0.4:  # 40% deviation threshold (more lenient)
                            order_errors.append(f"Word '{sample_word}' appears at {actual_position_ratio:.1%} but expected around {expected_position_ratio:.1%}")
                    
                    except ValueError:
                        # Word not found - this was already caught above
                        pass
            
            # Evaluate results
            if len(order_errors) > 5:  # Allow a few minor order issues
                print(f"❌ Significant word order issues detected:")
                for error in order_errors[:10]:  # Show first 10 errors
                    print(f"   {error}")
                print(f"   Total order issues: {len(order_errors)}")
                return False
            elif len(order_errors) > 0:
                print(f"⚠️  Minor word order issues detected ({len(order_errors)} issues):")
                for error in order_errors[:5]:
                    print(f"   {error}")
                print(f"✅ Word order generally preserved (within tolerance)")
                return True
            else:
                print(f"✅ Perfect word order preservation")
                print(f"   All {len(original_clean)} words appear in correct sequence")
                return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    def run_full_test(self) -> bool:
        """Run the complete test pipeline"""
        print("🚀 REAL VIDEO CAPTIONER TEST")
        print("=" * 60)
        print(f"📁 Video: {self.video_path}")
        print(f"💾 Output: {self.output_dir}")
        print()
        
        try:
            # Process video
            text, segments, stats = self.process_video()
            
            # Save outputs
            filename_base = Path(self.video_path).stem
            paths = self.save_outputs(text, segments, filename_base)
            
            print(f"\n📊 Processing Stats:")
            print(f"   Total time: {stats.total_time:.2f}s")
            print(f"   Audio duration: {stats.audio_duration:.2f}s")
            print(f"   Characters: {stats.characters_transcribed}")
            if stats.audio_duration > 0:
                speed_factor = stats.audio_duration / stats.total_time
                print(f"   Speed: {speed_factor:.1f}x realtime")
            
            # Run all tests
            tests = [
                ("VTT File Creation", lambda: self.test_vtt_file_creation(paths['vtt'])),
                ("Word Preservation", lambda: self.test_word_preservation(paths['txt'], paths['vtt'])),
                ("42 Character Limit", lambda: self.test_42_character_limit(paths['vtt'])),
                ("2 Line Limit", lambda: self.test_2_line_limit(paths['vtt'])),
                ("Word Order Preservation", lambda: self.test_word_order(paths['txt'], paths['vtt']))
            ]
            
            results = {}
            passed = 0
            
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
            
            for result in self.test_results:
                print(result)
            
            print(f"\n🎯 FINAL SCORE: {passed}/{len(tests)} tests passed ({passed/len(tests)*100:.1f}%)")
            
            success = passed == len(tests)
            if success:
                print("🎉 ALL TESTS PASSED! Real video processing successful.")
                print(f"\n📁 Generated files:")
                for format_type, path in paths.items():
                    size = path.stat().st_size
                    print(f"   {format_type.upper()}: {path} ({size} bytes)")
            else:
                print("⚠️  Some tests failed. Issues need to be addressed.")
            
            return success
            
        except Exception as e:
            print(f"💥 Test failed with error: {e}")
            return False


def main():
    """Main test function"""
    video_path = "/Users/sarah.applebaum/Documents/FinalRenders/HDRP Lighting Fundamentals/Camtasia/mp4/M1-11_Screen-Space-Reflection.mp4"
    
    print("🔧 Initializing Real Video Test...")
    
    test = RealVideoTest(video_path)
    success = test.run_full_test()
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
