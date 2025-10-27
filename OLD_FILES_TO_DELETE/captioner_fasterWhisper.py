# captioner_optimized_fixed.py
# Ultra-High Performance Video Captioner with Enhanced Error Handling
import sys
import os
import re
import subprocess
import tempfile
import shutil
import time
import threading
import queue
import gc
from datetime import timedelta
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                           QFileDialog, QProgressBar, QComboBox, QCheckBox,
                           QSpinBox, QGroupBox, QGridLayout, QListWidget,
                           QSplitter, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QFont
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa
import numpy as np
import re
from typing import List, Dict, Tuple, Set, Optional
from functools import lru_cache


# Configure logging for performance monitoring
logging.basicConfig(level=logging.DEBUG)  # <- Changed this line
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Track processing statistics for optimization"""
    start_time: float = 0
    end_time: float = 0
    audio_extraction_time: float = 0
    transcription_time: float = 0
    processing_time: float = 0
    file_size: int = 0
    audio_duration: float = 0
    characters_transcribed: int = 0
    
    @property
    def total_time(self) -> float:
        return max(0, self.end_time - self.start_time) if self.end_time and self.start_time else 0
    
    @property
    def chars_per_second(self) -> float:
        total = self.total_time
        return self.characters_transcribed / total if total > 0 else 0

def safe_float(value: Union[float, int, None], default: float = 0.0) -> float:
    """Safely convert value to float with default"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_timestamp_operation(a: Union[float, None], b: Union[float, None], operation: str = 'subtract') -> float:
    """Safely perform operations on timestamps that might be None"""
    a_val = safe_float(a, 0.0)
    b_val = safe_float(b, 0.0)
    
    if operation == 'subtract':
        return max(0, a_val - b_val)
    elif operation == 'add':
        return a_val + b_val
    else:
        return a_val

class OptimizedModelManager:
    """Model manager using faster-whisper for reliable word timestamps"""
    _instance = None
    _model = None
    _current_model_id = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_optimal_device_config():
        """Determine optimal device configuration for faster-whisper"""
        config = {}
        
        if torch.cuda.is_available():
            config['device'] = "cuda"
            config['compute_type'] = "float16"
            logger.info(f"🚀 CUDA GPU: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            # faster-whisper doesn't support MPS directly, use CPU with optimization
            config['device'] = "cpu"
            config['compute_type'] = "int8"  # Optimized for Apple Silicon
            logger.info("🍎 Apple Silicon - using optimized CPU mode")
        else:
            config['device'] = "cpu"
            config['compute_type'] = "int8"
            logger.info(f"🖥️  CPU ({os.cpu_count()} cores)")
        
        return config
    
    def load_model_optimized(self, model_id: str = "small") -> bool:
        """Load faster-whisper model"""
        try:
            if self._current_model_id == model_id and self._model is not None:
                logger.info(f"✅ Model already loaded: {model_id}")
                return True
            
            self.clear_model_aggressive()
            
            from faster_whisper import WhisperModel
            
            config = self.get_optimal_device_config()
            self._device = config['device']
            
            # Map full model names to faster-whisper names
            model_map = {
                'openai/whisper-tiny': 'tiny',
                'openai/whisper-base': 'base',
                'openai/whisper-small': 'small',
                'openai/whisper-medium': 'medium',
                'openai/whisper-large-v3': 'large-v3',
                'openai/whisper-large-v2': 'large-v2'
            }
            
            model_name = model_map.get(model_id, model_id)
            
            logger.info(f"🔄 Loading faster-whisper model: {model_name}")
            start_time = time.time()
            
            self._model = WhisperModel(
                model_name,
                device=config['device'],
                compute_type=config['compute_type'],
                cpu_threads=min(8, os.cpu_count()),
                num_workers=1
            )
            
            self._current_model_id = model_id
            load_time = time.time() - start_time
            
            logger.info(f"✅ Model loaded in {load_time:.2f}s on {self._device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.clear_model_aggressive()
            return False
    
    def clear_model_aggressive(self):
        """Cleanup model"""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._current_model_id = None
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model(self):
        return self._model
    
    @contextmanager
    def model_context(self, model_id: str):
        """Context manager for automatic model cleanup"""
        try:
            success = self.load_model_optimized(model_id)
            if not success:
                raise RuntimeError(f"Failed to load model: {model_id}")
            yield self._model
        finally:
            pass  # Keep model loaded for batch processing

class HighPerformanceAudioProcessor:
    """Optimized audio processing with caching and parallel processing"""

    progress_updated = pyqtSignal(int)
    file_progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    current_file_updated = pyqtSignal(str)
    file_completed = pyqtSignal(str, str, list, object)
    batch_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    performance_stats = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.model_manager = OptimizedModelManager()
        self.should_stop = False
        self.stats = {}
        self.temp_files = []
    
    _ffmpeg_cache = {}
    _audio_cache = {}
    
    @staticmethod
    def find_ffmpeg_optimized():
        """Cached FFmpeg detection with performance optimization"""
        if 'ffmpeg_path' in HighPerformanceAudioProcessor._ffmpeg_cache:
            cached_path = HighPerformanceAudioProcessor._ffmpeg_cache['ffmpeg_path']
            if cached_path and os.path.exists(cached_path):
                return cached_path
        
        logger.info("🔍 Optimized FFmpeg detection...")
        
        search_paths = []
        
        if hasattr(sys, '_MEIPASS'):
            search_paths.extend([
                Path(sys._MEIPASS) / "ffmpeg",
                Path(sys._MEIPASS) / "ffmpeg.exe",
            ])
        
        elif sys.executable.endswith('VideoCaption'):
            exe_path = Path(sys.executable)
            app_contents = exe_path.parent.parent
            search_paths.extend([
                app_contents / "Resources" / "ffmpeg",
                app_contents / "Frameworks" / "ffmpeg",
                exe_path.parent / "ffmpeg",
            ])
        
        search_paths.extend([
            Path("/opt/homebrew/bin/ffmpeg"),
            Path("/usr/local/bin/ffmpeg"),
            Path("/usr/bin/ffmpeg"),
        ])
        
        def test_ffmpeg_path(path):
            if not path or not path.exists():
                return None
            
            try:
                os.chmod(path, 0o755)
                result = subprocess.run(
                    [str(path), '-version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=5,
                    check=False
                )
                if result.returncode == 0 and 'ffmpeg version' in result.stdout.lower():
                    return str(path)
            except (subprocess.TimeoutExpired, Exception):
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {executor.submit(test_ffmpeg_path, path): path for path in search_paths}
            
            for future in as_completed(future_to_path, timeout=10):
                try:
                    result = future.result()
                    if result:
                        HighPerformanceAudioProcessor._ffmpeg_cache['ffmpeg_path'] = result
                        logger.info(f"✅ FFmpeg found: {result}")
                        return result
                except Exception:
                    continue
        
        system_ffmpeg = shutil.which('ffmpeg')
        if system_ffmpeg:
            HighPerformanceAudioProcessor._ffmpeg_cache['ffmpeg_path'] = system_ffmpeg
            logger.info(f"✅ FFmpeg in PATH: {system_ffmpeg}")
            return system_ffmpeg
        
        logger.error("❌ FFmpeg not found")
        return None
    
    @staticmethod
    def extract_audio_optimized(video_path: str, output_path: str = None) -> Tuple[str, float]:
        """High-performance audio extraction with caching and optimization"""
        try:
            cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
            if cache_key in HighPerformanceAudioProcessor._audio_cache:
                cached_path, duration = HighPerformanceAudioProcessor._audio_cache[cache_key]
                if os.path.exists(cached_path):
                    logger.info(f"🗄️  Using cached audio: {cached_path}")
                    return cached_path, duration
        except Exception as cache_error:
            logger.warning(f"Cache check failed: {cache_error}")
        
        ffmpeg_path = HighPerformanceAudioProcessor.find_ffmpeg_optimized()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found")
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        start_time = time.time()
        
        try:
            cmd = [
                ffmpeg_path,
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz (Whisper's native rate)
                '-ac', '1',  # Mono
                '-y',  # Overwrite
                '-loglevel', 'error',
                '-threads', str(min(4, os.cpu_count())),
                output_path
            ]
            
            logger.info(f"🎵 Extracting audio: {os.path.basename(video_path)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {result.stderr}")
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError("No audio output produced")
            
            duration = HighPerformanceAudioProcessor.get_audio_duration(output_path)
            extraction_time = time.time() - start_time
            
            # Cache the result
            try:
                cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
                HighPerformanceAudioProcessor._audio_cache[cache_key] = (output_path, duration)
            except Exception:
                pass  # Cache failure is not critical
            
            logger.info(f"✅ Audio extracted in {extraction_time:.2f}s (duration: {duration:.2f}s)")
            return output_path, duration
            
        except Exception as e:
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except Exception:
                    pass
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get audio duration efficiently with error handling"""
        try:
            duration = librosa.get_duration(path=audio_path)
            return safe_float(duration, 0.0)
        except Exception:
            ffmpeg_path = HighPerformanceAudioProcessor.find_ffmpeg_optimized()
            if ffmpeg_path:
                try:
                    cmd = [ffmpeg_path, '-i', audio_path, '-f', 'null', '-']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    duration_match = re.search(r'Duration: (\d+):(\d+):(\d+)\.(\d+)', result.stderr)
                    if duration_match:
                        h, m, s, ms = map(int, duration_match.groups())
                        return safe_float(h * 3600 + m * 60 + s + ms / 100)
                except Exception:
                    pass
        return 0.0

@dataclass
class WordInfo:
    """Cleaned word information with cached properties"""
    word: str
    start: float
    end: float
    _clean_word: Optional[str] = None
    
    @property
    def clean_word(self) -> str:
        """Cached cleaned word without punctuation"""
        if self._clean_word is None:
            self._clean_word = self.word.lower().strip('.,!?";:')
        return self._clean_word
    
    def to_dict(self) -> Dict:
        return {"word": self.word, "start": self.start, "end": self.end}

class OptimizedSubtitleFormatter:
    """Refactored high-performance subtitle formatter with single-pass processing"""
    
    # Compile regex patterns once at class level
    SENTENCE_SPLIT_PATTERN = re.compile(r'[.!?]+')
    NGRAM_PATTERN_CACHE = {}
    
    # Define constants at class level
    SENTENCE_STARTERS = frozenset([
        'and', 'but', 'so', 'yet', 'or', 'nor', 'for',
        'however', 'therefore', 'moreover', 'furthermore', 
        'nevertheless', 'additionally', 'meanwhile', 'consequently'
    ])
    
    COMMON_ABBREVS = frozenset([
        'mr.', 'mrs.', 'dr.', 'prof.', 'inc.', 'ltd.', 
        'co.', 'corp.', 'etc.', 'vs.', 'jr.', 'sr.'
    ])
    
    # Timing constants
    MIN_DURATION = 1.25
    MAX_DURATION = 8.0
    WORDS_PER_SECOND = 3.33
    
    @classmethod
    @lru_cache(maxsize=128)
    def _get_ngram_pattern(cls, ngram: str) -> re.Pattern:
        """Cache compiled regex patterns for n-grams"""
        escaped = re.escape(ngram)
        return re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
    
    @staticmethod
    def format_text_optimized(text: str, max_chars_per_line: int = 40, max_lines: int = 2) -> Tuple[str, List[str]]:
        """Optimized text formatting with single-pass processing"""
        if not text or not text.strip():
            return "", []
        
        words = text.split()
        if not words:
            return "", []
        
        lines = []
        current_line = ""
        used_words = 0
        
        for i, word in enumerate(words[:1000]):  # Safety limit
            test_line = f"{current_line} {word}" if current_line else word
            
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
                used_words = i + 1
            else:
                if current_line:
                    lines.append(current_line)
                    if len(lines) >= max_lines:
                        return "\n".join(lines), words[used_words:]
                
                current_line = word if len(word) <= max_chars_per_line else f"{word[:max_chars_per_line-3]}..."
                used_words = i + 1
                
                if len(current_line) > max_chars_per_line:
                    lines.append(current_line)
                    if len(lines) >= max_lines:
                        return "\n".join(lines), words[used_words:]
                    current_line = ""
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines), words[used_words:]
    
    @staticmethod
    def calculate_reading_time(text: str) -> float:
        """Calculate reading time based on word count and complexity"""
        if not text:
            return OptimizedSubtitleFormatter.MIN_DURATION
        
        words = len(text.split())
        base_time = words / OptimizedSubtitleFormatter.WORDS_PER_SECOND
        
        # Add time for punctuation pauses
        punctuation_time = sum(0.2 for char in text if char in '.,!?;:')
        
        # Calculate by character count as alternative
        char_time = len(text) * 0.05
        
        total_time = max(base_time + punctuation_time, char_time)
        
        return max(
            OptimizedSubtitleFormatter.MIN_DURATION,
            min(total_time, OptimizedSubtitleFormatter.MAX_DURATION)
        )
    
    @classmethod
    def clean_and_process_words(cls, words: List[Dict]) -> List[WordInfo]:
        """Single-pass word cleaning with all operations combined"""
        if not words:
            return []
        
        # First pass: convert to WordInfo and count frequencies
        word_infos = []
        word_frequencies = {}
        
        for word_dict in words:
            # FIX: Handle None values properly
            word = word_dict.get("word") or ""
            word = word.strip()
            
            if not word:
                continue
            
            word_info = WordInfo(
                word=word,
                start=safe_float(word_dict.get("start"), 0),
                end=safe_float(word_dict.get("end"), 0)
            )
            
            word_infos.append(word_info)
            clean = word_info.clean_word
            word_frequencies[clean] = word_frequencies.get(clean, 0) + 1
        
        if not word_infos:
            return []
        
        total_words = len(word_infos)
        
        # Second pass: filter excessive repetition and clean punctuation
        cleaned = []
        word_counts = {}
        
        for i, word_info in enumerate(word_infos):
            clean = word_info.clean_word
            frequency = word_frequencies.get(clean, 0)
            
            # Skip only truly excessive repetition (>50% of total)
            if frequency > total_words * 0.5:
                current_count = word_counts.get(clean, 0)
                if current_count >= 3:
                    continue
                word_counts[clean] = current_count + 1
            
            # Clean punctuation in single operation
            word = word_info.word
            if i < len(word_infos) - 1 and word.endswith('.'):
                next_info = word_infos[i + 1]
                next_word = next_info.word.lstrip('.,!?";:\'" ')
                
                if next_word:
                    first_char = next_word[0]
                    next_lower = next_word.lower()
                    
                    # Remove period if mid-sentence (lowercase, not sentence starter)
                    if first_char.islower() and next_lower not in cls.SENTENCE_STARTERS:
                        word = word[:-1]
                        word_info.word = word
            
            cleaned.append(word_info)
        
        return cleaned
    
    @staticmethod
    def is_sentence_boundary(word: str) -> bool:
        """Fast sentence boundary detection"""
        if not word:
            return False
        
        cleaned = word.rstrip('",\')"]}')
        
        if cleaned.endswith(('.', '!', '?', '...')):
            # Check if it's an abbreviation
            return cleaned.lower() not in OptimizedSubtitleFormatter.COMMON_ABBREVS
        
        return False
    
    @staticmethod
    def is_natural_pause(word: str) -> bool:
        """Detect natural pause points"""
        return word.rstrip().endswith((',', ';', ':', '--', '—'))
    
    @classmethod
    def create_optimized_segments(cls, words: List[Dict], max_chars: int = 80) -> List[Dict]:
        """Optimized segment creation with single-pass processing"""
        if not words:
            return []
        
        # Single-pass word cleaning
        cleaned_words = cls.clean_and_process_words(words)
        
        if not cleaned_words:
            return []
        
        segments = []
        current_segment = {
            "start": None,
            "end": None,
            "text": "",
            "word_count": 0
        }
        
        for i, word_info in enumerate(cleaned_words[:10000]):  # Safety limit
            word = word_info.word
            word_start = word_info.start
            word_end = word_info.end
            
            # Build potential text
            potential_text = f"{current_segment['text']} {word}" if current_segment['text'] else word
            
            # Decision factors
            would_exceed = len(potential_text) > max_chars
            is_sentence_end = cls.is_sentence_boundary(word)
            is_pause = cls.is_natural_pause(word)
            
            # Time gap check
            time_gap = False
            if current_segment["end"] is not None:
                gap = word_start - current_segment["end"]
                time_gap = gap > 2.0
            
            # Duration check
            current_duration = 0
            if current_segment["start"] is not None:
                current_duration = word_end - current_segment["start"]
            
            # FIX: Decide whether to break segment (more lenient time gap check)
            should_break = (
                (would_exceed and current_segment['text']) or
                (time_gap and current_segment['text']) or  # Fixed: removed length requirement
                (is_sentence_end and len(current_segment['text']) > 25) or
                (current_duration > 5.0 and (is_pause or len(current_segment['text']) > 50))
            )
            
            if should_break:
                if current_segment['text']:
                    segments.append({
                        "start": current_segment["start"],
                        "end": current_segment["end"],
                        "text": current_segment["text"].strip()
                    })
                
                # Start new segment
                current_segment = {
                    "start": word_start,
                    "end": word_end,
                    "text": word,
                    "word_count": 1
                }
            else:
                # Continue current segment
                if current_segment["start"] is None:
                    current_segment["start"] = word_start
                
                current_segment["text"] = potential_text
                current_segment["end"] = word_end
                current_segment["word_count"] += 1
        
        # Add final segment
        if current_segment['text']:
            segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "text": current_segment["text"].strip()
            })
        
        # Post-process segments
        return cls.post_process_segments(segments)
    
    @classmethod
    def post_process_segments(cls, segments: List[Dict]) -> List[Dict]:
        """Combined post-processing: deduplication, timing, punctuation"""
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: safe_float(x.get("start"), 0))
        
        processed = []
        seen_content = set()
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0)
            
            # Deduplicate using content hash
            content_hash = f"{text[:50]}_{start_time:.1f}"
            if content_hash in seen_content:
                continue
            
            seen_content.add(content_hash)
            
            # Optimize timing
            ideal_duration = cls.calculate_reading_time(text)
            proposed_end = start_time + ideal_duration
            
            # Check next segment for gap elimination
            if i + 1 < len(segments):
                next_start = safe_float(segments[i + 1].get("start"), proposed_end + 0.1)
                gap = next_start - proposed_end
                
                if gap > 0.1:
                    # Extend to next segment but leave small gap
                    extended_end = next_start - 0.05
                    if extended_end - start_time <= cls.MAX_DURATION:
                        proposed_end = extended_end
            
            # Ensure minimum duration
            if proposed_end - start_time < cls.MIN_DURATION:
                proposed_end = start_time + cls.MIN_DURATION
            
            # FIX: Ensure we don't overlap with previous segment
            if processed:
                previous_end = safe_float(processed[-1].get("end"), 0)
                if start_time < previous_end:
                    # Adjust start time to prevent overlap
                    start_time = previous_end + 0.05
                    # Recalculate end time
                    proposed_end = start_time + ideal_duration
                    if proposed_end - start_time < cls.MIN_DURATION:
                        proposed_end = start_time + cls.MIN_DURATION
            
            segment["start"] = start_time
            segment["end"] = proposed_end
            processed.append(segment)
        
        # Clean segment-end punctuation
        return cls.clean_segment_punctuation(processed)
    
    @classmethod
    def clean_segment_punctuation(cls, segments: List[Dict]) -> List[Dict]:
        """Remove inappropriate periods at segment boundaries"""
        if len(segments) <= 1:
            return segments
        
        for i in range(len(segments) - 1):
            current_text = segments[i].get("text", "").strip()
            next_text = segments[i + 1].get("text", "").strip()
            
            if not current_text or not next_text or not current_text.endswith('.'):
                continue
            
            # Get first word from next segment
            next_words = next_text.split()
            if not next_words:
                continue
            
            first_word = next_words[0].strip('.,!?";:\'"')
            
            # Remove period if mid-sentence (lowercase, not sentence starter)
            if first_word and first_word[0].islower() and first_word.lower() not in cls.SENTENCE_STARTERS:
                segments[i]["text"] = current_text[:-1].rstrip()
        
        return segments
    
    @classmethod
    def create_vtt_optimized(cls, segments: List[Dict], max_chars_per_line: int = 40) -> str:
        """Create optimized VTT format"""
        if not segments:
            return "WEBVTT\n\n"
        
        vtt_lines = ["WEBVTT", ""]
        
        for segment in sorted(segments, key=lambda x: safe_float(x.get("start"), 0)):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0.0)
            end_time = safe_float(segment.get("end"), start_time + cls.MIN_DURATION)
            
            # Ensure minimum duration
            if end_time - start_time < cls.MIN_DURATION:
                end_time = start_time + cls.MIN_DURATION
            
            # Format text
            formatted_text, _ = cls.format_text_optimized(text, max_chars_per_line)
            if not formatted_text:
                continue
            
            # Position based on line count
            line_count = formatted_text.count('\n') + 1
            position = {
                1: "align:middle line:90%",
                2: "align:middle line:84%"
            }.get(line_count, "align:middle line:80%")
            
            # Convert times
            start_str = cls.seconds_to_vtt_time(start_time)
            end_str = cls.seconds_to_vtt_time(end_time)
            
            vtt_lines.extend([
                f"{start_str} --> {end_str} {position}",
                formatted_text,
                ""
            ])
        
        return "\n".join(vtt_lines)
    
    @classmethod
    def create_srt_optimized(cls, segments: List[Dict], max_chars_per_line: int = 40) -> str:
        """Create optimized SRT format"""
        if not segments:
            return ""
        
        srt_lines = []
        counter = 1
        
        for segment in sorted(segments, key=lambda x: safe_float(x.get("start"), 0)):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0.0)
            end_time = safe_float(segment.get("end"), start_time + cls.MIN_DURATION)
            
            if end_time - start_time < cls.MIN_DURATION:
                end_time = start_time + cls.MIN_DURATION
            
            formatted_text, _ = cls.format_text_optimized(text, max_chars_per_line)
            if not formatted_text:
                continue
            
            start_str = cls.seconds_to_srt_time(start_time)
            end_str = cls.seconds_to_srt_time(end_time)
            
            srt_lines.extend([
                str(counter),
                f"{start_str} --> {end_str}",
                formatted_text,
                ""
            ])
            
            counter += 1
        
        return "\n".join(srt_lines)
    
    @staticmethod
    def seconds_to_vtt_time(seconds: float) -> str:
        """Convert seconds to VTT time format"""
        try:
            seconds = safe_float(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        except Exception:
            return "00:00:00.000"
    
    @staticmethod
    def seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT time format"""
        try:
            seconds = safe_float(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        except Exception:
            return "00:00:00,000"


class HighPerformanceBatchWorker(QObject):
    """Ultra-optimized batch worker with robust error handling and improved word reconciliation"""
    
    progress_updated = pyqtSignal(int)
    file_progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    current_file_updated = pyqtSignal(str)
    file_completed = pyqtSignal(str, str, list, object)
    batch_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    performance_stats = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.model_manager = OptimizedModelManager()
        self.should_stop = False
        self.stats = {}
        self.temp_files = []
        
    def stop_processing(self):
        self.should_stop = True
    
    def transcribe_batch_optimized(self, file_paths: List[str], model_id: str, 
                             return_timestamps: bool = True, max_chars_per_segment: int = 80):
        """Ultra-high performance batch transcription with faster-whisper"""
        results = {}
        successful = 0
        failed = 0
        total_start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting batch transcription with faster-whisper: {len(file_paths)} files")
            
            performance_stats = {
                'total_files': len(file_paths),
                'start_time': total_start_time,
                'files_processed': 0,
                'total_audio_duration': 0,
                'total_characters': 0,
                'average_speed': 0
            }
            
            self.setup_optimized_environment()
            
            self.status_updated.emit("Loading faster-whisper model...")
            
            # Load model using context manager
            with self.model_manager.model_context(model_id) as model:
                if model is None:
                    raise RuntimeError("Failed to load faster-whisper model")
                
                logger.info("✅ Faster-Whisper model loaded successfully")
                
                for i, file_path in enumerate(file_paths):
                    if self.should_stop:
                        break
                    
                    file_stats = ProcessingStats()
                    file_stats.start_time = time.time()
                    
                    try:
                        file_stats.file_size = os.path.getsize(file_path)
                    except Exception:
                        file_stats.file_size = 0
                    
                    filename = os.path.basename(file_path)
                    logger.info(f"📁 Processing {i+1}/{len(file_paths)}: {filename}")
                    
                    try:
                        self.current_file_updated.emit(f"Processing: {filename}")
                        self.file_progress_updated.emit(0)
                        
                        overall_progress = int(10 + (i / len(file_paths)) * 85)
                        self.progress_updated.emit(overall_progress)
                        
                        # Extract audio
                        audio_path, duration = self.extract_audio_optimized(file_path, file_stats)
                        file_stats.audio_duration = safe_float(duration, 0.0)
                        performance_stats['total_audio_duration'] += file_stats.audio_duration
                        
                        self.file_progress_updated.emit(30)
                        
                        # Transcribe with faster-whisper
                        text, segments = self.transcribe_optimized(
                            audio_path, model, return_timestamps, 
                            max_chars_per_segment, file_stats
                        )
                        
                        file_stats.characters_transcribed = len(text or "")
                        performance_stats['total_characters'] += file_stats.characters_transcribed
                        
                        self.file_progress_updated.emit(100)
                        
                        results[file_path] = {
                            'text': text,
                            'segments': segments,
                            'success': True,
                            'stats': file_stats
                        }
                        
                        file_stats.end_time = time.time()
                        successful += 1
                        performance_stats['files_processed'] += 1
                        
                        logger.info(f"✅ {filename}: {len(text or '')} chars, {len(segments)} segments")
                        
                        self.file_completed.emit(file_path, text or "", segments, file_stats)
                        self.cleanup_temp_files()
                        
                        # Update performance stats
                        current_time = time.time()
                        if performance_stats['total_audio_duration'] > 0:
                            performance_stats['average_speed'] = (
                                performance_stats['total_audio_duration'] / (current_time - total_start_time)
                            )
                        self.performance_stats.emit(performance_stats.copy())
                        
                    except Exception as e:
                        error_msg = f"Processing failed for {filename}: {str(e)}"
                        logger.error(f"❌ {error_msg}")
                        results[file_path] = {'error': error_msg, 'success': False}
                        failed += 1
                        self.error_occurred.emit(error_msg, file_path)
            
            total_time = time.time() - total_start_time
            performance_stats['end_time'] = time.time()
            performance_stats['total_time'] = total_time
            
            logger.info(f"🏁 Batch complete: {successful}/{len(file_paths)} successful in {total_time:.2f}s")
            
            summary = {
                'total': len(file_paths),
                'successful': successful,
                'failed': failed,
                'results': results,
                'performance': performance_stats
            }
            
            self.batch_finished.emit(summary)
            
        except Exception as e:
            error_msg = f"Critical batch error: {str(e)}"
            logger.error(f"💥 {error_msg}")
            self.error_occurred.emit(error_msg, "")
        finally:
            self.cleanup_temp_files()
    
    def setup_optimized_environment(self):
        """Setup optimized processing environment"""
        ffmpeg_path = HighPerformanceAudioProcessor.find_ffmpeg_optimized()
        if ffmpeg_path:
            env_vars = {
                'FFMPEG_BINARY': ffmpeg_path,
                'AUDIOREAD_FFMPEG_EXE': ffmpeg_path,
                'FFMPEG_EXECUTABLE': ffmpeg_path
            }
            for var, value in env_vars.items():
                os.environ[var] = value
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        torch.set_num_threads(min(8, os.cpu_count()))
    
    def extract_audio_optimized(self, video_path: str, file_stats: ProcessingStats) -> Tuple[str, float]:
        """Optimized audio extraction with stats tracking"""
        if self.is_audio_file(video_path):
            duration = HighPerformanceAudioProcessor.get_audio_duration(video_path)
            return video_path, duration
        
        extraction_start = time.time()
        audio_path, duration = HighPerformanceAudioProcessor.extract_audio_optimized(video_path)
        file_stats.audio_extraction_time = safe_timestamp_operation(time.time(), extraction_start, 'subtract')
        
        if audio_path != video_path:
            self.temp_files.append(audio_path)
        
        return audio_path, duration
    
    def transcribe_optimized(self, audio_path: str, model, return_timestamps: bool, 
                   max_chars: int, file_stats: ProcessingStats) -> Tuple[str, List[Dict]]:
        """
        High-performance transcription using faster-whisper - single pass for consistency
        """
        transcription_start = time.time()
        
        try:
            logger.info(f"🎤 Transcribing with faster-whisper (word timestamps: {return_timestamps})")
            
            # Single transcription pass
            segments_iter, info = model.transcribe(
                audio_path,
                word_timestamps=return_timestamps,
                language="en",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100
                )
            )
            
            all_segments = list(segments_iter)
            
            file_stats.transcription_time = safe_timestamp_operation(time.time(), transcription_start, 'subtract')
            
            # Extract word-level timestamps
            words = []
            if return_timestamps:
                for segment in all_segments:
                    if segment.words:
                        for word_info in segment.words:
                            words.append({
                                'word': word_info.word,
                                'start': word_info.start,
                                'end': word_info.end
                            })
            
            # Build complete text FROM word-level data
            complete_text = ' '.join([w.get('word', '').strip() for w in words]) if words else ""
            
            logger.info(f"✅ Faster-Whisper: {len(complete_text)} chars, {len(words)} words")
            
            if return_timestamps and words:
                # Merge compound words
                merged_words = self._merge_compound_words(words)
                
                if len(merged_words) < len(words):
                    logger.info(f"🔍 Merged {len(words) - len(merged_words)} compound words: {len(words)} → {len(merged_words)}")
                    complete_text = ' '.join([w.get('word', '').strip() for w in merged_words])
                
                # Create segments
                formatted_segments = OptimizedSubtitleFormatter.create_optimized_segments(merged_words, max_chars)
                
                # Auto-reconcile to ensure VTT matches TXT
                reconciled_segments = self._auto_reconcile_segments(complete_text, formatted_segments, max_chars_per_line=42)
                
                return complete_text, reconciled_segments
            else:
                formatted_segments = self.create_artificial_segments(complete_text, max_chars)
                return complete_text, formatted_segments
            
        except Exception as e:
            file_stats.transcription_time = safe_timestamp_operation(time.time(), transcription_start, 'subtract')
            raise RuntimeError(f"Faster-Whisper transcription failed: {str(e)}")



    def _reconcile_segments_with_complete_text(self, segments: List[Dict], complete_text: str) -> List[Dict]:
        """
        Compare segments with complete text and insert missing words into segment text.
        Segment timestamps remain unchanged - missing words appear during segment time.
        """
        complete_words = complete_text.split()
        
        # Build text from segments
        segment_text = ' '.join([seg.get('text', '') for seg in segments])
        segment_words = segment_text.split()
        
        logger.info(f"🔍 Reconciling segments: complete={len(complete_words)} words, segments={len(segment_words)} words")
        
        if len(complete_words) == len(segment_words):
            logger.info("✅ Word counts match, checking for differences...")
            
            # Check if words are actually the same
            complete_set = set(w.lower().strip('.,!?";:\'"') for w in complete_words)
            segment_set = set(w.lower().strip('.,!?";:\'"') for w in segment_words)
            
            missing = complete_set - segment_set
            if not missing:
                logger.info("✅ All words present in segments")
                return segments
            else:
                logger.warning(f"⚠️  Words in complete text but not in segments: {list(missing)[:10]}")
        
        # Find missing words and their context
        missing_words = self._find_missing_words_with_context(complete_words, segment_words)
        
        if not missing_words:
            logger.info("✅ No missing words found")
            return segments
        
        logger.warning(f"⚠️  Found {len(missing_words)} missing words to insert")
        
        # Insert missing words into appropriate segments
        reconciled = self._insert_missing_words_into_segments(segments, complete_text, missing_words)
        
        return reconciled

    def _find_missing_words_with_context(self, complete_words: List[str], segment_words: List[str]) -> List[Dict]:
        """
        Find missing words and their surrounding context for insertion.
        """
        missing = []
        
        complete_norm = [w.lower().strip('.,!?";:\'"') for w in complete_words]
        segment_norm = [w.lower().strip('.,!?";:\'"') for w in segment_words]
        
        logger.debug(f"Complete words: {complete_norm[:20]}...")
        logger.debug(f"Segment words: {segment_norm[:20]}...")
        
        seg_idx = 0
        
        for i, comp_word in enumerate(complete_norm):
            if seg_idx >= len(segment_norm):
                missing.append({
                    'word': complete_words[i],
                    'before': complete_words[i-1] if i > 0 else None,
                    'after': complete_words[i+1] if i < len(complete_words)-1 else None,
                    'position': i,
                    'type': 'missing'
                })
                logger.debug(f"  [{i}] '{complete_words[i]}' - missing (end of segment)")
                continue
            
            if comp_word == segment_norm[seg_idx]:
                logger.debug(f"  [{i}] '{comp_word}' == '{segment_norm[seg_idx]}' - match")
                seg_idx += 1
            elif self._are_equivalent_words(comp_word, segment_norm[seg_idx]):
                logger.debug(f"  [{i}] '{comp_word}' ≈ '{segment_norm[seg_idx]}' - equivalent")
                seg_idx += 1
            else:
                # Check if word appears later
                found_later = False
                for offset in range(1, min(5, len(segment_norm) - seg_idx)):
                    if comp_word == segment_norm[seg_idx + offset]:
                        logger.debug(f"  [{i}] '{comp_word}' found at offset {offset}, skipping {segment_norm[seg_idx:seg_idx+offset]}")
                        seg_idx += offset + 1
                        found_later = True
                        break
                
                if not found_later:
                    missing.append({
                        'word': complete_words[i],
                        'before': complete_words[i-1] if i > 0 else None,
                        'after': complete_words[i+1] if i < len(complete_words)-1 else None,
                        'position': i,
                        'type': 'missing'
                    })
                    logger.debug(f"  [{i}] '{complete_words[i]}' - MISSING (not found in next 5)")
        
        return missing
    
    def _are_equivalent_words(self, word1: str, word2: str) -> bool:
        """Check if two words are equivalent (e.g., number words vs digits)"""
        number_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12'
        }
        
        # Check if word1 is a number word and word2 is its digit equivalent
        if word1 in number_map and word2 == number_map[word1]:
            return True
        if word2 in number_map and word1 == number_map[word2]:
            return True
        
        return False

    def run_vtt_reconciliation(self, txt_path: str, vtt_path: str) -> bool:
        """
        Run the VTT reconciler as a separate post-processing step
        Returns True if successful, False otherwise
        """
        try:
            # Find the reconciler script
            reconciler_path = Path(__file__).parent / "vtt_reconciler.py"
            
            if not reconciler_path.exists():
                logger.warning(f"⚠️  VTT reconciler not found at {reconciler_path}")
                return False
            
            # Run reconciler as subprocess
            import subprocess
            
            result = subprocess.run(
                [sys.executable, str(reconciler_path), str(txt_path), str(vtt_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"✅ VTT reconciliation successful for {Path(vtt_path).name}")
                return True
            else:
                logger.error(f"❌ VTT reconciliation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error running VTT reconciler: {e}")
            return False

    
    def _segment_has_equivalent_word(self, seg_words: List[str], word: str) -> bool:
        """Check if segment has an equivalent version of the word"""
        word_norm = word.lower().strip('.,!?";:\'"')
        
        number_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10'
        }
        
        # Check if word is a number word
        if word_norm in number_map:
            digit_equiv = number_map[word_norm]
            seg_words_norm = [w.strip('.,!?";:\'"') for w in seg_words]
            if digit_equiv in seg_words_norm:
                return True
        
        return False



    def _insert_missing_words_into_segments(self, segments: List[Dict], complete_text: str, 
                                            missing_words: List[Dict]) -> List[Dict]:
        """
        Insert missing words into segment text based on context.
        Segment timestamps remain unchanged.
        """
        if not missing_words:
            return segments
        
        complete_words = complete_text.split()
        reconciled_segments = []
        
        for segment in segments:
            seg_text = segment.get('text', '')
            seg_words = seg_text.split()
            
            if not seg_words:
                reconciled_segments.append(segment)
                continue
            
            # Check if any missing words belong in this segment
            modified_text = seg_text
            insertions = []
            
            for missing in missing_words:
                before = missing.get('before')
                after = missing.get('after')
                word = missing.get('word')
                
                # Normalize for comparison
                before_norm = before.lower().strip('.,!?";:\'"') if before else None
                after_norm = after.lower().strip('.,!?";:\'"') if after else None
                seg_words_norm = [w.lower().strip('.,!?";:\'"') for w in seg_words]
                
                # Check if context words are in this segment
                before_in_seg = before_norm in seg_words_norm if before_norm else False
                after_in_seg = after_norm in seg_words_norm if after_norm else False
                
                if before_in_seg and after_in_seg:
                    # Both context words in segment - insert between them
                    insertions.append({
                        'word': word,
                        'before': before,
                        'after': after
                    })
                    logger.info(f"   ➕ Inserting '{word}' between '{before}' and '{after}' in segment")
                elif before_in_seg and not after:
                    # At end of complete text, before word in segment
                    insertions.append({
                        'word': word,
                        'before': before,
                        'after': None
                    })
                    logger.info(f"   ➕ Inserting '{word}' after '{before}' in segment")
                elif after_in_seg and not before:
                    # At start of complete text, after word in segment
                    insertions.append({
                        'word': word,
                        'before': None,
                        'after': after
                    })
                    logger.info(f"   ➕ Inserting '{word}' before '{after}' in segment")
            
            # Apply insertions
            for insertion in insertions:
                before = insertion.get('before')
                after = insertion.get('after')
                word = insertion.get('word')
                
                if before and after:
                    # Insert between
                    pattern = re.escape(before) + r'\s+' + re.escape(after)
                    modified_text = re.sub(
                        pattern, 
                        f"{before} {word} {after}", 
                        modified_text,
                        count=1
                    )
                elif before:
                    # Insert after
                    pattern = re.escape(before) + r'\b'
                    modified_text = re.sub(
                        pattern,
                        f"{before} {word}",
                        modified_text,
                        count=1
                    )
                elif after:
                    # Insert before
                    pattern = r'\b' + re.escape(after)
                    modified_text = re.sub(
                        pattern,
                        f"{word} {after}",
                        modified_text,
                        count=1
                    )
            
            # Create new segment with modified text but same timestamps
            reconciled_segment = segment.copy()
            reconciled_segment['text'] = modified_text
            reconciled_segments.append(reconciled_segment)
        
        return reconciled_segments
    
    # Add these methods to HighPerformanceBatchWorker class

    def _auto_reconcile_segments(self, complete_text: str, segments: List[Dict], max_chars_per_line: int = 42) -> List[Dict]:
        """
        Automatically reconcile segments with complete text and handle overflow
        """
        complete_words = complete_text.split()
        segment_text = ' '.join([seg.get('text', '') for seg in segments])
        segment_words = segment_text.split()
        
        if len(complete_words) == len(segment_words):
            logger.info("✅ Segment word count matches transcript")
            return segments
        
        logger.warning(f"⚠️  Reconciling segments: transcript={len(complete_words)} words, segments={len(segment_words)} words")
        
        # Find missing words
        missing_words = self._find_missing_words_simple(complete_words, segment_words)
        
        if not missing_words:
            return segments
        
        logger.info(f"🔧 Inserting {len(missing_words)} missing words: {[m['word'] for m in missing_words[:10]]}")
        
        # Insert missing words
        reconciled = self._insert_words_with_overflow_handling(segments, missing_words, max_chars_per_line)
        
        return reconciled

    def _find_missing_words_simple(self, complete_words: List[str], segment_words: List[str]) -> List[Dict]:
        """Find words in complete that are missing from segments"""
        def normalize(w):
            return w.lower().strip('.,!?";:\'"') if w else None
        
        complete_norm = [normalize(w) for w in complete_words]
        segment_norm = [normalize(w) for w in segment_words]
        
        missing = []
        seg_idx = 0
        
        for i, comp_word in enumerate(complete_norm):
            if not comp_word:
                continue
            
            found = False
            
            if seg_idx < len(segment_norm):
                if comp_word == segment_norm[seg_idx]:
                    seg_idx += 1
                    found = True
                else:
                    # Look ahead
                    for offset in range(1, min(6, len(segment_norm) - seg_idx)):
                        if comp_word == segment_norm[seg_idx + offset]:
                            seg_idx += offset + 1
                            found = True
                            break
            
            if not found:
                before_word = complete_words[i-1] if i > 0 else None
                after_word = complete_words[i+1] if i < len(complete_words)-1 else None
                
                missing.append({
                    'word': complete_words[i],
                    'before': before_word,
                    'after': after_word,
                    'position': i
                })
        
        return missing

    def _insert_words_with_overflow_handling(self, segments: List[Dict], missing_words: List[Dict], 
                                            max_chars_per_line: int) -> List[Dict]:
        """
        Insert missing words into segments with overflow handling
        """
        reconciled = []
        inserted_words = set()
        
        def normalize(w):
            return w.lower().strip('.,!?";:\'"') if w else None
        
        for seg_idx, segment in enumerate(segments):
            text = segment.get('text', '')
            start = segment.get('start')
            end = segment.get('end')
            
            # Get neighboring segments
            prev_segment = segments[seg_idx - 1] if seg_idx > 0 else None
            next_segment = segments[seg_idx + 1] if seg_idx < len(segments) - 1 else None
            
            # Try to insert missing words
            for miss_idx, missing in enumerate(missing_words):
                if miss_idx in inserted_words:
                    continue
                
                before = missing['before']
                after = missing['after']
                word = missing['word']
                
                before_norm = normalize(before)
                after_norm = normalize(after)
                
                segment_words = text.replace('\n', ' ').split()
                segment_words_norm = [normalize(w) for w in segment_words]
                
                before_in_seg = before_norm in segment_words_norm if before_norm else False
                after_in_seg = after_norm in segment_words_norm if after_norm else False
                
                inserted = False
                
                # Case 1: Both context words in segment
                if before_in_seg and after_in_seg and before and after:
                    new_text = self._insert_between_words(text, before, after, word)
                    if new_text != text:
                        text = new_text
                        inserted = True
                        logger.debug(f"   ➕ [{seg_idx}] Inserted '{word}' between '{before}' and '{after}'")
                
                # Case 2: Cross-segment insertion (before at end, after at start of next)
                elif before_in_seg and next_segment and not inserted:
                    next_text = next_segment.get('text', '').replace('\n', ' ')
                    next_words_norm = [normalize(w) for w in next_text.split()]
                    
                    if segment_words_norm and segment_words_norm[-1] == before_norm:
                        if after_norm in next_words_norm[:3]:
                            text = text.rstrip() + ' ' + word
                            inserted = True
                            logger.debug(f"   ➕ [{seg_idx}] Appended '{word}' (cross-segment)")
                
                # Case 3: Only before word
                elif before_in_seg and not after and not inserted:
                    new_text = self._insert_after_word(text, before, word)
                    if new_text != text:
                        text = new_text
                        inserted = True
                        logger.debug(f"   ➕ [{seg_idx}] Inserted '{word}' after '{before}'")
                
                # Case 4: Only after word
                elif after_in_seg and not before and not inserted:
                    new_text = self._insert_before_word(text, after, word)
                    if new_text != text:
                        text = new_text
                        inserted = True
                        logger.debug(f"   ➕ [{seg_idx}] Inserted '{word}' before '{after}'")
                
                if inserted:
                    inserted_words.add(miss_idx)
            
            # Check for overflow and reformat if needed
            if len(text) > max_chars_per_line * 2:  # Exceeds 2-line limit
                logger.debug(f"   ⚠️  Segment {seg_idx} overflow ({len(text)} chars), reformatting...")
                text = self._reformat_overflowed_segment(text, max_chars_per_line)
            
            reconciled.append({
                'start': start,
                'end': end,
                'text': text
            })
        
        logger.info(f"✅ Inserted {len(inserted_words)}/{len(missing_words)} words")
        return reconciled

    def _insert_between_words(self, text: str, before: str, after: str, word: str) -> str:
        """Insert word between two context words"""
        pattern = re.escape(before) + r'(\s+)' + re.escape(after)
        replacement = before + r'\1' + word + r' ' + after
        return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)

    def _insert_after_word(self, text: str, before: str, word: str) -> str:
        """Insert word after a context word"""
        pattern = r'\b' + re.escape(before) + r'\b'
        return re.sub(pattern, lambda m: m.group(0) + ' ' + word, text, count=1, flags=re.IGNORECASE)

    def _insert_before_word(self, text: str, after: str, word: str) -> str:
        """Insert word before a context word"""
        pattern = r'\b' + re.escape(after) + r'\b'
        return re.sub(pattern, lambda m: word + ' ' + m.group(0), text, count=1, flags=re.IGNORECASE)

    def _reformat_overflowed_segment(self, text: str, max_chars_per_line: int) -> str:
        """
        Reformat segment text that exceeds character limits
        Breaks into multiple lines respecting max_chars_per_line
        """
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}" if current_line else word
            
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                
                # Stop at 2 lines
                if len(lines) >= 2:
                    break
        
        if current_line and len(lines) < 2:
            lines.append(current_line)
        
        return '\n'.join(lines)





    
    def _reconcile_missing_words_improved(self, full_text: str, word_level: List[Dict]) -> List[Dict]:
        """
        Reconcile full text with word-level timestamps by:
        1. Merging split compound words in word_level
        2. Inserting missing words with interpolated timestamps
        """
        full_text_words = full_text.split()
        
        # Step 1: Try to merge compound words in word_level first
        merged_word_level = self._merge_compound_words(word_level)
        
        # Create cleaned version for matching
        word_level_cleaned = []
        for w in merged_word_level:
            word = w.get('word', '').strip()
            normalized = word.lower().strip('.,!?";:\'"').replace('-', '')  # Remove hyphens for matching
            word_level_cleaned.append(normalized)
        
        logger.info(f"🔍 Reconciling: {len(full_text_words)} full text words vs {len(merged_word_level)} timestamped words (after merging)")
        
        reconciled = []
        word_level_idx = 0
        missing_words = []
        
        for i, expected_word in enumerate(full_text_words):
            expected_clean = expected_word.lower().strip('.,!?";:\'"').replace('-', '')
            
            if not expected_clean:
                continue
            
            found = False
            
            # Try to find this word in merged_word_level
            if word_level_idx < len(merged_word_level):
                actual_clean = word_level_cleaned[word_level_idx]
                
                if expected_clean == actual_clean:
                    # Perfect match
                    reconciled.append(merged_word_level[word_level_idx])
                    word_level_idx += 1
                    found = True
                else:
                    # Look ahead up to 5 positions
                    for offset in range(1, min(6, len(merged_word_level) - word_level_idx)):
                        check_clean = word_level_cleaned[word_level_idx + offset]
                        
                        if expected_clean == check_clean:
                            # Found it ahead - we have extra words in word_level that aren't in full_text
                            # This shouldn't happen often, but add the skipped words anyway
                            for skip_idx in range(word_level_idx, word_level_idx + offset):
                                # Only add if the word seems valid (not fragments)
                                skip_word = merged_word_level[skip_idx].get('word', '').strip()
                                if len(skip_word) > 1 or skip_word.isalnum():
                                    reconciled.append(merged_word_level[skip_idx])
                            
                            # Add the found word
                            reconciled.append(merged_word_level[word_level_idx + offset])
                            word_level_idx += offset + 1
                            found = True
                            break
            
            if not found:
                # Word is genuinely missing - estimate timestamp
                estimated_timestamp = self._estimate_timestamp(
                    expected_word, 
                    reconciled, 
                    full_text_words, 
                    i, 
                    merged_word_level, 
                    word_level_idx
                )
                
                reconciled.append({
                    'word': expected_word,
                    'start': estimated_timestamp['start'],
                    'end': estimated_timestamp['end']
                })
                
                missing_words.append(expected_word)
                logger.info(f"➕ Added missing word '{expected_word}' at {estimated_timestamp['start']:.2f}s (estimated)")
        
        # Add any remaining word_level words (if any)
        while word_level_idx < len(merged_word_level):
            word = merged_word_level[word_level_idx].get('word', '').strip()
            # Only add if it's not a fragment
            if len(word) > 1 or word.isalnum():
                reconciled.append(merged_word_level[word_level_idx])
            word_level_idx += 1
        
        if missing_words:
            logger.warning(f"⚠️  Inserted {len(missing_words)} missing words with estimated timestamps: {missing_words[:10]}")
        
        logger.info(f"✅ Final reconciliation: {len(reconciled)} words total")
        
        return reconciled

    def _merge_compound_words(self, word_level: List[Dict]) -> List[Dict]:
        """
        Merge split compound words like 'high' + '-end' → 'high-end'
        Also handles other split patterns.
        """
        if not word_level:
            return []
        
        merged = []
        i = 0
        
        while i < len(word_level):
            current = word_level[i]
            current_word = current.get('word', '').strip()
            
            # Check if next word is a hyphenated fragment
            if i + 1 < len(word_level):
                next_word = word_level[i + 1].get('word', '').strip()
                
                # Pattern 1: "high" + "-end" → "high-end"
                if next_word.startswith('-') and len(next_word) > 1:
                    merged_word = current_word + next_word
                    merged_start = safe_float(current.get('start'), 0)
                    merged_end = safe_float(word_level[i + 1].get('end'), merged_start + 0.5)
                    
                    merged.append({
                        'word': merged_word,
                        'start': merged_start,
                        'end': merged_end
                    })
                    
                    logger.debug(f"   Merged '{current_word}' + '{next_word}' → '{merged_word}'")
                    i += 2  # Skip both words
                    continue
                
                # Pattern 2: "high-" + "end" → "high-end"
                elif current_word.endswith('-') and len(current_word) > 1:
                    merged_word = current_word + next_word
                    merged_start = safe_float(current.get('start'), 0)
                    merged_end = safe_float(word_level[i + 1].get('end'), merged_start + 0.5)
                    
                    merged.append({
                        'word': merged_word,
                        'start': merged_start,
                        'end': merged_end
                    })
                    
                    logger.debug(f"   Merged '{current_word}' + '{next_word}' → '{merged_word}'")
                    i += 2
                    continue
            
            # No merge needed, add as-is
            merged.append(current)
            i += 1
        
        if len(merged) < len(word_level):
            logger.info(f"   Merged {len(word_level) - len(merged)} split words: {len(word_level)} → {len(merged)}")
        
        return merged

    
    def _estimate_timestamp(self, word: str, reconciled: List[Dict], 
                           all_words: List[str], current_idx: int,
                           word_level: List[Dict], word_level_idx: int) -> Dict[str, float]:
        """
        Estimate timestamp for a missing word based on surrounding context.
        Uses interpolation between previous and next timestamped words.
        """
        word_duration = len(word) * 0.05  # Base duration: 50ms per character
        min_duration = 0.15  # Minimum 150ms
        max_duration = 0.50  # Maximum 500ms
        
        # Calculate estimated duration
        estimated_duration = max(min_duration, min(word_duration, max_duration))
        
        # Get previous word's timestamp
        prev_end = None
        if reconciled:
            prev_end = safe_float(reconciled[-1].get('end'), None)
        
        # Look ahead in word_level to find the next timestamped word
        next_start = None
        if word_level_idx < len(word_level):
            next_start = safe_float(word_level[word_level_idx].get('start'), None)
        
        # Case 1: We have both previous and next timestamps - INTERPOLATE
        if prev_end is not None and next_start is not None:
            gap = next_start - prev_end
            
            # Count how many missing words are in this gap
            missing_count = 1
            for j in range(current_idx + 1, len(all_words)):
                # Check if next word exists in word_level
                next_word_clean = all_words[j].lower().strip('.,!?";:\'"')
                if word_level_idx < len(word_level):
                    word_level_word = word_level[word_level_idx].get('word', '').strip()
                    word_level_clean = word_level_word.lower().strip('.,!?";:\'"')
                    if next_word_clean == word_level_clean:
                        break
                missing_count += 1
                if missing_count > 5:  # Safety limit
                    break
            
            # Divide the gap evenly among missing words
            word_gap = gap / (missing_count + 1)
            
            est_start = prev_end + word_gap * 0.5  # Small gap before word
            est_end = est_start + min(estimated_duration, word_gap * 0.8)
            
            logger.debug(f"   Interpolated '{word}': prev={prev_end:.2f}s, next={next_start:.2f}s, gap={gap:.2f}s, estimated={est_start:.2f}s")
        
        # Case 2: Only have previous timestamp - extrapolate forward
        elif prev_end is not None:
            est_start = prev_end + 0.05  # 50ms gap
            est_end = est_start + estimated_duration
            logger.debug(f"   Extrapolated '{word}' from previous: {est_start:.2f}s")
        
        # Case 3: Only have next timestamp - extrapolate backward
        elif next_start is not None:
            est_end = next_start - 0.05  # 50ms before next word
            est_start = max(0, est_end - estimated_duration)
            logger.debug(f"   Extrapolated '{word}' before next: {est_start:.2f}s")
        
        # Case 4: No surrounding timestamps - use position-based estimate
        else:
            est_start = current_idx * 0.35  # ~3 words per second
            est_end = est_start + estimated_duration
            logger.debug(f"   Position-based estimate for '{word}': {est_start:.2f}s")
        
        return {
            'start': est_start,
            'end': est_end
        }
    
    def create_artificial_segments(self, text: str, max_chars: int) -> List[Dict]:
        """Create artificial segments when no timestamps are available"""
        if not text or not text.strip():
            return []
        
        segments = []
        sentences = re.split(r'([.!?]+)', text)
        current_time = 0.0
        
        combined_sentences = []
        for i in range(0, len(sentences), 2):
            sentence = sentences[i].strip()
            if sentence and i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if sentence:
                combined_sentences.append(sentence)
        
        for sentence in combined_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(sentence) <= max_chars:
                duration = max(2.0, len(sentence) * 0.08)
                segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": sentence
                })
                current_time += duration
            else:
                # Split long sentences
                words = sentence.split()
                current_segment_text = ""
                
                for word in words:
                    potential_text = current_segment_text + (" " if current_segment_text else "") + word
                    
                    if len(potential_text) <= max_chars:
                        current_segment_text = potential_text
                    else:
                        if current_segment_text:
                            duration = max(2.0, len(current_segment_text) * 0.08)
                            segments.append({
                                "start": current_time,
                                "end": current_time + duration,
                                "text": current_segment_text
                            })
                            current_time += duration
                        
                        current_segment_text = word
                
                if current_segment_text:
                    duration = max(2.0, len(current_segment_text) * 0.08)
                    segments.append({
                        "start": current_time,
                        "end": current_time + duration,
                        "text": current_segment_text
                    })
                    current_time += duration
        
        return segments
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if file is already an audio file"""
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Could not cleanup {temp_file}: {e}")


class OptimizedMainWindow(QMainWindow):
    """Enhanced main window with performance monitoring and advanced UI"""
    
    def __init__(self):
        super().__init__()
        self.file_paths = []
        self.output_folder = ""
        self.batch_results = {}
        self.performance_data = {}
        self.thread = None
        self.worker = None
        self.init_ui_optimized()
        
    def init_ui_optimized(self):
        """Initialize optimized UI with performance monitoring"""
        self.setWindowTitle("Ultra-High Performance Video Captioner v2.1 - Improved Reconciliation")
        self.setGeometry(100, 100, 1200, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title with performance indicator
        title_layout = QHBoxLayout()
        title_label = QLabel("Ultra-High Performance Video Captioner (Improved)")
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        
        self.performance_label = QLabel("Ready")
        self.performance_label.setAlignment(Qt.AlignRight)
        self.performance_label.setStyleSheet("color: green; font-weight: bold;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(self.performance_label)
        layout.addLayout(title_layout)
        
        # Create main processing interface
        self.create_main_interface(layout)
        
        # Enhanced status bar
        self.create_enhanced_status_bar(layout)
    
    def create_main_interface(self, layout):
        """Create main processing interface"""
        # Configuration section
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout(config_group)
        
        # Model selection
        config_layout.addWidget(QLabel("AI Model:"), 0, 0)
        self.model_combo = QComboBox()
        models = [
            ("tiny", "Tiny - Fastest"),
            ("base", "Base - Fast"),
            ("small", "Small - Balanced"),
            ("medium", "Medium - Accurate"),
            ("large-v3", "Large V3 - Best Quality"),
            ("large-v2", "Large V2 - Alternative")
        ]
        for model_id, description in models:
            self.model_combo.addItem(f"{description}", model_id)
        self.model_combo.setCurrentIndex(2)  # Default to small
        config_layout.addWidget(self.model_combo, 0, 1, 1, 2)
        
        # Settings
        config_layout.addWidget(QLabel("Max chars/line:"), 1, 0)
        self.char_limit_spin = QSpinBox()
        self.char_limit_spin.setRange(20, 100)
        self.char_limit_spin.setValue(42)
        config_layout.addWidget(self.char_limit_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Max chars/subtitle:"), 1, 2)
        self.segment_limit_spin = QSpinBox()
        self.segment_limit_spin.setRange(40, 200)
        self.segment_limit_spin.setValue(84)
        config_layout.addWidget(self.segment_limit_spin, 1, 3)
        
        self.timestamps_check = QCheckBox("Generate timestamps")
        self.timestamps_check.setChecked(True)
        config_layout.addWidget(self.timestamps_check, 2, 0, 1, 2)
        
        layout.addWidget(config_group)
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        file_buttons_layout = QHBoxLayout()
        
        self.select_files_button = QPushButton("📁 Select Files")
        self.select_files_button.clicked.connect(self.select_multiple_files)
        
        self.select_folder_button = QPushButton("📂 Select Folder")
        self.select_folder_button.clicked.connect(self.select_folder)
        
        self.clear_files_button = QPushButton("🗑️ Clear")
        self.clear_files_button.clicked.connect(self.clear_files)
        
        self.test_system_button = QPushButton("🔧 Test System")
        self.test_system_button.clicked.connect(self.run_system_test)
        
        file_buttons_layout.addWidget(self.select_files_button)
        file_buttons_layout.addWidget(self.select_folder_button)
        file_buttons_layout.addWidget(self.clear_files_button)
        file_buttons_layout.addWidget(self.test_system_button)
        file_layout.addLayout(file_buttons_layout)
        
        # Output folder
        output_layout = QHBoxLayout()
        self.output_folder_label = QLabel("No output folder selected")
        self.output_folder_label.setStyleSheet("color: gray; padding: 8px; border: 1px dashed gray; border-radius: 4px;")
        self.select_output_button = QPushButton("📋 Output Folder")
        self.select_output_button.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(QLabel("Output:"))
        output_layout.addWidget(self.output_folder_label, 2)
        output_layout.addWidget(self.select_output_button, 1)
        file_layout.addLayout(output_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(160)
        file_layout.addWidget(QLabel("Selected Files:"))
        file_layout.addWidget(self.file_list)
        
        layout.addWidget(file_group)
        
        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Overall progress
        overall_layout = QHBoxLayout()
        overall_layout.addWidget(QLabel("Overall:"))
        self.overall_progress = QProgressBar()
        self.overall_progress.setVisible(False)
        self.eta_label = QLabel("")
        overall_layout.addWidget(self.overall_progress, 3)
        overall_layout.addWidget(self.eta_label, 1)
        progress_layout.addLayout(overall_layout)
        
        # Current file progress
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Current:"))
        self.file_progress = QProgressBar()
        self.file_progress.setVisible(False)
        self.speed_indicator = QLabel("")
        file_layout.addWidget(self.file_progress, 3)
        file_layout.addWidget(self.speed_indicator, 1)
        progress_layout.addLayout(file_layout)
        
        self.current_file_label = QLabel("Ready to process")
        self.current_file_label.setStyleSheet("font-style: italic; color: #666; padding: 4px;")
        progress_layout.addWidget(self.current_file_label)
        
        layout.addWidget(progress_group)
        
        # Results preview
        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("Transcription results will appear here...")
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Courier", 10))
        self.output_text.setMaximumHeight(200)
        layout.addWidget(self.output_text)
        
        # Control buttons
        self.create_control_buttons(layout)
    
    def create_control_buttons(self, layout):
        """Create control buttons"""
        button_layout = QHBoxLayout()
        
        self.transcribe_button = QPushButton("🚀 Start Processing")
        self.transcribe_button.clicked.connect(self.start_optimized_batch)
        self.transcribe_button.setEnabled(False)
        self.transcribe_button.setStyleSheet("""
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        self.save_txt_button = QPushButton("💾 Export TXT")
        self.save_txt_button.clicked.connect(lambda: self.save_all_optimized("txt"))
        self.save_txt_button.setEnabled(False)
        
        self.save_vtt_button = QPushButton("🎞️ Export VTT")
        self.save_vtt_button.clicked.connect(lambda: self.save_all_optimized("vtt"))
        self.save_vtt_button.setEnabled(False)
        
        self.save_srt_button = QPushButton("📝 Export SRT")
        self.save_srt_button.clicked.connect(lambda: self.save_all_optimized("srt"))
        self.save_srt_button.setEnabled(False)
        
        button_layout.addWidget(self.transcribe_button, 2)
        button_layout.addWidget(self.stop_button, 1)
        button_layout.addWidget(self.save_txt_button, 1)
        button_layout.addWidget(self.save_vtt_button, 1)
        button_layout.addWidget(self.save_srt_button, 1)
        
        layout.addLayout(button_layout)
    
    def create_enhanced_status_bar(self, layout):
        """Create enhanced status bar"""
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready - Improved Word Reconciliation")
        self.status_label.setStyleSheet("padding: 8px; background-color: #f8f9fa; border-radius: 4px;")
        
        self.performance_indicator = QLabel("System: Ready")
        self.performance_indicator.setStyleSheet("padding: 8px; background-color: #e8f5e8; border-radius: 4px; color: #2e7d2e;")
        
        status_layout.addWidget(self.status_label, 3)
        status_layout.addWidget(self.performance_indicator, 1)
        
        layout.addLayout(status_layout)
    
    def run_system_test(self):
        """Run system test"""
        self.status_label.setText("Running system test...")
        
        config = OptimizedModelManager.get_optimal_device_config()
        ffmpeg_path = HighPerformanceAudioProcessor.find_ffmpeg_optimized()
        ffmpeg_status = "✅ Ready" if ffmpeg_path else "❌ Missing"
        
        report = f"""System Test Results (v2.1 - Improved):

🔧 Hardware:
   Device: {config['device'].upper()}
   Precision: {config['compute_type']}
   FFmpeg: {ffmpeg_status}

⚡ Expected Performance:
   Tiny/Base: Good CPU performance
   Small: Balanced CPU/GPU
   Medium/Large: GPU recommended

💡 This version includes:
   - Improved word reconciliation with interpolation
   - Missing words detected and timestamps estimated
   - Better accuracy for word-level captions
   - Handles gaps in word-level timestamp data
"""
        
        QMessageBox.information(self, "System Test Results", report)
    
    def select_multiple_files(self):
        """Select multiple files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Media Files", 
            "", 
            "All Media (*.mp4 *.mov *.avi *.wav *.mp3 *.m4a *.flac *.mkv *.webm);;All Files (*)"
        )
        
        if file_paths:
            self.file_paths = file_paths
            self.update_file_list()
            self.check_ready_state()
    
    def select_folder(self):
        """Select folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if folder_path:
            media_extensions = {'.mp4', '.mov', '.avi', '.wav', '.mp3', '.m4a', '.flac', '.mkv', '.webm'}
            found_files = []
            
            for file_path in Path(folder_path).rglob("*"):
                if file_path.suffix.lower() in media_extensions:
                    found_files.append(str(file_path))
            
            if found_files:
                self.file_paths = found_files
                self.update_file_list()
                self.check_ready_state()
                self.status_label.setText(f"Found {len(found_files)} media files")
            else:
                QMessageBox.information(self, "No Media Files", "No supported media files found.")
    
    def select_output_folder(self):
        """Select output folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        
        if folder_path:
            self.output_folder = folder_path
            self.output_folder_label.setText(folder_path)
            self.output_folder_label.setStyleSheet(
                "color: #2e7d2e; padding: 8px; border: 2px solid #28a745; border-radius: 4px; background-color: #f8fff8;"
            )
            self.check_ready_state()
    
    def clear_files(self):
        """Clear files"""
        self.file_paths = []
        self.file_list.clear()
        self.batch_results = {}
        self.performance_data = {}
        self.output_text.clear()
        
        self.performance_indicator.setText("System: Ready")
        self.check_ready_state()
        self.status_label.setText("Files cleared")
    
    def update_file_list(self):
        """Update file list"""
        self.file_list.clear()
        for file_path in self.file_paths:
            self.file_list.addItem(os.path.basename(file_path))
    
    def check_ready_state(self):
        """Check ready state"""
        has_files = len(self.file_paths) > 0
        has_output = bool(self.output_folder)
        ready = has_files and has_output
        
        self.transcribe_button.setEnabled(ready)
        
        if ready:
            self.status_label.setText(f"Ready: {len(self.file_paths)} files selected")
        elif not has_files:
            self.status_label.setText("Select files to process")
        elif not has_output:
            self.status_label.setText("Select output folder")
    
    def start_optimized_batch(self):
        """Start processing"""
        if not self.file_paths or not self.output_folder:
            return
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.batch_results = {}
        self.performance_data = {}
        
        self.set_buttons_enabled(False)
        self.overall_progress.setVisible(True)
        self.file_progress.setVisible(True)
        self.overall_progress.setValue(0)
        self.file_progress.setValue(0)
        self.output_text.clear()
        self.stop_button.setEnabled(True)
        
        # Create worker
        self.thread = QThread()
        self.worker = HighPerformanceBatchWorker()
        self.worker.moveToThread(self.thread)
        
        # Connect signals
        self.thread.started.connect(lambda: self.worker.transcribe_batch_optimized(
            self.file_paths,
            self.model_combo.currentData(),
            self.timestamps_check.isChecked(),
            self.segment_limit_spin.value()
        ))
        
        self.worker.progress_updated.connect(self.overall_progress.setValue)
        self.worker.file_progress_updated.connect(self.file_progress.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.current_file_updated.connect(self.current_file_label.setText)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.performance_stats.connect(self.update_performance_stats)
        self.thread.finished.connect(self.cleanup_thread)
        
        self.thread.start()
        logger.info(f"🚀 Started processing {len(self.file_paths)} files")
    
    def update_performance_stats(self, stats: dict):
        """Update performance statistics"""
        self.performance_data.update(stats)
        
        if stats.get('average_speed', 0) > 0:
            speed = stats['average_speed']
            if speed >= 1.0:
                speed_color = "#28a745"  # Green
            elif speed >= 0.5:
                speed_color = "#ffc107"  # Yellow
            else:
                speed_color = "#dc3545"  # Red
            
            self.speed_indicator.setText(f"{speed:.1f}x")
            self.speed_indicator.setStyleSheet(f"color: {speed_color}; font-weight: bold;")
    
    def on_file_completed(self, file_path: str, text: str, segments: list, stats: ProcessingStats):
        """Handle file completion"""
        self.batch_results[file_path] = {
            'text': text,
            'segments': segments,
            'success': True,
            'stats': stats
        }
        
        filename = os.path.basename(file_path)
        speed_info = f"{stats.chars_per_second:.0f} chars/sec" if stats.chars_per_second > 0 else "N/A"
        
        self.output_text.append(f"✅ {filename}: {len(text)} chars, {len(segments)} segments, {speed_info}")
        
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)
    
    def on_batch_finished(self, summary: dict):
        """Handle batch completion"""
        self.batch_results.update(summary['results'])
        performance = summary.get('performance', {})
        
        self.overall_progress.setVisible(False)
        self.file_progress.setVisible(False)
        self.current_file_label.setText("✅ Processing complete!")
        
        # Enable export buttons
        has_successful = any(result.get('success', False) for result in self.batch_results.values())
        has_segments = any(
            result.get('success', False) and result.get('segments', []) 
            for result in self.batch_results.values()
        )
        
        self.save_txt_button.setEnabled(has_successful)
        self.save_vtt_button.setEnabled(has_segments)
        self.save_srt_button.setEnabled(has_segments)
        
        # Show summary
        total_time = performance.get('total_time', 0)
        avg_speed = performance.get('average_speed', 0)
        
        self.output_text.append(f"\n🏁 PROCESSING COMPLETE:")
        self.output_text.append(f"📊 Files: {summary['successful']}/{summary['total']} successful")
        self.output_text.append(f"⏱️ Time: {total_time:.1f} seconds")
        if avg_speed > 0:
            self.output_text.append(f"🚀 Speed: {avg_speed:.1f}x realtime")
        
        self.output_text.append(f"\n💾 Ready to export to: {self.output_folder}")
        
        if avg_speed > 0:
            self.performance_indicator.setText(f"Completed: {avg_speed:.1f}x realtime")
        
        # NEW: Run post-processing reconciliation
        if has_successful and has_segments:
            self.output_text.append(f"\n🔧 Running post-processing reconciliation...")
            
            # Use QTimer to run after UI updates
            QTimer.singleShot(100, self.run_post_processing_reconciliation)
        
        self.set_buttons_enabled(True)
        self.stop_button.setEnabled(False)

    
    def save_all_optimized(self, format_type: str):
        """Save all transcriptions"""
        if not self.batch_results or not self.output_folder:
            return
        
        successful_count = 0
        failed_count = 0
        
        self.status_label.setText(f"Exporting {format_type.upper()} files...")
        
        for file_path, result in self.batch_results.items():
            if not result.get('success', False):
                continue
            
            try:
                input_filename = Path(file_path).stem
                output_filename = f"{input_filename}.{format_type}"
                output_path = os.path.join(self.output_folder, output_filename)
                
                if format_type == "txt":
                    content = result['text']
                elif format_type == "vtt":
                    # Check if reconciled version exists
                    corrected_path = Path(self.output_folder) / f"{input_filename}_corrected.vtt"
                    
                    if corrected_path.exists():
                        # Copy reconciled version to final location
                        import shutil
                        shutil.copy2(corrected_path, output_path)
                        logger.info(f"✅ Using reconciled VTT for {input_filename}")
                        successful_count += 1
                        continue
                    else:
                        # Use original segments
                        segments = result.get('segments', [])
                        if segments:
                            content = OptimizedSubtitleFormatter.create_vtt_optimized(
                                segments, self.char_limit_spin.value()
                            )
                        else:
                            failed_count += 1
                            continue
                elif format_type == "srt":
                    segments = result.get('segments', [])
                    if segments:
                        content = OptimizedSubtitleFormatter.create_srt_optimized(
                            segments, self.char_limit_spin.value()
                        )
                    else:
                        failed_count += 1
                        continue
                
                with open(output_path, 'w', encoding='utf-8-sig') as f:
                    f.write(content)
                
                successful_count += 1
                
            except Exception as e:
                failed_count += 1
                logger.error(f"Export error for {file_path}: {e}")
        
        if successful_count > 0:
            self.status_label.setText(f"✅ Exported {successful_count} {format_type.upper()} files")
            self.performance_indicator.setText(f"Export: {successful_count} files saved")
        else:
            self.status_label.setText(f"❌ Export failed for {format_type.upper()} format")

    
    def on_error(self, error_message: str, file_path: str):
        """Handle errors"""
        timestamp = time.strftime('%H:%M:%S')
        if file_path:
            filename = os.path.basename(file_path)
            error_text = f"❌ [{timestamp}] {filename}: {error_message}"
        else:
            error_text = f"❌ [{timestamp}] System: {error_message}"
        
        self.output_text.append(error_text)
        logger.error(error_text)
    
    def stop_processing(self):
        """Stop processing"""
        if self.worker:
            self.worker.stop_processing()
        
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping...")
    
    def set_buttons_enabled(self, enabled: bool):
        """Set button states"""
        ready = enabled and len(self.file_paths) > 0 and bool(self.output_folder)
        
        self.transcribe_button.setEnabled(ready)
        self.select_files_button.setEnabled(enabled)
        self.select_folder_button.setEnabled(enabled)
        self.select_output_button.setEnabled(enabled)
        self.clear_files_button.setEnabled(enabled)
        self.test_system_button.setEnabled(enabled)
        
        self.model_combo.setEnabled(enabled)
        self.char_limit_spin.setEnabled(enabled)
        self.segment_limit_spin.setEnabled(enabled)
        self.timestamps_check.setEnabled(enabled)
    
    def cleanup_thread(self):
        """Clean up thread"""
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        if self.thread:
            self.thread.deleteLater()
            self.thread = None
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.thread and self.thread.isRunning():
            if self.worker:
                self.worker.stop_processing()
            self.thread.quit()
            self.thread.wait(5000)
        
        OptimizedModelManager().clear_model_aggressive()
        event.accept()

    def run_post_processing_reconciliation(self):
        """
        Run VTT reconciliation on all successfully processed files
        This runs AFTER all transcription and formatting is complete
        """
        if not self.batch_results or not self.output_folder:
            return
        
        logger.info("🔧 Starting post-processing VTT reconciliation...")
        
        reconciled_count = 0
        failed_count = 0
        
        for file_path, result in self.batch_results.items():
            if not result.get('success', False):
                continue
            
            try:
                # Generate expected output paths
                input_filename = Path(file_path).stem
                txt_path = Path(self.output_folder) / f"{input_filename}.txt"
                vtt_path = Path(self.output_folder) / f"{input_filename}.vtt"
                
                # Check if both files exist
                if not txt_path.exists():
                    logger.warning(f"⚠️  TXT not found: {txt_path.name}")
                    continue
                
                if not vtt_path.exists():
                    logger.warning(f"⚠️  VTT not found: {vtt_path.name}")
                    continue
                
                # Run reconciliation
                reconciler_path = Path(__file__).parent / "vtt_reconciler.py"
                
                if not reconciler_path.exists():
                    logger.error("❌ vtt_reconciler.py not found in script directory")
                    break
                
                import subprocess
                
                result = subprocess.run(
                    [sys.executable, str(reconciler_path), str(txt_path), str(vtt_path)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    reconciled_count += 1
                    logger.info(f"✅ Reconciled: {vtt_path.name}")
                    
                    # Parse output to show stats
                    if "Perfect match!" in result.stdout:
                        self.output_text.append(f"   ✅ {input_filename}: Perfect match")
                    elif "words difference" in result.stdout:
                        # Extract difference count
                        match = re.search(r'Still (\d+) words difference', result.stdout)
                        if match:
                            diff = match.group(1)
                            self.output_text.append(f"   ⚠️  {input_filename}: {diff} words still missing")
                else:
                    failed_count += 1
                    logger.error(f"❌ Reconciliation failed for {vtt_path.name}: {result.stderr}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error reconciling {file_path}: {e}")
        
        # Update status
        if reconciled_count > 0:
            self.output_text.append(f"\n🔧 Post-processing complete: {reconciled_count} VTT files reconciled")
            self.status_label.setText(f"✅ Processing complete with reconciliation ({reconciled_count} files)")
        
        if failed_count > 0:
            self.output_text.append(f"⚠️  {failed_count} files failed reconciliation")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    app.setApplicationName("Ultra-High Performance Video Captioner - Improved")
    app.setApplicationVersion("2.1-Improved")
    
    window = OptimizedMainWindow()
    window.show()
    
    logger.info("🚀 Ultra-High Performance Video Captioner v2.1 (Improved Reconciliation) started")
    OptimizedModelManager.get_optimal_device_config()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
