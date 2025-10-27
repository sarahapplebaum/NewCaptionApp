# captioner_refactored.py
# High Performance Video Captioner - Refactored for Efficiency
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
from faster_whisper import WhisperModel
import librosa
import numpy as np
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# SHARED UTILITIES (Consolidated)
# ========================================

# Pre-compiled regex patterns for performance
SENTENCE_SPLIT_PATTERN = re.compile(r'[.!?]+')
DURATION_PATTERN = re.compile(r'Duration: (\d+):(\d+):(\d+)\.(\d+)')

# Constants
SENTENCE_STARTERS = frozenset([
    'and', 'but', 'so', 'yet', 'or', 'nor', 'for',
    'however', 'therefore', 'moreover', 'furthermore', 
    'nevertheless', 'additionally', 'meanwhile', 'consequently'
])

COMMON_ABBREVS = frozenset([
    'mr.', 'mrs.', 'dr.', 'prof.', 'inc.', 'ltd.', 
    'co.', 'corp.', 'etc.', 'vs.', 'jr.', 'sr.'
])

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

def extract_words_from_result(result: Dict) -> List[Dict]:
    """
    Extract word-level timestamps with reconciliation for missing words
    Consolidated from duplicate implementations
    """
    try:
        # Get full text from Whisper
        full_text = result.get('text', '').strip()
        if not full_text:
            logger.warning("No text in Whisper result")
            return []
        
        full_text_words = full_text.split()
        
        # Extract word-level timestamps
        word_level = []
        
        if 'chunks' in result:
            for chunk in result['chunks']:
                try:
                    if isinstance(chunk, dict) and 'words' in chunk:
                        chunk_words = chunk['words']
                        if isinstance(chunk_words, list):
                            word_level.extend(chunk_words)
                    elif isinstance(chunk, dict) and 'timestamp' in chunk:
                        timestamp = chunk['timestamp']
                        if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                            start, end = timestamp[0], timestamp[1]
                            word_level.append({
                                'word': chunk.get('text', ''),
                                'start': safe_float(start, 0.0),
                                'end': safe_float(end, safe_float(start, 0.0) + 0.5)
                            })
                except Exception as chunk_error:
                    logger.warning(f"Error processing chunk: {chunk_error}")
                    continue
                    
        elif 'words' in result:
            result_words = result['words']
            if isinstance(result_words, list):
                word_level = result_words
        
        # Clean word-level data
        cleaned_word_level = []
        for word_info in word_level:
            try:
                if isinstance(word_info, dict) and word_info.get('word'):
                    cleaned_word = {
                        'word': str(word_info.get('word', '')).strip(),
                        'start': safe_float(word_info.get('start'), 0.0),
                        'end': safe_float(word_info.get('end'), safe_float(word_info.get('start'), 0.0) + 0.5)
                    }
                    if cleaned_word['word']:
                        cleaned_word_level.append(cleaned_word)
            except Exception as word_error:
                logger.warning(f"Error cleaning word: {word_error}")
                continue
        
        # Check for word count mismatch and reconcile
        word_diff = len(full_text_words) - len(cleaned_word_level)
        
        if word_diff != 0:
            logger.warning(f"⚠️  Word count mismatch! Full text: {len(full_text_words)}, "
                         f"Word-level: {len(cleaned_word_level)} (missing: {word_diff})")
            
            # Reconcile to find and add missing words
            reconciled = reconcile_words(full_text_words, cleaned_word_level)
            
            logger.info(f"✅ Reconciliation complete: {len(reconciled)} words total")
            return reconciled
        
        return cleaned_word_level
        
    except Exception as e:
        logger.error(f"Error extracting words from result: {e}")
        return []

def reconcile_words(full_text_words: List[str], word_level: List[Dict]) -> List[Dict]:
    """
    Enhanced word reconciliation with compound word handling and position-based insertion
    """
    logger.info(f"🔍 Reconciling {len(full_text_words)} full text words with {len(word_level)} timestamped words")
    
    # First pass: merge compound words in word_level (fix "high -end" → "high-end")
    word_level = merge_split_compound_words(word_level)
    
    # Create reconciliation structure
    reconciled = []
    word_level_idx = 0
    missing_words = []
    
    # Create normalized lookup for word_level
    word_level_normalized = []
    for w in word_level:
        word = w.get('word', '').strip()
        normalized = word.lower().strip('.,!?";:\'"')
        word_level_normalized.append(normalized)
    
    # Process each expected word
    for i, expected_word in enumerate(full_text_words):
        expected_clean = expected_word.lower().strip('.,!?";:\'"')
        
        if not expected_clean:
            continue
        
        found = False
        
        # Strategy 1: Direct match at current position
        if word_level_idx < len(word_level):
            actual_clean = word_level_normalized[word_level_idx]
            
            if expected_clean == actual_clean:
                reconciled.append(word_level[word_level_idx])
                word_level_idx += 1
                found = True
        
        # Strategy 2: Look ahead for the word (handle reordering)
        if not found and word_level_idx < len(word_level):
            for offset in range(1, min(8, len(word_level) - word_level_idx)):
                check_clean = word_level_normalized[word_level_idx + offset]
                
                if expected_clean == check_clean:
                    # Add all skipped words first
                    for skip_idx in range(word_level_idx, word_level_idx + offset):
                        reconciled.append(word_level[skip_idx])
                    
                    # Add the found word
                    reconciled.append(word_level[word_level_idx + offset])
                    word_level_idx += offset + 1
                    found = True
                    break
        
        # Strategy 3: Partial/compound word matching
        if not found and word_level_idx < len(word_level):
            actual_word = word_level[word_level_idx].get('word', '').lower().strip()
            
            # Check if expected word is contained in actual word or vice versa
            if (len(expected_clean) > 2 and expected_clean in actual_word) or \
               (len(actual_word) > 2 and actual_word in expected_clean):
                reconciled.append(word_level[word_level_idx])
                word_level_idx += 1
                found = True
        
        # Strategy 4: Position-based insertion for missing words
        if not found:
            # Find optimal insertion position based on surrounding context
            insertion_time = estimate_word_timestamp(expected_word, i, full_text_words, reconciled, word_level, word_level_idx)
            
            new_word_entry = {
                'word': expected_word,
                'start': insertion_time['start'],
                'end': insertion_time['end']
            }
            
            reconciled.append(new_word_entry)
            missing_words.append(expected_word)
            
            logger.warning(f"🚨 MISSING WORD DETECTED: '{expected_word}' inserted at {insertion_time['start']:.2f}s")
    
    # Add any remaining word_level words
    while word_level_idx < len(word_level):
        reconciled.append(word_level[word_level_idx])
        word_level_idx += 1
    
    # Sort by timestamp to maintain chronological order
    reconciled.sort(key=lambda x: safe_float(x.get('start'), 0))
    
    if missing_words:
        logger.warning(f"⚠️  Added {len(missing_words)} missing words with position-based insertion")
        logger.info(f"   Missing words: {missing_words}")
    
    logger.info(f"✅ Reconciliation complete: {len(reconciled)} words total")
    return reconciled

def merge_split_compound_words(word_level: List[Dict]) -> List[Dict]:
    """
    Enhanced compound word merging to handle various split patterns
    Handles: 'high' + '-end', 'high-' + 'end', 'high ' + '-end', etc.
    Also handles: 'g' + '-buffer', 'double' + '-clicking', '6' + '.2', etc.
    """
    if len(word_level) < 2:
        return word_level
    
    merged = []
    i = 0
    
    while i < len(word_level):
        current_word = word_level[i].get('word', '').strip()
        
        # FIRST: Always check for internal compound word issues
        if ' -' in current_word:
            # Fix "high -end" → "high-end" immediately
            fixed_word = current_word.replace(' -', '-')
            merged_entry = {
                'word': fixed_word,
                'start': safe_float(word_level[i].get('start'), 0),
                'end': safe_float(word_level[i].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
            }
            merged.append(merged_entry)
            logger.info(f"🔧 Fixed compound word: '{current_word}' → '{fixed_word}'")
            i += 1
            continue
        
        # Check if we can merge with next word
        if i + 1 < len(word_level):
            next_word = word_level[i + 1].get('word', '').strip()
            
            # Also fix next word if it has space issues
            if ' -' in next_word:
                next_word = next_word.replace(' -', '-')
            
            merged_word = None
            merged_entry = None
            
            # Pattern 1: "high" + "-end" → "high-end" (any word + hyphen-word)
            if next_word.startswith('-') and len(next_word) > 1:
                merged_word = current_word + next_word
                merged_entry = {
                    'word': merged_word,
                    'start': safe_float(word_level[i].get('start'), 0),
                    'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                }
                
            # Pattern 2: "high-" + "end" → "high-end"
            elif current_word.endswith('-') and len(current_word) > 1:
                merged_word = current_word + next_word
                merged_entry = {
                    'word': merged_word,
                    'start': safe_float(word_level[i].get('start'), 0),
                    'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                }
            
            # Pattern 3: Handle split words with space like "high " + "-end"
            elif current_word.endswith(' ') and next_word.startswith('-'):
                merged_word = current_word.rstrip() + next_word
                merged_entry = {
                    'word': merged_word,
                    'start': safe_float(word_level[i].get('start'), 0),
                    'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                }
            
            # Pattern 4: Number + period + number (e.g., "6" + ".2" → "6.2")
            elif current_word.replace(',', '').replace('.', '').isdigit() and next_word.startswith('.') and len(next_word) > 1:
                # Check if the part after the period is numeric
                after_period = next_word[1:].replace(',', '').replace('.', '')
                if after_period and (after_period.isdigit() or after_period[0].isdigit()):
                    merged_word = current_word + next_word
                    merged_entry = {
                        'word': merged_word,
                        'start': safe_float(word_level[i].get('start'), 0),
                        'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                    }
            
            # Pattern 5: Single letter/short word + hyphen word (e.g., "g" + "-buffer", "e" + "-mail")
            elif len(current_word) <= 2 and i + 1 < len(word_level) and next_word.startswith('-'):
                # Common patterns like g-buffer, e-mail, x-ray, etc.
                merged_word = current_word + next_word
                merged_entry = {
                    'word': merged_word,
                    'start': safe_float(word_level[i].get('start'), 0),
                    'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                }
            
            # Pattern 6: Word + hyphen at end of line (continuation)
            elif current_word.endswith('-') and not next_word.startswith('-'):
                # This might be a line-break hyphenation, check if it makes sense to merge
                combined = current_word[:-1] + next_word  # Remove hyphen and combine
                # Only merge if the combined word seems reasonable (not too long)
                if len(combined) < 20:  # Reasonable word length
                    merged_word = combined
                    merged_entry = {
                        'word': merged_word,
                        'start': safe_float(word_level[i].get('start'), 0),
                        'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                    }
            
            if merged_entry:
                merged.append(merged_entry)
                if merged_word:
                    logger.info(f"🔗 Merged compound word: '{current_word}' + '{next_word}' → '{merged_word}'")
                i += 2  # Skip the next word since we merged it
                continue
        
        # No merge needed, add current word as-is
        merged.append(word_level[i])
        i += 1
    
    # Second pass: ensure no words have space before hyphen
    final_merged = []
    for word_entry in merged:
        word = word_entry.get('word', '')
        if ' -' in word:
            word_entry['word'] = word.replace(' -', '-')
            logger.info(f"🔧 Final cleanup: '{word}' → '{word_entry['word']}'")
        final_merged.append(word_entry)
    
    return final_merged

def estimate_word_timestamp(word: str, word_index: int, all_words: List[str], 
                          current_reconciled: List[Dict], remaining_word_level: List[Dict], 
                          word_level_idx: int) -> Dict[str, float]:
    """
    Estimate timestamp for missing word based on position and context
    """
    word_duration = max(0.25, len(word) * 0.08)  # Minimum 0.25s, ~0.08s per character
    
    # Strategy 1: Base on last reconciled word
    if current_reconciled:
        last_word = current_reconciled[-1]
        last_end = safe_float(last_word.get('end'), 0)
        
        # Small gap between words
        est_start = last_end + 0.05
        est_end = est_start + word_duration
        
        return {'start': est_start, 'end': est_end}
    
    # Strategy 2: Base on next available word_level word
    if word_level_idx < len(remaining_word_level):
        next_word = remaining_word_level[word_level_idx]
        next_start = safe_float(next_word.get('start'), 0)
        
        # Insert before next word
        est_end = max(0, next_start - 0.05)
        est_start = max(0, est_end - word_duration)
        
        return {'start': est_start, 'end': est_end}
    
    # Strategy 3: Estimate based on word position
    est_start = word_index * 0.4  # Rough estimate
    est_end = est_start + word_duration
    
    return {'start': est_start, 'end': est_end}

def ensure_all_words_preserved(original_text: str, segments: List[Dict]) -> List[Dict]:
    """
    SIMPLIFIED POST-PROCESSING: Ensure all words preserved while maintaining chronological order
    """
    if not original_text or not segments:
        return segments
    
    logger.info("🔄 POST-PROCESSING: Simplified word preservation (chronological order priority)")
    
    # Extract all words from original text
    original_words = original_text.split()
    original_words_clean = [w.lower().strip('.,!?";:\'"') for w in original_words if w.strip()]
    
    # Extract all words from segments
    segment_text = ' '.join([s.get('text', '') for s in segments])
    segment_words = segment_text.split()
    segment_words_clean = [w.lower().strip('.,!?";:\'"') for w in segment_words if w.strip()]
    
    logger.info(f"   Original text: {len(original_words)} words")
    logger.info(f"   Segment text: {len(segment_words)} words")
    
    # Fix compound words in segment text first
    segment_text_fixed = segment_text.replace(' -', '-')
    if segment_text_fixed != segment_text:
        logger.info("🔧 Fixed compound words in segment text")
        # Update segments with fixed text
        for segment in segments:
            if segment.get('text'):
                segment['text'] = segment['text'].replace(' -', '-')
        
        # Re-extract words after fix
        segment_text = ' '.join([s.get('text', '') for s in segments])
        segment_words = segment_text.split()
        segment_words_clean = [w.lower().strip('.,!?";:\'"') for w in segment_words if w.strip()]
    
    # Count word occurrences
    original_counts = {}
    segment_counts = {}
    
    for word in original_words_clean:
        original_counts[word] = original_counts.get(word, 0) + 1
    
    for word in segment_words_clean:
        segment_counts[word] = segment_counts.get(word, 0) + 1
    
    # Find missing words
    missing_words = []
    for word, orig_count in original_counts.items():
        seg_count = segment_counts.get(word, 0)
        if seg_count < orig_count:
            missing_count = orig_count - seg_count
            # Find original words with proper capitalization
            for orig_word in original_words:
                orig_word_clean = orig_word.lower().strip('.,!?";:\'"')
                if orig_word_clean == word:
                    # Fix compound words before adding
                    fixed_word = orig_word.replace(' -', '-')
                    missing_words.extend([fixed_word] * missing_count)
                    break
            logger.warning(f"   Missing '{word}': need {missing_count} more occurrences")
    
    if not missing_words:
        logger.info("✅ POST-PROCESSING: All words already preserved - no changes needed")
        return segments
    
    logger.warning(f"🚨 POST-PROCESSING: Adding {len(missing_words)} missing words")
    logger.info(f"   Missing words: {missing_words[:15]}...")
    
    # SIMPLE STRATEGY: Append missing words to the last segment to preserve chronological order
    # This avoids complex positioning that can disrupt word order
    if segments and missing_words:
        last_segment = segments[-1]
        current_text = last_segment.get('text', '')
        
        # Add missing words with space separation
        missing_text = ' '.join(missing_words)
        if current_text:
            last_segment['text'] = f"{current_text} {missing_text}"
        else:
            last_segment['text'] = missing_text
        
        logger.info(f"✅ POST-PROCESSING: Added {len(missing_words)} words to final segment")
    
    return segments

def create_artificial_segments(text: str, max_chars: int) -> List[Dict]:
    """
    Create artificial segments when no timestamps are available
    Consolidated from duplicate implementations
    """
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

# ========================================
# DATA CLASSES
# ========================================

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

# ========================================
# MODEL MANAGEMENT
# ========================================

class FasterWhisperModelManager:
    """High-performance singleton model manager optimized for faster-whisper"""
    _instance = None
    _model = None
    _current_model_id = None
    _current_device = None
    _current_compute_type = None
    
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
            
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory < 6:
                config['compute_type'] = "int8"
                
            logger.info(f"🚀 CUDA GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
            
        else:
            config['device'] = "cpu"
            config['compute_type'] = "int8"  # Best performance on CPU
            
            # Set optimal CPU threads
            cpu_cores = os.cpu_count()
            config['cpu_threads'] = min(8, cpu_cores)
            
            logger.info(f"🖥️  CPU ({cpu_cores} cores, using {config['cpu_threads']} threads)")
        
        return config
    
    def load_model_optimized(self, model_id: str = "large-v3") -> bool:
        """Load faster-whisper model with optimizations"""
        try:
            # Convert OpenAI model names to faster-whisper format
            model_name = model_id.replace("openai/whisper-", "").replace("openai/", "")
            
            config = self.get_optimal_device_config()
            
            # Check if model is already loaded with same config
            if (self._current_model_id == model_name and 
                self._current_device == config['device'] and
                self._current_compute_type == config['compute_type'] and
                self._model is not None):
                logger.info(f"✅ Model already loaded: {model_name}")
                return True
            
            self.clear_model_aggressive()
            
            logger.info(f"🔄 Loading faster-whisper model: {model_name}")
            start_time = time.time()
            
            # Load faster-whisper model
            model_kwargs = {
                'device': config['device'],
                'compute_type': config['compute_type'],
                'download_root': None,  # Use default cache
            }
            
            # Add CPU-specific options
            if config['device'] == 'cpu':
                model_kwargs['cpu_threads'] = config.get('cpu_threads', 4)
            
            self._model = WhisperModel(model_name, **model_kwargs)
            
            self._current_model_id = model_name
            self._current_device = config['device']
            self._current_compute_type = config['compute_type']
            
            load_time = time.time() - start_time
            
            logger.info(f"✅ faster-whisper model loaded in {load_time:.2f}s on {config['device']} ({config['compute_type']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ faster-whisper model loading failed: {e}")
            self.clear_model_aggressive()
            return False
    
    def clear_model_aggressive(self):
        """Aggressive model cleanup with memory optimization"""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._current_model_id = None
        self._current_device = None
        self._current_compute_type = None
        
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
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

# Alias for backward compatibility
OptimizedModelManager = FasterWhisperModelManager

# ========================================
# AUDIO PROCESSING
# ========================================

class AudioProcessor:
    """Consolidated audio processing functionality"""
    
    _ffmpeg_cache = {}
    _audio_cache = {}
    
    @staticmethod
    def find_ffmpeg_optimized():
        """Cached FFmpeg detection with performance optimization"""
        if 'ffmpeg_path' in AudioProcessor._ffmpeg_cache:
            cached_path = AudioProcessor._ffmpeg_cache['ffmpeg_path']
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
                        AudioProcessor._ffmpeg_cache['ffmpeg_path'] = result
                        logger.info(f"✅ FFmpeg found: {result}")
                        return result
                except Exception:
                    continue
        
        system_ffmpeg = shutil.which('ffmpeg')
        if system_ffmpeg:
            AudioProcessor._ffmpeg_cache['ffmpeg_path'] = system_ffmpeg
            logger.info(f"✅ FFmpeg in PATH: {system_ffmpeg}")
            return system_ffmpeg
        
        logger.error("❌ FFmpeg not found")
        return None
    
    @staticmethod
    def extract_audio_optimized(video_path: str, output_path: str = None) -> Tuple[str, float]:
        """High-performance audio extraction with caching and optimization"""
        try:
            cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
            if cache_key in AudioProcessor._audio_cache:
                cached_path, duration = AudioProcessor._audio_cache[cache_key]
                if os.path.exists(cached_path):
                    logger.info(f"🗄️  Using cached audio: {cached_path}")
                    return cached_path, duration
        except Exception as cache_error:
            logger.warning(f"Cache check failed: {cache_error}")
        
        ffmpeg_path = AudioProcessor.find_ffmpeg_optimized()
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
            
            duration = AudioProcessor.get_audio_duration(output_path)
            extraction_time = time.time() - start_time
            
            # Cache the result
            try:
                cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
                AudioProcessor._audio_cache[cache_key] = (output_path, duration)
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
            ffmpeg_path = AudioProcessor.find_ffmpeg_optimized()
            if ffmpeg_path:
                try:
                    cmd = [ffmpeg_path, '-i', audio_path, '-f', 'null', '-']
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                    duration_match = DURATION_PATTERN.search(result.stderr)
                    if duration_match:
                        h, m, s, ms = map(int, duration_match.groups())
                        return safe_float(h * 3600 + m * 60 + s + ms / 100)
                except Exception:
                    pass
        return 0.0
    
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """Check if file is already an audio file"""
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
        return Path(file_path).suffix.lower() in audio_extensions

# ========================================
# SUBTITLE FORMATTING
# ========================================

class SubtitleFormatter:
    """Refactored high-performance subtitle formatter with single-pass processing"""
    
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
        """Direct word-moving approach: if word causes overflow, move to next line"""
        if not text or not text.strip():
            return "", []
        
        words = text.split()
        if not words:
            return "", []
        
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed the character limit
            test_line = f"{current_line} {word}" if current_line else word
            
            if len(test_line) <= max_chars_per_line:
                # Word fits, add it to current line
                current_line = test_line
            else:
                # Word would cause overflow, move to next line
                if current_line:  # Save current line if it has content
                    lines.append(current_line)
                
                # Start new line with the overflowing word
                # Check if single word exceeds limit (handle edge case)
                if len(word) > max_chars_per_line:
                    # Truncate single oversized word as last resort
                    current_line = f"{word[:max_chars_per_line-3]}..."
                else:
                    current_line = word
        
        # Add final line if it has content
        if current_line:
            lines.append(current_line)
        
        # Handle max_lines constraint by redistributing words
        if len(lines) > max_lines:
            # Collect all words from excess lines
            excess_words = []
            for i in range(max_lines, len(lines)):
                excess_words.extend(lines[i].split())
            
            # Keep only the allowed number of lines
            lines = lines[:max_lines]
            
            # Redistribute excess words back into existing lines
            for word in excess_words:
                # Try to fit word into existing lines that have space
                added = False
                for i in range(len(lines)):
                    test_line = f"{lines[i]} {word}"
                    if len(test_line) <= max_chars_per_line:
                        lines[i] = test_line
                        added = True
                        break
                
                # If no space available, add to last line to preserve word
                if not added:
                    lines[-1] = f"{lines[-1]} {word}"
        
        return "\n".join(lines), []
    
    @staticmethod
    def calculate_reading_time(text: str) -> float:
        """Calculate reading time based on word count and complexity"""
        if not text:
            return SubtitleFormatter.MIN_DURATION
        
        words = len(text.split())
        base_time = words / SubtitleFormatter.WORDS_PER_SECOND
        
        # Add time for punctuation pauses
        punctuation_time = sum(0.2 for char in text if char in '.,!?;:')
        
        # Calculate by character count as alternative
        char_time = len(text) * 0.05
        
        total_time = max(base_time + punctuation_time, char_time)
        
        return max(
            SubtitleFormatter.MIN_DURATION,
            min(total_time, SubtitleFormatter.MAX_DURATION)
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
        
        # Second pass: clean punctuation ONLY (preserve all words for accuracy)
        cleaned = []
        
        for i, word_info in enumerate(word_infos):
            # Clean punctuation in single operation
            word = word_info.word
            if i < len(word_infos) - 1 and word.endswith('.'):
                next_info = word_infos[i + 1]
                next_word = next_info.word.lstrip('.,!?";:\'" ')
                
                if next_word:
                    first_char = next_word[0]
                    next_lower = next_word.lower()
                    
                    # Remove period if mid-sentence (lowercase, not sentence starter)
                    if first_char.islower() and next_lower not in SENTENCE_STARTERS:
                        word = word[:-1]
                        word_info.word = word
            
            # PRESERVE ALL WORDS - no filtering for repetition to ensure accuracy
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
            return cleaned.lower() not in COMMON_ABBREVS
        
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
        
        for i, word_info in enumerate(cleaned_words):  # Process ALL words - no limits
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
            
            # Decide whether to break segment
            should_break = (
                (would_exceed and current_segment['text']) or
                (time_gap and current_segment['text']) or
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
        """Combined post-processing: deduplication, timing, punctuation, and merging short segments"""
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: safe_float(x.get("start"), 0))
        
        # First pass: deduplicate and initial timing
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
            
            # Ensure we don't overlap with previous segment
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
        
        # Second pass: merge short segments
        processed = cls.merge_short_segments(processed)
        
        # Third pass: ensure reading rate compliance
        processed = cls.enforce_reading_rate(processed)
        
        # Clean segment-end punctuation
        return cls.clean_segment_punctuation(processed)
    
    @classmethod
    def merge_short_segments(cls, segments: List[Dict]) -> List[Dict]:
        """Merge segments with very few words (1-2 words) with adjacent segments"""
        if len(segments) <= 1:
            return segments
        
        MIN_WORDS_PER_SEGMENT = 3  # Minimum words to keep as separate segment
        MAX_CHARS_PER_SEGMENT = 84  # Maximum characters per subtitle (2 lines * 42 chars)
        
        merged = []
        i = 0
        
        while i < len(segments):
            current_segment = segments[i]
            current_text = current_segment.get("text", "").strip()
            current_words = current_text.split()
            
            # If this segment has enough words or is the last segment, keep it
            if len(current_words) >= MIN_WORDS_PER_SEGMENT or i == len(segments) - 1:
                merged.append(current_segment)
                i += 1
                continue
            
            # Try to merge with previous segment first (if exists and has room)
            merged_with_previous = False
            if merged:
                prev_segment = merged[-1]
                prev_text = prev_segment.get("text", "").strip()
                combined_text = f"{prev_text} {current_text}"
                
                # Check if combined text fits within constraints
                if len(combined_text) <= MAX_CHARS_PER_SEGMENT:
                    # Check if it can be formatted within 2 lines of 42 chars
                    test_lines = []
                    test_words = combined_text.split()
                    current_line = ""
                    
                    for word in test_words:
                        test_line = f"{current_line} {word}" if current_line else word
                        if len(test_line) <= 42:
                            current_line = test_line
                        else:
                            test_lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        test_lines.append(current_line)
                    
                    # If it fits in 2 lines, merge
                    if len(test_lines) <= 2:
                        prev_segment["text"] = combined_text
                        prev_segment["end"] = current_segment.get("end", prev_segment.get("end"))
                        merged_with_previous = True
                        i += 1
                        continue
            
            # If couldn't merge with previous, try to merge with next segment
            if not merged_with_previous and i + 1 < len(segments):
                next_segment = segments[i + 1]
                next_text = next_segment.get("text", "").strip()
                combined_text = f"{current_text} {next_text}"
                
                # Check if combined text fits within constraints
                if len(combined_text) <= MAX_CHARS_PER_SEGMENT:
                    # Check if it can be formatted within 2 lines of 42 chars
                    test_lines = []
                    test_words = combined_text.split()
                    current_line = ""
                    
                    for word in test_words:
                        test_line = f"{current_line} {word}" if current_line else word
                        if len(test_line) <= 42:
                            current_line = test_line
                        else:
                            test_lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        test_lines.append(current_line)
                    
                    # If it fits in 2 lines, merge
                    if len(test_lines) <= 2:
                        # Create merged segment
                        merged_segment = {
                            "start": current_segment.get("start"),
                            "end": next_segment.get("end"),
                            "text": combined_text
                        }
                        merged.append(merged_segment)
                        i += 2  # Skip both segments
                        continue
            
            # If couldn't merge with either neighbor, keep as is
            merged.append(current_segment)
            i += 1
        
        return merged
    
    @classmethod
    def enforce_reading_rate(cls, segments: List[Dict]) -> List[Dict]:
        """Ensure reading rate doesn't exceed 25 characters per second"""
        MAX_CHARS_PER_SECOND = 25
        
        for segment in segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0.0)
            end_time = safe_float(segment.get("end"), start_time + cls.MIN_DURATION)
            
            # Calculate current reading rate
            duration = end_time - start_time
            if duration <= 0:
                duration = cls.MIN_DURATION
            
            char_count = len(text)
            current_rate = char_count / duration
            
            # If rate exceeds maximum, extend duration
            if current_rate > MAX_CHARS_PER_SECOND:
                # Calculate required duration
                required_duration = char_count / MAX_CHARS_PER_SECOND
                
                # Ensure minimum duration is respected
                required_duration = max(required_duration, cls.MIN_DURATION)
                
                # Update end time
                segment["end"] = start_time + required_duration
        
        # Now check for overlaps and adjust
        for i in range(1, len(segments)):
            current_segment = segments[i]
            prev_segment = segments[i - 1]
            
            current_start = safe_float(current_segment.get("start"), 0.0)
            prev_end = safe_float(prev_segment.get("end"), 0.0)
            
            # If there's an overlap, adjust current segment's start time
            if current_start < prev_end:
                gap = 0.05  # Small gap between segments
                current_segment["start"] = prev_end + gap
                
                # Recalculate end time to maintain duration
                text = current_segment.get("text", "").strip()
                char_count = len(text)
                required_duration = char_count / MAX_CHARS_PER_SECOND
                required_duration = max(required_duration, cls.MIN_DURATION)
                
                current_segment["end"] = current_segment["start"] + required_duration
        
        return segments
    
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
            if first_word and first_word[0].islower() and first_word.lower() not in SENTENCE_STARTERS:
                segments[i]["text"] = current_text[:-1].rstrip()
        
        return segments
    
    @classmethod
    def create_vtt(cls, segments: List[Dict], max_chars_per_line: int = 40) -> str:
        """Create VTT format with STRICT character limits and word preservation"""
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
            
            # Split text into words
            words = text.replace('\n', ' ').split()
            if not words:
                continue
            
            # Check if this segment needs to be split to preserve word order
            # Each subtitle can have max 2 lines of max_chars_per_line characters
            max_chars_per_subtitle = max_chars_per_line * 2
            
            # Create sub-segments that fit within constraints
            sub_segments = []
            current_words = []
            current_length = 0
            
            for word in words:
                word_len = len(word)
                
                # Check if adding this word would exceed subtitle capacity
                # Account for spaces between words
                space_needed = 1 if current_words else 0
                total_with_word = current_length + space_needed + word_len
                
                # Rough estimate of whether this will fit in 2 lines
                if current_words and total_with_word > max_chars_per_subtitle * 0.9:  # 90% capacity
                    # Create a sub-segment with current words
                    sub_segments.append(current_words)
                    current_words = [word]
                    current_length = word_len
                else:
                    current_words.append(word)
                    current_length += space_needed + word_len
            
            # Add remaining words
            if current_words:
                sub_segments.append(current_words)
            
            # Generate VTT entries for each sub-segment
            if len(sub_segments) == 1:
                # Original segment fits, process normally
                words = sub_segments[0]
                
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = f"{current_line} {word}" if current_line else word
                    
                    if len(test_line) <= max_chars_per_line:
                        current_line = test_line
                    else:
                        # Word causes overflow - push to next line
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # STRICT 2-line limit
                final_lines = lines[:2]
                
                formatted_text = '\n'.join(final_lines)
                
                if not formatted_text:
                    continue
                
                # Position based on line count
                line_count = len(final_lines)
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
            else:
                # Split into multiple subtitle entries to preserve word order
                segment_duration = end_time - start_time
                sub_duration = segment_duration / len(sub_segments)
                
                for i, word_group in enumerate(sub_segments):
                    sub_start = start_time + (i * sub_duration)
                    sub_end = sub_start + sub_duration
                    
                    # Ensure minimum duration
                    if sub_end - sub_start < cls.MIN_DURATION:
                        sub_end = sub_start + cls.MIN_DURATION
                    
                    # Format words into lines
                    lines = []
                    current_line = ""
                    
                    for word in word_group:
                        test_line = f"{current_line} {word}" if current_line else word
                        
                        if len(test_line) <= max_chars_per_line:
                            current_line = test_line
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    
                    if current_line:
                        lines.append(current_line)
                    
                    # Limit to 2 lines
                    final_lines = lines[:2]
                    
                    formatted_text = '\n'.join(final_lines)
                    
                    if not formatted_text:
                        continue
                    
                    # Position based on line count
                    line_count = len(final_lines)
                    position = {
                        1: "align:middle line:90%",
                        2: "align:middle line:84%"
                    }.get(line_count, "align:middle line:80%")
                    
                    # Convert times
                    start_str = cls.seconds_to_vtt_time(sub_start)
                    end_str = cls.seconds_to_vtt_time(sub_end)
                    
                    vtt_lines.extend([
                        f"{start_str} --> {end_str} {position}",
                        formatted_text,
                        ""
                    ])
        
        return "\n".join(vtt_lines)
    
    @classmethod
    def create_srt(cls, segments: List[Dict], max_chars_per_line: int = 40) -> str:
        """Create SRT format"""
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

# ========================================
# TRANSCRIPTION PROCESSING
# ========================================

class TranscriptionProcessor:
    """Consolidated transcription processing functionality"""
    
    def __init__(self):
        self.model_manager = OptimizedModelManager()
        self.temp_files = []
    
    def transcribe_optimized(self, audio_path: str, model, return_timestamps: bool, 
                           max_chars: int, file_stats: ProcessingStats) -> Tuple[str, List[Dict]]:
        """High-performance transcription with faster-whisper"""
        transcription_start = time.time()
        
        try:
            # Configure faster-whisper transcription options (minimal for maximum compatibility)
            transcribe_kwargs = {}
            
            # Only add explicitly supported parameters
            if return_timestamps:
                transcribe_kwargs['word_timestamps'] = True
            
            # Optional: add beam_size only if we want to override default
            # transcribe_kwargs['beam_size'] = 5
            
            logger.info(f"🎙️ Transcribing with faster-whisper...")
            
            # Transcribe with faster-whisper
            segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)
            
            # Convert iterator to list and process results
            transcribed_segments = list(segments_iter)
            
            file_stats.transcription_time = safe_timestamp_operation(time.time(), transcription_start, 'subtract')
            
            # Extract full text
            text = " ".join([segment.text.strip() for segment in transcribed_segments if segment.text.strip()])
            
            segments = []
            
            if return_timestamps and transcribed_segments:
                # Convert faster-whisper segments to our format
                words = []
                
                for segment in transcribed_segments:
                    if hasattr(segment, 'words') and segment.words:
                        # Extract word-level timestamps
                        for word in segment.words:
                            words.append({
                                'word': word.word,
                                'start': safe_float(word.start, 0.0),
                                'end': safe_float(word.end, safe_float(word.start, 0.0) + 0.5)
                            })
                    else:
                        # Fallback: create artificial word timestamps for the segment
                        segment_text = segment.text.strip()
                        if segment_text:
                            segment_words = segment_text.split()
                            segment_duration = safe_float(segment.end, 0.0) - safe_float(segment.start, 0.0)
                            words_per_second = len(segment_words) / max(segment_duration, 0.1)
                            
                            current_time = safe_float(segment.start, 0.0)
                            for word in segment_words:
                                word_duration = min(1.0 / max(words_per_second, 1.0), 0.8)
                                words.append({
                                    'word': word,
                                    'start': current_time,
                                    'end': current_time + word_duration
                                })
                                current_time += word_duration
                
                # IMMEDIATELY merge compound words after extraction
                if words:
                    words = merge_split_compound_words(words)
                
                if words:
                    logger.info(f"✅ Extracted {len(words)} words from faster-whisper")
                    
                    # VALIDATION: Check word preservation before segmentation
                    raw_words = text.split()
                    extracted_words = [w['word'] for w in words]
                    
                    logger.info(f"🔍 Word preservation check:")
                    logger.info(f"   Raw transcript: {len(raw_words)} words")
                    logger.info(f"   Extracted words: {len(extracted_words)} words")
                    
                    segments = SubtitleFormatter.create_optimized_segments(words, max_chars)
                    
                    # VALIDATION: Check final word count in segments
                    segment_text = ' '.join([s.get('text', '') for s in segments])
                    segment_words = segment_text.split()
                    
                    logger.info(f"   Final segments: {len(segment_words)} words")
                    
                    if len(segment_words) < len(raw_words):
                        missing_count = len(raw_words) - len(segment_words)
                        logger.warning(f"⚠️  WORD LOSS DETECTED: {missing_count} words missing from final segments!")
                        
                        # Log missing words for debugging
                        raw_set = set(w.lower().strip('.,!?";:') for w in raw_words)
                        segment_set = set(w.lower().strip('.,!?";:') for w in segment_words)
                        missing_words = raw_set - segment_set
                        
                        if missing_words:
                            logger.warning(f"   Missing words: {list(missing_words)[:20]}")
                    else:
                        logger.info(f"✅ Word preservation successful - no words lost!")
                else:
                    logger.warning("No words extracted from faster-whisper, creating artificial segments")
                    segments = create_artificial_segments(text, max_chars)
            elif transcribed_segments:
                # Create segments from faster-whisper segments without word-level timestamps
                for segment in transcribed_segments:
                    if segment.text.strip():
                        segments.append({
                            "start": safe_float(segment.start, 0.0),
                            "end": safe_float(segment.end, safe_float(segment.start, 0.0) + 2.0),
                            "text": segment.text.strip()
                        })
                
                # Post-process the segments
                segments = SubtitleFormatter.post_process_segments(segments)
            
            logger.info(f"✅ faster-whisper transcription complete: {len(text)} chars, {len(segments)} segments")
            
            # POST-PROCESSING: Ensure all words from original text are preserved in segments
            if segments:
                segments = ensure_all_words_preserved(text, segments)
            
            return text, segments
            
        except Exception as e:
            file_stats.transcription_time = safe_timestamp_operation(time.time(), transcription_start, 'subtract')
            raise RuntimeError(f"faster-whisper transcription failed: {str(e)}")
    
    def extract_audio_optimized(self, video_path: str, file_stats: ProcessingStats) -> Tuple[str, float]:
        """Optimized audio extraction with stats tracking"""
        if AudioProcessor.is_audio_file(video_path):
            duration = AudioProcessor.get_audio_duration(video_path)
            return video_path, duration
        
        extraction_start = time.time()
        audio_path, duration = AudioProcessor.extract_audio_optimized(video_path)
        file_stats.audio_extraction_time = safe_timestamp_operation(time.time(), extraction_start, 'subtract')
        
        if audio_path != video_path:
            self.temp_files.append(audio_path)
        
        return audio_path, duration
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files[:]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"⚠️ Could not cleanup {temp_file}: {e}")

# ========================================
# BATCH PROCESSING WORKER
# ========================================

class BatchWorker(QObject):
    """Optimized batch worker with consolidated functionality"""
    
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
        self.processor = TranscriptionProcessor()
        self.should_stop = False
        self.stats = {}
        
    def stop_processing(self):
        self.should_stop = True
    
    def transcribe_batch_optimized(self, file_paths: List[str], model_id: str, 
                                 return_timestamps: bool = True, max_chars_per_segment: int = 80):
        """Ultra-high performance batch transcription"""
        results = {}
        successful = 0
        failed = 0
        total_start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting batch transcription of {len(file_paths)} files")
            
            performance_stats = {
                'total_files': len(file_paths),
                'start_time': total_start_time,
                'files_processed': 0,
                'total_audio_duration': 0,
                'total_characters': 0,
                'average_speed': 0
            }
            
            self.setup_optimized_environment()
            
            self.status_updated.emit("Loading AI model...")
            with self.model_manager.model_context(model_id) as pipeline:
                if pipeline is None:
                    raise RuntimeError("Failed to load pipeline")
                
                logger.info("✅ Model loaded successfully")
                
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
                        audio_path, duration = self.processor.extract_audio_optimized(file_path, file_stats)
                        file_stats.audio_duration = safe_float(duration, 0.0)
                        performance_stats['total_audio_duration'] += file_stats.audio_duration
                        
                        self.file_progress_updated.emit(30)
                        
                        # Transcribe
                        text, segments = self.processor.transcribe_optimized(audio_path, pipeline, 
                                                                           return_timestamps, max_chars_per_segment, 
                                                                           file_stats)
                        
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
                        
                        self.processor.cleanup_temp_files()
                        
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
            self.processor.cleanup_temp_files()
    
    def setup_optimized_environment(self):
        """Setup optimized processing environment"""
        ffmpeg_path = AudioProcessor.find_ffmpeg_optimized()
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

# ========================================
# MAIN APPLICATION UI
# ========================================

class MainWindow(QMainWindow):
    """Enhanced main window with performance monitoring"""
    
    def __init__(self):
        super().__init__()
        self.file_paths = []
        self.output_folder = ""
        self.batch_results = {}
        self.performance_data = {}
        self.thread = None
        self.worker = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("High Performance Video Captioner - Refactored")
        self.setGeometry(100, 100, 1200, 900)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title_label = QLabel("High Performance Video Captioner (Refactored)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        layout.addWidget(title_label)
        
        # Configuration section
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout(config_group)
        
        # Model selection
        config_layout.addWidget(QLabel("AI Model:"), 0, 0)
        self.model_combo = QComboBox()
        models = [
            ("openai/whisper-tiny", "Fastest"),
            ("openai/whisper-base", "Fast"),
            ("openai/whisper-small", "Balanced"),
            ("openai/whisper-medium", "Accurate"),
            ("openai/whisper-large-v3", "Best Quality")
        ]
        for model_id, description in models:
            self.model_combo.addItem(f"{model_id.split('/')[-1]} - {description}", model_id)
        self.model_combo.setCurrentIndex(2)
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
        
        file_buttons_layout.addWidget(self.select_files_button)
        file_buttons_layout.addWidget(self.select_folder_button)
        file_buttons_layout.addWidget(self.clear_files_button)
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
        self.overall_progress = QProgressBar()
        self.overall_progress.setVisible(False)
        progress_layout.addWidget(QLabel("Overall Progress:"))
        progress_layout.addWidget(self.overall_progress)
        
        # Current file progress
        self.file_progress = QProgressBar()
        self.file_progress.setVisible(False)
        progress_layout.addWidget(QLabel("Current File:"))
        progress_layout.addWidget(self.file_progress)
        
        self.current_file_label = QLabel("Ready to process")
        self.current_file_label.setStyleSheet("font-style: italic; color: #666; padding: 4px;")
        progress_layout.addWidget(self.current_file_label)
        
        layout.addWidget(progress_group)
        
        # Results preview
        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("Transcription results will appear here...")
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))
        self.output_text.setMaximumHeight(200)
        layout.addWidget(self.output_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.transcribe_button = QPushButton("🚀 Start Processing")
        self.transcribe_button.clicked.connect(self.start_batch_processing)
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
        self.save_txt_button.clicked.connect(lambda: self.save_all_results("txt"))
        self.save_txt_button.setEnabled(False)
        
        self.save_vtt_button = QPushButton("🎞️ Export VTT")
        self.save_vtt_button.clicked.connect(lambda: self.save_all_results("vtt"))
        self.save_vtt_button.setEnabled(False)
        
        self.save_srt_button = QPushButton("📝 Export SRT")
        self.save_srt_button.clicked.connect(lambda: self.save_all_results("srt"))
        self.save_srt_button.setEnabled(False)
        
        self.test_button = QPushButton("🧪 Run Tests")
        self.test_button.clicked.connect(self.run_tests)
        self.test_button.setStyleSheet("""
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                font-weight: bold; 
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        
        button_layout.addWidget(self.transcribe_button, 2)
        button_layout.addWidget(self.stop_button, 1)
        button_layout.addWidget(self.save_txt_button, 1)
        button_layout.addWidget(self.save_vtt_button, 1)
        button_layout.addWidget(self.save_srt_button, 1)
        button_layout.addWidget(self.test_button, 1)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready - Refactored Version")
        self.status_label.setStyleSheet("padding: 8px; background-color: #f8f9fa; border-radius: 4px;")
        layout.addWidget(self.status_label)
    
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
    
    def start_batch_processing(self):
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
        self.worker = BatchWorker()
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
        self.thread.finished.connect(self.cleanup_thread)
        
        self.thread.start()
        logger.info(f"🚀 Started processing {len(self.file_paths)} files")
    
    def on_file_completed(self, file_path: str, text: str, segments: list, stats: ProcessingStats):
        """Handle file completion"""
        filename = os.path.basename(file_path)
        self.output_text.append(f"✅ {filename}: {len(text)} chars, {len(segments)} segments")
        
        # Auto-save TXT and generate VTT
        try:
            stem = Path(file_path).stem
            
            # Save TXT
            txt_path = Path(self.output_folder) / f"{stem}.txt"
            with open(txt_path, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            
            # Generate VTT with conservative character limit
            if segments:
                vtt_path = Path(self.output_folder) / f"{stem}.vtt"
                conservative_char_limit = max(20, self.char_limit_spin.value() - 25)
                vtt_content = SubtitleFormatter.create_vtt(segments, conservative_char_limit)
                
                # Save initial VTT
                with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(vtt_content)
                
                # POST-PROCESSING RECONCILIATION
                try:
                    reconciled_vtt = self.post_process_reconcile_vtt(txt_path, vtt_path)
                    if reconciled_vtt:
                        with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                            f.write(reconciled_vtt)
                        self.output_text.append(f"🔄 {filename}: Post-processing reconciliation applied")
                except Exception as reconcile_error:
                    logger.warning(f"Post-processing reconciliation failed for {filename}: {reconcile_error}")
                
                # AUTOMATIC TESTING of the generated VTT
                self.output_text.append(f"🧪 Testing {filename}...")
                test_passed = self.test_single_vtt(vtt_path, txt_path)
                
                # Store result with test status
                self.batch_results[file_path] = {
                    'text': text,
                    'segments': segments,
                    'success': True,
                    'test_passed': test_passed,
                    'vtt_path': str(vtt_path),
                    'txt_path': str(txt_path)
                }
                
                if test_passed:
                    self.output_text.append(f"✅ {filename}: All tests passed!")
                else:
                    self.output_text.append(f"⚠️ {filename}: Some tests failed - see details above")
            else:
                # No segments, just store text result
                self.batch_results[file_path] = {
                    'text': text,
                    'segments': segments,
                    'success': True,
                    'test_passed': None,
                    'txt_path': str(txt_path)
                }
        
        except Exception as e:
            logger.error(f"Processing error for {filename}: {e}")
            self.batch_results[file_path] = {
                'text': text,
                'segments': segments,
                'success': False,
                'error': str(e)
            }
    
    def on_batch_finished(self, summary: dict):
        """Handle batch completion"""
        self.batch_results.update(summary['results'])
        
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
        self.output_text.append(f"\n🏁 PROCESSING COMPLETE:")
        self.output_text.append(f"📊 Files: {summary['successful']}/{summary['total']} successful")
        self.output_text.append(f"💾 Results saved to: {self.output_folder}")
        
        self.set_buttons_enabled(True)
        self.stop_button.setEnabled(False)
    
    def save_all_results(self, format_type: str):
        """Save all files in specified format"""
        if not self.batch_results or not self.output_folder:
            return
        
        count = 0
        
        for file_path, result in self.batch_results.items():
            if not result.get('success'):
                continue
            
            try:
                filename = Path(file_path).stem
                output_path = Path(self.output_folder) / f"{filename}.{format_type}"
                
                if format_type == "txt":
                    content = result['text']
                elif format_type == "vtt":
                    # FIXED: Check if corrected VTT already exists on disk from post-processing
                    existing_vtt_path = Path(self.output_folder) / f"{filename}.vtt"
                    if existing_vtt_path.exists():
                        # Use the already-corrected VTT file instead of regenerating
                        logger.info(f"📋 Using existing corrected VTT for {filename}")
                        try:
                            with open(existing_vtt_path, 'r', encoding='utf-8-sig') as f:
                                content = f.read()
                            # If file already exists and has content, we're done
                            if content.strip():
                                count += 1
                                continue
                        except Exception as read_error:
                            logger.warning(f"Could not read existing VTT {filename}: {read_error}")
                    
                    # Fallback: generate from segments if no corrected VTT exists
                    segments = result.get('segments', [])
                    if segments:
                        content = SubtitleFormatter.create_vtt(segments, self.char_limit_spin.value())
                    else:
                        continue
                elif format_type == "srt":
                    segments = result.get('segments', [])
                    if segments:
                        content = SubtitleFormatter.create_srt(segments, self.char_limit_spin.value())
                    else:
                        continue
                
                with open(output_path, 'w', encoding='utf-8-sig') as f:
                    f.write(content)
                
                count += 1
                
            except Exception as e:
                logger.error(f"Save error for {Path(file_path).name}: {e}")
        
        if count > 0:
            self.status_label.setText(f"✅ Saved {count} {format_type.upper()} files")
        else:
            self.status_label.setText(f"ℹ️  No {format_type.upper()} files to save")
    
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
    
    def post_process_reconcile_vtt(self, txt_path: Path, vtt_path: Path) -> Optional[str]:
        """
        DISABLED: Post-processing reconciliation was causing word duplication
        The VTT generation already preserves all words correctly.
        """
        # DISABLED - This reconciliation logic was causing corruption by adding duplicate words
        # The word preservation is already handled correctly in the initial transcription
        return None
    
    def _parse_vtt_time(self, time_str: str) -> float:
        """Parse VTT time format to seconds"""
        try:
            # Format: HH:MM:SS.mmm
            parts = time_str.split(':')
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds_parts = parts[2].split('.')
                seconds = int(seconds_parts[0])
                milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                
                return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        except Exception:
            pass
        return 0.0
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format seconds to VTT time format"""
        try:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        except Exception:
            return "00:00:00.000"
    
    def _apply_word_pushing_format(self, text: str, max_chars_per_line: int) -> List[str]:
        """
        Apply word-pushing format to ensure character limit compliance.
        When a word causes overflow, move it to the next line. If that causes
        overflow again, continue moving words until all lines comply.
        """
        if not text or not text.strip():
            return []
        
        words = text.split()
        if not words:
            return []
        
        lines = []
        current_line = ""
        
        for word in words:
            # Check if adding this word would exceed the character limit
            test_line = f"{current_line} {word}" if current_line else word
            
            if len(test_line) <= max_chars_per_line:
                # Word fits, add it to current line
                current_line = test_line
            else:
                # Word would cause overflow, move to next line
                if current_line:  # Save current line if it has content
                    lines.append(current_line)
                
                # Start new line with the overflowing word
                current_line = word
                
                # Check if this single word exceeds limit (handle edge case)
                if len(current_line) > max_chars_per_line:
                    # Truncate oversized word as last resort
                    current_line = f"{word[:max_chars_per_line-3]}..."
        
        # Add final line if it has content
        if current_line:
            lines.append(current_line)
        
        # Now apply recursive word-pushing to ensure ALL lines comply
        final_lines = []
        
        for line in lines:
            if len(line) <= max_chars_per_line:
                final_lines.append(line)
            else:
                # Line exceeds limit - apply word-pushing within this line
                line_words = line.split()
                pushed_lines = []
                temp_line = ""
                
                for word in line_words:
                    test_line = f"{temp_line} {word}" if temp_line else word
                    
                    if len(test_line) <= max_chars_per_line:
                        temp_line = test_line
                    else:
                        # Word causes overflow - push to next line
                        if temp_line:
                            pushed_lines.append(temp_line)
                        
                        # Handle single oversized word
                        if len(word) > max_chars_per_line:
                            temp_line = f"{word[:max_chars_per_line-3]}..."
                        else:
                            temp_line = word
                
                if temp_line:
                    pushed_lines.append(temp_line)
                
                # Recursively check each pushed line for compliance
                for pushed_line in pushed_lines:
                    if len(pushed_line) <= max_chars_per_line:
                        final_lines.append(pushed_line)
                    else:
                        # Recursively apply word-pushing if still exceeds limit
                        recursive_lines = self._apply_word_pushing_format(pushed_line, max_chars_per_line)
                        final_lines.extend(recursive_lines)
        
        # Limit to max 2 lines for VTT compliance
        if len(final_lines) > 2:
            # Merge excess lines back into the last allowed line to preserve words
            excess_words = []
            for i in range(2, len(final_lines)):
                excess_words.extend(final_lines[i].split())
            
            final_lines = final_lines[:2]
            
            # Try to fit excess words back into existing lines
            for word in excess_words:
                added = False
                for i in range(len(final_lines)):
                    test_line = f"{final_lines[i]} {word}"
                    if len(test_line) <= max_chars_per_line:
                        final_lines[i] = test_line
                        added = True
                        break
                
                # If no space available, add to last line to preserve word
                if not added:
                    final_lines[-1] = f"{final_lines[-1]} {word}"
        
        return final_lines
    
    def _regenerate_vtt_with_updated_text(self, original_vtt: str, updated_segments: List[Dict]) -> str:
        """
        Regenerate VTT content with updated segment texts while preserving timing
        Uses word-pushing to ensure 42 character limit compliance
        """
        try:
            lines = original_vtt.split('\n')
            new_lines = []
            segment_idx = 0
            in_segment_text = False
            
            for line in lines:
                original_line = line
                line_stripped = line.strip()
                
                # Keep WEBVTT header and empty lines
                if not line_stripped or line_stripped.startswith('WEBVTT') or line_stripped.startswith('NOTE'):
                    new_lines.append(original_line)
                    continue
                
                # Check if it's a timestamp line
                if '-->' in line_stripped:
                    # Add timestamp line as-is
                    new_lines.append(original_line)
                    in_segment_text = True
                    
                elif in_segment_text and line_stripped:
                    # This is the start of segment text - replace with updated text
                    if segment_idx < len(updated_segments):
                        # Apply word-pushing formatting to ensure 42 character compliance
                        updated_text = updated_segments[segment_idx]['text']
                        formatted_lines = self._apply_word_pushing_format(updated_text, 42)
                        
                        # Add the formatted text (may be multiple lines)
                        for text_line in formatted_lines:
                            new_lines.append(text_line)
                        
                        segment_idx += 1
                        
                        # Skip original segment text lines until we hit empty line
                        in_segment_text = "skip"
                    else:
                        # No more updated segments, keep original
                        new_lines.append(original_line)
                        
                elif in_segment_text == "skip" and not line_stripped:
                    # End of segment we're skipping - add empty line and reset
                    new_lines.append("")
                    in_segment_text = False
                    
                elif in_segment_text == "skip":
                    # Skip original segment text lines
                    continue
                    
                elif in_segment_text and not line_stripped:
                    # Empty line - end of segment
                    new_lines.append("")
                    in_segment_text = False
                    
                else:
                    # Other lines, keep as-is
                    new_lines.append(original_line)
            
            return "\n".join(new_lines)
            
        except Exception as e:
            logger.error(f"Error regenerating VTT: {e}")
            return original_vtt  # Return original if regeneration fails
    
    def test_single_vtt(self, vtt_path: Path, txt_path: Path) -> bool:
        """Test a single VTT file automatically after generation"""
        try:
            # Import locally to avoid circular import
            from test_runner import CaptionerTestRunner
            
            runner = CaptionerTestRunner()
            
            # Test the specific file
            success, output = runner.test_existing_files(str(vtt_path), str(txt_path))
            
            # Extract key results from output
            test_lines = output.split('\n')
            failed_tests = []
            
            for line in test_lines:
                if '❌' in line and 'Test' in line:
                    # Extract the test name that failed
                    failed_tests.append(line.strip())
            
            # Display compact results
            if success:
                self.output_text.append(f"   ✓ All tests passed")
            else:
                self.output_text.append(f"   ✗ {len(failed_tests)} tests failed:")
                for failed in failed_tests[:3]:  # Show first 3 failed tests
                    self.output_text.append(f"      {failed}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error testing VTT: {e}")
            self.output_text.append(f"   ✗ Test error: {e}")
            return False
    
    def run_tests(self):
        """Run tests and display results in output window"""
        self.output_text.clear()
        self.status_label.setText("Running tests...")
        
        # Import locally to avoid circular import
        from test_runner import CaptionerTestRunner
        
        # Create test runner
        runner = CaptionerTestRunner()
        
        # Check what to test
        if self.output_folder and os.path.exists(self.output_folder):
            # Option 1: Test all files in output folder
            vtt_files = list(Path(self.output_folder).glob("*.vtt"))
            
            if vtt_files:
                self.output_text.append(f"📂 Testing {len(vtt_files)} files from output folder: {self.output_folder}\n")
                success, output = runner.test_output_folder(self.output_folder)
            else:
                # No VTT files in output folder, run synthetic tests
                self.output_text.append("📝 No VTT files found in output folder. Running synthetic tests...\n")
                success, output = runner.run_quick_tests()
        elif self.batch_results:
            # Option 2: Test files from current batch results
            self.output_text.append(f"📊 Testing {len(self.batch_results)} files from current batch\n")
            
            all_success = True
            output_lines = ["🧪 BATCH RESULTS VALIDATION", "="*60]
            
            for file_path, result in self.batch_results.items():
                if result.get('success'):
                    # Look for saved VTT/TXT files
                    filename = Path(file_path).stem
                    vtt_path = Path(self.output_folder) / f"{filename}.vtt"
                    txt_path = Path(self.output_folder) / f"{filename}.txt"
                    
                    if vtt_path.exists():
                        output_lines.append(f"\n📄 Testing: {filename}")
                        file_success, file_output = runner.test_existing_files(
                            str(vtt_path), 
                            str(txt_path) if txt_path.exists() else None
                        )
                        output_lines.append(file_output)
                        
                        if not file_success:
                            all_success = False
            
            success = all_success
            output = "\n".join(output_lines)
        else:
            # Option 3: Run synthetic tests
            self.output_text.append("📝 Running tests with synthetic data (no files to test)\n")
            success, output = runner.run_quick_tests()
        
        # Display results
        self.output_text.append(output)
        
        # Update status
        if success:
            self.status_label.setText("✅ All tests passed!")
        else:
            self.status_label.setText("❌ Some tests failed - see output for details")
        
        # Scroll to bottom to show summary
        cursor = self.output_text.textCursor()
        cursor.movePosition(cursor.End)
        self.output_text.setTextCursor(cursor)
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.thread and self.thread.isRunning():
            if self.worker:
                self.worker.stop_processing()
            self.thread.quit()
            self.thread.wait(5000)
        
        OptimizedModelManager().clear_model_aggressive()
        event.accept()

# ========================================
# CLI PROCESSING
# ========================================

def process_single_video_cli(video_path: str, output_folder: str, model_id: str = "small", 
                           max_chars_per_line: int = 42, max_chars_per_segment: int = 84,
                           generate_timestamps: bool = True) -> bool:
    """
    Process a single video file from command line
    Returns True if successful, False otherwise
    """
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize components
        model_manager = FasterWhisperModelManager()
        processor = TranscriptionProcessor()
        stats = ProcessingStats()
        stats.start_time = time.time()
        
        filename = os.path.basename(video_path)
        logger.info(f"🎬 Processing: {filename}")
        
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"❌ File not found: {video_path}")
            return False
        
        # Load model
        with model_manager.model_context(model_id) as model:
            if model is None:
                logger.error("❌ Failed to load AI model")
                return False
            
            # Extract audio
            logger.info("🎵 Extracting audio...")
            audio_path, duration = processor.extract_audio_optimized(video_path, stats)
            stats.audio_duration = duration
            logger.info(f"✅ Audio extracted: {duration:.2f}s")
            
            # Transcribe
            logger.info("🎙️ Transcribing...")
            text, segments = processor.transcribe_optimized(
                audio_path, model, generate_timestamps, max_chars_per_segment, stats
            )
            
            stats.characters_transcribed = len(text or "")
            stats.end_time = time.time()
            
            logger.info(f"✅ Transcription complete: {len(text or '')} chars, {len(segments)} segments")
            
            # Save outputs
            stem = Path(video_path).stem
            
            # Save TXT
            txt_path = Path(output_folder) / f"{stem}.txt"
            with open(txt_path, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            logger.info(f"💾 Saved transcript: {txt_path}")
            
            # Save VTT
            if segments and generate_timestamps:
                vtt_path = Path(output_folder) / f"{stem}.vtt"
                vtt_content = SubtitleFormatter.create_vtt(segments, max_chars_per_line)
                with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(vtt_content)
                logger.info(f"💾 Saved VTT: {vtt_path}")
                
                # Save SRT
                srt_path = Path(output_folder) / f"{stem}.srt"
                srt_content = SubtitleFormatter.create_srt(segments, max_chars_per_line)
                with open(srt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(srt_content)
                logger.info(f"💾 Saved SRT: {srt_path}")
            
            # Clean up
            processor.cleanup_temp_files()
            
            # Report stats
            logger.info(f"📊 Processing stats:")
            logger.info(f"   Total time: {stats.total_time:.2f}s")
            logger.info(f"   Audio duration: {stats.audio_duration:.2f}s")
            logger.info(f"   Speed: {stats.audio_duration / stats.total_time:.1f}x realtime")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        return False
    finally:
        # Ensure cleanup
        try:
            processor.cleanup_temp_files()
        except:
            pass

# ========================================
# MAIN FUNCTION
# ========================================

def main():
    """Main function with CLI and GUI support"""
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="High Performance Video Captioner - Generate subtitles from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video with default settings
  python captioner.py input.mp4 -o output_folder

  # Process with specific model and settings
  python captioner.py input.mp4 -o output_folder -m large-v3 --max-chars 50

  # Process without timestamps (transcript only)
  python captioner.py input.mp4 -o output_folder --no-timestamps

  # Launch GUI mode
  python captioner.py --gui
        """
    )
    
    # Add arguments
    parser.add_argument('input', nargs='?', help='Input MP4 file path')
    parser.add_argument('-o', '--output', help='Output folder for generated files')
    parser.add_argument('-m', '--model', default='small', 
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model to use (default: small)')
    parser.add_argument('--max-chars', type=int, default=42,
                       help='Maximum characters per line (default: 42)')
    parser.add_argument('--max-segment-chars', type=int, default=84,
                       help='Maximum characters per subtitle segment (default: 84)')
    parser.add_argument('--no-timestamps', action='store_true',
                       help='Generate transcript only without timestamps')
    parser.add_argument('--gui', action='store_true',
                       help='Launch GUI mode (default if no input file provided)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine mode
    if args.gui or (not args.input):
        # GUI mode
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        app.setApplicationName("High Performance Video Captioner - Refactored")
        app.setApplicationVersion("2.0")
        
        window = MainWindow()
        window.show()
        
        logger.info("🚀 High Performance Video Captioner v2.0 (GUI Mode)")
        OptimizedModelManager.get_optimal_device_config()
        
        sys.exit(app.exec_())
    else:
        # CLI mode
        if not args.output:
            parser.error("Output folder is required in CLI mode")
        
        logger.info("🚀 High Performance Video Captioner v2.0 (CLI Mode)")
        OptimizedModelManager.get_optimal_device_config()
        
        # Process the video
        success = process_single_video_cli(
            video_path=args.input,
            output_folder=args.output,
            model_id=args.model,
            max_chars_per_line=args.max_chars,
            max_chars_per_segment=args.max_segment_chars,
            generate_timestamps=not args.no_timestamps
        )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
