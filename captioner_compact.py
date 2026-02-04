# captioner_compact.py - Optimized Video Captioner with reduced redundancy
import sys
import os
import re
import subprocess
import tempfile
import shutil
import time
import gc
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging

# Configure logging EARLY
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(tempfile.gettempdir(), 'videocaptioner_debug.log'), mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Log startup information
logger.info("="*60)
logger.info("Video Captioner Starting...")
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {sys.platform}")
logger.info(f"Frozen: {getattr(sys, 'frozen', False)}")
if hasattr(sys, '_MEIPASS'):
    logger.info(f"PyInstaller temp dir: {sys._MEIPASS}")
logger.info("="*60)

# IMPORTANT: Import PyTorch BEFORE PyQt5 to avoid WinError 1114 on Windows
# See: https://github.com/pytorch/pytorch/issues/166628
# PyTorch 2.9.0 has a bug where importing after PyQt5 causes DLL initialization failures

# Import PyTorch with detailed error handling
try:
    logger.info("Importing PyTorch...")
    import torch
    logger.info(f"✓ PyTorch {torch.__version__} imported successfully")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("  Running in CPU-only mode")
except ImportError as e:
    logger.error(f"✗ Failed to import PyTorch: {e}")
    logger.error("This usually indicates missing DLL dependencies on Windows.")
    logger.error("Please ensure Visual C++ Redistributable 2019+ is installed:")
    logger.error("https://aka.ms/vs/17/release/vc_redist.x64.exe")
    
    # Create error dialog if PyQt5 is available
    try:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "PyTorch Import Error", 
            f"Failed to load PyTorch library:\n\n{str(e)}\n\n"
            "This is usually caused by missing Visual C++ Redistributable.\n"
            "Please install VC++ 2019 or later from:\n"
            "https://aka.ms/vs/17/release/vc_redist.x64.exe")
    except:
        pass
    sys.exit(1)
except OSError as e:
    logger.error(f"✗ OSError loading PyTorch DLLs: {e}")
    logger.error("DLL initialization failed. Possible causes:")
    logger.error("  1. Missing Visual C++ Redistributable")
    logger.error("  2. Corrupted PyTorch installation")
    logger.error("  3. Incompatible CUDA drivers (if using GPU)")
    
    if "WinError 1114" in str(e):
        logger.error("\nWinError 1114 detected - DLL initialization failure")
        logger.error("This is often caused by:")
        logger.error("  - UPX compression (should be disabled in build)")
        logger.error("  - Missing libiomp5md.dll or other Intel OpenMP libraries")
        logger.error("  - Incorrect DLL search paths")
    
    try:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "DLL Loading Error", 
            f"Failed to initialize PyTorch DLLs:\n\n{str(e)}\n\n"
            "Please ensure:\n"
            "1. Visual C++ Redistributable 2019+ is installed\n"
            "2. All files were extracted together (don't move .exe alone)\n"
            "3. Your antivirus isn't blocking the application")
    except:
        pass
    sys.exit(1)

# NOW import PyQt5 AFTER PyTorch (this is safe and avoids WinError 1114)
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                               QFileDialog, QProgressBar, QComboBox, QCheckBox,
                               QSpinBox, QGroupBox, QGridLayout, QListWidget, QMessageBox)
    from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThread
    from PyQt5.QtGui import QFont
    logger.info("✓ PyQt5 imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import PyQt5: {e}")
    sys.exit(1)

# Import faster-whisper with error handling
try:
    logger.info("Importing faster-whisper...")
    from faster_whisper import WhisperModel
    logger.info("✓ faster-whisper imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import faster-whisper: {e}")
    try:
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "Import Error", 
            f"Failed to load faster-whisper:\n\n{str(e)}")
    except:
        pass
    sys.exit(1)

# Import other dependencies
try:
    import librosa
    from functools import lru_cache
    from difflib import SequenceMatcher
    import csv
    logger.info("✓ All dependencies imported successfully")
except ImportError as e:
    logger.error(f"✗ Failed to import dependency: {e}")
    sys.exit(1)

logger.info("="*60)
logger.info("All imports completed successfully")
logger.info(f"Debug log file: {os.path.join(tempfile.gettempdir(), 'videocaptioner_debug.log')}")
logger.info("="*60)

# ========================================
# CONSTANTS & UTILITIES
# ========================================

DURATION_PATTERN = re.compile(r'Duration: (\d+):(\d+):(\d+)\.(\d+)')
SENTENCE_STARTERS = frozenset(['and', 'but', 'so', 'yet', 'or', 'nor', 'for', 'however', 'therefore'])
COMMON_ABBREVS = frozenset(['mr.', 'mrs.', 'dr.', 'prof.', 'inc.', 'ltd.', 'co.', 'corp.', 'etc.', 'vs.', 'jr.', 'sr.'])

def safe_float(value: Union[float, int, None], default: float = 0.0) -> float:
    """Safely convert value to float with default"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def merge_split_compound_words(word_level: List[Dict]) -> List[Dict]:
    """Merge split compound words like 'high' + '-end' → 'high-end'"""
    if len(word_level) < 2:
        return word_level
    
    merged = []
    i = 0
    
    while i < len(word_level):
        current_word = word_level[i].get('word', '').strip()
        
        # Fix internal compound word issues first
        if ' -' in current_word:
            fixed_word = current_word.replace(' -', '-')
            merged.append({
                'word': fixed_word,
                'start': safe_float(word_level[i].get('start'), 0),
                'end': safe_float(word_level[i].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
            })
            logger.info(f"🔧 Fixed compound word: '{current_word}' → '{fixed_word}'")
            i += 1
            continue
        
        # Check if we can merge with next word
        if i + 1 < len(word_level):
            next_word = word_level[i + 1].get('word', '').strip()
            
            # Fix next word if needed
            if ' -' in next_word:
                next_word = next_word.replace(' -', '-')
            
            merged_word = None
            merged_entry = None
            
            # Pattern matching for compound words
            if (next_word.startswith('-') and len(next_word) > 1) or \
               (current_word.endswith('-') and len(current_word) > 1) or \
               (current_word.endswith(' ') and next_word.startswith('-')) or \
               (current_word.replace(',', '').replace('.', '').isdigit() and next_word.startswith('.') and len(next_word) > 1) or \
               (len(current_word) <= 2 and next_word.startswith('-')):
                
                # Determine merged word
                if next_word.startswith('-') or current_word.endswith('-'):
                    merged_word = current_word.rstrip() + next_word if next_word.startswith('-') else current_word + next_word
                elif current_word.endswith('-') and not next_word.startswith('-'):
                    merged_word = current_word[:-1] + next_word  # Remove hyphen for line-break hyphenation
                else:
                    merged_word = current_word + next_word
                
                merged_entry = {
                    'word': merged_word,
                    'start': safe_float(word_level[i].get('start'), 0),
                    'end': safe_float(word_level[i + 1].get('end'), safe_float(word_level[i].get('start'), 0) + 0.7)
                }
            
            if merged_entry:
                merged.append(merged_entry)
                logger.info(f"🔗 Merged: '{current_word}' + '{next_word}' → '{merged_word}'")
                i += 2
                continue
        
        merged.append(word_level[i])
        i += 1
    
    # Final cleanup pass
    return [{**entry, 'word': entry['word'].replace(' -', '-')} for entry in merged]

# ========================================
# VOCABULARY CORRECTION
# ========================================

class VocabularyCorrector:
    """Unity-specific vocabulary correction system"""
    
    def __init__(self, csv_path: str, similarity_threshold: float = 0.85, enable_fallback: bool = True):
        """
        Initialize vocabulary corrector
        
        Args:
            csv_path: Path to CSV file with vocabulary terms
            similarity_threshold: Minimum similarity for fuzzy matching (0.0-1.0)
            enable_fallback: Enable title case fallback for unknown terms
        """
        self.csv_path = csv_path
        self.similarity_threshold = similarity_threshold
        self.enable_fallback = enable_fallback
        
        # Storage
        self.terms = []  # Original terms
        self.term_lookup = {}  # lowercase -> correct case mapping
        self.multi_word_terms = {}  # lowercase multi-word -> correct case
        self.fuzzy_cache = {}  # Cache for fuzzy matches
        self.correction_log = []  # Track corrections made
        
        # Load vocabulary
        self._load_csv()
        self._build_indices()
        
        logger.info(f"📚 Vocabulary loaded: {len(self.terms)} terms ({len(self.multi_word_terms)} multi-word)")
    
    def _load_csv(self):
        """Load terms from CSV file"""
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get('Term', '').strip()
                    if term and term not in ['Term', '']:  # Skip header duplicates
                        self.terms.append(term)
            
            logger.info(f"✅ Loaded {len(self.terms)} vocabulary terms from {self.csv_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load vocabulary file: {e}")
            self.terms = []
    
    def _build_indices(self):
        """Build lookup indices for fast matching"""
        for term in self.terms:
            # Single-word and multi-word indexing
            term_lower = term.lower()
            self.term_lookup[term_lower] = term
            
            # Index multi-word terms (2-3 words)
            words = term.split()
            if 2 <= len(words) <= 3:
                self.multi_word_terms[term_lower] = term
    
    def get_initial_prompt(self, max_terms: int = 50) -> str:
        """
        Generate initial prompt for Whisper with top Unity terms
        
        Args:
            max_terms: Maximum number of terms to include
            
        Returns:
            Context string for Whisper
        """
        if not self.terms:
            return ""
        
        # Prioritize common Unity terms (shorter, more frequently used)
        priority_terms = sorted(self.terms, key=lambda x: (len(x), x))[:max_terms]
        
        prompt = "This video discusses Unity game engine, including: " + ", ".join(priority_terms[:30])
        return prompt
    
    def correct_word(self, word: str) -> Tuple[str, str]:
        """
        Correct a single word with punctuation preservation
        
        Args:
            word: Word to correct (may include punctuation)
            
        Returns:
            Tuple of (corrected_word, correction_type)
            correction_type: 'exact', 'fuzzy', 'fallback', or 'none'
        """
        if not word:
            return word, 'none'
        
        # Separate punctuation
        leading_punct = ''
        trailing_punct = ''
        
        # Extract leading punctuation
        while word and not word[0].isalnum():
            leading_punct += word[0]
            word = word[1:]
        
        # Extract trailing punctuation
        while word and not word[-1].isalnum():
            trailing_punct = word[-1] + trailing_punct
            word = word[:-1]
        
        if not word:
            return leading_punct + trailing_punct, 'none'
        
        # Try exact match (case-insensitive)
        word_lower = word.lower()
        if word_lower in self.term_lookup:
            corrected = self.term_lookup[word_lower]
            return leading_punct + corrected + trailing_punct, 'exact'
        
        # Try fuzzy match (cached)
        cache_key = word_lower
        if cache_key in self.fuzzy_cache:
            cached_result = self.fuzzy_cache[cache_key]
            if cached_result:
                return leading_punct + cached_result + trailing_punct, 'fuzzy'
        else:
            # Perform fuzzy matching
            best_match = None
            best_score = 0.0
            
            for term_lower, correct_term in self.term_lookup.items():
                # Skip multi-word terms in single-word matching
                if ' ' in term_lower:
                    continue
                
                # Calculate similarity
                similarity = SequenceMatcher(None, word_lower, term_lower).ratio()
                
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = correct_term
            
            # Cache result
            self.fuzzy_cache[cache_key] = best_match
            
            if best_match:
                return leading_punct + best_match + trailing_punct, 'fuzzy'
        
        # Try title case fallback for Unity-like terms
        if self.enable_fallback and self._looks_like_unity_term(word):
            corrected = self._apply_title_case(word)
            return leading_punct + corrected + trailing_punct, 'fallback'
        
        # No correction
        return leading_punct + word + trailing_punct, 'none'
    
    def _looks_like_unity_term(self, word: str) -> bool:
        """Check if word looks like it should be a Unity term"""
        if len(word) < 3:
            return False
        
        # Check for camelCase or PascalCase patterns
        has_capitals = any(c.isupper() for c in word)
        has_lowercase = any(c.islower() for c in word)
        
        # Already properly cased
        if has_capitals and has_lowercase and word[0].isupper():
            return False
        
        # All lowercase single word that could be a class/component name
        if word.islower() and len(word) >= 5:
            # Common Unity suffixes/patterns
            unity_patterns = ['mesh', 'shader', 'texture', 'sprite', 'script', 'object', 
                            'system', 'manager', 'controller', 'renderer', 'collider']
            return any(pattern in word for pattern in unity_patterns)
        
        return False
    
    def _apply_title_case(self, word: str) -> str:
        """Apply intelligent title case to word"""
        # Handle special cases
        if word.lower() in ['vr', 'ar', 'xr', 'ui', 'api', 'fps', 'hdr', 'gpu', 'cpu', 'ai']:
            return word.upper()
        
        # PascalCase for compound-looking words
        if len(word) > 8 and not any(c.isupper() for c in word):
            # Try to detect word boundaries heuristically
            # For now, just capitalize first letter
            return word[0].upper() + word[1:]
        
        return word[0].upper() + word[1:]
    
    def correct_word_list(self, words: List[Dict]) -> List[Dict]:
        """
        Correct a list of word dictionaries with timestamps
        
        Args:
            words: List of dicts with 'word', 'start', 'end' keys
            
        Returns:
            Corrected word list with same structure
        """
        if not words:
            return words
        
        corrected_words = []
        self.correction_log = []  # Reset log
        i = 0
        
        while i < len(words):
            # Try multi-word matching first (2-3 words)
            multi_word_match = self._try_multi_word_match(words, i)
            
            if multi_word_match:
                corrected_entry, words_consumed = multi_word_match
                corrected_words.append(corrected_entry)
                i += words_consumed
                continue
            
            # Single word correction
            word_dict = words[i]
            original_word = word_dict.get('word', '').strip()
            
            if original_word:
                corrected_word, correction_type = self.correct_word(original_word)
                
                # Log correction if changed
                if corrected_word != original_word and correction_type != 'none':
                    self.correction_log.append({
                        'original': original_word,
                        'corrected': corrected_word,
                        'type': correction_type,
                        'position': i
                    })
                
                corrected_words.append({
                    'word': corrected_word,
                    'start': safe_float(word_dict.get('start'), 0),
                    'end': safe_float(word_dict.get('end'), 0)
                })
            else:
                corrected_words.append(word_dict)
            
            i += 1
        
        return corrected_words
    
    def _try_multi_word_match(self, words: List[Dict], index: int) -> Optional[Tuple[Dict, int]]:
        """
        Try to match 2-3 consecutive words as a multi-word term
        
        Args:
            words: Full word list
            index: Current position
            
        Returns:
            Tuple of (merged_word_dict, words_consumed) or None
        """
        # Try 3-word match first, then 2-word
        for word_count in [3, 2]:
            if index + word_count > len(words):
                continue
            
            # Get consecutive words
            word_group = words[index:index + word_count]
            combined_text = ' '.join([w.get('word', '').strip() for w in word_group])
            combined_lower = combined_text.lower()
            
            # Strip punctuation for matching
            combined_clean = combined_lower.strip('.,!?";:\'"')
            
            # Check if it's a known multi-word term
            if combined_clean in self.multi_word_terms:
                correct_term = self.multi_word_terms[combined_clean]
                
                # Log correction
                self.correction_log.append({
                    'original': combined_text,
                    'corrected': correct_term,
                    'type': 'multi-word',
                    'position': index,
                    'words_consumed': word_count
                })
                
                # Preserve trailing punctuation from last word
                last_word = word_group[-1].get('word', '')
                trailing_punct = ''
                while last_word and not last_word[-1].isalnum():
                    trailing_punct = last_word[-1] + trailing_punct
                    last_word = last_word[:-1]
                
                # Create merged entry
                merged_entry = {
                    'word': correct_term + trailing_punct,
                    'start': safe_float(word_group[0].get('start'), 0),
                    'end': safe_float(word_group[-1].get('end'), 0)
                }
                
                return (merged_entry, word_count)
        
        return None
    
    def get_correction_summary(self) -> str:
        """Get a summary of corrections made"""
        if not self.correction_log:
            return "No corrections made"
        
        summary_lines = [f"✏️  Vocabulary Corrections: {len(self.correction_log)} changes"]
        
        # Group by type
        by_type = {}
        for correction in self.correction_log:
            corr_type = correction['type']
            by_type.setdefault(corr_type, []).append(correction)
        
        # Show sample corrections by type
        for corr_type, corrections in by_type.items():
            count = len(corrections)
            samples = corrections[:3]  # Show first 3
            
            if corr_type == 'exact':
                summary_lines.append(f"  • {count} exact matches")
            elif corr_type == 'fuzzy':
                summary_lines.append(f"  • {count} fuzzy matches")
                for corr in samples:
                    summary_lines.append(f"    - '{corr['original']}' → '{corr['corrected']}'")
            elif corr_type == 'multi-word':
                summary_lines.append(f"  • {count} multi-word terms")
                for corr in samples:
                    summary_lines.append(f"    - '{corr['original']}' → '{corr['corrected']}'")
            elif corr_type == 'fallback':
                summary_lines.append(f"  • {count} fallback capitalizations")
                for corr in samples:
                    summary_lines.append(f"    - '{corr['original']}' → '{corr['corrected']}'")
        
        return '\n'.join(summary_lines)
    
    def log_corrections_verbose(self):
        """Log all corrections to logger"""
        if not self.correction_log:
            return
        
        logger.info("✏️  Vocabulary Corrections Applied:")
        for correction in self.correction_log:
            corr_type = correction['type']
            original = correction['original']
            corrected = correction['corrected']
            
            if corr_type == 'multi-word':
                logger.info(f"  🔗 Multi-word: '{original}' → '{corrected}'")
            elif corr_type == 'fuzzy':
                logger.info(f"  🔍 Fuzzy match: '{original}' → '{corrected}'")
            elif corr_type == 'exact':
                logger.info(f"  ✓ Exact: '{original}' → '{corrected}'")
            elif corr_type == 'fallback':
                logger.info(f"  📝 Fallback: '{original}' → '{corrected}'")

# ========================================
# DATA CLASSES
# ========================================

@dataclass
class ProcessingStats:
    """Track processing statistics"""
    start_time: float = 0
    end_time: float = 0
    audio_extraction_time: float = 0
    transcription_time: float = 0
    file_size: int = 0
    audio_duration: float = 0
    characters_transcribed: int = 0
    
    @property
    def total_time(self) -> float:
        return max(0, self.end_time - self.start_time) if self.end_time and self.start_time else 0

@dataclass
class WordInfo:
    """Word information with timestamps"""
    word: str
    start: float
    end: float
    
    @property
    def clean_word(self) -> str:
        return self.word.lower().strip('.,!?";:')
    
    def to_dict(self) -> Dict:
        return {"word": self.word, "start": self.start, "end": self.end}

# ========================================
# MODEL MANAGEMENT
# ========================================

class FasterWhisperModelManager:
    """Singleton model manager for faster-whisper"""
    _instance = None
    _model = None
    _current_model_id = None
    _current_device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @staticmethod
    def get_optimal_device_config():
        """Determine optimal device configuration"""
        config = {}
        
        if torch.cuda.is_available():
            config['device'] = "cuda"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            config['compute_type'] = "int8" if gpu_memory < 6 else "float16"
            logger.info(f"🚀 CUDA GPU: {torch.cuda.get_device_name()} ({gpu_memory:.1f}GB)")
        else:
            config['device'] = "cpu"
            config['compute_type'] = "int8"
            config['cpu_threads'] = min(8, os.cpu_count())
            logger.info(f"🖥️  CPU ({os.cpu_count()} cores, using {config['cpu_threads']} threads)")
        
        return config
    
    def load_model_optimized(self, model_id: str = "large-v3") -> bool:
        """Load faster-whisper model"""
        try:
            model_name = model_id.replace("openai/whisper-", "").replace("openai/", "")
            config = self.get_optimal_device_config()
            
            # Check if already loaded
            if (self._current_model_id == model_name and 
                self._current_device == config['device'] and
                self._model is not None):
                logger.info(f"✅ Model already loaded: {model_name}")
                return True
            
            self.clear_model()
            
            logger.info(f"🔄 Loading faster-whisper model: {model_name}")
            start_time = time.time()
            
            model_kwargs = {
                'device': config['device'],
                'compute_type': config['compute_type'],
            }
            
            if config['device'] == 'cpu':
                model_kwargs['cpu_threads'] = config.get('cpu_threads', 4)
            
            self._model = WhisperModel(model_name, **model_kwargs)
            self._current_model_id = model_name
            self._current_device = config['device']
            
            logger.info(f"✅ Model loaded in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.clear_model()
            return False
    
    def clear_model(self):
        """Clear model and free memory"""
        if self._model is not None:
            del self._model
            self._model = None
        
        self._current_model_id = None
        self._current_device = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model(self):
        return self._model
    
    @contextmanager
    def model_context(self, model_id: str):
        """Context manager for model usage"""
        try:
            if not self.load_model_optimized(model_id):
                raise RuntimeError(f"Failed to load model: {model_id}")
            yield self._model
        finally:
            pass  # Keep model loaded for batch processing

# ========================================
# AUDIO PROCESSING
# ========================================

class AudioProcessor:
    """Audio extraction and processing"""
    
    _ffmpeg_cache = {}
    _audio_cache = {}
    
    @staticmethod
    def find_ffmpeg():
        """Find FFmpeg executable with cross-platform support"""
        import platform
        
        if 'ffmpeg_path' in AudioProcessor._ffmpeg_cache:
            cached = AudioProcessor._ffmpeg_cache['ffmpeg_path']
            if cached and os.path.exists(cached):
                return cached
        
        paths = []
        is_windows = platform.system() == 'Windows'
        ffmpeg_name = 'ffmpeg.exe' if is_windows else 'ffmpeg'
        
        # Priority 1: Bundled FFmpeg (PyInstaller)
        if hasattr(sys, '_MEIPASS'):
            bundle_path = Path(sys._MEIPASS)
            paths.extend([
                bundle_path / ffmpeg_name,
                bundle_path / 'ffmpeg' / ffmpeg_name,
                bundle_path / 'bin' / ffmpeg_name,
            ])
        
        # Priority 2: Application directory (for portable installs)
        if getattr(sys, 'frozen', False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).parent
        
        paths.extend([
            app_dir / ffmpeg_name,
            app_dir / 'ffmpeg' / ffmpeg_name,
            app_dir / 'bin' / ffmpeg_name,
        ])
        
        # Priority 3: Platform-specific common locations
        if is_windows:
            # Windows common paths
            paths.extend([
                Path('C:/ffmpeg/bin/ffmpeg.exe'),
                Path('C:/Program Files/ffmpeg/bin/ffmpeg.exe'),
                Path('C:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe'),
            ])
            # Windows user paths
            local_app_data = os.environ.get('LOCALAPPDATA', '')
            if local_app_data:
                paths.extend([
                    Path(local_app_data) / 'Microsoft' / 'WinGet' / 'Links' / 'ffmpeg.exe',
                    Path(local_app_data) / 'Programs' / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
                ])
            # Chocolatey path
            choco_path = os.environ.get('ChocolateyInstall', 'C:\\ProgramData\\chocolatey')
            paths.append(Path(choco_path) / 'bin' / 'ffmpeg.exe')
        else:
            # macOS and Linux paths
            paths.extend([
                Path('/opt/homebrew/bin/ffmpeg'),  # macOS Apple Silicon
                Path('/usr/local/bin/ffmpeg'),      # macOS Intel / Homebrew
                Path('/usr/bin/ffmpeg'),            # Linux system
                Path('/snap/bin/ffmpeg'),           # Linux Snap
            ])
        
        # Test each path
        for path in paths:
            if path.exists():
                try:
                    # Use shell=True on Windows to handle .exe properly
                    result = subprocess.run(
                        [str(path), '-version'], 
                        capture_output=True, 
                        text=True, 
                        timeout=10,
                        creationflags=subprocess.CREATE_NO_WINDOW if is_windows else 0
                    )
                    if result.returncode == 0:
                        AudioProcessor._ffmpeg_cache['ffmpeg_path'] = str(path)
                        logger.info(f"✅ FFmpeg found: {path}")
                        return str(path)
                except Exception as e:
                    logger.debug(f"FFmpeg test failed for {path}: {e}")
                    continue
        
        # Fallback: Try system PATH
        system_ffmpeg = shutil.which('ffmpeg')
        if system_ffmpeg:
            try:
                result = subprocess.run(
                    [system_ffmpeg, '-version'], 
                    capture_output=True, 
                    text=True, 
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if is_windows else 0
                )
                if result.returncode == 0:
                    AudioProcessor._ffmpeg_cache['ffmpeg_path'] = system_ffmpeg
                    logger.info(f"✅ FFmpeg found in PATH: {system_ffmpeg}")
                    return system_ffmpeg
            except Exception as e:
                logger.debug(f"System FFmpeg test failed: {e}")
        
        logger.error("❌ FFmpeg not found. Please install FFmpeg and add it to PATH.")
        logger.error("   Windows: choco install ffmpeg  or  winget install ffmpeg")
        logger.error("   macOS: brew install ffmpeg")
        logger.error("   Linux: sudo apt install ffmpeg")
        return None
    
    @staticmethod
    def extract_audio(video_path: str, output_path: str = None) -> Tuple[str, float]:
        """Extract audio from video"""
        # Check cache
        try:
            cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
            if cache_key in AudioProcessor._audio_cache:
                cached_path, duration = AudioProcessor._audio_cache[cache_key]
                if os.path.exists(cached_path):
                    return cached_path, duration
        except:
            pass
        
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if not ffmpeg_path:
            raise RuntimeError("FFmpeg not found")
        
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        try:
            cmd = [
                ffmpeg_path, '-i', video_path, '-vn',
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-y', '-loglevel', 'error', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            duration = AudioProcessor.get_audio_duration(output_path)
            
            # Cache result
            try:
                cache_key = f"{video_path}:{os.path.getmtime(video_path)}"
                AudioProcessor._audio_cache[cache_key] = (output_path, duration)
            except:
                pass
            
            return output_path, duration
            
        except Exception as e:
            if os.path.exists(output_path):
                try:
                    os.unlink(output_path)
                except:
                    pass
            raise RuntimeError(f"Audio extraction failed: {str(e)}")
    
    @staticmethod
    def get_audio_duration(audio_path: str) -> float:
        """Get audio duration"""
        try:
            return safe_float(librosa.get_duration(path=audio_path), 0.0)
        except:
            # Fallback to FFmpeg
            ffmpeg_path = AudioProcessor.find_ffmpeg()
            if ffmpeg_path:
                try:
                    result = subprocess.run([ffmpeg_path, '-i', audio_path, '-f', 'null', '-'], 
                                          capture_output=True, text=True, timeout=10)
                    match = DURATION_PATTERN.search(result.stderr)
                    if match:
                        h, m, s, ms = map(int, match.groups())
                        return h * 3600 + m * 60 + s + ms / 100
                except:
                    pass
        return 0.0
    
    @staticmethod
    def is_audio_file(file_path: str) -> bool:
        """Check if file is audio"""
        return Path(file_path).suffix.lower() in {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.wma'}

# ========================================
# SUBTITLE FORMATTING
# ========================================

class SubtitleFormatter:
    """Subtitle formatting and segmentation"""
    
    MIN_DURATION = 1.25  # For normal segments
    MIN_DURATION_SINGLE_WORD = 0.4  # For single words
    MAX_DURATION = 8.0
    WORDS_PER_SECOND = 3.33
    MAX_CHARS_PER_SECOND = 25
    
    @staticmethod
    def format_text_lines(text: str, max_chars: int = 40, max_lines: int = 2) -> List[str]:
        """Format text into lines with word-pushing"""
        if not text or not text.strip():
            return []
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}" if current_line else word
            
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word[:max_chars-3] + "..." if len(word) > max_chars else word
        
        if current_line:
            lines.append(current_line)
        
        # Enforce max lines
        if len(lines) > max_lines:
            # Redistribute excess words
            excess_words = []
            for i in range(max_lines, len(lines)):
                excess_words.extend(lines[i].split())
            
            lines = lines[:max_lines]
            
            for word in excess_words:
                for i in range(len(lines)):
                    test_line = f"{lines[i]} {word}"
                    if len(test_line) <= max_chars:
                        lines[i] = test_line
                        break
        
        return lines
    
    @staticmethod
    def calculate_reading_time(text: str) -> float:
        """Calculate ideal reading time"""
        if not text:
            return SubtitleFormatter.MIN_DURATION
        
        words = len(text.split())
        base_time = words / SubtitleFormatter.WORDS_PER_SECOND
        char_time = len(text) * 0.05
        
        total_time = max(base_time, char_time)
        return max(SubtitleFormatter.MIN_DURATION, min(total_time, SubtitleFormatter.MAX_DURATION))
    
    @classmethod
    def clean_words(cls, words: List[Dict]) -> List[WordInfo]:
        """Clean and convert words to WordInfo objects"""
        if not words:
            return []
        
        word_infos = []
        
        for word_dict in words:
            word = word_dict.get("word", "").strip()
            if not word:
                continue
            
            word_infos.append(WordInfo(
                word=word,
                start=safe_float(word_dict.get("start"), 0),
                end=safe_float(word_dict.get("end"), 0)
            ))
        
        # Clean mid-sentence periods
        for i in range(len(word_infos) - 1):
            if word_infos[i].word.endswith('.'):
                next_word = word_infos[i + 1].word.lstrip('.,!?";:\'" ')
                if next_word and next_word[0].islower() and next_word.lower() not in SENTENCE_STARTERS:
                    word_infos[i].word = word_infos[i].word[:-1]
        
        return word_infos
    
    @classmethod
    def create_optimized_segments(cls, words: List[Dict], max_chars: int = 80) -> List[Dict]:
        """Create segments from words with optimal timing"""
        if not words:
            return []
        
        # Clean words first
        cleaned_words = cls.clean_words(words)
        if not cleaned_words:
            return []
        
        # Be more conservative with segment length to avoid splitting in VTT
        # Since VTT uses 42 chars per line and 2 lines, aim for ~70 chars max
        # to leave room for word wrapping
        safe_max_chars = min(max_chars, 70)
        
        segments = []
        current_segment = {
            "start": None,
            "end": None,
            "text": "",
            "word_count": 0
        }
        
        for word_info in cleaned_words:
            word = word_info.word
            word_start = word_info.start
            word_end = word_info.end
            
            # Build potential text
            potential_text = f"{current_segment['text']} {word}" if current_segment['text'] else word
            
            # Check if adding this word would create formatting issues
            # Test if it would fit in 2 lines of 42 chars each
            test_lines = cls.format_text_lines(potential_text, 42, 2)
            would_fit_properly = len(test_lines) <= 2 and all(len(line) <= 42 for line in test_lines)
            
            # Check if we should break
            would_exceed = len(potential_text) > safe_max_chars or not would_fit_properly
            time_gap = current_segment["end"] is not None and (word_start - current_segment["end"]) > 2.0
            
            # Sentence boundaries
            is_sentence_end = word.rstrip('",\')"]}').endswith(('.', '!', '?', '...')) and \
                            word.lower() not in COMMON_ABBREVS
            is_pause = word.rstrip().endswith((',', ';', ':', '--', '—'))
            
            # Duration check
            current_duration = 0
            if current_segment["start"] is not None:
                current_duration = word_end - current_segment["start"]
            
            # Natural break points - prefer breaking at punctuation
            natural_break = is_sentence_end or (is_pause and current_segment['word_count'] >= 5)
            
            should_break = (
                (would_exceed and current_segment['text']) or
                (time_gap and current_segment['text']) or
                (natural_break and len(current_segment['text']) > 20) or
                (current_duration > 5.0 and len(current_segment['text']) > 40)
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
        """Post-process segments for timing, merging, and deduplication"""
        if not segments:
            return []
        
        # Sort by start time
        segments.sort(key=lambda x: safe_float(x.get("start"), 0))
        
        # Remove duplicates and adjust timing
        processed = []
        seen_content = set()
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0)
            end_time = safe_float(segment.get("end"), start_time + 0.5)
            
            # Deduplicate
            content_hash = f"{text[:50]}_{start_time:.1f}"
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Calculate ideal duration for comparison
            ideal_duration = cls.calculate_reading_time(text)
            actual_duration = end_time - start_time
            
            # Determine appropriate minimum duration based on content
            word_count = len(text.split())
            if word_count <= 1:
                min_duration = cls.MIN_DURATION_SINGLE_WORD
            elif word_count <= 2:
                min_duration = 0.8
            else:
                min_duration = cls.MIN_DURATION
            
            # Only adjust if the actual duration is problematic
            if actual_duration < min_duration:
                # Too short - extend to appropriate minimum
                end_time = start_time + min_duration
            elif actual_duration > cls.MAX_DURATION:
                # Too long - cap at maximum
                end_time = start_time + cls.MAX_DURATION
            
            # Check for gaps with next segment
            if i + 1 < len(segments):
                next_start = safe_float(segments[i + 1].get("start"), end_time + 0.1)
                gap = next_start - end_time
                
                # Only extend if there's a small gap and we're not making it too long
                if gap > 0.1 and gap < 1.0:
                    proposed_end = next_start - 0.05
                    # Don't extend beyond reasonable reading time
                    if proposed_end - start_time <= ideal_duration * 1.5:
                        end_time = proposed_end
            
            # Prevent overlap with previous
            if processed:
                prev_end = safe_float(processed[-1].get("end"), 0)
                if start_time < prev_end:
                    start_time = prev_end + 0.05
                    # Maintain the actual duration when shifting
                    end_time = start_time + actual_duration
            
            segment["start"] = start_time
            segment["end"] = end_time
            processed.append(segment)
        
        # Merge short segments
        processed = cls.merge_short_segments(processed)
        
        # Enforce reading rate
        processed = cls.enforce_reading_rate(processed)
        
        # Clean punctuation
        return cls.clean_segment_punctuation(processed)
    
    @classmethod
    def merge_short_segments(cls, segments: List[Dict]) -> List[Dict]:
        """Merge segments with very few words"""
        if len(segments) <= 1:
            return segments
        
        MIN_WORDS = 3
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            current_text = current.get("text", "").strip()
            current_words = current_text.split()
            
            # If segment has enough words, keep it
            if len(current_words) >= MIN_WORDS:
                merged.append(current)
                i += 1
                continue
            
            # Short segment - must merge with neighbors
            merged_with_prev = False
            merged_with_next = False
            
            # Try merging with previous segment first
            if merged:
                prev_text = merged[-1].get("text", "")
                combined = prev_text + " " + current_text
                
                # Check if it formats correctly into 2 lines
                lines = cls.format_text_lines(combined, 42, 2)
                
                if len(lines) <= 2:
                    # Merge with previous
                    merged[-1]["text"] = combined
                    merged[-1]["end"] = current.get("end")
                    merged_with_prev = True
                    i += 1
                    continue
            
            # If couldn't merge with previous, try next segment
            if not merged_with_prev and i + 1 < len(segments):
                next_segment = segments[i + 1]
                next_text = next_segment.get("text", "")
                combined = current_text + " " + next_text
                
                # Check if it formats correctly into 2 lines
                lines = cls.format_text_lines(combined, 42, 2)
                
                if len(lines) <= 2:
                    # Merge current with next
                    merged.append({
                        "start": current.get("start"),
                        "end": next_segment.get("end"),
                        "text": combined
                    })
                    merged_with_next = True
                    i += 2
                    continue
            
            # If it's the last segment and couldn't merge, try harder to merge with previous
            if i == len(segments) - 1 and merged and not merged_with_prev:
                # Force merge with previous even if it exceeds ideal formatting
                prev_text = merged[-1].get("text", "")
                combined = prev_text + " " + current_text
                
                # Check with relaxed line limit for orphaned words
                lines = cls.format_text_lines(combined, 42, 3)  # Allow 3 lines for edge cases
                
                if len(lines) <= 3:
                    # Still merge to avoid orphaned single words
                    merged[-1]["text"] = combined
                    merged[-1]["end"] = current.get("end")
                    i += 1
                    continue
            
            # Only keep as standalone if we absolutely can't merge
            # This should be rare - only for very long surrounding segments
            merged.append(current)
            i += 1
        
        return merged
    
    @classmethod
    def enforce_reading_rate(cls, segments: List[Dict]) -> List[Dict]:
        """Ensure reading rate doesn't exceed limit"""
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start = safe_float(segment.get("start"), 0.0)
            end = safe_float(segment.get("end"), start + cls.MIN_DURATION)
            actual_duration = end - start
            
            # Calculate required duration based on reading speed
            char_count = len(text)
            word_count = len(text.split())
            
            # Dynamic minimum based on content
            if word_count <= 1:
                content_min = cls.MIN_DURATION_SINGLE_WORD
            elif word_count <= 2:
                content_min = 0.8
            else:
                content_min = cls.MIN_DURATION
            
            min_required_duration = char_count / cls.MAX_CHARS_PER_SECOND
            min_required_duration = max(min_required_duration, content_min)
            
            # Only extend if the actual duration is too short for comfortable reading
            if actual_duration < min_required_duration:
                # Try to preserve the original ratio if we need to extend
                segment["end"] = start + min_required_duration
            
            # Fix overlaps without changing duration unnecessarily
            if i > 0:
                prev_end = safe_float(segments[i-1].get("end"), 0.0)
                if segment["start"] < prev_end:
                    overlap = prev_end - segment["start"]
                    segment["start"] = prev_end + 0.05
                    # Maintain duration when shifting
                    segment["end"] = segment["start"] + actual_duration
        
        return segments
    
    @classmethod
    def clean_segment_punctuation(cls, segments: List[Dict]) -> List[Dict]:
        """Remove inappropriate periods at segment boundaries"""
        if len(segments) <= 1:
            return segments
        
        for i in range(len(segments) - 1):
            current_text = segments[i].get("text", "").strip()
            next_text = segments[i + 1].get("text", "").strip()
            
            if current_text.endswith('.') and next_text:
                first_word = next_text.split()[0].strip('.,!?";:\'"')
                
                if first_word and first_word[0].islower() and first_word.lower() not in SENTENCE_STARTERS:
                    segments[i]["text"] = current_text[:-1].rstrip()
        
        return segments
    
    @classmethod
    def create_vtt(cls, segments: List[Dict], max_chars_per_line: int = 40) -> str:
        """Create VTT format"""
        if not segments:
            return "WEBVTT\n\n"
        
        vtt_lines = ["WEBVTT", ""]
        
        for segment in sorted(segments, key=lambda x: safe_float(x.get("start"), 0)):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0.0)
            end_time = safe_float(segment.get("end"), start_time + cls.MIN_DURATION)
            
            # Format text into lines
            lines = cls.format_text_lines(text, max_chars_per_line, 2)
            if not lines:
                continue
            
            # Check if needs splitting for word order preservation
            words = text.split()
            max_subtitle_chars = max_chars_per_line * 2 * 0.9  # 90% capacity
            
            if len(text) > max_subtitle_chars:
                # Split into multiple subtitles, avoiding single-word segments
                sub_segments = []
                current_words = []
                current_length = 0
                
                for i, word in enumerate(words):
                    word_len = len(word)
                    space_needed = 1 if current_words else 0
                    
                    if current_words and current_length + space_needed + word_len > max_subtitle_chars:
                        # Before creating a new segment, check if we'd be leaving a single word
                        remaining_words = words[i:]
                        
                        # If we're about to create a segment and there's only one word left,
                        # try to fit it in the current segment or merge with previous
                        if len(remaining_words) == 1:
                            # Try to keep it with current words if possible
                            test_text = ' '.join(current_words + [word])
                            test_lines = cls.format_text_lines(test_text, max_chars_per_line, 3)
                            if len(test_lines) <= 3:  # Allow 3 lines for edge cases
                                current_words.append(word)
                                break
                        
                        sub_segments.append(current_words)
                        current_words = [word]
                        current_length = word_len
                    else:
                        current_words.append(word)
                        current_length += space_needed + word_len
                
                if current_words:
                    # If the last segment would be a single word, merge it with previous
                    if len(current_words) == 1 and sub_segments:
                        # Try to merge with the last sub-segment
                        last_segment = sub_segments[-1]
                        combined = last_segment + current_words
                        combined_text = ' '.join(combined)
                        
                        # Check if it fits reasonably
                        test_lines = cls.format_text_lines(combined_text, max_chars_per_line, 3)
                        if len(test_lines) <= 3:
                            sub_segments[-1] = combined
                        else:
                            # If can't merge, at least ensure minimum 2 words
                            if len(last_segment) > 3:
                                # Take a word from the previous segment
                                word_to_move = last_segment.pop()
                                current_words.insert(0, word_to_move)
                            sub_segments.append(current_words)
                    else:
                        sub_segments.append(current_words)
                
                # Final pass: ensure no single-word segments
                final_segments = []
                for i, word_group in enumerate(sub_segments):
                    if len(word_group) == 1 and i > 0:
                        # Try to merge with previous
                        final_segments[-1].extend(word_group)
                    elif len(word_group) == 1 and i < len(sub_segments) - 1:
                        # Merge with next
                        sub_segments[i + 1] = word_group + sub_segments[i + 1]
                    else:
                        final_segments.append(word_group)
                
                sub_segments = final_segments
                
                # Calculate proper timing for each sub-segment based on word count
                # This gives more natural timing than equal division
                segment_duration = end_time - start_time
                total_words = len(words)
                
                current_time = start_time
                for j, word_group in enumerate(sub_segments):
                    # Calculate duration based on proportion of words
                    word_proportion = len(word_group) / total_words
                    sub_duration = segment_duration * word_proportion
                    
                    # Ensure minimum duration for readability
                    if len(word_group) == 1:
                        sub_duration = max(sub_duration, cls.MIN_DURATION_SINGLE_WORD)
                    elif len(word_group) <= 2:
                        sub_duration = max(sub_duration, 0.8)
                    else:
                        sub_duration = max(sub_duration, 1.0)
                    
                    sub_start = current_time
                    sub_end = current_time + sub_duration
                    
                    # Ensure we don't exceed the segment's end time on the last sub-segment
                    if j == len(sub_segments) - 1:
                        sub_end = end_time
                    
                    sub_text = ' '.join(word_group)
                    sub_lines = cls.format_text_lines(sub_text, max_chars_per_line, 2)
                    
                    if sub_lines:
                        formatted_text = '\n'.join(sub_lines)
                        line_count = len(sub_lines)
                        position = "align:middle line:90%" if line_count == 1 else "align:middle line:84%"
                        
                        vtt_lines.extend([
                            f"{cls.seconds_to_vtt_time(sub_start)} --> {cls.seconds_to_vtt_time(sub_end)} {position}",
                            formatted_text,
                            ""
                        ])
                    
                    current_time = sub_end
            else:
                # Single subtitle
                formatted_text = '\n'.join(lines)
                line_count = len(lines)
                position = "align:middle line:90%" if line_count == 1 else "align:middle line:84%"
                
                vtt_lines.extend([
                    f"{cls.seconds_to_vtt_time(start_time)} --> {cls.seconds_to_vtt_time(end_time)} {position}",
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
            
            lines = cls.format_text_lines(text, max_chars_per_line, 2)
            if not lines:
                continue
            
            srt_lines.extend([
                str(counter),
                f"{cls.seconds_to_srt_time(start_time)} --> {cls.seconds_to_srt_time(end_time)}",
                '\n'.join(lines),
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
        except:
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
        except:
            return "00:00:00,000"

# ========================================
# TRANSCRIPTION PROCESSING
# ========================================

class TranscriptionProcessor:
    """Transcription processing with faster-whisper"""
    
    def __init__(self, vocabulary_corrector: Optional[VocabularyCorrector] = None):
        self.model_manager = FasterWhisperModelManager()
        self.temp_files = []
        self.vocabulary_corrector = vocabulary_corrector
    
    def transcribe(self, audio_path: str, model, return_timestamps: bool, 
                  max_chars: int, file_stats: ProcessingStats,
                  initial_prompt: str = None) -> Tuple[str, List[Dict]]:
        """Transcribe audio with faster-whisper"""
        transcription_start = time.time()
        
        try:
            # Configure options
            transcribe_kwargs = {}
            if return_timestamps:
                transcribe_kwargs['word_timestamps'] = True
            
            # Add custom initial prompt if provided
            if initial_prompt and initial_prompt.strip():
                transcribe_kwargs['initial_prompt'] = initial_prompt.strip()
                logger.info(f"📚 Using custom context prompt for transcription")
            # Add vocabulary guidance if available (fallback)
            elif self.vocabulary_corrector:
                vocab_prompt = self.vocabulary_corrector.get_initial_prompt()
                if vocab_prompt:
                    transcribe_kwargs['initial_prompt'] = vocab_prompt
                    logger.info("📚 Using vocabulary guidance for transcription")
            
            logger.info(f"🎙️ Transcribing with faster-whisper...")
            
            # Transcribe
            segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)
            transcribed_segments = list(segments_iter)
            
            file_stats.transcription_time = time.time() - transcription_start
            
            # Extract text
            text = " ".join([segment.text.strip() for segment in transcribed_segments if segment.text.strip()])
            
            segments = []
            
            if return_timestamps and transcribed_segments:
                # Extract words
                words = []
                
                for segment in transcribed_segments:
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            words.append({
                                'word': word.word,
                                'start': safe_float(word.start, 0.0),
                                'end': safe_float(word.end, safe_float(word.start, 0.0) + 0.5)
                            })
                    else:
                        # Fallback: artificial word timestamps
                        segment_text = segment.text.strip()
                        if segment_text:
                            segment_words = segment_text.split()
                            segment_duration = safe_float(segment.end, 0.0) - safe_float(segment.start, 0.0)
                            word_duration = segment_duration / max(len(segment_words), 1)
                            
                            current_time = safe_float(segment.start, 0.0)
                            for word in segment_words:
                                words.append({
                                    'word': word,
                                    'start': current_time,
                                    'end': current_time + word_duration
                                })
                                current_time += word_duration
                
                # Apply vocabulary correction if available (before merging compounds)
                if words and self.vocabulary_corrector:
                    logger.info("📚 Applying vocabulary corrections...")
                    words = self.vocabulary_corrector.correct_word_list(words)
                    
                    # Log corrections
                    if self.vocabulary_corrector.correction_log:
                        self.vocabulary_corrector.log_corrections_verbose()
                        summary = self.vocabulary_corrector.get_correction_summary()
                        logger.info(summary)
                
                # Merge compound words
                if words:
                    words = merge_split_compound_words(words)
                    logger.info(f"✅ Extracted {len(words)} words")
                    
                    # Create segments
                    segments = SubtitleFormatter.create_optimized_segments(words, max_chars)
                    
                    # Fix compound words in segments
                    for segment in segments:
                        if segment.get('text'):
                            segment['text'] = segment['text'].replace(' -', '-')
            
            logger.info(f"✅ Transcription complete: {len(text)} chars, {len(segments)} segments")
            return text, segments
            
        except Exception as e:
            file_stats.transcription_time = time.time() - transcription_start
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def extract_audio(self, video_path: str, file_stats: ProcessingStats) -> Tuple[str, float]:
        """Extract audio with stats tracking"""
        if AudioProcessor.is_audio_file(video_path):
            duration = AudioProcessor.get_audio_duration(video_path)
            return video_path, duration
        
        extraction_start = time.time()
        audio_path, duration = AudioProcessor.extract_audio(video_path)
        file_stats.audio_extraction_time = time.time() - extraction_start
        
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
# BATCH PROCESSING
# ========================================

class BatchWorker(QObject):
    """Batch processing worker"""
    
    progress_updated = pyqtSignal(int)
    file_progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    current_file_updated = pyqtSignal(str)
    file_completed = pyqtSignal(str, str, list, object)
    batch_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.model_manager = FasterWhisperModelManager()
        self.processor = TranscriptionProcessor()
        self.should_stop = False
    
    def stop_processing(self):
        self.should_stop = True
    
    def transcribe_batch(self, file_paths: List[str], model_id: str, 
                        return_timestamps: bool = True, max_chars_per_segment: int = 80,
                        initial_prompt: str = None, vocab_settings: dict = None):
        """Process batch of files
        
        Args:
            file_paths: List of file paths to process
            model_id: Whisper model ID
            return_timestamps: Whether to generate timestamps
            max_chars_per_segment: Maximum characters per subtitle segment
            initial_prompt: Context prompt for Whisper
            vocab_settings: Dictionary with vocabulary correction settings:
                - enabled: bool
                - csv_path: str (path to vocabulary CSV)
                - sensitivity: float (0.0-1.0)
                - title_case_fallback: bool
        """
        results = {}
        successful = 0
        failed = 0
        total_start_time = time.time()
        
        # Initialize vocabulary corrector if enabled
        vocabulary_corrector = None
        if vocab_settings and vocab_settings.get('enabled') and vocab_settings.get('csv_path'):
            try:
                vocabulary_corrector = VocabularyCorrector(
                    csv_path=vocab_settings['csv_path'],
                    similarity_threshold=vocab_settings.get('sensitivity', 0.85),
                    enable_fallback=vocab_settings.get('title_case_fallback', True)
                )
                self.processor.vocabulary_corrector = vocabulary_corrector
                logger.info(f"📚 Vocabulary correction enabled with {len(vocabulary_corrector.terms)} terms")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize vocabulary corrector: {e}")
        
        try:
            logger.info(f"🚀 Starting batch transcription of {len(file_paths)} files")
            if initial_prompt:
                logger.info(f"📚 Using context prompt: {initial_prompt[:50]}...")
            
            self.status_updated.emit("Loading AI model...")
            with self.model_manager.model_context(model_id) as model:
                if model is None:
                    raise RuntimeError("Failed to load model")
                
                logger.info("✅ Model loaded successfully")
                
                for i, file_path in enumerate(file_paths):
                    if self.should_stop:
                        break
                    
                    file_stats = ProcessingStats()
                    file_stats.start_time = time.time()
                    
                    try:
                        file_stats.file_size = os.path.getsize(file_path)
                    except:
                        file_stats.file_size = 0
                    
                    filename = os.path.basename(file_path)
                    logger.info(f"📁 Processing {i+1}/{len(file_paths)}: {filename}")
                    
                    try:
                        self.current_file_updated.emit(f"Processing: {filename}")
                        self.file_progress_updated.emit(0)
                        
                        overall_progress = int(10 + (i / len(file_paths)) * 85)
                        self.progress_updated.emit(overall_progress)
                        
                        # Extract audio
                        audio_path, duration = self.processor.extract_audio(file_path, file_stats)
                        file_stats.audio_duration = duration
                        
                        self.file_progress_updated.emit(30)
                        
                        # Transcribe with context prompt
                        text, segments = self.processor.transcribe(audio_path, model, 
                                                                  return_timestamps, max_chars_per_segment, 
                                                                  file_stats, initial_prompt)
                        
                        file_stats.characters_transcribed = len(text or "")
                        
                        self.file_progress_updated.emit(100)
                        
                        results[file_path] = {
                            'text': text,
                            'segments': segments,
                            'success': True,
                            'stats': file_stats
                        }
                        
                        file_stats.end_time = time.time()
                        successful += 1
                        
                        logger.info(f"✅ {filename}: {len(text or '')} chars, {len(segments)} segments")
                        
                        self.file_completed.emit(file_path, text or "", segments, file_stats)
                        
                        self.processor.cleanup_temp_files()
                        
                    except Exception as e:
                        error_msg = f"Processing failed for {filename}: {str(e)}"
                        logger.error(f"❌ {error_msg}")
                        results[file_path] = {'error': error_msg, 'success': False}
                        failed += 1
                        self.error_occurred.emit(error_msg, file_path)
            
            total_time = time.time() - total_start_time
            
            logger.info(f"🏁 Batch complete: {successful}/{len(file_paths)} successful in {total_time:.2f}s")
            
            summary = {
                'total': len(file_paths),
                'successful': successful,
                'failed': failed,
                'results': results
            }
            
            self.batch_finished.emit(summary)
            
        except Exception as e:
            error_msg = f"Critical batch error: {str(e)}"
            logger.error(f"💥 {error_msg}")
            self.error_occurred.emit(error_msg, "")
        finally:
            self.processor.cleanup_temp_files()
    
    def setup_environment(self):
        """Setup processing environment"""
        ffmpeg_path = AudioProcessor.find_ffmpeg()
        if ffmpeg_path:
            for var in ['FFMPEG_BINARY', 'AUDIOREAD_FFMPEG_EXE', 'FFMPEG_EXECUTABLE']:
                os.environ[var] = ffmpeg_path
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        torch.set_num_threads(min(8, os.cpu_count()))

# ========================================
# MAIN GUI
# ========================================

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.file_paths = []
        self.output_folder = ""
        self.batch_results = {}
        self.thread = None
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Video Captioner - Compact Version")
        self.setGeometry(100, 100, 1000, 700)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Title
        title_label = QLabel("Video Captioner (Compact)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title_label)
        
        # Configuration
        config_group = QGroupBox("Configuration")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("AI Model:"), 0, 0)
        self.model_combo = QComboBox()
        models = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText("small")
        config_layout.addWidget(self.model_combo, 0, 1)
        
        config_layout.addWidget(QLabel("Max chars/line:"), 1, 0)
        self.char_limit_spin = QSpinBox()
        self.char_limit_spin.setRange(20, 100)
        self.char_limit_spin.setValue(42)
        config_layout.addWidget(self.char_limit_spin, 1, 1)
        
        self.timestamps_check = QCheckBox("Generate timestamps")
        self.timestamps_check.setChecked(True)
        config_layout.addWidget(self.timestamps_check, 2, 0, 1, 2)
        
        # Context prompt input
        config_layout.addWidget(QLabel("Context Prompt (optional):"), 3, 0, 1, 2)
        self.context_prompt_input = QTextEdit()
        self.context_prompt_input.setPlaceholderText(
            "e.g., This video is about the Unity game engine, covering GameObjects, Prefabs, and C# scripting..."
        )
        self.context_prompt_input.setMaximumHeight(60)
        self.context_prompt_input.setToolTip(
            "Provide context to help faster-whisper better recognize domain-specific vocabulary.\n"
            "Example: 'This video covers Unity game engine topics including Rigidbody, Animator, and NavMesh.'"
        )
        config_layout.addWidget(self.context_prompt_input, 4, 0, 1, 2)
        
        context_help_label = QLabel("💡 Helps improve accuracy for specialized vocabulary")
        context_help_label.setStyleSheet("color: gray; font-size: 11px;")
        config_layout.addWidget(context_help_label, 5, 0, 1, 2)
        
        layout.addWidget(config_group)
        
        # Vocabulary Correction section
        vocab_group = QGroupBox("Vocabulary Correction (Post-Processing)")
        vocab_layout = QGridLayout(vocab_group)
        
        # Enable checkbox
        self.vocab_correction_check = QCheckBox("Enable vocabulary correction")
        self.vocab_correction_check.setToolTip(
            "Apply post-processing vocabulary correction using a CSV word list.\n"
            "This corrects transcription errors after the AI generates the initial text."
        )
        self.vocab_correction_check.stateChanged.connect(self.on_vocab_correction_toggled)
        vocab_layout.addWidget(self.vocab_correction_check, 0, 0, 1, 2)
        
        # CSV file picker
        vocab_layout.addWidget(QLabel("Vocabulary CSV:"), 1, 0)
        self.vocab_file_layout = QHBoxLayout()
        self.vocab_file_label = QLabel("No file selected")
        self.vocab_file_label.setStyleSheet("color: gray;")
        self.vocab_file_button = QPushButton("📂 Select CSV")
        self.vocab_file_button.clicked.connect(self.select_vocab_file)
        self.vocab_file_button.setEnabled(False)
        self.vocab_file_layout.addWidget(self.vocab_file_label, 1)
        self.vocab_file_layout.addWidget(self.vocab_file_button)
        vocab_layout.addLayout(self.vocab_file_layout, 1, 1)
        
        # Sensitivity slider
        vocab_layout.addWidget(QLabel("Fuzzy Match Sensitivity:"), 2, 0)
        sensitivity_layout = QHBoxLayout()
        self.sensitivity_slider = QSpinBox()
        self.sensitivity_slider.setRange(70, 100)
        self.sensitivity_slider.setValue(85)
        self.sensitivity_slider.setSuffix("%")
        self.sensitivity_slider.setToolTip(
            "Minimum similarity for fuzzy matching (70-100%).\n"
            "Higher = stricter matching (fewer false positives)\n"
            "Lower = looser matching (catches more variations)"
        )
        self.sensitivity_slider.setEnabled(False)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        
        self.sensitivity_label = QLabel("(85% recommended)")
        self.sensitivity_label.setStyleSheet("color: gray; font-size: 11px;")
        sensitivity_layout.addWidget(self.sensitivity_label)
        sensitivity_layout.addStretch()
        vocab_layout.addLayout(sensitivity_layout, 2, 1)
        
        # Title case fallback
        self.title_case_fallback_check = QCheckBox("Enable title case fallback for unknown terms")
        self.title_case_fallback_check.setChecked(True)
        self.title_case_fallback_check.setEnabled(False)
        self.title_case_fallback_check.setToolTip(
            "Capitalize words that look like Unity terms but aren't in the vocabulary list."
        )
        vocab_layout.addWidget(self.title_case_fallback_check, 3, 0, 1, 2)
        
        vocab_help_label = QLabel("💡 Use after context prompting if vocabulary still needs correction")
        vocab_help_label.setStyleSheet("color: gray; font-size: 11px;")
        vocab_layout.addWidget(vocab_help_label, 4, 0, 1, 2)
        
        # Store vocabulary file path
        self.vocab_file_path = None
        
        layout.addWidget(vocab_group)
        
        # File selection
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout(file_group)
        
        file_buttons_layout = QHBoxLayout()
        
        self.select_files_button = QPushButton("📁 Select Files")
        self.select_files_button.clicked.connect(self.select_files)
        
        self.clear_files_button = QPushButton("🗑️ Clear")
        self.clear_files_button.clicked.connect(self.clear_files)
        
        file_buttons_layout.addWidget(self.select_files_button)
        file_buttons_layout.addWidget(self.clear_files_button)
        file_layout.addLayout(file_buttons_layout)
        
        # Output folder
        output_layout = QHBoxLayout()
        self.output_folder_label = QLabel("No output folder selected")
        self.select_output_button = QPushButton("📋 Output Folder")
        self.select_output_button.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(QLabel("Output:"))
        output_layout.addWidget(self.output_folder_label, 1)
        output_layout.addWidget(self.select_output_button)
        file_layout.addLayout(output_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        file_layout.addWidget(QLabel("Selected Files:"))
        file_layout.addWidget(self.file_list)
        
        layout.addWidget(file_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.overall_progress = QProgressBar()
        self.overall_progress.setVisible(False)
        progress_layout.addWidget(QLabel("Overall Progress:"))
        progress_layout.addWidget(self.overall_progress)
        
        self.file_progress = QProgressBar()
        self.file_progress.setVisible(False)
        progress_layout.addWidget(QLabel("Current File:"))
        progress_layout.addWidget(self.file_progress)
        
        self.current_file_label = QLabel("Ready to process")
        progress_layout.addWidget(self.current_file_label)
        
        layout.addWidget(progress_group)
        
        # Output text
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 9))
        self.output_text.setMaximumHeight(150)
        layout.addWidget(self.output_text)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.transcribe_button = QPushButton("🚀 Start Processing")
        self.transcribe_button.clicked.connect(self.start_processing)
        self.transcribe_button.setEnabled(False)
        
        self.stop_button = QPushButton("⏹️ Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        self.save_txt_button = QPushButton("💾 Export TXT")
        self.save_txt_button.clicked.connect(lambda: self.save_results("txt"))
        self.save_txt_button.setEnabled(False)
        
        self.save_vtt_button = QPushButton("🎞️ Export VTT")
        self.save_vtt_button.clicked.connect(lambda: self.save_results("vtt"))
        self.save_vtt_button.setEnabled(False)
        
        self.save_srt_button = QPushButton("📝 Export SRT")
        self.save_srt_button.clicked.connect(lambda: self.save_results("srt"))
        self.save_srt_button.setEnabled(False)
        
        button_layout.addWidget(self.transcribe_button, 2)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_txt_button)
        button_layout.addWidget(self.save_vtt_button)
        button_layout.addWidget(self.save_srt_button)
        
        layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
    
    def select_files(self):
        """Select media files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Media Files", 
            "", 
            "Media Files (*.mp4 *.mov *.avi *.mkv *.webm *.wav *.mp3 *.m4a *.flac);;All Files (*)"
        )
        
        if file_paths:
            self.file_paths = file_paths
            self.update_file_list()
            self.check_ready_state()
    
    def select_output_folder(self):
        """Select output folder"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        
        if folder_path:
            self.output_folder = folder_path
            self.output_folder_label.setText(folder_path)
            self.check_ready_state()
    
    def clear_files(self):
        """Clear selected files"""
        self.file_paths = []
        self.file_list.clear()
        self.batch_results = {}
        self.output_text.clear()
        self.check_ready_state()
    
    def update_file_list(self):
        """Update file list display"""
        self.file_list.clear()
        for file_path in self.file_paths:
            self.file_list.addItem(os.path.basename(file_path))
    
    def on_vocab_correction_toggled(self, state):
        """Handle vocabulary correction checkbox toggle"""
        enabled = state == Qt.Checked
        self.vocab_file_button.setEnabled(enabled)
        self.sensitivity_slider.setEnabled(enabled)
        self.title_case_fallback_check.setEnabled(enabled)
        
        if not enabled:
            self.vocab_file_path = None
            self.vocab_file_label.setText("No file selected")
            self.vocab_file_label.setStyleSheet("color: gray;")
    
    def select_vocab_file(self):
        """Select vocabulary CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Vocabulary CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            self.vocab_file_path = file_path
            # Show just the filename, not full path
            filename = os.path.basename(file_path)
            self.vocab_file_label.setText(filename)
            self.vocab_file_label.setStyleSheet("color: green; font-weight: bold;")
            self.vocab_file_label.setToolTip(file_path)
            
            # Log the selection
            logger.info(f"📚 Vocabulary file selected: {file_path}")
    
    def check_ready_state(self):
        """Check if ready to process"""
        ready = len(self.file_paths) > 0 and bool(self.output_folder)
        self.transcribe_button.setEnabled(ready)
        
        if ready:
            self.status_label.setText(f"Ready: {len(self.file_paths)} files selected")
        elif not self.file_paths:
            self.status_label.setText("Select files to process")
        else:
            self.status_label.setText("Select output folder")
    
    def start_processing(self):
        """Start batch processing"""
        if not self.file_paths or not self.output_folder:
            return
        
        os.makedirs(self.output_folder, exist_ok=True)
        
        self.batch_results = {}
        self.set_buttons_enabled(False)
        self.overall_progress.setVisible(True)
        self.file_progress.setVisible(True)
        self.overall_progress.setValue(0)
        self.file_progress.setValue(0)
        self.output_text.clear()
        self.stop_button.setEnabled(True)
        
        # Create worker thread
        self.thread = QThread()
        self.worker = BatchWorker()
        self.worker.moveToThread(self.thread)
        
        # Get context prompt from input
        context_prompt = self.context_prompt_input.toPlainText().strip()
        
        # Build vocabulary settings if enabled
        vocab_settings = None
        if self.vocab_correction_check.isChecked() and self.vocab_file_path:
            vocab_settings = {
                'enabled': True,
                'csv_path': self.vocab_file_path,
                'sensitivity': self.sensitivity_slider.value() / 100.0,  # Convert percentage to decimal
                'title_case_fallback': self.title_case_fallback_check.isChecked()
            }
            logger.info(f"📚 Vocabulary correction enabled: {os.path.basename(self.vocab_file_path)}")
        
        # Connect signals
        self.thread.started.connect(lambda: self.worker.transcribe_batch(
            self.file_paths,
            self.model_combo.currentText(),
            self.timestamps_check.isChecked(),
            84,  # max chars per segment
            context_prompt if context_prompt else None,
            vocab_settings
        ))
        
        self.worker.progress_updated.connect(self.overall_progress.setValue)
        self.worker.file_progress_updated.connect(self.file_progress.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.current_file_updated.connect(self.current_file_label.setText)
        self.worker.file_completed.connect(self.on_file_completed)
        self.worker.batch_finished.connect(self.on_batch_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.thread.finished.connect(self.cleanup_thread)
        
        self.worker.setup_environment()
        self.thread.start()
    
    def on_file_completed(self, file_path: str, text: str, segments: list, stats: ProcessingStats):
        """Handle file completion"""
        filename = os.path.basename(file_path)
        self.output_text.append(f"✅ {filename}: {len(text)} chars, {len(segments)} segments")
        
        # Auto-save files
        try:
            stem = Path(file_path).stem
            
            # Save TXT
            txt_path = Path(self.output_folder) / f"{stem}.txt"
            with open(txt_path, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            
            # Save VTT
            if segments:
                vtt_path = Path(self.output_folder) / f"{stem}.vtt"
                vtt_content = SubtitleFormatter.create_vtt(segments, self.char_limit_spin.value())
                with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(vtt_content)
            
            self.batch_results[file_path] = {
                'text': text,
                'segments': segments,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Auto-save error for {filename}: {e}")
    
    def on_batch_finished(self, summary: dict):
        """Handle batch completion"""
        self.batch_results.update(summary['results'])
        
        self.overall_progress.setVisible(False)
        self.file_progress.setVisible(False)
        self.current_file_label.setText("✅ Processing complete!")
        
        # Enable export buttons
        has_successful = any(r.get('success') for r in self.batch_results.values())
        self.save_txt_button.setEnabled(has_successful)
        self.save_vtt_button.setEnabled(has_successful)
        self.save_srt_button.setEnabled(has_successful)
        
        # Show summary
        self.output_text.append(f"\n🏁 COMPLETE: {summary['successful']}/{summary['total']} successful")
        self.output_text.append(f"💾 Results saved to: {self.output_folder}")
        
        self.set_buttons_enabled(True)
        self.stop_button.setEnabled(False)
    
    def save_results(self, format_type: str):
        """Save all results in specified format"""
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
                elif format_type == "vtt" and result.get('segments'):
                    content = SubtitleFormatter.create_vtt(result['segments'], self.char_limit_spin.value())
                elif format_type == "srt" and result.get('segments'):
                    content = SubtitleFormatter.create_srt(result['segments'], self.char_limit_spin.value())
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
        if file_path:
            filename = os.path.basename(file_path)
            error_text = f"❌ {filename}: {error_message}"
        else:
            error_text = f"❌ System: {error_message}"
        
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
        self.select_output_button.setEnabled(enabled)
        self.clear_files_button.setEnabled(enabled)
        
        self.model_combo.setEnabled(enabled)
        self.char_limit_spin.setEnabled(enabled)
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
        
        FasterWhisperModelManager().clear_model()
        event.accept()

# ========================================
# CLI PROCESSING
# ========================================

def process_single_video_cli(video_path: str, output_folder: str, model_id: str = "small", 
                           max_chars_per_line: int = 42, max_chars_per_segment: int = 84,
                           generate_timestamps: bool = True, context_prompt: str = None,
                           vocab_csv: str = None, vocab_sensitivity: float = 0.85,
                           vocab_fallback: bool = True) -> bool:
    """
    Process a single video file from command line
    
    Args:
        video_path: Path to input video/audio file
        output_folder: Output folder for generated files
        model_id: Whisper model to use
        max_chars_per_line: Maximum characters per subtitle line
        max_chars_per_segment: Maximum characters per subtitle segment
        generate_timestamps: Whether to generate timestamps
        context_prompt: Context prompt for Whisper
        vocab_csv: Path to vocabulary CSV for post-processing correction
        vocab_sensitivity: Fuzzy match sensitivity (0.0-1.0)
        vocab_fallback: Enable title case fallback for unknown terms
        
    Returns True if successful, False otherwise
    """
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize vocabulary corrector if provided
        vocabulary_corrector = None
        if vocab_csv and os.path.exists(vocab_csv):
            try:
                vocabulary_corrector = VocabularyCorrector(
                    csv_path=vocab_csv,
                    similarity_threshold=vocab_sensitivity,
                    enable_fallback=vocab_fallback
                )
                logger.info(f"📚 Vocabulary correction enabled with {len(vocabulary_corrector.terms)} terms")
            except Exception as e:
                logger.warning(f"⚠️ Could not load vocabulary file: {e}")
        
        # Initialize components
        model_manager = FasterWhisperModelManager()
        processor = TranscriptionProcessor(vocabulary_corrector=vocabulary_corrector)
        stats = ProcessingStats()
        stats.start_time = time.time()
        
        filename = os.path.basename(video_path)
        logger.info(f"🎬 Processing: {filename}")
        
        if context_prompt:
            logger.info(f"📚 Using context prompt: {context_prompt[:50]}...")
        
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
            audio_path, duration = processor.extract_audio(video_path, stats)
            stats.audio_duration = duration
            logger.info(f"✅ Audio extracted: {duration:.2f}s")
            
            # Transcribe
            logger.info("🎙️ Transcribing...")
            text, segments = processor.transcribe(
                audio_path, model, generate_timestamps, max_chars_per_segment, stats, context_prompt
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
        description="Video Captioner - Generate subtitles from video files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video with default settings
  python captioner_compact.py input.mp4 -o output_folder

  # Process with specific model and settings
  python captioner_compact.py input.mp4 -o output_folder -m large-v3 --max-chars 50

  # Process without timestamps (transcript only)
  python captioner_compact.py input.mp4 -o output_folder --no-timestamps

  # Process with context prompt for better vocabulary recognition
  python captioner_compact.py input.mp4 -o output_folder --prompt "This video covers Unity game engine"

  # Launch GUI mode
  python captioner_compact.py --gui
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
    parser.add_argument('--prompt', '-p', type=str, default=None,
                       help='Context prompt to help faster-whisper recognize domain-specific vocabulary')
    parser.add_argument('--vocab-csv', type=str, default=None,
                       help='Path to vocabulary CSV file for post-processing correction')
    parser.add_argument('--vocab-sensitivity', type=int, default=85,
                       help='Fuzzy match sensitivity percentage (70-100, default: 85)')
    parser.add_argument('--no-vocab-fallback', action='store_true',
                       help='Disable title case fallback for unknown terms')
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
        
        app.setApplicationName("Video Captioner - Compact")
        app.setApplicationVersion("1.0")
        
        window = MainWindow()
        window.show()
        
        logger.info("🚀 Video Captioner (Compact) GUI Mode")
        FasterWhisperModelManager.get_optimal_device_config()
        
        sys.exit(app.exec_())
    else:
        # CLI mode
        if not args.output:
            parser.error("Output folder is required in CLI mode")
        
        logger.info("🚀 Video Captioner (Compact) CLI Mode")
        FasterWhisperModelManager.get_optimal_device_config()
        
        # Process the video
        success = process_single_video_cli(
            video_path=args.input,
            output_folder=args.output,
            model_id=args.model,
            max_chars_per_line=args.max_chars,
            max_chars_per_segment=args.max_segment_chars,
            generate_timestamps=not args.no_timestamps,
            context_prompt=args.prompt,
            vocab_csv=args.vocab_csv,
            vocab_sensitivity=args.vocab_sensitivity / 100.0,  # Convert percentage to decimal
            vocab_fallback=not args.no_vocab_fallback
        )
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
