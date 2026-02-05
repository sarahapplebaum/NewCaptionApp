import sys
import os
import re
import csv
import logging
import tempfile
import codecs
from typing import List, Dict, Tuple, Union, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

# Constants
DURATION_PATTERN = re.compile(r'Duration: (\d+):(\d+):(\d+)\.(\d+)')
SENTENCE_STARTERS = frozenset(['and', 'but', 'so', 'yet', 'or', 'nor', 'for', 'however', 'therefore'])
COMMON_ABBREVS = frozenset(['mr.', 'mrs.', 'dr.', 'prof.', 'inc.', 'ltd.', 'co.', 'corp.', 'etc.', 'vs.', 'jr.', 'sr.'])

logger = logging.getLogger(__name__)

def configure_logging(verbose: bool = False):
    """Configure logging with UTF-8 support"""
    # Force UTF-8 encoding for console output on Windows
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except (AttributeError, OSError):
            # Python < 3.7 or reconfigure not available
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(tempfile.gettempdir(), 'videocaptioner_debug.log'), 
                mode='w',
                encoding='utf-8'
            )
        ]
    )

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
            # logger.debug(f"🔧 Fixed compound word: '{current_word}' → '{fixed_word}'")
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
                # logger.debug(f"🔗 Merged: '{current_word}' + '{next_word}' → '{merged_word}'")
                i += 2
                continue
        
        merged.append(word_level[i])
        i += 1
    
    # Final cleanup pass
    return [{**entry, 'word': entry['word'].replace(' -', '-')} for entry in merged]

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

class VocabularyCorrector:
    """Unity-specific vocabulary correction system"""
    
    def __init__(self, csv_path: str, similarity_threshold: float = 0.85, enable_fallback: bool = True):
        self.csv_path = csv_path
        self.similarity_threshold = similarity_threshold
        self.enable_fallback = enable_fallback
        
        self.terms = []
        self.term_lookup = {}
        self.multi_word_terms = {}
        self.fuzzy_cache = {}
        self.correction_log = []
        
        self._load_csv()
        self._build_indices()
        
        logger.info(f"📚 Vocabulary loaded: {len(self.terms)} terms ({len(self.multi_word_terms)} multi-word)")
    
    def _load_csv(self):
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    term = row.get('Term', '').strip()
                    if term and term not in ['Term', '']:
                        self.terms.append(term)
            logger.info(f"✅ Loaded {len(self.terms)} vocabulary terms from {self.csv_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not load vocabulary file: {e}")
            self.terms = []
    
    def _build_indices(self):
        for term in self.terms:
            term_lower = term.lower()
            self.term_lookup[term_lower] = term
            words = term.split()
            if 2 <= len(words) <= 3:
                self.multi_word_terms[term_lower] = term
    
    def get_initial_prompt(self, max_terms: int = 50) -> str:
        if not self.terms:
            return ""
        priority_terms = sorted(self.terms, key=lambda x: (len(x), x))[:max_terms]
        prompt = "This video discusses Unity game engine, including: " + ", ".join(priority_terms[:30])
        return prompt
    
    def correct_word(self, word: str) -> Tuple[str, str]:
        if not word:
            return word, 'none'
        
        leading_punct = ''
        trailing_punct = ''
        
        while word and not word[0].isalnum():
            leading_punct += word[0]
            word = word[1:]
        
        while word and not word[-1].isalnum():
            trailing_punct = word[-1] + trailing_punct
            word = word[:-1]
        
        if not word:
            return leading_punct + trailing_punct, 'none'
        
        word_lower = word.lower()
        if word_lower in self.term_lookup:
            corrected = self.term_lookup[word_lower]
            return leading_punct + corrected + trailing_punct, 'exact'
        
        cache_key = word_lower
        if cache_key in self.fuzzy_cache:
            cached_result = self.fuzzy_cache[cache_key]
            if cached_result:
                return leading_punct + cached_result + trailing_punct, 'fuzzy'
        else:
            best_match = None
            best_score = 0.0
            
            for term_lower, correct_term in self.term_lookup.items():
                if ' ' in term_lower:
                    continue
                similarity = SequenceMatcher(None, word_lower, term_lower).ratio()
                if similarity > best_score and similarity >= self.similarity_threshold:
                    best_score = similarity
                    best_match = correct_term
            
            self.fuzzy_cache[cache_key] = best_match
            if best_match:
                return leading_punct + best_match + trailing_punct, 'fuzzy'
        
        if self.enable_fallback and self._looks_like_unity_term(word):
            corrected = self._apply_title_case(word)
            return leading_punct + corrected + trailing_punct, 'fallback'
        
        return leading_punct + word + trailing_punct, 'none'
    
    def _looks_like_unity_term(self, word: str) -> bool:
        if len(word) < 3:
            return False
        has_capitals = any(c.isupper() for c in word)
        has_lowercase = any(c.islower() for c in word)
        if has_capitals and has_lowercase and word[0].isupper():
            return False
        if word.islower() and len(word) >= 5:
            unity_patterns = ['mesh', 'shader', 'texture', 'sprite', 'script', 'object', 
                            'system', 'manager', 'controller', 'renderer', 'collider']
            return any(pattern in word for pattern in unity_patterns)
        return False
    
    def _apply_title_case(self, word: str) -> str:
        if word.lower() in ['vr', 'ar', 'xr', 'ui', 'api', 'fps', 'hdr', 'gpu', 'cpu', 'ai']:
            return word.upper()
        if len(word) > 8 and not any(c.isupper() for c in word):
            return word[0].upper() + word[1:]
        return word[0].upper() + word[1:]
    
    def correct_word_list(self, words: List[Dict]) -> List[Dict]:
        if not words:
            return words
        
        corrected_words = []
        self.correction_log = []
        i = 0
        
        while i < len(words):
            multi_word_match = self._try_multi_word_match(words, i)
            if multi_word_match:
                corrected_entry, words_consumed = multi_word_match
                corrected_words.append(corrected_entry)
                i += words_consumed
                continue
            
            word_dict = words[i]
            original_word = word_dict.get('word', '').strip()
            
            if original_word:
                corrected_word, correction_type = self.correct_word(original_word)
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
        for word_count in [3, 2]:
            if index + word_count > len(words):
                continue
            
            word_group = words[index:index + word_count]
            combined_text = ' '.join([w.get('word', '').strip() for w in word_group])
            combined_lower = combined_text.lower()
            combined_clean = combined_lower.strip('.,!?";:\'"')
            
            if combined_clean in self.multi_word_terms:
                correct_term = self.multi_word_terms[combined_clean]
                self.correction_log.append({
                    'original': combined_text,
                    'corrected': correct_term,
                    'type': 'multi-word',
                    'position': index,
                    'words_consumed': word_count
                })
                
                last_word = word_group[-1].get('word', '')
                trailing_punct = ''
                while last_word and not last_word[-1].isalnum():
                    trailing_punct = last_word[-1] + trailing_punct
                    last_word = last_word[:-1]
                
                merged_entry = {
                    'word': correct_term + trailing_punct,
                    'start': safe_float(word_group[0].get('start'), 0),
                    'end': safe_float(word_group[-1].get('end'), 0)
                }
                return (merged_entry, word_count)
        return None
    
    def get_correction_summary(self) -> str:
        if not self.correction_log:
            return "No corrections made"
        
        summary_lines = [f"✏️  Vocabulary Corrections: {len(self.correction_log)} changes"]
        by_type = {}
        for correction in self.correction_log:
            corr_type = correction['type']
            by_type.setdefault(corr_type, []).append(correction)
        
        for corr_type, corrections in by_type.items():
            count = len(corrections)
            samples = corrections[:3]
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

class SubtitleFormatter:
    """Subtitle formatting and segmentation"""
    
    MIN_DURATION = 1.25
    MIN_DURATION_SINGLE_WORD = 0.4
    MAX_DURATION = 8.0
    WORDS_PER_SECOND = 3.33
    MAX_CHARS_PER_SECOND = 25
    
    @staticmethod
    def format_text_lines(text: str, max_chars: int = 40, max_lines: int = 2) -> List[str]:
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
        
        if len(lines) > max_lines:
            # Simple truncation/redistribution strategy
            lines = lines[:max_lines]
        
        return lines
    
    @staticmethod
    def calculate_reading_time(text: str) -> float:
        if not text:
            return SubtitleFormatter.MIN_DURATION
        words = len(text.split())
        base_time = words / SubtitleFormatter.WORDS_PER_SECOND
        char_time = len(text) * 0.05
        total_time = max(base_time, char_time)
        return max(SubtitleFormatter.MIN_DURATION, min(total_time, SubtitleFormatter.MAX_DURATION))
    
    @classmethod
    def clean_words(cls, words: List[Dict]) -> List[WordInfo]:
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
        
        for i in range(len(word_infos) - 1):
            if word_infos[i].word.endswith('.'):
                next_word = word_infos[i + 1].word.lstrip('.,!?";:\'" ')
                if next_word and next_word[0].islower() and next_word.lower() not in SENTENCE_STARTERS:
                    word_infos[i].word = word_infos[i].word[:-1]
        return word_infos
    
    @classmethod
    def create_optimized_segments(cls, words: List[Dict], max_chars: int = 80) -> List[Dict]:
        if not words:
            return []
        
        cleaned_words = cls.clean_words(words)
        if not cleaned_words:
            return []
        
        safe_max_chars = min(max_chars, 70)
        segments = []
        current_segment = {"start": None, "end": None, "text": "", "word_count": 0}
        
        for word_info in cleaned_words:
            word = word_info.word
            word_start = word_info.start
            word_end = word_info.end
            
            potential_text = f"{current_segment['text']} {word}" if current_segment['text'] else word
            test_lines = cls.format_text_lines(potential_text, 42, 2)
            would_fit_properly = len(test_lines) <= 2 and all(len(line) <= 42 for line in test_lines)
            
            would_exceed = len(potential_text) > safe_max_chars or not would_fit_properly
            time_gap = current_segment["end"] is not None and (word_start - current_segment["end"]) > 2.0
            
            is_sentence_end = word.rstrip('",\')"]}').endswith(('.', '!', '?', '...')) and \
                            word.lower() not in COMMON_ABBREVS
            is_pause = word.rstrip().endswith((',', ';', ':', '--', '—'))
            
            current_duration = 0
            if current_segment["start"] is not None:
                current_duration = word_end - current_segment["start"]
            
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
                current_segment = {"start": word_start, "end": word_end, "text": word, "word_count": 1}
            else:
                if current_segment["start"] is None:
                    current_segment["start"] = word_start
                current_segment["text"] = potential_text
                current_segment["end"] = word_end
                current_segment["word_count"] += 1
        
        if current_segment['text']:
            segments.append({
                "start": current_segment["start"],
                "end": current_segment["end"],
                "text": current_segment["text"].strip()
            })
        
        return cls.post_process_segments(segments)
    
    @classmethod
    def post_process_segments(cls, segments: List[Dict]) -> List[Dict]:
        if not segments:
            return []
        
        segments.sort(key=lambda x: safe_float(x.get("start"), 0))
        processed = []
        seen_content = set()
        
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0)
            end_time = safe_float(segment.get("end"), start_time + 0.5)
            
            content_hash = f"{text[:50]}_{start_time:.1f}"
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            ideal_duration = cls.calculate_reading_time(text)
            actual_duration = end_time - start_time
            
            word_count = len(text.split())
            if word_count <= 1:
                min_duration = cls.MIN_DURATION_SINGLE_WORD
            elif word_count <= 2:
                min_duration = 0.8
            else:
                min_duration = cls.MIN_DURATION
            
            if actual_duration < min_duration:
                end_time = start_time + min_duration
            elif actual_duration > cls.MAX_DURATION:
                end_time = start_time + cls.MAX_DURATION
            
            if i + 1 < len(segments):
                next_start = safe_float(segments[i + 1].get("start"), end_time + 0.1)
                gap = next_start - end_time
                if gap > 0.1 and gap < 1.0:
                    proposed_end = next_start - 0.05
                    if proposed_end - start_time <= ideal_duration * 1.5:
                        end_time = proposed_end
            
            if processed:
                prev_end = safe_float(processed[-1].get("end"), 0)
                if start_time < prev_end:
                    start_time = prev_end + 0.05
                    end_time = start_time + actual_duration
            
            segment["start"] = start_time
            segment["end"] = end_time
            processed.append(segment)
        
        processed = cls.merge_short_segments(processed)
        processed = cls.enforce_reading_rate(processed)
        return cls.clean_segment_punctuation(processed)
    
    @classmethod
    def merge_short_segments(cls, segments: List[Dict]) -> List[Dict]:
        if len(segments) <= 1:
            return segments
        
        MIN_WORDS = 3
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            current_text = current.get("text", "").strip()
            current_words = current_text.split()
            
            if len(current_words) >= MIN_WORDS:
                merged.append(current)
                i += 1
                continue
            
            merged_with_prev = False
            
            if merged:
                prev_text = merged[-1].get("text", "")
                combined = prev_text + " " + current_text
                lines = cls.format_text_lines(combined, 42, 2)
                if len(lines) <= 2:
                    merged[-1]["text"] = combined
                    merged[-1]["end"] = current.get("end")
                    merged_with_prev = True
                    i += 1
                    continue
            
            if not merged_with_prev and i + 1 < len(segments):
                next_segment = segments[i + 1]
                next_text = next_segment.get("text", "")
                combined = current_text + " " + next_text
                lines = cls.format_text_lines(combined, 42, 2)
                if len(lines) <= 2:
                    merged.append({
                        "start": current.get("start"),
                        "end": next_segment.get("end"),
                        "text": combined
                    })
                    i += 2
                    continue
            
            if i == len(segments) - 1 and merged and not merged_with_prev:
                prev_text = merged[-1].get("text", "")
                combined = prev_text + " " + current_text
                lines = cls.format_text_lines(combined, 42, 3)
                if len(lines) <= 3:
                    merged[-1]["text"] = combined
                    merged[-1]["end"] = current.get("end")
                    i += 1
                    continue
            
            merged.append(current)
            i += 1
        return merged
    
    @classmethod
    def enforce_reading_rate(cls, segments: List[Dict]) -> List[Dict]:
        for i, segment in enumerate(segments):
            text = segment.get("text", "").strip()
            if not text:
                continue
            start = safe_float(segment.get("start"), 0.0)
            end = safe_float(segment.get("end"), start + cls.MIN_DURATION)
            actual_duration = end - start
            
            word_count = len(text.split())
            if word_count <= 1:
                content_min = cls.MIN_DURATION_SINGLE_WORD
            elif word_count <= 2:
                content_min = 0.8
            else:
                content_min = cls.MIN_DURATION
            
            min_required_duration = len(text) / cls.MAX_CHARS_PER_SECOND
            min_required_duration = max(min_required_duration, content_min)
            
            if actual_duration < min_required_duration:
                segment["end"] = start + min_required_duration
            
            if i > 0:
                prev_end = safe_float(segments[i-1].get("end"), 0.0)
                if segment["start"] < prev_end:
                    segment["start"] = prev_end + 0.05
                    segment["end"] = segment["start"] + actual_duration
        return segments
    
    @classmethod
    def clean_segment_punctuation(cls, segments: List[Dict]) -> List[Dict]:
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
        if not segments:
            return "WEBVTT\n\n"
        
        vtt_lines = ["WEBVTT", ""]
        
        for segment in sorted(segments, key=lambda x: safe_float(x.get("start"), 0)):
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            start_time = safe_float(segment.get("start"), 0.0)
            end_time = safe_float(segment.get("end"), start_time + cls.MIN_DURATION)
            
            # Simple VTT generation - omitting the complex splitting logic for brevity/reliability
            # as it was quite long and potentially buggy in migration. 
            # Reusing format_text_lines which handles basic wrapping.
            lines = cls.format_text_lines(text, max_chars_per_line, 2)
            if not lines:
                continue
                
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
        try:
            seconds = safe_float(seconds, 0.0)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        except:
            return "00:00:00,000"
