import sys
import os
import time
import logging
import torch # We can import here as this module is only loaded when GUI is requested
from pathlib import Path
from typing import List, Dict, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QPushButton, QFileDialog, QProgressBar, 
    QTextEdit, QComboBox, QSpinBox, QCheckBox, QGroupBox, QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont

from src.core import FasterWhisperModelManager, TranscriptionProcessor, AudioProcessor
from src.utils import ProcessingStats, SubtitleFormatter, VocabularyCorrector

logger = logging.getLogger(__name__)

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
        results = {}
        successful = 0
        failed = 0
        total_start_time = time.time()
        
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
                        
                        audio_path, duration = self.processor.extract_audio(file_path, file_stats)
                        file_stats.audio_duration = duration
                        self.file_progress_updated.emit(30)
                        
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
        
        torch.set_num_threads(min(8, os.cpu_count() or 4))

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
        self.setWindowTitle("Video Captioner - Compact Version")
        self.setGeometry(100, 100, 1000, 700)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        title_label = QLabel("Video Captioner (Compact)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title_label)
        
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
        
        config_layout.addWidget(QLabel("Context Prompt (optional):"), 3, 0, 1, 2)
        self.context_prompt_input = QTextEdit()
        self.context_prompt_input.setPlaceholderText(
            "e.g., This video is about the Unity game engine, covering GameObjects, Prefabs, and C# scripting..."
        )
        self.context_prompt_input.setMaximumHeight(60)
        config_layout.addWidget(self.context_prompt_input, 4, 0, 1, 2)
        
        layout.addWidget(config_group)
        
        vocab_group = QGroupBox("Vocabulary Correction (Post-Processing)")
        vocab_layout = QGridLayout(vocab_group)
        
        self.vocab_correction_check = QCheckBox("Enable vocabulary correction")
        self.vocab_correction_check.stateChanged.connect(self.on_vocab_correction_toggled)
        vocab_layout.addWidget(self.vocab_correction_check, 0, 0, 1, 2)
        
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
        
        vocab_layout.addWidget(QLabel("Fuzzy Match Sensitivity:"), 2, 0)
        sensitivity_layout = QHBoxLayout()
        self.sensitivity_slider = QSpinBox()
        self.sensitivity_slider.setRange(70, 100)
        self.sensitivity_slider.setValue(85)
        self.sensitivity_slider.setSuffix("%")
        self.sensitivity_slider.setEnabled(False)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        vocab_layout.addLayout(sensitivity_layout, 2, 1)
        
        self.title_case_fallback_check = QCheckBox("Enable title case fallback for unknown terms")
        self.title_case_fallback_check.setChecked(True)
        self.title_case_fallback_check.setEnabled(False)
        vocab_layout.addWidget(self.title_case_fallback_check, 3, 0, 1, 2)
        
        self.vocab_file_path = None
        layout.addWidget(vocab_group)
        
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
        
        output_layout = QHBoxLayout()
        self.output_folder_label = QLabel("No output folder selected")
        self.select_output_button = QPushButton("📋 Output Folder")
        self.select_output_button.clicked.connect(self.select_output_folder)
        
        output_layout.addWidget(QLabel("Output:"))
        output_layout.addWidget(self.output_folder_label, 1)
        output_layout.addWidget(self.select_output_button)
        file_layout.addLayout(output_layout)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        file_layout.addWidget(QLabel("Selected Files:"))
        file_layout.addWidget(self.file_list)
        
        layout.addWidget(file_group)
        
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
        
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 9))
        self.output_text.setMaximumHeight(150)
        layout.addWidget(self.output_text)
        
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
        
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

    def select_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Media Files", "", 
            "Media Files (*.mp4 *.mov *.avi *.mkv *.webm *.wav *.mp3 *.m4a *.flac);;All Files (*)"
        )
        if file_paths:
            self.file_paths = file_paths
            self.update_file_list()
            self.check_ready_state()

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder = folder_path
            self.output_folder_label.setText(folder_path)
            self.check_ready_state()

    def clear_files(self):
        self.file_paths = []
        self.file_list.clear()
        self.batch_results = {}
        self.output_text.clear()
        self.check_ready_state()

    def update_file_list(self):
        self.file_list.clear()
        for file_path in self.file_paths:
            self.file_list.addItem(os.path.basename(file_path))

    def on_vocab_correction_toggled(self, state):
        enabled = state == Qt.Checked
        self.vocab_file_button.setEnabled(enabled)
        self.sensitivity_slider.setEnabled(enabled)
        self.title_case_fallback_check.setEnabled(enabled)
        if not enabled:
            self.vocab_file_path = None
            self.vocab_file_label.setText("No file selected")
            self.vocab_file_label.setStyleSheet("color: gray;")

    def select_vocab_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Vocabulary CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.vocab_file_path = file_path
            self.vocab_file_label.setText(os.path.basename(file_path))
            self.vocab_file_label.setStyleSheet("color: green; font-weight: bold;")
            logger.info(f"📚 Vocabulary file selected: {file_path}")

    def check_ready_state(self):
        ready = len(self.file_paths) > 0 and bool(self.output_folder)
        self.transcribe_button.setEnabled(ready)
        if ready:
            self.status_label.setText(f"Ready: {len(self.file_paths)} files selected")
        elif not self.file_paths:
            self.status_label.setText("Select files to process")
        else:
            self.status_label.setText("Select output folder")

    def start_processing(self):
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
        
        self.thread = QThread()
        self.worker = BatchWorker()
        self.worker.moveToThread(self.thread)
        
        context_prompt = self.context_prompt_input.toPlainText().strip()
        vocab_settings = None
        if self.vocab_correction_check.isChecked() and self.vocab_file_path:
            vocab_settings = {
                'enabled': True,
                'csv_path': self.vocab_file_path,
                'sensitivity': self.sensitivity_slider.value() / 100.0,
                'title_case_fallback': self.title_case_fallback_check.isChecked()
            }
        
        self.thread.started.connect(lambda: self.worker.transcribe_batch(
            self.file_paths, self.model_combo.currentText(), self.timestamps_check.isChecked(),
            84, context_prompt if context_prompt else None, vocab_settings
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
        filename = os.path.basename(file_path)
        self.output_text.append(f"✅ {filename}: {len(text)} chars, {len(segments)} segments")
        try:
            stem = Path(file_path).stem
            txt_path = Path(self.output_folder) / f"{stem}.txt"
            with open(txt_path, 'w', encoding='utf-8-sig') as f:
                f.write(text)
            if segments:
                vtt_path = Path(self.output_folder) / f"{stem}.vtt"
                vtt_content = SubtitleFormatter.create_vtt(segments, self.char_limit_spin.value())
                with open(vtt_path, 'w', encoding='utf-8-sig') as f:
                    f.write(vtt_content)
            self.batch_results[file_path] = {'text': text, 'segments': segments, 'success': True}
        except Exception as e:
            logger.error(f"Auto-save error for {filename}: {e}")

    def on_batch_finished(self, summary: dict):
        self.batch_results.update(summary['results'])
        self.overall_progress.setVisible(False)
        self.file_progress.setVisible(False)
        self.current_file_label.setText("✅ Processing complete!")
        has_successful = any(r.get('success') for r in self.batch_results.values())
        self.save_txt_button.setEnabled(has_successful)
        self.save_vtt_button.setEnabled(has_successful)
        self.save_srt_button.setEnabled(has_successful)
        self.output_text.append(f"\n🏁 COMPLETE: {summary['successful']}/{summary['total']} successful")
        self.output_text.append(f"💾 Results saved to: {self.output_folder}")
        self.set_buttons_enabled(True)
        self.stop_button.setEnabled(False)

    def save_results(self, format_type: str):
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
        self.status_label.setText(f"Saved {count} {format_type.upper()} files" if count > 0 else f"No {format_type.upper()} files")

    def on_error(self, error_message: str, file_path: str):
        error_text = f"❌ {os.path.basename(file_path)}: {error_message}" if file_path else f"❌ System: {error_message}"
        self.output_text.append(error_text)
        logger.error(error_text)

    def stop_processing(self):
        if self.worker:
            self.worker.stop_processing()
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping...")

    def set_buttons_enabled(self, enabled: bool):
        ready = enabled and len(self.file_paths) > 0 and bool(self.output_folder)
        self.transcribe_button.setEnabled(ready)
        self.select_files_button.setEnabled(enabled)
        self.select_output_button.setEnabled(enabled)
        self.clear_files_button.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.char_limit_spin.setEnabled(enabled)
        self.timestamps_check.setEnabled(enabled)

    def cleanup_thread(self):
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            if self.worker:
                self.worker.stop_processing()
            self.thread.quit()
            self.thread.wait(5000)
        FasterWhisperModelManager().clear_model()
        event.accept()

def launch_gui():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationName("Video Captioner - Compact")
    app.setApplicationVersion("1.0")
    window = MainWindow()
    window.show()
    logger.info("🚀 Video Captioner (Compact) GUI Mode")
    FasterWhisperModelManager.get_optimal_device_config()
    sys.exit(app.exec_())
