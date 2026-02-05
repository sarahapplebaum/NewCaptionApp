import sys
import argparse
import logging
from src.utils import configure_logging

logger = logging.getLogger(__name__)

def main():
    """Main function with CLI and GUI support"""
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
    
    # Configure logging
    configure_logging(args.verbose)
    
    # Determine mode
    if args.gui or (not args.input):
        # GUI mode - lazy import
        try:
            from src.gui import launch_gui
            launch_gui()
        except ImportError as e:
            logger.error(f"Failed to import GUI module: {e}")
            logger.error("Please ensure you have installed the required dependencies: pip install PyQt5")
            sys.exit(1)
        except Exception as e:
            logger.error(f"GUI Error: {e}")
            sys.exit(1)
    else:
        # CLI mode
        if not args.output:
            parser.error("Output folder is required in CLI mode")
        
        logger.info("🚀 Video Captioner (Compact) CLI Mode")
        
        try:
            # Lazy import core
            from src.core import process_single_video_cli, FasterWhisperModelManager
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
                vocab_sensitivity=args.vocab_sensitivity / 100.0,
                vocab_fallback=not args.no_vocab_fallback
            )
            
            sys.exit(0 if success else 1)
        except ImportError as e:
            logger.error(f"Failed to import Core module: {e}")
            logger.error("Please ensure you have installed: torch, faster-whisper")
            sys.exit(1)

if __name__ == '__main__':
    main()
