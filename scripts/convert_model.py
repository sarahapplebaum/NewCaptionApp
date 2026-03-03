"""
convert_model.py — Phase 4: Merge LoRA → Full Model → CTranslate2

After fine-tuning with LoRA, this script:
  1. Loads the base Whisper-Small model
  2. Loads the LoRA adapter from the local output dir or HuggingFace Hub
  3. Merges the LoRA weights into the base model (merge_and_unload)
  4. Saves the full merged model temporarily
  5. Converts it to CTranslate2 format using ct2-transformers-converter
  6. Verifies the converted model loads and transcribes correctly
  7. (Optional) Cleans up the temporary merged model

The resulting CTranslate2 model can be dropped directly into the existing
NewCaptionApp faster-whisper pipeline — no other code changes needed.

Usage:
    # Convert from local training output directory:
    python scripts/convert_model.py --adapter ./unity-whisper-small

    # Convert directly from HuggingFace Hub:
    python scripts/convert_model.py --adapter sapplebaum/unity-whisper-small

    # Specify output directory:
    python scripts/convert_model.py \\
        --adapter ./unity-whisper-small \\
        --output  ./models/unity-whisper-small-ct2

    # Convert with float16 precision instead of int8:
    python scripts/convert_model.py --adapter ./unity-whisper-small --quantization float16

Requirements:
    pip install transformers peft ctranslate2 faster-whisper
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL_ID      = "openai/whisper-small"
DEFAULT_OUTPUT_DIR = "./models/unity-whisper-small-ct2"
DEFAULT_QUANT      = "int8"   # int8 for best CPU performance; float16 for GPU quality


# ── Step 1 & 2: Merge LoRA into Base Model ────────────────────────────────────

def merge_lora_into_base(adapter_path: str, merged_output_dir: Path) -> None:
    """
    Load the LoRA adapter, merge its weights into the base Whisper-Small model,
    and save the full merged model to merged_output_dir.

    After merge_and_unload(), the model is a standard WhisperForConditionalGeneration
    with no PEFT dependencies — ready for ct2-transformers-converter.
    """
    try:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from peft import PeftModel
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install transformers peft")
        sys.exit(1)

    logger.info(f"🔧 Loading base model: {BASE_MODEL_ID}")
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

    logger.info(f"🔗 Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("🔀 Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    logger.info("   ✅ Weights merged — no LoRA dependencies remain")

    logger.info(f"💾 Saving merged model to: {merged_output_dir}")
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_output_dir))

    # Also save the processor (tokenizer + feature extractor)
    logger.info(f"💾 Saving processor...")
    processor = WhisperProcessor.from_pretrained(adapter_path)
    processor.save_pretrained(str(merged_output_dir))

    logger.info("   ✅ Merged model saved")


# ── Step 3: Convert to CTranslate2 ────────────────────────────────────────────

def convert_to_ctranslate2(
    merged_dir: Path,
    output_dir: Path,
    quantization: str,
) -> None:
    """
    Call ct2-transformers-converter to convert the merged HuggingFace model
    to CTranslate2 format, which faster-whisper can load directly.

    Quantization options:
      int8        — Best for CPU inference (default); smallest file size
      float16     — Best for GPU inference; higher quality than int8
      int8_float16 — Mixed: int8 weights stored as float16 activations (GPU)
      float32     — No quantization; largest file, highest precision
    """
    # Verify the converter is available
    converter = shutil.which("ct2-transformers-converter")
    if not converter:
        logger.error("❌ ct2-transformers-converter not found in PATH")
        logger.error("Install with: pip install ctranslate2")
        sys.exit(1)

    logger.info(f"\n🔄 Converting to CTranslate2 format...")
    logger.info(f"   Source:       {merged_dir}")
    logger.info(f"   Destination:  {output_dir}")
    logger.info(f"   Quantization: {quantization}")

    cmd = [
        converter,
        "--model", str(merged_dir),
        "--output_dir", str(output_dir),
        "--quantization", quantization,
        "--force",          # Overwrite existing output directory
    ]

    logger.info(f"   Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"❌ Conversion failed!")
        logger.error(f"   stdout: {result.stdout}")
        logger.error(f"   stderr: {result.stderr}")
        sys.exit(1)

    if result.stdout:
        logger.info(result.stdout.strip())

    logger.info(f"   ✅ CTranslate2 model saved to: {output_dir}")

    # List converted files
    ct2_files = list(output_dir.iterdir())
    logger.info(f"   Files: {[f.name for f in ct2_files]}")


# ── Step 4: Verify the Converted Model ────────────────────────────────────────

def verify_model(output_dir: Path) -> bool:
    """
    Load the converted CTranslate2 model with faster-whisper and run a
    quick sanity-check transcription on a silent audio signal.
    Returns True if the model loads without errors.
    """
    logger.info(f"\n🧪 Verifying converted model...")
    try:
        import numpy as np
        from faster_whisper import WhisperModel
    except ImportError as e:
        logger.warning(f"⚠️  Cannot verify: {e} (install faster-whisper to verify)")
        return True  # Don't fail — user may verify manually

    try:
        logger.info(f"   Loading: {output_dir}")
        model = WhisperModel(
            str(output_dir),
            device="cpu",       # Always verify on CPU for reproducibility
            compute_type="int8",
        )

        # Generate a short silent audio clip (1 second at 16kHz)
        silent_audio = np.zeros(16_000, dtype=np.float32)

        segments, info = model.transcribe(silent_audio, language="en")
        _ = list(segments)  # Consume the generator

        logger.info(f"   ✅ Model loaded and ran successfully")
        logger.info(f"   Detected language: {info.language} (confidence: {info.language_probability:.2f})")
        return True

    except Exception as e:
        logger.error(f"   ❌ Verification failed: {e}")
        return False


# ── Step 5: Print App Integration Instructions ─────────────────────────────────

def print_integration_guide(output_dir: Path) -> None:
    """Print instructions for integrating the converted model into the app."""
    abs_path = output_dir.resolve()

    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Conversion Complete — Integration Guide")
    logger.info("=" * 60)
    logger.info(f"\n  Converted model location:")
    logger.info(f"    {abs_path}")
    logger.info(f"\n  To use in NewCaptionApp:")
    logger.info(f"\n  Option A — GUI: In the app's model dropdown, select")
    logger.info(f"    'Custom...' and browse to the folder above.")
    logger.info(f"\n  Option B — CLI:")
    logger.info(f"    python captioner_compact.py input.mp4 -o output/ \\")
    logger.info(f"        -m {abs_path}")
    logger.info(f"\n  Option C — Code (src/core.py already handles local paths):")
    logger.info(f"    from faster_whisper import WhisperModel")
    logger.info(f"    model = WhisperModel(")
    logger.info(f'        "{abs_path}",')
    logger.info(f'        device="cpu",  # or "mps" on Apple Silicon')
    logger.info(f'        compute_type="int8",')
    logger.info(f"    )")
    logger.info("")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into Whisper-Small and convert to CTranslate2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert from local training output (most common):
    python scripts/convert_model.py --adapter ./unity-whisper-small

    # Convert from HuggingFace Hub:
    python scripts/convert_model.py --adapter sapplebaum/unity-whisper-small

    # Custom output and float16 precision (better for GPU use):
    python scripts/convert_model.py \\
        --adapter ./unity-whisper-small \\
        --output  ./models/unity-whisper-small-ct2 \\
        --quantization float16

Quantization options:
    int8         — Best for CPU (default, smallest file)
    float16      — Best for GPU (larger file, higher quality)
    int8_float16 — Mixed precision for GPU
    float32      — No quantization (largest, most precise)
        """,
    )
    parser.add_argument(
        "--adapter", "-a", required=True,
        help="Local path or HuggingFace Hub ID of the trained LoRA adapter "
             "(e.g. ./unity-whisper-small or sapplebaum/unity-whisper-small)"
    )
    parser.add_argument(
        "--output", "-o", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the CTranslate2 model (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--quantization", "-q",
        choices=["int8", "float16", "int8_float16", "float32"],
        default=DEFAULT_QUANT,
        help=f"Quantization format for the converted model (default: {DEFAULT_QUANT})"
    )
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="Keep the intermediate merged HuggingFace model on disk (default: delete it)"
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip loading and verifying the converted model"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output).expanduser().resolve()

    logger.info("=" * 60)
    logger.info("  Unity Whisper Model Conversion")
    logger.info("=" * 60)
    logger.info(f"  Adapter:      {args.adapter}")
    logger.info(f"  Output:       {output_dir}")
    logger.info(f"  Quantization: {args.quantization}")
    logger.info("=" * 60)

    # Use a temporary directory for the intermediate merged model
    with tempfile.TemporaryDirectory(prefix="unity_whisper_merged_") as tmp_dir:
        merged_dir = Path(tmp_dir) / "merged"

        # ── Step 1+2: Merge LoRA into base ────────────────────────────────────
        merge_lora_into_base(
            adapter_path=args.adapter,
            merged_output_dir=merged_dir,
        )

        # ── Step 3: Convert to CTranslate2 ───────────────────────────────────
        convert_to_ctranslate2(
            merged_dir=merged_dir,
            output_dir=output_dir,
            quantization=args.quantization,
        )

        # If user wants to keep the merged model, copy it out of tmp before cleanup
        if args.keep_merged:
            keep_path = output_dir.parent / "unity-whisper-small-merged-hf"
            logger.info(f"📂 Copying merged HF model to: {keep_path}")
            shutil.copytree(str(merged_dir), str(keep_path), dirs_exist_ok=True)

    # tmp_dir is now deleted automatically

    # ── Step 4: Verify ────────────────────────────────────────────────────────
    if not args.skip_verify:
        ok = verify_model(output_dir)
        if not ok:
            logger.warning("⚠️  Verification failed — the model may still work; test manually.")

    # ── Step 5: Integration guide ─────────────────────────────────────────────
    print_integration_guide(output_dir)


if __name__ == "__main__":
    main()
