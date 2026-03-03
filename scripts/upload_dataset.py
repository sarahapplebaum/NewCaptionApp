"""
upload_dataset.py — Phase 2: AudioFolder → HuggingFace Hub

Validates and uploads the prepared AudioFolder dataset to:
    https://huggingface.co/datasets/sapplebaum/unity-pt-videos

Usage:
    # Login first (one-time):
    huggingface-cli login

    # Then upload:
    python scripts/upload_dataset.py --dataset ./unity_dataset

    # Append new clips to an existing dataset on the Hub:
    python scripts/upload_dataset.py --dataset ./unity_dataset --append

Requirements:
    pip install datasets huggingface_hub soundfile
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

HF_REPO_ID = "sapplebaum/unity-pt-videos"
HF_REPO_TYPE = "dataset"


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_dataset(dataset_dir: Path) -> bool:
    """
    Validate the AudioFolder structure before upload.
    Returns True if the dataset looks correct.
    """
    logger.info("🔍 Validating dataset structure...")
    ok = True

    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error(f"  ❌ metadata.csv not found in {dataset_dir}")
        return False

    # Parse metadata and verify each file exists
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "file_name" not in (reader.fieldnames or []):
            logger.error("  ❌ metadata.csv missing 'file_name' column")
            return False
        if "transcription" not in (reader.fieldnames or []):
            logger.error("  ❌ metadata.csv missing 'transcription' column")
            return False
        rows = list(reader)

    if not rows:
        logger.error("  ❌ metadata.csv is empty")
        return False

    logger.info(f"  📄 metadata.csv: {len(rows)} entries")

    missing = 0
    empty_text = 0
    train_count = 0
    test_count = 0

    for row in rows:
        file_name = row.get("file_name", "").strip()
        transcription = row.get("transcription", "").strip()

        wav_path = dataset_dir / file_name
        if not wav_path.exists():
            logger.warning(f"  ⚠️  Missing audio file: {file_name}")
            missing += 1
        elif wav_path.stat().st_size < 1000:
            logger.warning(f"  ⚠️  Suspiciously small file: {file_name}")

        if not transcription:
            logger.warning(f"  ⚠️  Empty transcription for: {file_name}")
            empty_text += 1

        if file_name.startswith("train/"):
            train_count += 1
        elif file_name.startswith("test/"):
            test_count += 1

    logger.info(f"  🎵 Train clips: {train_count}")
    logger.info(f"  🎵 Test clips:  {test_count}")

    if missing > 0:
        logger.error(f"  ❌ {missing} audio files referenced in metadata but not found on disk")
        ok = False

    if empty_text > 0:
        logger.warning(f"  ⚠️  {empty_text} entries have empty transcriptions (will be skipped by trainer)")

    if ok:
        logger.info("  ✅ Dataset structure valid")

    return ok


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_dataset(dataset_dir: Path, append: bool, private: bool) -> None:
    """
    Load the AudioFolder dataset and push it to HuggingFace Hub.
    """
    try:
        from datasets import load_dataset, Audio
        from huggingface_hub import HfApi, login
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error("Install with: pip install datasets huggingface_hub soundfile")
        sys.exit(1)

    # ── Verify HF credentials ──────────────────────────────────────────────────
    api = HfApi()
    try:
        user = api.whoami()
        logger.info(f"✅ Logged in as: {user['name']}")
    except Exception:
        logger.error("❌ Not logged in to HuggingFace.")
        logger.error("   Run: huggingface-cli login")
        sys.exit(1)

    # ── Ensure the dataset repo exists ─────────────────────────────────────────
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE)
        logger.info(f"📦 Repository exists: {HF_REPO_ID}")
    except Exception:
        logger.info(f"📦 Creating new dataset repository: {HF_REPO_ID}")
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            private=private,
            exist_ok=True,
        )

    # ── Load local AudioFolder ─────────────────────────────────────────────────
    logger.info(f"\n📂 Loading AudioFolder from: {dataset_dir}")
    logger.info("   (This may take a moment for large datasets...)")

    dataset = load_dataset(
        "audiofolder",
        data_dir=str(dataset_dir),
        trust_remote_code=True,
    )

    # Cast audio column to ensure 16kHz
    logger.info("🔊 Casting audio to 16kHz...")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    logger.info(f"\n📊 Dataset splits loaded:")
    for split_name, split_data in dataset.items():
        logger.info(f"   {split_name}: {len(split_data)} examples")

    # ── Push to Hub ────────────────────────────────────────────────────────────
    logger.info(f"\n🚀 Pushing to Hub: {HF_REPO_ID}")
    logger.info("   (Large datasets may take several minutes...)")

    push_kwargs = {
        "repo_id": HF_REPO_ID,
        "private": private,
    }

    dataset.push_to_hub(**push_kwargs)

    hub_url = f"https://huggingface.co/datasets/{HF_REPO_ID}"
    logger.info(f"\n✅ Dataset successfully uploaded!")
    logger.info(f"   View at: {hub_url}")
    logger.info(f"\nNext step — fine-tune the model:")
    logger.info(f"   python src/training.py")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=f"Upload AudioFolder dataset to HuggingFace Hub ({HF_REPO_ID})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Before running this script, authenticate with HuggingFace:
    huggingface-cli login

Examples:
    # Standard upload
    python scripts/upload_dataset.py --dataset ./unity_dataset

    # Upload as public dataset
    python scripts/upload_dataset.py --dataset ./unity_dataset --public

Target repository:
    https://huggingface.co/datasets/{HF_REPO_ID}
        """,
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to the prepared AudioFolder dataset directory"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing dataset on the Hub (default: overwrite)"
    )
    parser.add_argument(
        "--public", action="store_true",
        help="Make the dataset public (default: private)"
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip local dataset validation before uploading"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dataset_dir = Path(args.dataset).expanduser().resolve()

    if not dataset_dir.exists():
        logger.error(f"❌ Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  Unity Whisper Dataset Upload")
    logger.info("=" * 60)
    logger.info(f"  Source:     {dataset_dir}")
    logger.info(f"  Target:     https://huggingface.co/datasets/{HF_REPO_ID}")
    logger.info(f"  Visibility: {'public' if args.public else 'private'}")
    logger.info("=" * 60)

    if not args.skip_validation:
        valid = validate_dataset(dataset_dir)
        if not valid:
            logger.error("\n❌ Validation failed. Fix errors before uploading.")
            logger.error("   Use --skip-validation to bypass (not recommended).")
            sys.exit(1)

    upload_dataset(
        dataset_dir=dataset_dir,
        append=args.append,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
