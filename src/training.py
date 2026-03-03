"""
training.py — Phase 3: Fine-tune Whisper-Small with LoRA for Unity vocabulary

Fine-tunes openai/whisper-small using Low-Rank Adaptation (LoRA) via the PEFT
library on the Unity tutorial audio dataset stored at:
    https://huggingface.co/datasets/sapplebaum/unity-pt-videos

The fine-tuned adapter is saved and pushed to:
    https://huggingface.co/sapplebaum/unity-whisper-small

Optimised for Apple M3 Max (MPS backend, bf16, no CUDA required).

Usage:
    # Ensure you are logged in first:
    huggingface-cli login

    # Run training:
    python src/training.py

    # Resume from a checkpoint:
    python src/training.py --resume-from-checkpoint ./unity-whisper-small/checkpoint-500

    # Dry-run (2 steps only, no push):
    python src/training.py --dry-run

Requirements:
    pip install transformers datasets evaluate jiwer accelerate peft \
                soundfile tensorboard
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_MODEL_ID   = "openai/whisper-small"
DATASET_REPO_ID = "sapplebaum/unity-pt-videos"
HUB_MODEL_ID    = "sapplebaum/unity-whisper-small"
OUTPUT_DIR      = "./unity-whisper-small"
LANGUAGE        = "english"
TASK            = "transcribe"

# LoRA hyperparameters
LORA_RANK       = 32     # Number of decomposition dimensions (higher = more capacity)
LORA_ALPHA      = 64     # Scaling factor (typically 2× rank)
LORA_DROPOUT    = 0.05
# Only adapt the query and value projection layers — standard for Whisper LoRA
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training hyperparameters (tuned for M3 Max, 64GB unified memory)
TRAIN_BATCH_SIZE       = 16   # Fits comfortably in MPS memory at bf16
EVAL_BATCH_SIZE        = 8
GRADIENT_ACCUM_STEPS   = 1    # Effective batch = 16; increase if memory is tight
LEARNING_RATE          = 1e-5
WARMUP_STEPS           = 200
MAX_STEPS              = 4000  # ~4 passes over a ~1000-clip dataset; adjust as needed
SAVE_EVAL_STEPS        = 500
LOGGING_STEPS          = 25
GENERATION_MAX_LENGTH  = 225


# ── Data Collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input features and labels to a uniform length within each batch.
    Labels are padded with -100 so that CrossEntropyLoss ignores padding tokens.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], "torch.Tensor"]]]
    ) -> Dict[str, "torch.Tensor"]:
        import torch

        # ── Input features (log-mel spectrograms) ─────────────────────────────
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # ── Labels (tokenised transcriptions) ─────────────────────────────────
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id (0) with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Whisper decoder always starts with the BOS token; strip it from labels
        # to avoid the model learning to predict BOS as the first output token.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── Preprocessing ──────────────────────────────────────────────────────────────

def make_prepare_dataset_fn(processor):
    """Return a dataset-mapping function bound to the given processor."""

    def prepare_dataset(batch):
        """Convert raw audio + text into model-ready input_features and labels."""
        audio = batch["audio"]

        # Extract log-mel spectrogram features (float32 array of shape [80, 3000])
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]

        # Tokenise the transcription text into label ids
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

        return batch

    return prepare_dataset


# ── WER Metric ─────────────────────────────────────────────────────────────────

def make_compute_metrics_fn(processor):
    """Return a metric-computation function bound to the given processor."""
    import evaluate
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids

        # -100 → pad token so the tokenizer can decode labels
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        logger.info(f"  📊 WER: {wer:.2f}%")
        return {"wer": wer}

    return compute_metrics


# ── Device Detection ───────────────────────────────────────────────────────────

def detect_device() -> str:
    """
    Detect the best available device.
    Returns 'mps' on Apple Silicon, 'cuda' on NVIDIA, else 'cpu'.
    """
    import torch
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("🍎 Apple MPS (Metal) backend detected — using MPS")
        return "mps"
    if torch.cuda.is_available():
        logger.info(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name()}")
        return "cuda"
    logger.info("🖥️  No GPU found — falling back to CPU (training will be slow)")
    return "cpu"


# ── Main Training Function ─────────────────────────────────────────────────────

def train(
    resume_from_checkpoint: str = None,
    dry_run: bool = False,
    push_to_hub: bool = True,
) -> None:
    """
    Main fine-tuning entry point.

    Steps:
      1. Load dataset from HuggingFace Hub
      2. Load Whisper processor (feature extractor + tokenizer)
      3. Preprocess the dataset (audio → mel spectrogram, text → token ids)
      4. Load base Whisper-Small model
      5. Wrap model with LoRA via PEFT
      6. Configure Seq2SeqTrainer and train
      7. Push adapter + tokenizer to HuggingFace Hub
    """

    # ── Imports ────────────────────────────────────────────────────────────────
    try:
        import torch
        from datasets import load_dataset, Audio
        from transformers import (
            WhisperProcessor,
            WhisperForConditionalGeneration,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.error(
            "Install with:\n"
            "  pip install transformers datasets evaluate jiwer accelerate peft "
            "soundfile tensorboard"
        )
        sys.exit(1)

    device = detect_device()
    use_bf16 = device in ("mps", "cuda")  # bf16 works on MPS and modern CUDA

    # ── 1. Load Dataset ────────────────────────────────────────────────────────
    logger.info(f"\n📂 Loading dataset: {DATASET_REPO_ID}")
    dataset = load_dataset(DATASET_REPO_ID, trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    logger.info(f"   Train: {len(dataset['train'])} examples")
    logger.info(f"   Test:  {len(dataset['test'])} examples")

    # Tiny subset for dry-run validation
    if dry_run:
        logger.warning("⚠️  DRY RUN — using only 4 train + 2 test examples")
        dataset["train"] = dataset["train"].select(range(min(4, len(dataset["train"]))))
        dataset["test"]  = dataset["test"].select(range(min(2, len(dataset["test"]))))

    # ── 2. Load Processor ──────────────────────────────────────────────────────
    logger.info(f"\n🔧 Loading processor: {BASE_MODEL_ID}")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL_ID,
        language=LANGUAGE,
        task=TASK,
    )

    # ── 3. Preprocess Dataset ──────────────────────────────────────────────────
    logger.info("\n⚙️  Preprocessing dataset (audio → mel + text → tokens)...")
    prepare_fn = make_prepare_dataset_fn(processor)

    # num_proc > 1 can cause issues with soundfile on macOS; keep at 1 for safety
    dataset = dataset.map(
        prepare_fn,
        remove_columns=dataset.column_names["train"],
        num_proc=1,
        desc="Preparing dataset",
    )
    logger.info("   ✅ Preprocessing complete")

    # ── 4. Load Base Model ─────────────────────────────────────────────────────
    logger.info(f"\n🤖 Loading base model: {BASE_MODEL_ID}")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)

    # Configure decoder to always transcribe in English
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None  # Let LoRA learn freely

    # ── 5. Apply LoRA ──────────────────────────────────────────────────────────
    logger.info(f"\n🔗 Applying LoRA (r={LORA_RANK}, α={LORA_ALPHA})")
    logger.info(f"   Target modules: {LORA_TARGET_MODULES}")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows ~1% of params are trainable

    # ── 6. Data Collator ───────────────────────────────────────────────────────
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ── 7. Training Arguments ──────────────────────────────────────────────────
    max_steps = 4 if dry_run else MAX_STEPS

    # Note on gradient_checkpointing:
    # LoRA + MPS + gradient_checkpointing can be unstable. We disable it here
    # for MPS; it's safe to re-enable for CUDA if VRAM is constrained.
    use_gradient_checkpointing = (device == "cuda")

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch sizes
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,

        # Precision (bf16 for MPS/CUDA, fp32 for CPU)
        bf16=use_bf16,
        fp16=False,  # fp16 is unstable on MPS; always use bf16

        # Learning rate schedule
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS if not dry_run else 0,
        max_steps=max_steps,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=SAVE_EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_EVAL_STEPS,
        save_total_limit=3,           # Keep only the 3 most recent checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,      # Lower WER is better

        # Generation (needed for predict_with_generate)
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,

        # Logging
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],

        # Gradient checkpointing (disabled on MPS, optional on CUDA)
        gradient_checkpointing=use_gradient_checkpointing,

        # HuggingFace Hub
        push_to_hub=push_to_hub and not dry_run,
        hub_model_id=HUB_MODEL_ID if (push_to_hub and not dry_run) else None,

        # Dataloader
        dataloader_num_workers=0,     # 0 avoids multiprocessing conflicts on macOS
        dataloader_pin_memory=False,  # Pin memory not supported on MPS
    )

    # ── 8. Trainer ─────────────────────────────────────────────────────────────
    compute_metrics_fn = make_compute_metrics_fn(processor)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        processing_class=processor.feature_extractor,
    )

    # ── 9. Train ───────────────────────────────────────────────────────────────
    logger.info("\n🚀 Starting training...")
    logger.info(f"   Model:         {BASE_MODEL_ID} + LoRA")
    logger.info(f"   Dataset:       {DATASET_REPO_ID}")
    logger.info(f"   Device:        {device} | bf16={use_bf16}")
    logger.info(f"   Max steps:     {max_steps}")
    logger.info(f"   Batch size:    {TRAIN_BATCH_SIZE} (per device)")
    logger.info(f"   Output dir:    {OUTPUT_DIR}")
    logger.info(f"   Push to Hub:   {push_to_hub and not dry_run} → {HUB_MODEL_ID}")
    logger.info("")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── 10. Save & Push ────────────────────────────────────────────────────────
    logger.info("\n💾 Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    if push_to_hub and not dry_run:
        logger.info(f"🚀 Pushing to HuggingFace Hub: {HUB_MODEL_ID}")
        trainer.push_to_hub()
        processor.push_to_pretrained(HUB_MODEL_ID)
        logger.info(f"✅ Model available at: https://huggingface.co/{HUB_MODEL_ID}")

    logger.info(f"\n🎉 Training complete!")
    logger.info(f"   Adapter saved to: {OUTPUT_DIR}/")
    logger.info(f"\nNext step — convert for deployment:")
    logger.info(f"   python scripts/convert_model.py --adapter {OUTPUT_DIR}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-Small with LoRA for Unity vocabulary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Before running, log in to HuggingFace:
    huggingface-cli login

Examples:
    # Standard training run
    python src/training.py

    # Dry run (4 steps, no upload) to verify setup
    python src/training.py --dry-run

    # Resume from checkpoint
    python src/training.py --resume-from-checkpoint ./unity-whisper-small/checkpoint-500

    # Train but don't push to Hub
    python src/training.py --no-push
        """,
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        metavar="CHECKPOINT_DIR",
        help="Path to a checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run only 4 training steps with 6 examples to verify the setup"
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Do not push the trained model to HuggingFace Hub"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug-level logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("  Unity Whisper Fine-Tuning (LoRA)")
    logger.info("=" * 60)
    logger.info(f"  Base model:  {BASE_MODEL_ID}")
    logger.info(f"  Dataset:     {DATASET_REPO_ID}")
    logger.info(f"  Output:      {HUB_MODEL_ID}")
    logger.info(f"  LoRA rank:   {LORA_RANK} | alpha: {LORA_ALPHA}")
    logger.info(f"  Dry run:     {args.dry_run}")
    logger.info("=" * 60)

    train(
        resume_from_checkpoint=args.resume_from_checkpoint,
        dry_run=args.dry_run,
        push_to_hub=not args.no_push,
    )


if __name__ == "__main__":
    main()
