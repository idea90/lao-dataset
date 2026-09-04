"""
Native PyTorch Fine-Tuning Script for Lao Language & Grammar AI Models.

Features:
- Native PyTorch training loop with DataLoader & dynamic batch padding
- PEFT / LoRA parameter-efficient fine-tuning (under 4GB VRAM requirement)
- Masked Loss (-100 on prompt tokens; loss computed strictly on assistant answers)
- Mixed precision training (fp16 / bf16) with PyTorch GradScaler
- Cosine learning rate scheduling with warmup
- Epoch-level validation with cross-entropy loss & perplexity tracking
- Checkpoint saving and post-training interactive Lao generation test
"""

import argparse
import io
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure Windows PowerShell UTF-8 output
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LaoPyTorchTrainer")


def check_dependencies():
    """Verify PyTorch and Transformers installation."""
    missing = []
    try:
        import torch
    except ImportError:
        missing.append("torch")
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    try:
        import peft
    except ImportError:
        missing.append("peft")

    if missing:
        logger.error(f"Missing required libraries: {missing}. Please install via: pip install {' '.join(missing)}")
        sys.exit(1)


check_dependencies()

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


SYSTEM_PROMPT = (
    "You are an expert AI linguistic assistant specializing in the Lao language (ພາສາລາວ), "
    "grammar rules, syntactic structures, classifiers, phonology, and vocabulary."
)


class LaoInstructTorchDataset(Dataset):
    """
    PyTorch Dataset for Lao instruction-tuning data.
    Constructs conversations and masks prompt tokens with -100 so loss
    is calculated solely on the model's generated response.
    """
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[Dict[str, Any]] = []

        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]

        # Extract prompt and answer
        if "messages" in item and len(item["messages"]) >= 2:
            user_text = item["messages"][0]["content"]
            assistant_text = item["messages"][1]["content"]
        else:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            user_text = f"{inst}\n\n{inp}".strip() if inp else inst
            assistant_text = item.get("output", "")

        # Format prompt with ChatML / standard template
        prompt_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": assistant_text}]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt_str = self.tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            full_str = self.tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        else:
            # Fallback ChatML style formatting
            prompt_str = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            full_str = f"{prompt_str}{assistant_text}<|im_end|>\n"

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)

        # Truncate if exceeding max_length
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]

        # Construct labels: -100 for all prompt tokens, actual token IDs for assistant response
        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]

        # Ensure EOS token at the end
        if self.tokenizer.eos_token_id and full_ids[-1] != self.tokenizer.eos_token_id and len(full_ids) < self.max_length:
            full_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        attention_mask = [1] * len(full_ids)

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class DynamicPaddingCollator:
    """Dynamically pads each batch to the longest sequence in that batch."""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_max_len = max(x["input_ids"].size(0) for x in batch)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = batch_max_len - seq_len

            # Pad input_ids with pad_token_id
            padded_input_ids.append(torch.cat([
                item["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ]))

            # Pad attention_mask with 0
            padded_attention_mask.append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long)
            ]))

            # Pad labels with -100 (ignored by CrossEntropyLoss)
            padded_labels.append(torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long)
            ]))

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels)
        }


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Run validation loop and return average loss & perplexity."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                total_steps += 1

    avg_loss = total_loss / max(1, total_steps)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent overflow
    return avg_loss, perplexity


def run_sample_generation(model, tokenizer, device: torch.device, prompts: List[str]):
    """Test model generation on Lao linguistic prompts."""
    model.eval()
    print("\n" + "=" * 70)
    print("      🧪 SAMPLE INFERENCE TEST (LAO GRAMMAR & LANGUAGE)")
    print("=" * 70)

    for idx, prompt_text in enumerate(prompts, start=1):
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            input_text = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        # Slice off input tokens to get only response
        generated_ids = output_tokens[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"\n[Test Prompt {idx}]: {prompt_text}")
        print(f"[Model Output]:\n{response}\n")
    print("=" * 70 + "\n")


def train(args):
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device selection
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Using GPU: {gpu_name} (Total VRAM: {gpu_mem:.2f} GB)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU for training.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Prepare PyTorch Datasets & DataLoaders
    train_file = Path(args.train_file)
    val_file = Path(args.val_file)

    logger.info(f"Loading training data from {train_file}...")
    train_dataset = LaoInstructTorchDataset(train_file, tokenizer, max_length=args.max_length)
    val_dataset = LaoInstructTorchDataset(val_file, tokenizer, max_length=args.max_length) if val_file.exists() else None

    collator = DynamicPaddingCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator
    ) if val_dataset else None

    # 3. Load Model with Base Weights
    logger.info(f"Loading base model: {args.model_name}...")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if args.cpu:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    # 4. Configure LoRA
    logger.info(f"Configuring LoRA adapter (rank={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.to(device)

    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable: {100 * trainable_params / all_params:.2f}%"
    )

    # 5. Optimizer & Cosine Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_training_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_training_steps)
    )

    # Mixed precision scaler
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 6. PyTorch Training Loop
    logger.info(f"Starting PyTorch training for {args.epochs} epochs ({total_training_steps} optimization steps)...")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum_steps

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                step_count += 1

            if step % args.log_interval == 0 or step == len(train_loader):
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch [{epoch}/{args.epochs}] | Step [{step}/{len(train_loader)}] | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f} | LR: {current_lr:.2e}"
                )

        epoch_train_loss = running_loss / len(train_loader)

        # Validation Step
        if val_loader:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            logger.info(
                f">>> Epoch {epoch} Complete | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}"
            )
            history.append({
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl
            })

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_dir = output_dir / "best_checkpoint"
                logger.info(f"New best val loss ({val_loss:.4f})! Saving to {best_ckpt_dir}...")
                model.save_pretrained(best_ckpt_dir)
                tokenizer.save_pretrained(best_ckpt_dir)
        else:
            logger.info(f">>> Epoch {epoch} Complete | Train Loss: {epoch_train_loss:.4f}")
            history.append({"epoch": epoch, "train_loss": epoch_train_loss})

    # Save final model
    final_ckpt_dir = output_dir / "final_model"
    logger.info(f"Saving final trained adapter to {final_ckpt_dir}...")
    model.save_pretrained(final_ckpt_dir)
    tokenizer.save_pretrained(final_ckpt_dir)

    # Save training metrics history
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("PyTorch fine-tuning completed successfully!")

    # 7. Post-Training Sample Inference Test
    sample_prompts = [
        "Explain the canonical word order in Lao grammar with an example.",
        "Choose the correct classifier for counting dogs: 'ຂ້ອຍມີຫມາ 2 [___]'",
        "What tone does the high consonant 'ຂ' produce in an unmarked syllable?"
    ]
    run_sample_generation(model, tokenizer, device, sample_prompts)


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Fine-Tuning Script for Lao Language & Grammar",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Model & Data paths
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model from Hugging Face (e.g. Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--train-file", type=str, default="data/final/splits/instruct_train.jsonl",
                        help="Path to instruction training JSONL file")
    parser.add_argument("--val-file", type=str, default="data/final/splits/instruct_val.jsonl",
                        help="Path to instruction validation JSONL file")
    parser.add_argument("--output-dir", type=str, default="models/lao-llama-adapter",
                        help="Directory to save fine-tuned LoRA weights")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate for LoRA")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.10, help="Linear warmup ratio")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")

    # LoRA settings
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-interval", type=int, default=5, help="Steps between logging")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training instead of CUDA")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
