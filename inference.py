"""
Interactive Inference & Evaluation CLI for Fine-Tuned Lao Language Models.

Allows testing the fine-tuned LoRA model interactively or evaluating it
against test split benchmarks (data/final/splits/instruct_test.jsonl).
"""

import argparse
import io
import json
import sys
from pathlib import Path
from typing import List, Optional

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = (
    "You are an expert AI linguistic assistant specializing in the Lao language (ພາສາລາວ), "
    "grammar rules, syntactic structures, classifiers, phonology, and vocabulary."
)


def load_fine_tuned_model(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    device: Optional[torch.device] = None
):
    """Load base model and optionally attach fine-tuned PEFT LoRA adapter."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    if device.type == "cpu":
        dtype = torch.float32

    print(f"[*] Loading tokenizer: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[*] Loading base model: {base_model_name} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    if adapter_path and Path(adapter_path).exists():
        print(f"[*] Loading LoRA adapter weights from: {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        print("[!] No adapter path provided or not found. Running with base model weights.")
        model = base_model

    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_response(
    model,
    tokenizer,
    device: torch.device,
    user_prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> str:
    """Generate a response for a Lao linguistic query."""
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        input_text = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_ids = output_tokens[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_benchmark(model, tokenizer, device, test_file: Path, max_samples: int = 10):
    """Run model on test split samples and display comparison."""
    if not test_file.exists():
        print(f"[!] Test file not found: {test_file}")
        return

    print("\n" + "=" * 75)
    print(f"       📊 BENCHMARK TEST ON TEST SPLIT: {test_file.name}")
    print("=" * 75)

    with open(test_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()][:max_samples]

    for idx, item in enumerate(lines, start=1):
        inst = item.get("instruction", "")
        ground_truth = item.get("output", "")
        prediction = generate_response(model, tokenizer, device, inst, max_new_tokens=200)

        print(f"\n--- [Test Sample {idx}/{len(lines)}] Category: {item.get('category', 'general')} ---")
        print(f"Instruction:\n  {inst}")
        print(f"\nModel Prediction:\n  {prediction}")
        print(f"\nGround Truth (Reference):\n  {ground_truth}")
        print("-" * 75)


def interactive_chat(model, tokenizer, device):
    """Launch interactive CLI chat session."""
    print("\n" + "=" * 75)
    print("   🇱🇦 Lao Language & Grammar AI Assistant - Interactive Session")
    print("   Type your questions about Lao grammar, classifiers, or vocabulary.")
    print("   Type 'exit' or 'quit' to end.")
    print("=" * 75 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye! (ລາກ່ອນ)\n")
                break

            response = generate_response(model, tokenizer, device, user_input)
            print(f"\nLao AI Assistant >\n{response}\n")
            print("-" * 75)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.\n")
            break


def main():
    parser = argparse.ArgumentParser(description="Lao AI Interactive Inference & Test CLI")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="models/lao-llama-adapter/final_model")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on test split")
    parser.add_argument("--test-file", type=str, default="data/final/splits/instruct_test.jsonl")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples to benchmark")

    args = parser.parse_args()

    model, tokenizer, device = load_fine_tuned_model(args.base_model, args.adapter_path)

    if args.benchmark:
        run_benchmark(model, tokenizer, device, Path(args.test_file), max_samples=args.samples)
    else:
        interactive_chat(model, tokenizer, device)


if __name__ == "__main__":
    main()
