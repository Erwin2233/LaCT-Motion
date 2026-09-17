"""Test generation quality of a CoconutMotion checkpoint.

Runs generation on a subset of training data using the same parameters as
GRPO training, to diagnose whether the checkpoint itself has generation issues
or whether problems only appear during GRPO.

Two modes are tested for each sample:
  1. Greedy (temperature=0, do_sample=False) — baseline quality
  2. Sampling (same params as GRPO config) — GRPO-identical generation

Usage:
    python test_generation.py --config options/grpo/t2m_grpo.yaml [--num_samples 20] [--seed 42]

Output:
    - Console: per-sample diagnostics + summary statistics
    - File: outputs/grpo/test_generation.jsonl (detailed records)
"""

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from coconut_motion import CoconutMotion
from dataset import (
    get_dataset,
    get_question_latent_dataset,
    MotionCollator,
)
from utils import parse_motion_tokens

# ---------- Patterns ----------
_MOTION_TAG_PATTERN = re.compile(r"<Motion>(.*?)</Motion>", re.DOTALL)
_MOTION_CODE_PATTERN = re.compile(r"<Motion_\d+>")
_INPUT_CAPTION_PATTERN = re.compile(
    r"###\s*Input:\s*(.*?)(?:<\|im_end\|>|$)",
    flags=re.IGNORECASE | re.DOTALL,
)


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)


def setup_tokenizer_and_model(configs):
    """Same as train_grpo.py — must match exactly."""
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_id, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft_path = getattr(configs, "sft_checkpoint", None)
    if sft_path and os.path.exists(sft_path):
        print(f"Loading SFT checkpoint from: {sft_path}")
        model = AutoModelForCausalLM.from_pretrained(
            sft_path,
            torch_dtype=torch.bfloat16 if configs.bf16 else torch.float32,
            attn_implementation=getattr(
                configs, "attn_implementation", "flash_attention_2"
            ),
            trust_remote_code=True,
        )
    else:
        print(f"Loading base model: {configs.model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            configs.model_id,
            torch_dtype=torch.bfloat16 if configs.bf16 else torch.float32,
            attn_implementation=getattr(
                configs, "attn_implementation", "flash_attention_2"
            ),
            trust_remote_code=True,
        )

    tokenizer.add_tokens(["<Motion>", "</Motion>"])
    for i in range(configs.nb_code):
        tokenizer.add_tokens([f"<Motion_{i}>"])
    tokenizer.add_tokens(["<think>", "</think>"])
    tokenizer.add_tokens(["<|start-latent|>", "<|end-latent|>", "<|latent|>"])
    model.resize_token_embeddings(len(tokenizer))

    embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"[setup] tokenizer vocab={len(tokenizer)}, embedding rows={embed_size}")
    assert embed_size == len(tokenizer)

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    print(f"[setup] latent_id={latent_id}, start_id={start_id}, end_id={end_id}")

    return model, tokenizer, latent_id, start_id, end_id


def extract_caption(question: str) -> str:
    match = _INPUT_CAPTION_PATTERN.search(question)
    if match:
        return match.group(1).strip()
    return ""


@torch.no_grad()
def generate_one(
    model, input_ids, attention_mask, tokenizer,
    max_new_tokens, temperature, do_sample, top_p, top_k,
    num_generations, force_motion_start,
):
    """Generate completions for a single batch, return list of decoded strings."""
    batch_size = input_ids.shape[0]

    if force_motion_start:
        motion_start_id = tokenizer.convert_tokens_to_ids("<Motion>")
        motion_col = torch.full(
            (batch_size, 1), motion_start_id,
            device=input_ids.device, dtype=input_ids.dtype,
        )
        input_ids = torch.cat([input_ids, motion_col], dim=1)
        attn_col = torch.ones(
            (batch_size, 1), device=input_ids.device, dtype=attention_mask.dtype,
        )
        attention_mask = torch.cat([attention_mask, attn_col], dim=1)

    prompt_len = input_ids.shape[1]

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        num_generations=num_generations,
    )

    results = []
    for i in range(batch_size * num_generations):
        comp_ids = outputs[i, prompt_len:].tolist()
        trimmed = []
        for tid in comp_ids:
            if tid == tokenizer.eos_token_id:
                break
            trimmed.append(tid)
        text = ("<Motion>" if force_motion_start else "") + tokenizer.decode(
            trimmed, skip_special_tokens=False
        )
        results.append(text)
    return results


def analyze_completion(text):
    """Return diagnostic dict for a single completion."""
    has_tag = bool(_MOTION_TAG_PATTERN.search(text))
    codes = _MOTION_CODE_PATTERN.findall(text)
    n_codes = len(codes)
    # Check for non-motion text (garbage) before </Motion>
    inner = ""
    m = _MOTION_TAG_PATTERN.search(text)
    if m:
        inner = m.group(1)
    non_code_chars = _MOTION_CODE_PATTERN.sub("", inner).strip()
    has_garbage = len(non_code_chars) > 0
    return {
        "has_motion_tag": has_tag,
        "n_codes": n_codes,
        "has_garbage": has_garbage,
        "garbage_text": non_code_chars[:100] if has_garbage else "",
        "length": len(text),
    }


def main():
    parser = argparse.ArgumentParser(description="Test CoconutMotion generation quality")
    parser.add_argument("--config", type=str, default="options/grpo/t2m_grpo.yaml")
    parser.add_argument("--num_samples", type=int, default=3000,
                        help="Number of samples to test")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for generation test (default: 32)")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="G per prompt for sampling (default: from config)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="test_generation.jsonl",
                        help="Output JSONL path (default: <save_path>/test_generation.jsonl)")
    parser.add_argument("--device", type=str, default="cuda:4")
    args = parser.parse_args()

    configs = load_config(args.config)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # Generation params — identical to GRPO
    temperature = configs.temperature         # 0.7
    top_p = getattr(configs, "top_p", 1.0)    # 0.9
    top_k = getattr(configs, "top_k", 0)      # 0
    max_new_tokens = configs.max_new_tokens    # 256
    G = args.num_generations or configs.num_generations  # 8
    force_motion = getattr(configs, "force_motion_start", True)

    print("=" * 70)
    print("CoconutMotion Generation Test")
    print("=" * 70)
    print(f"  Checkpoint:      {getattr(configs, 'sft_checkpoint', configs.model_id)}")
    print(f"  Num samples:     {args.num_samples}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Device:          {device}")
    print()
    print("  GRPO-identical sampling params:")
    print(f"    temperature:   {temperature}")
    print(f"    top_p:         {top_p}")
    print(f"    top_k:         {top_k}")
    print(f"    max_new_tokens:{max_new_tokens}")
    print(f"    num_generations (G): {G}")
    print(f"    force_motion_start:  {force_motion}")
    print("=" * 70)

    # === Model ===
    base_model, tokenizer, latent_id, start_id, end_id = setup_tokenizer_and_model(configs)
    coconut_model = CoconutMotion(
        base_model, latent_id, start_id, end_id, tokenizer.eos_token_id
    )
    coconut_model = coconut_model.to(device)
    coconut_model.eval()

    # === Dataset ===
    max_steps = getattr(configs, "max_think_steps", 20)
    train_tokenized = get_dataset(configs.train_path, tokenizer, max_steps=max_steps)
    scheduled_stage = configs.max_latent_stage

    train_dataset = get_question_latent_dataset(
        scheduled_stage, train_tokenized, configs, start_id, latent_id, end_id
    )

    # Ground truth + captions
    train_gts = {}
    for sample in train_tokenized:
        answer_text = tokenizer.decode(sample["answer_tokenized"], skip_special_tokens=False)
        train_gts[sample["idx"]] = parse_motion_tokens(answer_text)

    with open(configs.train_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    train_captions = {
        idx: extract_caption(sample.get("question", ""))
        for idx, sample in enumerate(raw_data)
    }

    collator = MotionCollator(tokenizer=tokenizer, latent_id=latent_id)

    num_samples = min(args.num_samples, len(train_dataset))

    # === Output file ===
    save_dir = getattr(configs, "save_path", "./outputs/grpo")
    os.makedirs(save_dir, exist_ok=True)
    out_path = args.output or os.path.join(save_dir, "test_generation.jsonl")
    out_file = open(out_path, "w", encoding="utf-8")

    # === Run tests ===
    # Stats accumulators
    stats = {
        "greedy": {"total": 0, "has_tag": 0, "has_codes": 0, "has_garbage": 0, "total_codes": 0},
        "sampling": {"total": 0, "has_tag": 0, "has_codes": 0, "has_garbage": 0, "total_codes": 0},
    }

    batch_size = max(1, args.batch_size)
    print(f"\nTesting {num_samples} samples (batch_size={batch_size})...\n")

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_features = []
        batch_meta = []

        for sample_i in range(batch_start, batch_end):
            sample = train_dataset[sample_i]
            idx = sample["idx"]
            caption = train_captions.get(idx, "")
            gt_codes = train_gts.get(idx, [])

            batch_features.append({
                "input_ids": list(sample["input_ids"]),
                "attention_mask": list(sample["attention_mask"]),
                "position_ids": list(sample["position_ids"]),
                "idx": idx,
            })
            batch_meta.append({
                "sample_i": sample_i,
                "idx": idx,
                "caption": caption,
                "gt_n_codes": len(gt_codes),
            })

        batch = collator(batch_features)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        actual_batch = input_ids.shape[0]

        print(
            f"=== Batch {batch_start // batch_size + 1} | "
            f"samples {batch_start + 1}-{batch_end} / {num_samples} ==="
        )

        # --- Greedy for whole batch ---
        greedy_results = generate_one(
            coconut_model, input_ids, attention_mask, tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.0, do_sample=False, top_p=1.0, top_k=0,
            num_generations=1, force_motion_start=force_motion,
        )

        # --- Sampling (G completions) for whole batch ---
        sampling_results = generate_one(
            coconut_model, input_ids, attention_mask, tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True,
            top_p=top_p, top_k=top_k,
            num_generations=G, force_motion_start=force_motion,
        )

        for local_i in range(actual_batch):
            meta = batch_meta[local_i]
            sample_i = meta["sample_i"]
            idx = meta["idx"]
            caption = meta["caption"]
            gt_n_codes = meta["gt_n_codes"]

            print(f"--- Sample {sample_i+1}/{num_samples} (idx={idx}) ---")
            print(f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
            print(f"  GT codes: {gt_n_codes}")

            greedy_text = greedy_results[local_i]
            greedy_diag = analyze_completion(greedy_text)
            stats["greedy"]["total"] += 1
            stats["greedy"]["has_tag"] += int(greedy_diag["has_motion_tag"])
            stats["greedy"]["has_codes"] += int(greedy_diag["n_codes"] > 0)
            stats["greedy"]["has_garbage"] += int(greedy_diag["has_garbage"])
            stats["greedy"]["total_codes"] += greedy_diag["n_codes"]

            print(f"  [Greedy]   tag={greedy_diag['has_motion_tag']} | "
                  f"codes={greedy_diag['n_codes']} | "
                  f"garbage={greedy_diag['has_garbage']}")
            if greedy_diag["has_garbage"]:
                print(f"             garbage: {greedy_diag['garbage_text']}")
            print(f"             text: {greedy_text[:200]}{'...' if len(greedy_text) > 200 else ''}")

            out_file.write(json.dumps({
                "sample_idx": idx, "caption": caption,
                "gt_n_codes": gt_n_codes,
                "mode": "greedy", "gen_id": 0,
                "completion": greedy_text,
                **{f"diag_{k}": v for k, v in greedy_diag.items()},
            }, ensure_ascii=False) + "\n")

            sample_sampling_results = sampling_results[local_i * G: (local_i + 1) * G]
            for g, samp_text in enumerate(sample_sampling_results):
                samp_diag = analyze_completion(samp_text)
                stats["sampling"]["total"] += 1
                stats["sampling"]["has_tag"] += int(samp_diag["has_motion_tag"])
                stats["sampling"]["has_codes"] += int(samp_diag["n_codes"] > 0)
                stats["sampling"]["has_garbage"] += int(samp_diag["has_garbage"])
                stats["sampling"]["total_codes"] += samp_diag["n_codes"]

                if g < 3:  # Print first 3 sampling results
                    print(f"  [Sample {g}] tag={samp_diag['has_motion_tag']} | "
                          f"codes={samp_diag['n_codes']} | "
                          f"garbage={samp_diag['has_garbage']}")
                    if samp_diag["has_garbage"]:
                        print(f"             garbage: {samp_diag['garbage_text']}")
                    print(f"             text: {samp_text[:200]}{'...' if len(samp_text) > 200 else ''}")

                out_file.write(json.dumps({
                    "sample_idx": idx, "caption": caption,
                    "gt_n_codes": gt_n_codes,
                    "mode": "sampling", "gen_id": g,
                    "completion": samp_text,
                    **{f"diag_{k}": v for k, v in samp_diag.items()},
                }, ensure_ascii=False) + "\n")

            if len(sample_sampling_results) > 3:
                # Summary for remaining samples
                remaining = sample_sampling_results[3:]
                n_tag = sum(1 for t in remaining if _MOTION_TAG_PATTERN.search(t))
                n_codes_avg = sum(
                    len(_MOTION_CODE_PATTERN.findall(t)) for t in remaining
                ) / max(len(remaining), 1)
                print(f"  [Sample 3-{G-1}] {n_tag}/{len(remaining)} have tags | avg_codes={n_codes_avg:.1f}")

            print()

        out_file.flush()

    out_file.close()

    # === Summary ===
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for mode in ["greedy", "sampling"]:
        s = stats[mode]
        total = max(s["total"], 1)
        avg_codes = s["total_codes"] / total
        print(f"\n  [{mode.upper()}] ({s['total']} completions)")
        print(f"    Has <Motion>...</Motion> tag:  {s['has_tag']}/{s['total']} ({100*s['has_tag']/total:.1f}%)")
        print(f"    Has motion codes:              {s['has_codes']}/{s['total']} ({100*s['has_codes']/total:.1f}%)")
        print(f"    Has garbage text in tag:       {s['has_garbage']}/{s['total']} ({100*s['has_garbage']/total:.1f}%)")
        print(f"    Avg codes per completion:      {avg_codes:.1f}")

    print(f"\n  Detailed results saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
