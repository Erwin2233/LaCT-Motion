"""Batch evaluation with standard motion generation metrics.

Loads a trained CoconutMotion checkpoint and evaluates on the HumanML3D test
set using Motion-R1's evaluation protocol: FID, R-Precision (Top-1/2/3),
Matching Score, and Diversity.

Generation is fully batched (batch_size=32) following Motion-R1's eval_LM.py.
Each batch of captions is tokenized, padded with the MotionCollator for latent
alignment, and decoded in a single model.generate() call.

Usage:
    python evaluate.py configs/t2m_coconut.yaml \
        --checkpoint checkpoints/checkpoint-epoch33

    # With custom repeat times for confidence intervals
    python evaluate.py configs/t2m_coconut.yaml \
        --checkpoint checkpoints/checkpoint-epoch33 \
        --repeat 20 --output eval_results.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import linalg
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------
from build_data import build_question, load_vqvae
from dataset import MotionCollator
from inference import load_model
from utils import Config, parse_motion_tokens, set_seed

# ---------------------------------------------------------------------------
# Evaluation infrastructure (copied from Motion-R1 into eval_infra/)
# ---------------------------------------------------------------------------
from eval_infra import dataset_TM_eval
from eval_infra.evaluator_wrapper import EvaluatorModelWrapper
from eval_infra.get_eval_option import get_opt
from eval_infra.word_vectorizer import WordVectorizer

# ---------------------------------------------------------------------------
# Paths to Motion-R1 *data* resources (checkpoints, glove, datasets).
# Only data files are read from Motion-R1 — no Python imports.
# ---------------------------------------------------------------------------
from _paths import project_path

RESOURCE_ROOT = project_path()

GLOVE_DIR = os.path.join(RESOURCE_ROOT, "glove")
EVAL_OPT_PATH = os.path.join(
    RESOURCE_ROOT, "checkpoints", "t2m", "Comp_v6_KLD005", "opt.txt"
)
DEFAULT_VQVAE_PATH = os.path.join(RESOURCE_ROOT, "ckpt", "vqvae.pth")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress verbose Kineto stage logs when profiler mode is enabled.
os.environ.setdefault("KINETO_LOG_LEVEL", "5")


# ---------------------------------------------------------------------------
# Metric functions (from Motion-R1/utils/evaluation.py)
# ---------------------------------------------------------------------------


def euclidean_distance_matrix(matrix1, matrix2):
    """Pairwise Euclidean distance: (N1, D), (N2, D) -> (N1, N2)."""
    d1 = -2 * np.dot(matrix1, matrix2.T)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)
    d3 = np.sum(np.square(matrix2), axis=1)
    dists = np.sqrt(d1 + d2 + d3)
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = correct_vec | bool_mat[:, i]
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(
        activation[first_indices] - activation[second_indices], axis=1
    )
    return dist.mean()


def calculate_multimodality(activation, multimodality_times):
    """Multimodality metric (same as Motion-Agent).

    Args:
        activation: (N, num_repeats, D) — motion embeddings from multiple
            generations of the same text prompt.
        multimodality_times: number of random pairs to draw from the repeats.

    Returns:
        Scalar mean L2 distance between random pairs of generations.
    """
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def _is_cuda_device(device):
    dev = torch.device(device)
    return dev.type == "cuda" and torch.cuda.is_available()


def _cuda_index(device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    return dev.index if dev.index is not None else torch.cuda.current_device()


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def batch_generate(
    model, tokenizer, captions, configs,
    latent_id, start_id, end_id, collator, device,
    profile_flops=False,
):
    """Build batched input_ids from captions and generate in one forward pass.

    Returns:
        motion_code_batch: List[Tensor] of motion code indices per sample.
        perf_stats: Dict with generated token counts, latency, and optional FLOPs.
    """
    features = []
    for caption in captions:
        prompt = build_question(caption)
        question_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if getattr(configs, "cot", False) or getattr(configs, "no_cot", False):
            tokens = question_ids
        else:
            n_latent = configs.max_latent_stage * configs.c_thought
            tokens = question_ids + [start_id] + [latent_id] * n_latent + [end_id]
        features.append({
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        })

    batch = collator(features)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    position_ids = batch["position_ids"].to(device)

    start_gen = time.perf_counter()
    flops_total = None
    generation_kwargs = {
        "position_ids": position_ids,
        "max_new_tokens": configs.max_new_tokens,
        "synced_gpus": False,
    }
    if getattr(configs, "do_sample", False):
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = getattr(configs, "temperature", 1.0)
        generation_kwargs["top_p"] = getattr(configs, "top_p", 1.0)
        generation_kwargs["top_k"] = getattr(configs, "top_k", 0)
        generation_kwargs["repetition_penalty"] = getattr(
            configs, "repetition_penalty", 1.0
        )

    with torch.inference_mode():
        if profile_flops:
            try:
                profiler_activities = [torch.profiler.ProfilerActivity.CPU]
                if _is_cuda_device(device):
                    profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)
                with torch.profiler.profile(
                    activities=profiler_activities,
                    with_flops=True,
                    profile_memory=False,
                    record_shapes=False,
                ) as prof:
                    outputs = model.generate(
                        input_ids, attention_mask,
                        **generation_kwargs,
                    )
                flops_total = float(
                    sum(
                        evt.flops
                        for evt in prof.key_averages()
                        if getattr(evt, "flops", None)
                    )
                )
            except Exception as exc:
                print(
                    f"[Perf] WARNING: FLOPs profiling failed ({exc}). "
                    "Continue without FLOPs."
                )
                outputs = model.generate(
                    input_ids, attention_mask,
                    **generation_kwargs,
                )
        else:
            outputs = model.generate(
                input_ids, attention_mask,
                **generation_kwargs,
            )

    latency_sec = time.perf_counter() - start_gen

    generated_tokens = 0
    if outputs.shape[1] > input_ids.shape[1]:
        generated_part = outputs[:, input_ids.shape[1]:]
        if tokenizer.pad_token_id is None:
            generated_tokens = int(generated_part.numel())
        else:
            generated_tokens = int(
                (generated_part != tokenizer.pad_token_id).sum().item()
            )

    motion_code_batch = []
    for i in range(len(captions)):
        generated_text = tokenizer.decode(outputs[i], skip_special_tokens=False)
        codes = parse_motion_tokens(generated_text)
        if len(codes) == 0:
            motion_code_batch.append(
                torch.empty(0, dtype=torch.long, device=device)
            )
        else:
            motion_code_batch.append(
                torch.tensor(codes, dtype=torch.long, device=device)
            )

    perf_stats = {
        "input_tokens": int(attention_mask.sum().item()),
        "generated_tokens": generated_tokens,
        "latency_sec": latency_sec,
        "num_samples": len(captions),
        "flops_total": flops_total,
    }
    return motion_code_batch, perf_stats


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluation_test(
    test_loader, model, tokenizer, vqvae, configs,
    latent_id, start_id, end_id, collator, eval_wrapper, device,
    mm_num_repeats=0, mm_num_times=10,
    flops_per_sample_hint=None,
    enable_flops_profile=True,
    flops_mode="estimate",
    model_param_count=None,
):
    """Full evaluation pass following Motion-R1 protocol.

    When mm_num_repeats > 0, each batch is generated mm_num_repeats extra times
    to compute the multimodality metric (following Motion-Agent's protocol).

    Args:
        mm_num_repeats: Number of generation repeats per prompt for multimodality.
            0 = disabled. Must be > mm_num_times to enable calculation.
        mm_num_times: Number of random pairs to draw for multimodality distance.

    Returns:
        (fid, diversity, top1, top2, top3, matching_score_pred, multimodality,
         motion_emb_cos, semantic_cos, perf_dict)
    """
    model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    motion_emb_cos_sum = 0.0    # sum of cos(f_m(pred), f_m(gt))
    semantic_cos_sum = 0.0      # sum of cos(f_m(pred), f_text(caption))
    nb_sample = 0
    total_input_tokens = 0
    total_generated_tokens = 0
    total_gen_time_sec = 0.0
    total_generated_samples = 0
    flops_per_sample = flops_per_sample_hint

    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(_cuda_index(device))

    # Total generation passes: 1 (normal) + mm_num_repeats (for multimodality)
    num_mm_iters = max(mm_num_repeats, 1)

    for batch in tqdm(test_loader, desc="Evaluating"):
        word_embeddings, pos_one_hots, caption, sent_len, pose, m_length, token, name, cot = batch
        bs, seq = pose.shape[:2]

        motion_multimodality_batch = []
        for mm_i in range(num_mm_iters):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1]), device=device)
            pred_len = torch.ones(bs, dtype=torch.long)

            # --- 1. Batch generation ---
            motion_code_batch, perf_stats = batch_generate(
                model, tokenizer, caption, configs,
                latent_id, start_id, end_id, collator, device,
                profile_flops=(
                    flops_mode == "profile"
                    and enable_flops_profile
                    and flops_per_sample is None
                ),
            )
            total_input_tokens += perf_stats["input_tokens"]
            total_generated_tokens += perf_stats["generated_tokens"]
            total_gen_time_sec += perf_stats["latency_sec"]
            total_generated_samples += perf_stats["num_samples"]
            if (
                flops_per_sample is None
                and perf_stats["flops_total"] is not None
                and perf_stats["num_samples"] > 0
            ):
                flops_per_sample = (
                    perf_stats["flops_total"] / perf_stats["num_samples"]
                )

            # --- 2. Per-sample VQ-VAE decoding ---
            for k in range(bs):
                try:
                    index_motion = motion_code_batch[k]
                    if index_motion.numel() == 0:
                        raise RuntimeError("empty motion")
                    pred_pose = vqvae.forward_decoder(index_motion.unsqueeze(0))
                except Exception:
                    index_motion = torch.ones(1, 1, device=device, dtype=torch.long)
                    pred_pose = vqvae.forward_decoder(index_motion)

                cur_len = pred_pose.shape[1]
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k : k + 1, : min(cur_len, seq)] = pred_pose[:, :seq]

            # --- 3. Co-embeddings ---
            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len,
            )

            # Collect embedding for multimodality: (bs, 1, dim)
            if mm_num_repeats > 0:
                motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            # Only compute other metrics on the first pass
            if mm_i == 0:
                pose = pose.to(device).float()
                et, em = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pose, m_length,
                )

                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # --- 4. Per-sample cosine similarities ---
                batch_motion_cos = F.cosine_similarity(em_pred, em, dim=-1)
                motion_emb_cos_sum += batch_motion_cos.sum().item()
                batch_semantic_cos = F.cosine_similarity(em_pred, et_pred, dim=-1)
                semantic_cos_sum += batch_semantic_cos.sum().item()

                print(
                    f"  [batch {nb_sample // bs + 1}] "
                    f"motion_emb_cos={batch_motion_cos.mean().item():.4f}, "
                    f"semantic_cos={batch_semantic_cos.mean().item():.4f}"
                )

                # --- 5. Per-batch R-precision ---
                temp_R, temp_match = calculate_R_precision(
                    et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True,
                )
                R_precision_real += temp_R
                matching_score_real += temp_match

                temp_R, temp_match = calculate_R_precision(
                    et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True,
                )
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        # Cat all repeats for this batch: (bs, num_mm_iters, dim)
        if mm_num_repeats > 0:
            motion_multimodality.append(
                torch.cat(motion_multimodality_batch, dim=1)
            )

    # --- 6. Aggregate across all batches ---
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    diversity_times = 300 if nb_sample > 300 else 100
    diversity_real = calculate_diversity(motion_annotation_np, diversity_times)
    diversity = calculate_diversity(motion_pred_np, diversity_times)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    motion_emb_cos = motion_emb_cos_sum / nb_sample
    semantic_cos = semantic_cos_sum / nb_sample

    multimodality = 0
    if mm_num_repeats > 0:
        # (total_samples, num_mm_iters, dim)
        motion_multimodality_np = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        print(f"[MModality] shape={motion_multimodality_np.shape}, "
              f"num_repeats={mm_num_repeats}, mm_num_times={mm_num_times}")
        if motion_multimodality_np.shape[1] > mm_num_times:
            multimodality = calculate_multimodality(motion_multimodality_np, mm_num_times)
        else:
            print(f"[MModality] WARNING: num_repeats ({mm_num_repeats}) <= "
                  f"mm_num_times ({mm_num_times}), skipping. "
                  f"Increase --mm-num-repeats.")
        print(f"[MModality] multimodality = {multimodality:.4f}")

    msg = (
        f"--> \t FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, "
        f"matching_score_pred. {matching_score_pred:.4f}, "
        f"motion_emb_cos. {motion_emb_cos:.4f}, "
        f"semantic_cos. {semantic_cos:.4f}, "
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)

    per_sample_latency_ms = (
        1000.0 * total_gen_time_sec / total_generated_samples
        if total_generated_samples > 0 else 0.0
    )
    generated_tokens_per_sample = (
        total_generated_tokens / total_generated_samples
        if total_generated_samples > 0 else 0.0
    )
    peak_gpu_memory_bytes = (
        int(torch.cuda.max_memory_allocated(_cuda_index(device)))
        if _is_cuda_device(device) else 0
    )
    peak_gpu_memory_gb = peak_gpu_memory_bytes / (1024 ** 3)
    if flops_mode == "estimate" and model_param_count is not None and total_generated_samples > 0:
        avg_tokens_per_sample = (
            (total_input_tokens + total_generated_tokens) / total_generated_samples
        )
        # Lightweight inference FLOPs estimate for autoregressive decoding.
        flops_per_sample = 2.0 * float(model_param_count) * float(avg_tokens_per_sample)
    flops_msg = f"{flops_per_sample:.3e}" if flops_per_sample is not None else "N/A"
    flops_name = "FLOPs/sample(est.)" if flops_mode == "estimate" else "FLOPs/sample"
    print(
        "[Perf] "
        f"{flops_name}={flops_msg}, "
        f"Peak GPU Memory={peak_gpu_memory_gb:.3f} GB, "
        f"Generated Tokens={total_generated_tokens} "
        f"({generated_tokens_per_sample:.2f}/sample), "
        f"Per-sample Latency={per_sample_latency_ms:.2f} ms"
    )

    perf_dict = {
        "flops_per_sample": flops_per_sample,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "generated_tokens_total": int(total_generated_tokens),
        "generated_tokens_per_sample": float(generated_tokens_per_sample),
        "per_sample_latency_ms": float(per_sample_latency_ms),
        "generated_samples_total": int(total_generated_samples),
        "flops_mode": flops_mode,
    }

    return (
        fid, diversity, R_precision[0], R_precision[1], R_precision[2],
        matching_score_pred, multimodality, motion_emb_cos, semantic_cos,
        perf_dict,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Latent-CoT-Motion: evaluate with standard motion metrics",
    )
    parser.add_argument("config_file", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint", required=True, help="Trained checkpoint directory",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of evaluation repeats for confidence intervals",
    )
    parser.add_argument(
        "--output", default=None, help="Save results JSON to this path",
    )
    parser.add_argument("--vqvae-path", default=DEFAULT_VQVAE_PATH)
    parser.add_argument("--eval-opt-path", default=EVAL_OPT_PATH)
    parser.add_argument("--glove-dir", default=GLOVE_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--multimodality", action="store_true",
        help="Enable multimodality metric (generates each prompt multiple times)",
    )
    parser.add_argument(
        "--mm-num-repeats", type=int, default=20,
        help="Number of generation repeats per prompt for multimodality (default: 20)",
    )
    parser.add_argument(
        "--mm-num-times", type=int, default=10,
        help="Number of random pairs to draw for multimodality distance (default: 10)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flops-mode",
        choices=["estimate", "profile", "off"],
        default="estimate",
        help=(
            "FLOPs mode: estimate (fast, default), "
            "profile (slow, torch.profiler), off (disable FLOPs)"
        ),
    )
    parser.add_argument(
        "--unimo-sampling",
        action="store_true",
        help="Use UniMo eval generation settings: do_sample=True, max_new_tokens=512.",
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)
    generation_strategy = "default"
    if args.unimo_sampling:
        configs.do_sample = True
        configs.max_new_tokens = 512
        configs.temperature = 0.7
        configs.top_p = 0.8
        configs.top_k = 20
        configs.repetition_penalty = 1.05
        generation_strategy = "unimo_sampling"
        print(
            "Using UniMo sampling generation: "
            "do_sample=True, max_new_tokens=512, temperature=0.7, "
            "top_p=0.8, top_k=20, repetition_penalty=1.05"
        )
    set_seed(args.seed)

    device = args.device

    # --- Load our model ---
    print("Loading model...")
    model, tokenizer, latent_id, start_id, end_id = load_model(
        args.checkpoint, configs, device,
    )
    model_param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {model_param_count:,}")

    # --- Load VQ-VAE ---
    print("Loading VQ-VAE...")
    vqvae = load_vqvae(args.vqvae_path, device)

    # --- Build collator for batch generation ---
    collator = MotionCollator(
        tokenizer=tokenizer,
        latent_id=latent_id,
    )

    # --- Load Motion-R1 evaluation infrastructure ---
    # Temporarily use the project root for resource path resolution
    # (dataset_TM_eval reads data from ./dataset/HumanML3D/, evaluator from
    # ./checkpoints/...).
    print("Loading evaluator...")
    orig_cwd = os.getcwd()
    os.chdir(RESOURCE_ROOT)

    w_vectorizer = WordVectorizer(args.glove_dir, "our_vab")
    wrapper_opt = get_opt(args.eval_opt_path, torch.device(device))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    down_t = 2  # VQ-VAE downsampling factor (same as build_data.py)
    unit_length = 2 ** down_t  # = 4

    print(
        "Loading test DataLoader "
        "(batch_size=%d, shuffle=True, drop_last=True)..." % args.batch_size
    )
    test_loader = dataset_TM_eval.DATALoader(
        "t2m", "test", args.batch_size, w_vectorizer, unit_length=unit_length,
    )

    os.chdir(orig_cwd)

    # --- Run evaluation ---
    repeat_time = args.repeat
    fid_list, div_list = [], []
    top1_list, top2_list, top3_list = [], [], []
    matching_list, multi_list = [], []
    motion_emb_cos_list, semantic_cos_list = [], []
    flops_per_sample_list = []
    peak_gpu_memory_bytes_list = []
    generated_tokens_total_list = []
    generated_tokens_per_sample_list = []
    per_sample_latency_ms_list = []
    cached_flops_per_sample = None

    start_time = time.time()
    for i in range(repeat_time):
        if repeat_time > 1:
            print(f"\n===== Repeat {i + 1}/{repeat_time} =====")

        fid, diversity, top1, top2, top3, matching, multi, motion_emb_cos, semantic_cos, perf = evaluation_test(
            test_loader, model, tokenizer, vqvae, configs,
            latent_id, start_id, end_id, collator, eval_wrapper, device,
            mm_num_repeats=args.mm_num_repeats if args.multimodality else 0,
            mm_num_times=args.mm_num_times,
            flops_per_sample_hint=(
                cached_flops_per_sample if args.flops_mode == "profile" else None
            ),
            enable_flops_profile=(
                args.flops_mode == "profile" and cached_flops_per_sample is None
            ),
            flops_mode=args.flops_mode,
            model_param_count=model_param_count if args.flops_mode == "estimate" else None,
        )
        if (
            args.flops_mode == "profile"
            and cached_flops_per_sample is None
            and perf["flops_per_sample"] is not None
        ):
            cached_flops_per_sample = perf["flops_per_sample"]
        fid_list.append(fid)
        div_list.append(diversity)
        top1_list.append(top1)
        top2_list.append(top2)
        top3_list.append(top3)
        matching_list.append(matching)
        multi_list.append(multi)
        motion_emb_cos_list.append(motion_emb_cos)
        semantic_cos_list.append(semantic_cos)
        flops_per_sample_list.append(
            np.nan if perf["flops_per_sample"] is None else perf["flops_per_sample"]
        )
        peak_gpu_memory_bytes_list.append(perf["peak_gpu_memory_bytes"])
        generated_tokens_total_list.append(perf["generated_tokens_total"])
        generated_tokens_per_sample_list.append(perf["generated_tokens_per_sample"])
        per_sample_latency_ms_list.append(perf["per_sample_latency_ms"])

    elapsed = time.time() - start_time

    # --- Print final results ---
    fid_arr = np.array(fid_list)
    div_arr = np.array(div_list)
    top1_arr = np.array(top1_list)
    top2_arr = np.array(top2_list)
    top3_arr = np.array(top3_list)
    matching_arr = np.array(matching_list)
    multi_arr = np.array(multi_list)
    motion_emb_cos_arr = np.array(motion_emb_cos_list)
    semantic_cos_arr = np.array(semantic_cos_list)
    flops_arr = np.array(flops_per_sample_list, dtype=np.float64)
    peak_gpu_memory_arr = np.array(peak_gpu_memory_bytes_list, dtype=np.float64)
    generated_tokens_total_arr = np.array(generated_tokens_total_list, dtype=np.float64)
    generated_tokens_per_sample_arr = np.array(
        generated_tokens_per_sample_list, dtype=np.float64
    )
    per_sample_latency_ms_arr = np.array(per_sample_latency_ms_list, dtype=np.float64)

    def _ci(arr):
        return np.std(arr) * 1.96 / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

    def _mean_valid(arr):
        valid = arr[np.isfinite(arr)]
        return float(np.mean(valid)) if len(valid) > 0 else None

    def _ci_valid(arr):
        valid = arr[np.isfinite(arr)]
        return (
            float(np.std(valid) * 1.96 / np.sqrt(len(valid)))
            if len(valid) > 1 else 0.0
        )

    flops_mean = _mean_valid(flops_arr)
    flops_ci = _ci_valid(flops_arr)
    peak_gpu_memory_max = (
        float(np.max(peak_gpu_memory_arr)) if len(peak_gpu_memory_arr) > 0 else 0.0
    )
    peak_gpu_memory_mean = (
        float(np.mean(peak_gpu_memory_arr)) if len(peak_gpu_memory_arr) > 0 else 0.0
    )

    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(f"  FID:             {np.mean(fid_arr):.4f} +/- {_ci(fid_arr):.4f}")
    print(f"  Diversity:       {np.mean(div_arr):.4f} +/- {_ci(div_arr):.4f}")
    print(f"  Top-1:           {np.mean(top1_arr):.4f} +/- {_ci(top1_arr):.4f}")
    print(f"  Top-2:           {np.mean(top2_arr):.4f} +/- {_ci(top2_arr):.4f}")
    print(f"  Top-3:           {np.mean(top3_arr):.4f} +/- {_ci(top3_arr):.4f}")
    print(f"  Matching Score:  {np.mean(matching_arr):.4f} +/- {_ci(matching_arr):.4f}")
    print(f"  Multimodality:   {np.mean(multi_arr):.4f} +/- {_ci(multi_arr):.4f}")
    print(f"  Motion Emb Cos:  {np.mean(motion_emb_cos_arr):.4f} +/- {_ci(motion_emb_cos_arr):.4f}")
    print(f"  Semantic Cos:    {np.mean(semantic_cos_arr):.4f} +/- {_ci(semantic_cos_arr):.4f}")
    if flops_mean is None:
        print("  FLOPs/sample:    N/A")
    else:
        print(f"  FLOPs/sample:    {flops_mean:.3e} +/- {flops_ci:.3e}")
    print(
        f"  Peak GPU Memory: {peak_gpu_memory_max / (1024 ** 3):.3f} GB (max), "
        f"{peak_gpu_memory_mean / (1024 ** 3):.3f} GB (mean)"
    )
    print(
        f"  Gen Tokens:      {np.mean(generated_tokens_total_arr):.1f} total/repeat, "
        f"{np.mean(generated_tokens_per_sample_arr):.2f} /sample"
    )
    print(
        f"  Latency/sample:  {np.mean(per_sample_latency_ms_arr):.2f} ms +/- "
        f"{_ci(per_sample_latency_ms_arr):.2f} ms"
    )
    print(f"  Time:            {elapsed:.1f}s")
    print(f"  Repeats:         {repeat_time}")
    print(f"{'=' * 60}")

    msg_final = (
        f"FID. {np.mean(fid_arr):.3f}, conf. {_ci(fid_arr):.3f}, "
        f"Diversity. {np.mean(div_arr):.3f}, conf. {_ci(div_arr):.3f}, "
        f"TOP1. {np.mean(top1_arr):.3f}, conf. {_ci(top1_arr):.3f}, "
        f"TOP2. {np.mean(top2_arr):.3f}, conf. {_ci(top2_arr):.3f}, "
        f"TOP3. {np.mean(top3_arr):.3f}, conf. {_ci(top3_arr):.3f}, "
        f"Matching. {np.mean(matching_arr):.3f}, conf. {_ci(matching_arr):.3f}, "
        f"Multi. {np.mean(multi_arr):.3f}, conf. {_ci(multi_arr):.3f}, "
        f"MotionEmbCos. {np.mean(motion_emb_cos_arr):.3f}, conf. {_ci(motion_emb_cos_arr):.3f}, "
        f"SemanticCos. {np.mean(semantic_cos_arr):.3f}, conf. {_ci(semantic_cos_arr):.3f}, "
        f"GenTok/sample. {np.mean(generated_tokens_per_sample_arr):.2f}, "
        f"Latency/sample(ms). {np.mean(per_sample_latency_ms_arr):.2f}, "
        f"PeakMem(GB,max). {peak_gpu_memory_max / (1024 ** 3):.3f}, "
    )
    if flops_mean is None:
        msg_final += "FLOPs/sample. N/A"
    else:
        msg_final += f"FLOPs/sample. {flops_mean:.3e}"
    print(msg_final)

    # --- Save results ---
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        results = {
            "fid": float(np.mean(fid_arr)),
            "fid_ci": float(_ci(fid_arr)),
            "diversity": float(np.mean(div_arr)),
            "diversity_ci": float(_ci(div_arr)),
            "top1": float(np.mean(top1_arr)),
            "top1_ci": float(_ci(top1_arr)),
            "top2": float(np.mean(top2_arr)),
            "top2_ci": float(_ci(top2_arr)),
            "top3": float(np.mean(top3_arr)),
            "top3_ci": float(_ci(top3_arr)),
            "matching_score": float(np.mean(matching_arr)),
            "matching_score_ci": float(_ci(matching_arr)),
            "multimodality": float(np.mean(multi_arr)),
            "multimodality_ci": float(_ci(multi_arr)),
            "motion_emb_cos": float(np.mean(motion_emb_cos_arr)),
            "motion_emb_cos_ci": float(_ci(motion_emb_cos_arr)),
            "semantic_cos": float(np.mean(semantic_cos_arr)),
            "semantic_cos_ci": float(_ci(semantic_cos_arr)),
            "flops_per_sample": float(flops_mean) if flops_mean is not None else None,
            "flops_per_sample_ci": float(flops_ci) if flops_mean is not None else None,
            "peak_gpu_memory_bytes_max": int(peak_gpu_memory_max),
            "peak_gpu_memory_mb_max": float(peak_gpu_memory_max / (1024 ** 2)),
            "peak_gpu_memory_bytes_mean": int(peak_gpu_memory_mean),
            "generated_tokens_total": float(np.mean(generated_tokens_total_arr)),
            "generated_tokens_total_ci": float(_ci(generated_tokens_total_arr)),
            "generated_tokens_per_sample": float(np.mean(generated_tokens_per_sample_arr)),
            "generated_tokens_per_sample_ci": float(_ci(generated_tokens_per_sample_arr)),
            "per_sample_latency_ms": float(np.mean(per_sample_latency_ms_arr)),
            "per_sample_latency_ms_ci": float(_ci(per_sample_latency_ms_arr)),
            "flops_mode": args.flops_mode,
            "generation_strategy": generation_strategy,
            "do_sample": bool(getattr(configs, "do_sample", False)),
            "max_new_tokens": int(configs.max_new_tokens),
            "temperature": float(getattr(configs, "temperature", 0.0)),
            "top_p": float(getattr(configs, "top_p", 1.0)),
            "top_k": int(getattr(configs, "top_k", 0)),
            "repetition_penalty": float(getattr(configs, "repetition_penalty", 1.0)),
            "repeat_time": repeat_time,
            "checkpoint": args.checkpoint,
            "elapsed_seconds": elapsed,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
