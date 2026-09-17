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


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


def batch_generate(
    model, tokenizer, captions, configs,
    latent_id, start_id, end_id, collator, device,
):
    """Build batched input_ids from captions and generate in one forward pass.

    Returns:
        motion_code_batch: List[Tensor] of motion code indices per sample.
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

    with torch.inference_mode():
        outputs = model.generate(
            input_ids, attention_mask,
            position_ids=position_ids,
            max_new_tokens=configs.max_new_tokens,
            synced_gpus=False,
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
    return motion_code_batch


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluation_test(
    test_loader, model, tokenizer, vqvae, configs,
    latent_id, start_id, end_id, collator, eval_wrapper, device,
):
    """Full evaluation pass following Motion-R1 protocol.

    Returns:
        (fid, diversity, top1, top2, top3, matching_score_pred, multimodality)
    """
    model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    nb_sample = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        word_embeddings, pos_one_hots, caption, sent_len, pose, m_length, token, name, cot = batch
        bs, seq = pose.shape[:2]

        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1]), device=device)
        pred_len = torch.ones(bs, dtype=torch.long)

        # --- 1. Batch generation ---
        motion_code_batch = batch_generate(
            model, tokenizer, caption, configs,
            latent_id, start_id, end_id, collator, device,
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

        pose = pose.to(device).float()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pose, m_length,
        )

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        # --- 4. Per-batch R-precision ---
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

    # --- 5. Aggregate across all batches ---
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

    multimodality = 0  # disabled for cost reasons (same as Motion-R1)

    msg = (
        f"--> \t FID. {fid:.4f}, "
        f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
        f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
        f"matching_score_real. {matching_score_real:.4f}, "
        f"matching_score_pred. {matching_score_pred:.4f}, "
        f"multimodality. {multimodality:.4f}"
    )
    print(msg)

    return (
        fid, diversity, R_precision[0], R_precision[1], R_precision[2],
        matching_score_pred, multimodality,
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)
    configs = Config(config_dict)
    set_seed(args.seed)

    device = args.device

    # --- Load our model ---
    print("Loading model...")
    model, tokenizer, latent_id, start_id, end_id = load_model(
        args.checkpoint, configs, device,
    )

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

    print("Loading test DataLoader (batch_size=%d, drop_last=True)..." % args.batch_size)
    test_loader = dataset_TM_eval.DATALoader(
        "t2m", "test", args.batch_size, w_vectorizer, unit_length=unit_length,
    )

    os.chdir(orig_cwd)

    # --- Run evaluation ---
    repeat_time = args.repeat
    fid_list, div_list = [], []
    top1_list, top2_list, top3_list = [], [], []
    matching_list, multi_list = [], []

    start_time = time.time()
    for i in range(repeat_time):
        if repeat_time > 1:
            print(f"\n===== Repeat {i + 1}/{repeat_time} =====")

        fid, diversity, top1, top2, top3, matching, multi = evaluation_test(
            test_loader, model, tokenizer, vqvae, configs,
            latent_id, start_id, end_id, collator, eval_wrapper, device,
        )
        fid_list.append(fid)
        div_list.append(diversity)
        top1_list.append(top1)
        top2_list.append(top2)
        top3_list.append(top3)
        matching_list.append(matching)
        multi_list.append(multi)

    elapsed = time.time() - start_time

    # --- Print final results ---
    fid_arr = np.array(fid_list)
    div_arr = np.array(div_list)
    top1_arr = np.array(top1_list)
    top2_arr = np.array(top2_list)
    top3_arr = np.array(top3_list)
    matching_arr = np.array(matching_list)
    multi_arr = np.array(multi_list)

    def _ci(arr):
        return np.std(arr) * 1.96 / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print("Final Results:")
    print(f"  FID:             {np.mean(fid_arr):.4f} +/- {_ci(fid_arr):.4f}")
    print(f"  Diversity:       {np.mean(div_arr):.4f} +/- {_ci(div_arr):.4f}")
    print(f"  Top-1:           {np.mean(top1_arr):.4f} +/- {_ci(top1_arr):.4f}")
    print(f"  Top-2:           {np.mean(top2_arr):.4f} +/- {_ci(top2_arr):.4f}")
    print(f"  Top-3:           {np.mean(top3_arr):.4f} +/- {_ci(top3_arr):.4f}")
    print(f"  Matching Score:  {np.mean(matching_arr):.4f} +/- {_ci(matching_arr):.4f}")
    print(f"  Multimodality:   {np.mean(multi_arr):.4f} +/- {_ci(multi_arr):.4f}")
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
        f"Multi. {np.mean(multi_arr):.3f}, conf. {_ci(multi_arr):.3f}"
    )
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
            "repeat_time": repeat_time,
            "checkpoint": args.checkpoint,
            "elapsed_seconds": elapsed,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
