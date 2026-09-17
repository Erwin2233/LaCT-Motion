"""Dataset and collator with Coconut-style curriculum learning.

Loads JSON data, tokenizes questions/steps/answers, and applies
stage-based latent token replacement for curriculum training.

Reference: Coconut/dataset.py
"""

import itertools
import json
import random
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


def _truncate_feature_dict(feature: dict, max_seq_len: int) -> dict:
    """Truncate sequence fields to max_seq_len (prefix-preserving)."""
    if max_seq_len <= 0:
        return feature

    if "input_ids" in feature and len(feature["input_ids"]) > max_seq_len:
        feature["input_ids"] = feature["input_ids"][:max_seq_len]
        if "labels" in feature:
            feature["labels"] = feature["labels"][:max_seq_len]
        if "attention_mask" in feature:
            feature["attention_mask"] = feature["attention_mask"][:max_seq_len]
        if "position_ids" in feature:
            feature["position_ids"] = feature["position_ids"][:max_seq_len]
    return feature


def get_dataset(path, tokenizer, max_size=1_000_000_000, max_steps=20):
    """Load and tokenize the T2M dataset JSON.

    Each sample is tokenized into:
      - question_tokenized: token IDs for the chat prompt
      - steps_tokenized: list of token ID lists, one per thinking step
      - answer_tokenized: token IDs for <Motion>...<|im_end|>
      - idx: sample index

    Args:
        path: Path to JSON file.
        tokenizer: HuggingFace tokenizer.
        max_size: Maximum samples to load.
        max_steps: Cap on thinking steps per sample.
    """

    def tokenize_sample(sample):
        question_tokenized = tokenizer.encode(
            sample["question"], add_special_tokens=False
        )

        steps = sample["steps"][:max_steps]
        steps_tokenized = [
            tokenizer.encode(s.strip() + "\n", add_special_tokens=False)
            for s in steps
        ]

        answer_tokenized = tokenizer.encode(
            sample["answer"], add_special_tokens=False
        )

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }

    with open(path, encoding="utf-8") as f:
        data = json.load(f)[:max_size]
    if len(data) == 0:
        raise ValueError(
            f"Dataset is empty: {path}\n"
            "Check that the JSON file contains a non-empty list of samples."
        )
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})

    # All ranks process independently; tokenization is fully deterministic.
    dataset = dataset.map(
        tokenize_sample,
        remove_columns=list(dataset.features),
        num_proc=16,
    )

    return dataset


def get_cot_latent_dataset(
    scheduled_stage,
    base_dataset,
    configs,
    start_id,
    latent_id,
    end_id,
    think_open_ids=None,
    think_close_ids=None,
    no_special_marker=False,
    shuffle=False,
):
    """Build curriculum training dataset with latent token replacement.

    At stage k:
      - Skip first k reasoning steps
      - Insert k * c_thought latent tokens
      - Keep remaining steps wrapped in <think>...</think>
      - Append answer (motion tokens)
      - Labels: -100 on question + latent region, train on rest

    Args:
        scheduled_stage: Current curriculum stage (epoch // epochs_per_stage).
        base_dataset: Tokenized dataset from get_dataset().
        configs: Config object with c_thought, max_latent_stage, etc.
        start_id: Token ID for <|start-latent|>.
        latent_id: Token ID for <|latent|>.
        end_id: Token ID for <|end-latent|>.
        think_open_ids: Token IDs for "<think>\n".
        think_close_ids: Token IDs for "\n</think>".
        no_special_marker: If True, skip latent boundary markers.
        shuffle: Shuffle the resulting dataset.
    """
    n_additional_tokens = 0 if no_special_marker else 2  # start + end markers

    def process_dataset(sample):
        # Use per-sample deterministic RNG so that multiprocessing workers
        # (num_proc>1) produce consistent results regardless of shard assignment.
        rng = random.Random(sample["idx"] * 31337 + scheduled_stage)

        # Optionally sample a random stage (only for coconut mode;
        # cot/no_cot modes must keep their fixed stage to stay pure).
        if (
            not getattr(configs, "cot", False)
            and not getattr(configs, "no_cot", False)
            and rng.random() < configs.uniform_prob
        ):
            scheduled_stage_to_train = rng.choice(
                list(range(len(sample["steps_tokenized"]) + 1))
            )
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > configs.max_latent_stage:
            n_skip_steps = 10000  # skip all
            if configs.pad_latent_to_max:
                n_latent_tokens = configs.max_latent_stage
            else:
                n_latent_tokens = min(
                    len(sample["steps_tokenized"]), configs.max_latent_stage
                )
        else:
            n_skip_steps = scheduled_stage_to_train
            n_latent_tokens = scheduled_stage_to_train

        # no_cot mode must skip all explicit reasoning steps and latent tokens.
        if getattr(configs, "no_cot", False):
            n_skip_steps = 10_000
            n_latent_tokens = 0

        n_latent_tokens *= configs.c_thought

        # Build remaining steps token sequence
        remaining_steps = list(
            itertools.chain.from_iterable(
                sample["steps_tokenized"][n_skip_steps:]
            )
        )

        # Wrap remaining steps with <think>...</think> if any exist
        if len(remaining_steps) > 0 and think_open_ids and think_close_ids:
            remaining_steps = think_open_ids + remaining_steps + think_close_ids

        # Build full token sequence
        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + remaining_steps
            + sample["answer_tokenized"]
        )

        # Build labels: -100 for question + latent region
        n_masked = (
            len(sample["question_tokenized"])
            + n_latent_tokens
            + n_additional_tokens
        )
        labels = [-100] * n_masked + tokens[n_masked:]

        feature = {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }
        return _truncate_feature_dict(
            feature, int(getattr(configs, "max_seq_len", 0) or 0)
        )

    # All ranks process independently; per-sample deterministic RNG
    # (seeded by sample idx + scheduled_stage) ensures identical results.
    dataset = base_dataset.map(
        process_dataset,
        remove_columns=list(base_dataset.features),
        num_proc=16,
    )
    if shuffle:
        dataset = dataset.shuffle(seed=scheduled_stage)

    return dataset


def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    configs,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):
    """Build evaluation dataset: question + latent tokens only.

    Used for generation-based evaluation where the model must produce
    the answer (motion tokens) from only the question and latent reasoning.
    """

    def process_dataset(sample):
        if configs.pad_latent_to_max:
            max_latent_stage = configs.max_latent_stage
        else:
            max_latent_stage = min(
                configs.max_latent_stage, len(sample["steps_tokenized"])
            )

        k = min(max_latent_stage, scheduled_stage)
        k *= configs.c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        feature = {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }
        return _truncate_feature_dict(
            feature, int(getattr(configs, "max_seq_len", 0) or 0)
        )

    return base_dataset_valid.map(
        process_dataset,
        remove_columns=list(base_dataset_valid.features),
        num_proc=16,
    )


@dataclass
class MotionCollator:
    """Batch collator that aligns latent token positions for KV cache reuse.

    Left-pads sequences so that the first <|latent|> token aligns across
    all batch elements, maximizing KV cache reuse during multi-pass forward.

    Example alignment:
        xxxxxxxxxx<latent><latent>xxxxx--
        -----xxxxx<latent>xxxxxxxx-------
        ---xxxxxxx<latent><latent>xxxxxxx

    Reference: Coconut/dataset.py MyCollator
    """

    tokenizer: PreTrainedTokenizerBase
    latent_id: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features, return_tensors=None):
        assert self.tokenizer.padding_side == "right"

        # Shallow-copy features to avoid mutating Dataset cache (P2-17).
        features = [
            {k: list(v) if isinstance(v, list) else v for k, v in f.items()}
            for f in features
        ]

        # Find the earliest latent position in each sample
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature[
                        "input_ids"
                    ].index(self.latent_id)
                else:
                    # Non-latent samples must be padded to the same alignment
                    # point so the multi-pass forward compute ranges are
                    # consistent across the batch (P0-3).
                    n_tok_pad = latest_earliest_latent

                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]
                if "labels" in feature:
                    feature["labels"] = [
                        self.label_pad_token_id
                    ] * n_tok_pad + feature["labels"]
                feature["attention_mask"] = [0] * n_tok_pad + feature[
                    "attention_mask"
                ]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids"
            }
            for feature in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        # Manually pad labels and position_ids
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None

        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )

        if labels is not None:
            max_label_length = max(len(lbl) for lbl in labels)
            batch["labels"] = [
                lbl + [self.label_pad_token_id] * (max_label_length - len(lbl))
                for lbl in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(pos) for pos in position_ids)
            batch["position_ids"] = [
                pos + [0] * (max_pos_length - len(pos))
                for pos in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch
