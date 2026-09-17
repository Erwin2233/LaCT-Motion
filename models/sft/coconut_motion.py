"""Coconut model wrapper adapted for Qwen 2.5.

Implements the multi-pass forward mechanism from the Coconut paper
("Training Large Language Models to Reason in a Continuous Latent Space"),
adapted for Qwen architecture with DynamicCache KV-cache handling.

Reference: https://github.com/facebookresearch/Coconut
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torch.nn import CrossEntropyLoss
from transformers import DynamicCache

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 40


def top_p_filtering(logits, top_p=0.9):
    """Nucleus sampling: keep tokens with cumulative probability <= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs - sorted_probs >= top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k_filtering(logits, top_k=50):
    """Top-k sampling: keep only top_k highest-probability tokens."""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.clone()
    logits[indices_to_remove] = float("-inf")
    return logits


def apply_repetition_penalty(logits, generated, repetition_penalty=1.0):
    """Apply HF-style repetition penalty to tokens already in each sequence."""
    if repetition_penalty is None or repetition_penalty == 1.0:
        return logits
    logits = logits.clone()
    vocab_size = logits.size(-1)
    for row, token_ids in enumerate(generated):
        for token_id in set(token_ids):
            if 0 <= token_id < vocab_size:
                score = logits[row, token_id]
                logits[row, token_id] = (
                    score * repetition_penalty
                    if score < 0
                    else score / repetition_penalty
                )
    return logits


def sample_from_logits(
    logits,
    temperature=1.0,
    do_sample=False,
    top_p=1.0,
    top_k=0,
    generated=None,
    repetition_penalty=1.0,
):
    """Decode next token with greedy or UniMo-style sampling."""
    logits = apply_repetition_penalty(logits, generated or [], repetition_penalty)
    if do_sample:
        if temperature is None or temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / temperature
        if top_p < 1.0:
            logits = top_p_filtering(logits, top_p)
        if top_k > 0:
            logits = top_k_filtering(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.argmax(logits, dim=-1)


def _to_legacy_cache(kv_cache):
    """Convert KV cache to tuple-of-tuples format if needed."""
    if kv_cache is None:
        return None
    # Already tuple-of-tuples
    if isinstance(kv_cache, (tuple, list)) and len(kv_cache) > 0:
        if isinstance(kv_cache[0], (tuple, list)):
            return kv_cache
    # DynamicCache or similar
    if hasattr(kv_cache, "to_legacy_cache"):
        return kv_cache.to_legacy_cache()
    if hasattr(kv_cache, "key_cache"):
        return tuple(
            (kv_cache.key_cache[i], kv_cache.value_cache[i])
            for i in range(len(kv_cache))
        )
    return kv_cache


def _slice_kv_cache(kv_cache, max_pos: int):
    """Slice KV cache to first max_pos positions.

    Returns a DynamicCache compatible with transformers >= 4.36.
    """
    legacy = _to_legacy_cache(kv_cache)
    cache = DynamicCache()
    for key_states, value_states in legacy:
        cache.update(
            key_states[:, :, :max_pos, :],
            value_states[:, :, :max_pos, :],
            layer_idx=len(cache),
        )
    return cache


class CoconutMotion(nn.Module):
    """Coconut wrapper for Qwen 2.5 CausalLM.

    Performs multi-pass forward with hidden state replacement at latent
    token positions. Each latent token's embedding is replaced by the
    last hidden state of the preceding position from the previous pass.

    Args:
        base_causallm: The base causal language model (Qwen2ForCausalLM)
        latent_token_id: Token ID for <|latent|>
        start_latent_id: Token ID for <|start-latent|>
        end_latent_id: Token ID for <|end-latent|>
        eos_token_id: Token ID for EOS
    """

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.loss_fct = CrossEntropyLoss()

        # Qwen uses model.get_input_embeddings()
        self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        """Multi-pass forward with hidden state replacement.

        For each latent token position:
        1. Forward-pass up to that position
        2. Extract the last hidden state from the preceding position
        3. Replace the latent token's embedding with that hidden state
        4. Reuse KV cache for already-computed positions

        Final pass processes remaining tokens after all latent replacements.
        Loss is computed on the concatenated logits from all passes.
        """
        logits = []

        # Find all latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            # Verify all batch elements have latent tokens starting at
            # the same column (requires aligned collation).
            first_positions = [lst[0] for lst in latent_lists if lst]
            assert len(set(first_positions)) <= 1, (
                f"Latent positions are not aligned across batch: {first_positions}. "
                "Check MotionCollator left-padding alignment."
            )
            # Only compute up to the earliest latent token position
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                # First forward pass: no KV cache
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                # Subsequent passes: reuse KV cache
                past_key_values = _slice_kv_cache(
                    kv_cache, next_compute_range[0]
                )

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )

                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            # Update compute range for next pass
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            # Get last-layer hidden states and KV cache
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Determine which latent positions to fill in this pass
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Clone + advanced indexing avoids expensive Python-level tensor
            # reconstruction while keeping autograd graph correctness.
            if filling_indices:
                device = inputs_embeds.device
                batch_idxs = torch.tensor(
                    [idx[0] for idx in filling_indices],
                    device=device,
                    dtype=torch.long,
                )
                token_idxs = torch.tensor(
                    [idx[1] for idx in filling_indices],
                    device=device,
                    dtype=torch.long,
                )
                source_idxs = token_idxs - 1 - hidden_states_offset
                new_embeds = inputs_embeds.clone()
                new_embeds[batch_idxs, token_idxs] = hidden_states[
                    batch_idxs, source_idxs
                ]
                inputs_embeds = new_embeds

        # Final pass: process remaining tokens
        final_past_kv = None
        if kv_cache is not None:
            final_past_kv = _slice_kv_cache(kv_cache, next_compute_range[0])

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=(
                attention_mask[:, : next_compute_range[1]]
                if kv_cache is not None
                else attention_mask[
                    :, next_compute_range[0] : next_compute_range[1]
                ]
            ),
            position_ids=position_ids[
                :, next_compute_range[0] : next_compute_range[1]
            ],
            past_key_values=final_past_kv,
            output_hidden_states=False,
            use_cache=False,
        )

        logits.append(outputs.logits)

        # Concatenate logits from all passes and compute loss
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def _forward_latent_for_generation(
        self, input_ids, attention_mask, position_ids
    ):
        """Inference-only latent forward that returns reusable KV cache."""
        assert not torch.is_grad_enabled(), (
            "_forward_latent_for_generation must be called under "
            "no_grad or inference_mode"
        )
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = _slice_kv_cache(
                    kv_cache, next_compute_range[0]
                )
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            for instance_idx, mask_list in enumerate(latent_lists):
                if len(mask_list) > pass_idx:
                    token_idx = mask_list[pass_idx]
                    inputs_embeds[instance_idx, token_idx, :] = hidden_states[
                        instance_idx, token_idx - 1 - hidden_states_offset, :
                    ]

        self.gen_forward_cnt += max_n_latents

        final_past_kv = None
        if kv_cache is not None:
            final_past_kv = _slice_kv_cache(kv_cache, next_compute_range[0])

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=(
                attention_mask[:, : next_compute_range[1]]
                if kv_cache is not None
                else attention_mask[
                    :, next_compute_range[0] : next_compute_range[1]
                ]
            ),
            position_ids=position_ids[
                :, next_compute_range[0] : next_compute_range[1]
            ],
            past_key_values=final_past_kv,
            output_hidden_states=False,
            use_cache=True,
        )
        self.gen_forward_cnt += 1
        return outputs.past_key_values, outputs.logits, inputs_embeds

    def generate(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        max_new_tokens=256,
        output_embedding=False,
        synced_gpus=False,
        temperature=1.0,
        do_sample=False,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        **kwargs,
    ):
        """Autoregressive generation after latent reasoning.

        Supports arbitrary batch sizes. For each sample in the batch:
        1. Forward-pass through latent tokens (multi-pass)
        2. Decode new tokens using KV cache until EOS

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            position_ids: (batch_size, seq_len) or None (auto-computed)
            temperature: Sampling temperature.
            do_sample: Whether to sample instead of greedy decoding.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling threshold (0 = disabled).
            repetition_penalty: HF-style repetition penalty.
        """
        self.gen_forward_cnt = 0
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Compute position_ids from attention_mask if not provided
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        with torch.inference_mode():
            past_key_values, last_logits, inputs_embeds = (
                self._forward_latent_for_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            )
            # Initialize per-sample generated token lists (input + generated)
            generated = [input_ids[i].tolist() for i in range(batch_size)]
            next_tokens = sample_from_logits(
                last_logits[:, -1, :],
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                generated=generated,
                repetition_penalty=repetition_penalty,
            )
            done = next_tokens == self.eos_token_id
            next_tokens_cpu = next_tokens.cpu().tolist()
            done_cpu = done.cpu().tolist()
            for i in range(batch_size):
                if not done_cpu[i]:
                    generated[i].append(next_tokens_cpu[i])

            # Next position per sample = number of real tokens so far
            cur_positions = attention_mask.sum(dim=-1)  # (batch_size,)

            # Pre-allocate full decode attention mask to avoid repeated cat.
            max_total_len = attention_mask.shape[1] + max_new_tokens
            decode_attn = attention_mask.new_zeros((batch_size, max_total_len))
            decode_attn[:, : attention_mask.shape[1]] = attention_mask
            decode_len = attention_mask.shape[1]

            # Decode remaining tokens with KV cache
            for _ in range(max_new_tokens - 1):
                if done.all():
                    break

                new_token_embed = self.embedding(next_tokens.unsqueeze(1))
                pos_ids = cur_positions.unsqueeze(1)

                decode_attn[:, decode_len] = 1
                decode_len += 1

                out = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    attention_mask=decode_attn[:, :decode_len],
                    past_key_values=past_key_values,
                    position_ids=pos_ids,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                self.gen_forward_cnt += 1
                cur_positions = cur_positions + 1

                next_tokens = sample_from_logits(
                    out.logits[:, -1, :],
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    generated=generated,
                    repetition_penalty=repetition_penalty,
                )

                # Track EOS
                done = done | (next_tokens == self.eos_token_id)

                # Batch GPU→CPU transfer (avoid per-element sync)
                next_tokens_cpu = next_tokens.cpu().tolist()
                done_cpu = done.cpu().tolist()
                for i in range(batch_size):
                    if not done_cpu[i]:
                        generated[i].append(next_tokens_cpu[i])

                # Feed EOS for done samples in next iteration
                next_tokens[done] = self.eos_token_id

        # Sync forward count across GPUs for FSDP
        if synced_gpus:
            with torch.inference_mode():
                while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                    self.gen_forward_cnt += 1
                    dummy_embed = self.embedding(
                        torch.full(
                            (batch_size, 1), self.eos_token_id,
                            device=device, dtype=torch.long,
                        )
                    )
                    _ = self.base_causallm(
                        inputs_embeds=dummy_embed,
                        attention_mask=torch.ones(
                            batch_size, 1, device=device, dtype=torch.long,
                        ),
                        use_cache=False,
                    )

        # Pad results to same length
        max_len = max(len(g) for g in generated)
        result = torch.full(
            (batch_size, max_len), self.eos_token_id,
            device=device, dtype=torch.long,
        )
        for i in range(batch_size):
            result[i, :len(generated[i])] = torch.tensor(
                generated[i], device=device, dtype=torch.long,
            )

        if output_embedding:
            return result, inputs_embeds
        return result

    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)
