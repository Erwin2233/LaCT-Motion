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


def top_p_filtering(logits, top_p=0.9):
    """Nucleus sampling: keep tokens with cumulative probability <= top_p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above top_p (keep first above)
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k_filtering(logits, top_k=50):
    """Top-k sampling: keep only top_k highest probability tokens."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.clone()
    logits[indices_to_remove] = float('-inf')
    return logits

def sample_from_logits(logits, temperature=0.0, do_sample=False, top_p=1.0, top_k=0):
# def sample_from_logits(
#     logits,
#     temperature=0.0,
#     do_sample=False,
#     top_p=1.0,
#     top_k=0,
#     forbidden_token_ids=None,
# ):
    """Unified sampling: greedy or temperature sampling with top_p/top_k."""
    # if forbidden_token_ids:
    #     blocked = [
    #         int(tid)
    #         for tid in forbidden_token_ids
    #         if tid is not None and 0 <= int(tid) < logits.size(-1)
    #     ]
    #     if blocked:
    #         logits = logits.clone()
    #         logits[:, blocked] = float("-inf")

    if temperature > 0 and do_sample:
        logits = logits / temperature
        if top_p < 1.0:
            logits = top_p_filtering(logits, top_p)
        if top_k > 0:
            logits = top_k_filtering(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        return torch.argmax(logits, dim=-1)

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 40


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

    # def _get_latent_lists(self, input_ids):
    #     """Return latent token positions inside the prompt latent span."""
    #     latent_lists = []
    #     for row in input_ids:
    #         start_positions = (row == self.start_latent_id).nonzero(as_tuple=True)[0]
    #         end_positions = (row == self.end_latent_id).nonzero(as_tuple=True)[0]

    #         if start_positions.numel() > 0:
    #             start = int(start_positions[0].item()) + 1
    #             end_after_start = end_positions[end_positions >= start]
    #             end = (
    #                 int(end_after_start[0].item())
    #                 if end_after_start.numel() > 0
    #                 else row.size(0)
    #             )
    #             latent_positions = (
    #                 (row[start:end] == self.latent_token_id)
    #                 .nonzero(as_tuple=True)[0]
    #                 + start
    #             )
    #         else:
    #             # no_special_marker compatibility: all latent tokens are prompt latents.
    #             latent_positions = (row == self.latent_token_id).nonzero(as_tuple=True)[0]

    #         latent_lists.append([int(pos.item()) for pos in latent_positions])
    #     return latent_lists

    def forward(self, input_ids, attention_mask, labels, position_ids, compute_ce_loss=True, **kwargs):
        """Multi-pass forward with hidden state replacement.

        For each latent token position:
        1. Forward the prefix through the *transformer only* (no lm_head)
        2. Extract the last hidden state from the preceding position
        3. Replace the latent token's embedding with that hidden state

        After all latent replacements, a single forward through the full
        CausalLM (with lm_head) produces logits for the entire sequence.
        This avoids computing the expensive lm_head on every intermediate
        pass, saving ~10 GB per pass at large batch sizes.

        Compatible with gradient checkpointing (use_cache is not needed
        since each intermediate pass re-forwards from position 0).
        """
        # Find all latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        # latent_lists = self._get_latent_lists(input_ids)
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            # Verify all batch elements have latent tokens starting at
            # the same column (requires aligned collation).
            first_positions = [lst[0] for lst in latent_lists if lst]
            assert len(set(first_positions)) <= 1, (
                f"Latent positions are not aligned across batch: {first_positions}. "
                "Check MotionCollator left-padding alignment."
            )
        # --- Intermediate latent passes (transformer only, no lm_head) ---
        # Compute up to each latent position, extract hidden states, and
        # replace the latent token embedding.  We use the underlying
        # transformer (self.base_causallm.model) to skip the lm_head and
        # save memory.  Each pass forwards from position 0 so that
        # gradient checkpointing (which disables use_cache) works correctly.
        next_end = latent_indices[:, 1].min().item() if max_n_latents > 0 else 0

        for pass_idx in range(max_n_latents):
            # Forward prefix through transformer layers only (no lm_head)
            model_out = self.base_causallm.model(
                inputs_embeds=inputs_embeds[:, :next_end, :],
                attention_mask=attention_mask[:, :next_end],
                position_ids=position_ids[:, :next_end],
                use_cache=False,
            )
            hidden_states = model_out.last_hidden_state

            # Advance end position for next pass
            next_end = next_end + 1 if pass_idx + 1 < max_n_latents else next_end

            # Determine which latent positions to fill in this pass
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]
            # if not filling_indices:
            #     continue

            # compute_end = max(token_idx for _, token_idx in filling_indices)
            # if compute_end <= 0:
            #     raise IndexError(
            #         f"Latent token cannot be filled from previous position: "
            #         f"pass={pass_idx}, filling_indices={filling_indices}"
            #     )

            # # Forward prefix through transformer layers only (no lm_head)
            # model_out = self.base_causallm.model(
            #     inputs_embeds=inputs_embeds[:, :compute_end, :],
            #     attention_mask=attention_mask[:, :compute_end],
            #     position_ids=position_ids[:, :compute_end],
            #     use_cache=False,
            # )
            # hidden_states = model_out.last_hidden_state

            # Replace latent token embeddings with hidden states from
            # the preceding position.  Clone + advanced indexing keeps
            # autograd graph correct.
            if filling_indices:
                device = inputs_embeds.device
                batch_idxs = torch.tensor(
                    [idx[0] for idx in filling_indices],
                    device=device,
                    dtype=torch.long,
            # device = inputs_embeds.device
            # batch_idxs = torch.tensor(
            #     [idx[0] for idx in filling_indices],
            #     device=device,
            #     dtype=torch.long,
            # )
            # token_idxs = torch.tensor(
            #     [idx[1] for idx in filling_indices],
            #     device=device,
            #     dtype=torch.long,
            # )
            # source_idxs = token_idxs - 1
            # if (
            #     source_idxs.min().item() < 0
            #     or source_idxs.max().item() >= hidden_states.shape[1]
            # ):
            #     raise IndexError(
            #         f"Latent hidden-state source index out of range: "
            #         f"pass={pass_idx}, compute_end={compute_end}, "
            #         f"token_idxs={token_idxs.tolist()}, "
            #         f"source_idxs={source_idxs.tolist()}, "
            #         f"hidden_states.shape={tuple(hidden_states.shape)}"
                )
                token_idxs = torch.tensor(
                    [idx[1] for idx in filling_indices],
                    device=device,
                    dtype=torch.long,
                )
                # hidden_states covers [0, next_end), offset is always 0
                source_idxs = token_idxs - 1
                new_embeds = inputs_embeds.clone()
                new_embeds[batch_idxs, token_idxs] = hidden_states[
                    batch_idxs, source_idxs
                ]
                inputs_embeds = new_embeds
            # new_embeds = inputs_embeds.clone()
            # new_embeds[batch_idxs, token_idxs] = hidden_states[
            #     batch_idxs, source_idxs
            # ]
            # inputs_embeds = new_embeds

        # --- Final pass: full CausalLM forward (with lm_head) ---
        # Process the *entire* sequence with the (potentially modified)
        # embeddings.  Gradient checkpointing is active here, saving
        # memory for the long completion portion.
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        logits = outputs.logits

        # Only compute CE loss when needed (SFT training).  During GRPO the
        # loss is computed externally via grpo_loss.py, so skipping this
        # avoids allocating a huge shift_logits tensor (~B*L*V) whose
        # computation graph would never be freed.
        loss = None
        if compute_ce_loss:
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
        """Inference-only latent forward that returns reusable KV cache.

        Returns:
            past_key_values: KV cache after processing the full prompt.
            next_token_logits: (B, V) logits at each sample's last *real* token
                position (not the right-padding position).
            inputs_embeds: Prompt embeddings after latent replacement.
        """
        assert not torch.is_grad_enabled(), (
            "_forward_latent_for_generation must be called under "
            "no_grad or inference_mode"
        )
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        # latent_lists = self._get_latent_lists(input_ids)
        max_n_latents = max((len(lst) for lst in latent_lists), default=0)

        next_compute_range = (0, input_ids.shape[1])

        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # next_compute_range = (0, min(lst[0] for lst in latent_lists if lst))

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
                    source_idx = token_idx - 1 - hidden_states_offset
                    if source_idx < 0 or source_idx >= hidden_states.shape[1]:
                        raise IndexError(
                            f"[_forward_latent] hidden_states OOB: "
                            f"instance={instance_idx}, pass={pass_idx}, "
                            f"token_idx={token_idx}, offset={hidden_states_offset}, "
                            f"source_idx={source_idx}, "
                            f"hidden_states.shape={hidden_states.shape}"
                        )
                    inputs_embeds[instance_idx, token_idx, :] = hidden_states[
                        instance_idx, source_idx, :
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

        # Select logits from each sample's last valid (attention=1) token.
        # This avoids using the right-padding column when prompts in a batch
        # have different effective lengths.
        if not attention_mask.bool().any(dim=-1).all():
            raise ValueError(
                "Found an empty prompt in _forward_latent_for_generation: "
                "attention_mask has a row with no valid tokens."
            )
        reversed_last_pos = attention_mask.flip(-1).argmax(dim=-1)
        abs_last_pos = attention_mask.size(1) - 1 - reversed_last_pos
        local_last_pos = abs_last_pos - next_compute_range[0]
        if (local_last_pos < 0).any() or (
            local_last_pos >= outputs.logits.size(1)
        ).any():
            raise ValueError(
                "Last-token position is outside final logits chunk: "
                f"chunk_start={next_compute_range[0]}, "
                f"chunk_len={outputs.logits.size(1)}, "
                f"abs_last_pos={abs_last_pos.tolist()}, "
                f"local_last_pos={local_last_pos.tolist()}"
            )
        batch_indices = torch.arange(
            outputs.logits.size(0), device=outputs.logits.device
        )
        next_token_logits = outputs.logits[batch_indices, local_last_pos]

        self.gen_forward_cnt += 1
        return outputs.past_key_values, next_token_logits, inputs_embeds

    @staticmethod
    def _expand_kv_cache(past_key_values, factor):
        """Repeat each KV cache entry ``factor`` times along the batch dim."""
        if past_key_values is None or factor == 1:
            return past_key_values
        # transformers DynamicCache.batch_repeat_interleave() mutates in place
        # and returns None; keep compatibility if a future cache type returns
        # itself.
        expanded = past_key_values.batch_repeat_interleave(factor)
        return past_key_values if expanded is None else expanded

    def generate(
        self,
        input_ids,
        attention_mask,
        position_ids=None,
        max_new_tokens=256,
        output_embedding=False,
        synced_gpus=False,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        top_k=0,
        num_generations=1,
        **kwargs,
    ):
        """Autoregressive generation after latent reasoning.

        Supports arbitrary batch sizes. For each sample in the batch:
        1. Forward-pass through latent tokens (multi-pass)
        2. Decode new tokens using KV cache until EOS (greedy or sampling)

        When ``num_generations`` > 1 the latent forward is computed once and
        the resulting KV cache is replicated so that all G completions per
        prompt share the same prefix computation.  This avoids redundant
        multi-pass latent forwards and gives a large speed-up.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            position_ids: (batch_size, seq_len) or None (auto-computed)
            temperature: Sampling temperature (0.0 = greedy)
            do_sample: Whether to use sampling instead of greedy
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold (0 = disabled)
            num_generations: G, number of completions per prompt (default 1)
        """
        self.gen_forward_cnt = 0
        batch_size = input_ids.shape[0]
        device = input_ids.device
        G = num_generations

        # Compute position_ids from attention_mask if not provided
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

        with torch.inference_mode():
            # --- Latent forward (computed once for B prompts) ---
            past_key_values, next_token_logits, inputs_embeds = (
                self._forward_latent_for_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
            )

            # --- Expand to B*G for parallel multi-generation ---
            if G > 1:
                past_key_values = self._expand_kv_cache(past_key_values, G)
                next_token_logits = next_token_logits.repeat_interleave(G, dim=0)
                attention_mask = attention_mask.repeat_interleave(G, dim=0)
                input_ids = input_ids.repeat_interleave(G, dim=0)

            total_batch = batch_size * G

            # First token: sample from logits
            # forbidden_token_ids = [
            #     self.latent_token_id,
            #     self.start_latent_id,
            #     self.end_latent_id,
            # ]
            next_tokens = sample_from_logits(
                next_token_logits,
                temperature,
                do_sample,
                top_p,
                top_k,
                # forbidden_token_ids=forbidden_token_ids,
            )

            # Initialize per-sample generated token lists (input + generated)
            generated = [input_ids[i].tolist() for i in range(total_batch)]
            done = next_tokens == self.eos_token_id
            next_tokens_cpu = next_tokens.cpu().tolist()
            done_cpu = done.cpu().tolist()
            for i in range(total_batch):
                if not done_cpu[i]:
                    generated[i].append(next_tokens_cpu[i])

            # Next position per sample = number of real tokens so far
            cur_positions = attention_mask.sum(dim=-1)  # (total_batch,)

            # Pre-allocate full decode attention mask to avoid repeated cat.
            max_total_len = attention_mask.shape[1] + max_new_tokens
            decode_attn = attention_mask.new_zeros((total_batch, max_total_len))
            decode_attn[:, : attention_mask.shape[1]] = attention_mask
            decode_len = attention_mask.shape[1]

            # Greedily decode remaining tokens with KV cache
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
                    temperature,
                    do_sample,
                    top_p,
                    top_k,
                    # forbidden_token_ids=forbidden_token_ids,
                )

                # Track EOS
                done = done | (next_tokens == self.eos_token_id)

                # Batch GPU→CPU transfer (avoid per-element sync)
                next_tokens_cpu = next_tokens.cpu().tolist()
                done_cpu = done.cpu().tolist()
                for i in range(total_batch):
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
                            (total_batch, 1), self.eos_token_id,
                            device=device, dtype=torch.long,
                        )
                    )
                    _ = self.base_causallm(
                        inputs_embeds=dummy_embed,
                        attention_mask=torch.ones(
                            total_batch, 1, device=device, dtype=torch.long,
                        ),
                        use_cache=False,
                    )

        # Pad results to same length
        max_len = max(len(g) for g in generated)
        result = torch.full(
            (total_batch, max_len), self.eos_token_id,
            device=device, dtype=torch.long,
        )
        for i in range(total_batch):
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
