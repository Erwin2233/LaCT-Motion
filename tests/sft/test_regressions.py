import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coconut_motion import CoconutMotion, Outputs, _slice_kv_cache  # noqa: E402
from dataset import get_cot_latent_dataset, get_question_latent_dataset  # noqa: E402
from train import (  # noqa: E402
    _compute_warmup_lr,
    _build_dataloader_kwargs,
    build_epoch_schedule,
    resolve_start_epoch,
)


class _ToyCausalLM(nn.Module):
    """Minimal CausalLM stub with cache semantics for regression tests."""

    def __init__(self, vocab_size=32, hidden_size=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            e = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            p = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            self.embed.weight.copy_(torch.sin(e).view(vocab_size, hidden_size) * 0.05)
            self.proj.weight.copy_(torch.cos(p).view(vocab_size, hidden_size) * 0.05)

    def get_input_embeddings(self):
        return self.embed

    @staticmethod
    def _to_legacy(past_key_values):
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        return past_key_values

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_hidden_states=False,
        use_cache=False,
    ):
        legacy = self._to_legacy(past_key_values)
        batch_size, seq_len, hidden_size = inputs_embeds.shape

        if legacy is None:
            prefix_key = inputs_embeds.new_zeros((batch_size, 1, 0, hidden_size))
            prefix_val = inputs_embeds.new_zeros((batch_size, 1, 0, hidden_size))
            prefix_state = inputs_embeds.new_zeros((batch_size, hidden_size))
        else:
            prefix_key, prefix_val = legacy[0]
            if prefix_key.shape[2] == 0:
                prefix_state = inputs_embeds.new_zeros((batch_size, hidden_size))
            else:
                prefix_state = prefix_key[:, 0, -1, :]

        if position_ids is None:
            position_ids = torch.arange(
                seq_len, device=inputs_embeds.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        pos_term = position_ids.to(inputs_embeds.dtype).unsqueeze(-1) * 0.01
        hidden = torch.cumsum(inputs_embeds + pos_term, dim=1) + prefix_state.unsqueeze(1)
        logits = self.proj(hidden)

        out_past = None
        if use_cache:
            new_key = torch.cat([prefix_key, hidden.unsqueeze(1)], dim=2)
            new_val = torch.cat([prefix_val, (hidden * 0.5).unsqueeze(1)], dim=2)
            out_past = ((new_key, new_val),)

        out_hidden_states = [hidden] if output_hidden_states else None
        return SimpleNamespace(
            logits=logits,
            hidden_states=out_hidden_states,
            past_key_values=out_past,
        )


def _forward_reference_python_rebuild(
    model, input_ids, attention_mask, labels, position_ids
):
    """Reference implementation of legacy forward() replacement path."""
    logits = []
    latent_indices = (input_ids == model.latent_token_id).nonzero()

    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max([len(lst) for lst in latent_lists]) if latent_lists else 0

    next_compute_range = (0, input_ids.shape[1])
    inputs_embeds = model.embedding(input_ids)
    if max_n_latents > 0:
        next_compute_range = (0, latent_indices[:, 1].min().item())

    kv_cache = None
    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = model.base_causallm(
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
            past_key_values = _slice_kv_cache(kv_cache, next_compute_range[0])
            outputs = model.base_causallm(
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

        filling_indices = [
            (instance_idx, mask_list[pass_idx])
            for instance_idx, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]

        tensor_list = [
            [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
            for batch_idx in range(inputs_embeds.shape[0])
        ]
        for batch_idx, token_idx in filling_indices:
            tensor_list[batch_idx][token_idx] = hidden_states[
                batch_idx, token_idx - 1 - hidden_states_offset, :
            ]
        inputs_embeds = torch.stack(
            [
                torch.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ]
        )

    final_past_kv = None
    if kv_cache is not None:
        final_past_kv = _slice_kv_cache(kv_cache, next_compute_range[0])

    outputs = model.base_causallm(
        inputs_embeds=inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :],
        attention_mask=(
            attention_mask[:, : next_compute_range[1]]
            if kv_cache is not None
            else attention_mask[:, next_compute_range[0] : next_compute_range[1]]
        ),
        position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
        past_key_values=final_past_kv,
        output_hidden_states=False,
        use_cache=False,
    )
    logits.append(outputs.logits)
    logits = torch.cat(logits, dim=-2)

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)


def _generate_reference_prefill(
    model, input_ids, attention_mask, position_ids, max_new_tokens
):
    """Reference generate path using legacy double-prefill flow."""
    labels = input_ids.clone()
    outputs = model.forward(input_ids, attention_mask, labels, position_ids)
    inputs_embeds = outputs.inputs_embeds

    prefill_out = model.base_causallm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_key_values = prefill_out.past_key_values
    next_tokens = torch.argmax(prefill_out.logits[:, -1, :], dim=-1)

    batch_size = input_ids.shape[0]
    device = input_ids.device
    generated = [input_ids[i].tolist() for i in range(batch_size)]
    done = next_tokens == model.eos_token_id
    next_tokens_cpu = next_tokens.cpu().tolist()
    done_cpu = done.cpu().tolist()
    for i in range(batch_size):
        if not done_cpu[i]:
            generated[i].append(next_tokens_cpu[i])

    cur_positions = attention_mask.sum(dim=-1)
    decode_attn = attention_mask.clone()

    for _ in range(max_new_tokens - 1):
        if done.all():
            break
        new_token_embed = model.embedding(next_tokens.unsqueeze(1))
        pos_ids = cur_positions.unsqueeze(1)
        decode_attn = torch.cat(
            [
                decode_attn,
                torch.ones(batch_size, 1, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        out = model.base_causallm(
            inputs_embeds=new_token_embed,
            attention_mask=decode_attn,
            past_key_values=past_key_values,
            position_ids=pos_ids,
            use_cache=True,
        )
        past_key_values = out.past_key_values
        cur_positions = cur_positions + 1
        next_tokens = torch.argmax(out.logits[:, -1, :], dim=-1)
        done = done | (next_tokens == model.eos_token_id)
        next_tokens_cpu = next_tokens.cpu().tolist()
        done_cpu = done.cpu().tolist()
        for i in range(batch_size):
            if not done_cpu[i]:
                generated[i].append(next_tokens_cpu[i])
        next_tokens[done] = model.eos_token_id

    max_len = max(len(g) for g in generated)
    result = torch.full(
        (batch_size, max_len), model.eos_token_id, device=device, dtype=torch.long
    )
    for i in range(batch_size):
        result[i, : len(generated[i])] = torch.tensor(
            generated[i], device=device, dtype=torch.long
        )

    return result, inputs_embeds


class DatasetRegressionTests(unittest.TestCase):
    def test_no_cot_skips_all_reasoning_steps(self):
        base = Dataset.from_dict(
            {
                "question_tokenized": [[10, 11]],
                "steps_tokenized": [[[21, 22], [31, 32]]],
                "answer_tokenized": [[41, 42]],
                "idx": [0],
            }
        )
        cfg = SimpleNamespace(
            uniform_prob=0.0,
            max_latent_stage=8,
            pad_latent_to_max=True,
            c_thought=2,
            no_cot=True,
            max_seq_len=128,
        )
        out = get_cot_latent_dataset(
            scheduled_stage=3,
            base_dataset=base,
            configs=cfg,
            start_id=100,
            latent_id=101,
            end_id=102,
            think_open_ids=[201],
            think_close_ids=[202],
            no_special_marker=True,
            shuffle=False,
        )

        sample = out[0]
        self.assertEqual(sample["input_ids"], [10, 11, 41, 42])
        self.assertEqual(sample["labels"], [-100, -100, 41, 42])
        self.assertNotIn(201, sample["input_ids"])
        self.assertNotIn(202, sample["input_ids"])
        self.assertNotIn(101, sample["input_ids"])

    def test_max_seq_len_is_enforced_in_cot_dataset(self):
        base = Dataset.from_dict(
            {
                "question_tokenized": [[1, 2, 3]],
                "steps_tokenized": [[[4, 5, 6], [7, 8, 9]]],
                "answer_tokenized": [[10, 11, 12]],
                "idx": [0],
            }
        )
        cfg = SimpleNamespace(
            uniform_prob=0.0,
            max_latent_stage=8,
            pad_latent_to_max=True,
            c_thought=1,
            no_cot=False,
            max_seq_len=5,
        )
        out = get_cot_latent_dataset(
            scheduled_stage=0,
            base_dataset=base,
            configs=cfg,
            start_id=100,
            latent_id=101,
            end_id=102,
            think_open_ids=[201],
            think_close_ids=[202],
            no_special_marker=False,
            shuffle=False,
        )
        sample = out[0]

        self.assertEqual(len(sample["input_ids"]), 5)
        self.assertEqual(len(sample["labels"]), 5)
        self.assertEqual(len(sample["attention_mask"]), 5)
        self.assertEqual(len(sample["position_ids"]), 5)

    def test_question_dataset_respects_no_special_marker(self):
        base = Dataset.from_dict(
            {
                "question_tokenized": [[10, 11]],
                "steps_tokenized": [[[21], [31], [41]]],
                "answer_tokenized": [[51]],
                "idx": [0],
            }
        )
        cfg = SimpleNamespace(
            max_latent_stage=8,
            pad_latent_to_max=True,
            c_thought=2,
            max_seq_len=64,
        )
        with_markers = get_question_latent_dataset(
            scheduled_stage=2,
            base_dataset_valid=base,
            configs=cfg,
            start_id=100,
            latent_id=101,
            end_id=102,
            no_special_marker=False,
        )[0]["input_ids"]
        without_markers = get_question_latent_dataset(
            scheduled_stage=2,
            base_dataset_valid=base,
            configs=cfg,
            start_id=100,
            latent_id=101,
            end_id=102,
            no_special_marker=True,
        )[0]["input_ids"]

        self.assertIn(100, with_markers)
        self.assertIn(102, with_markers)
        self.assertNotIn(100, without_markers)
        self.assertNotIn(102, without_markers)


class TrainHelperTests(unittest.TestCase):
    def test_warmup_lr(self):
        self.assertAlmostEqual(_compute_warmup_lr(2e-5, 200, 1), 1e-7)
        self.assertAlmostEqual(_compute_warmup_lr(2e-5, 200, 200), 2e-5)
        self.assertAlmostEqual(_compute_warmup_lr(2e-5, 200, 500), 2e-5)
        self.assertAlmostEqual(_compute_warmup_lr(2e-5, 0, 1), 2e-5)

    def test_resolve_start_epoch(self):
        self.assertEqual(resolve_start_epoch(0, False, 9), 10)
        self.assertEqual(resolve_start_epoch(0, True, 9), 9)
        self.assertEqual(resolve_start_epoch(5, False, 9), 5)
        self.assertEqual(resolve_start_epoch(0, False, None), 0)

    def test_build_epoch_schedule(self):
        self.assertEqual(list(build_epoch_schedule(3, 6, False)), [3, 4, 5])
        self.assertEqual(list(build_epoch_schedule(6, 6, False)), [])
        self.assertEqual(build_epoch_schedule(9, 30, True), [9])

    def test_dataloader_kwargs_no_worker_only_fields_when_zero_workers(self):
        cfg = SimpleNamespace(
            train_num_workers=0,
            dataloader_prefetch_factor=8,
            dataloader_persistent_workers=True,
        )
        kwargs = _build_dataloader_kwargs(
            cfg, num_workers_key="train_num_workers", default_num_workers=8
        )
        self.assertEqual(kwargs["num_workers"], 0)
        self.assertTrue(kwargs["pin_memory"])
        self.assertNotIn("prefetch_factor", kwargs)
        self.assertNotIn("persistent_workers", kwargs)

    def test_dataloader_kwargs_include_worker_only_fields_when_workers_positive(self):
        cfg = SimpleNamespace(
            val_num_workers=3,
            dataloader_prefetch_factor=6,
            dataloader_persistent_workers=False,
        )
        kwargs = _build_dataloader_kwargs(
            cfg, num_workers_key="val_num_workers", default_num_workers=2
        )
        self.assertEqual(kwargs["num_workers"], 3)
        self.assertEqual(kwargs["prefetch_factor"], 6)
        self.assertFalse(kwargs["persistent_workers"])

    def test_dataloader_kwargs_backward_compatible_defaults(self):
        cfg = SimpleNamespace()
        kwargs = _build_dataloader_kwargs(
            cfg,
            num_workers_key="train_num_workers",
            default_num_workers=8,
            drop_last=True,
        )
        self.assertEqual(kwargs["num_workers"], 8)
        self.assertEqual(kwargs["prefetch_factor"], 4)
        self.assertTrue(kwargs["persistent_workers"])
        self.assertTrue(kwargs["drop_last"])


class CoconutMotionRegressionTests(unittest.TestCase):
    def setUp(self):
        base = _ToyCausalLM(vocab_size=32, hidden_size=12)
        self.model = CoconutMotion(
            base_causallm=base,
            latent_token_id=29,
            start_latent_id=30,
            end_latent_id=31,
            eos_token_id=0,
        )

    def test_forward_matches_legacy_python_rebuild(self):
        input_ids = torch.tensor(
            [
                [5, 30, 29, 31, 7, 8],
                [9, 30, 29, 31, 6, 7],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long
        ).unsqueeze(0).repeat(input_ids.shape[0], 1)
        labels = input_ids.clone()

        out_new = self.model.forward(input_ids, attention_mask, labels, position_ids)
        out_ref = _forward_reference_python_rebuild(
            self.model, input_ids, attention_mask, labels, position_ids
        )

        self.assertTrue(torch.allclose(out_new.logits, out_ref.logits, atol=1e-6))
        self.assertTrue(torch.allclose(out_new.inputs_embeds, out_ref.inputs_embeds, atol=1e-6))
        self.assertTrue(torch.allclose(out_new.loss, out_ref.loss, atol=1e-6))

    def test_generate_matches_legacy_double_prefill_flow(self):
        input_ids = torch.tensor(
            [
                [4, 30, 29, 29, 31, 6],
                [7, 30, 29, 31, 5, 6],
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long
        ).unsqueeze(0).repeat(input_ids.shape[0], 1)
        max_new_tokens = 6

        with torch.inference_mode():
            out_new, emb_new = self.model.generate(
                input_ids,
                attention_mask,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                output_embedding=True,
            )
            out_ref, emb_ref = _generate_reference_prefill(
                self.model,
                input_ids,
                attention_mask,
                position_ids,
                max_new_tokens=max_new_tokens,
            )

        self.assertTrue(torch.equal(out_new, out_ref))
        self.assertTrue(torch.allclose(emb_new, emb_ref, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
