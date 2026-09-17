"""Check DDP updates against a single-process global-batch reference on CPU."""

from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from training.grpo.gradient_accumulation import forward_with_gradient_sync


def _model():
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.4, -0.2]], dtype=torch.float64))
        model.bias.fill_(0.1)
    return model


def _batch(step, rank):
    x = torch.tensor([[step + 1.0, rank * 2.0 - 0.5]], dtype=torch.float64)
    target = torch.tensor([[0.3 * step - rank]], dtype=torch.float64)
    return x, target


def _worker(rank, world_size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size,
        timeout=timedelta(seconds=45),
    )
    try:
        # Include normal accumulation, no accumulation, and incomplete final windows.
        for accumulation, total_microbatches in [(1, 4), (3, 8), (8, 10)]:
            model = DistributedDataParallel(_model())
            reference = _model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
            reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.02)
            for step in range(total_microbatches):
                boundary = (step + 1) % accumulation == 0 or step + 1 == total_microbatches
                x, target = _batch(step, rank)
                prediction = forward_with_gradient_sync(
                    model, sync_gradients=boundary, input=x,
                )
                (prediction.sub(target).square().mean() / accumulation).backward()

                batches = [_batch(step, other) for other in range(world_size)]
                global_x = torch.cat([batch[0] for batch in batches])
                global_target = torch.cat([batch[1] for batch in batches])
                reference_loss = reference(global_x).sub(global_target).square().mean()
                (reference_loss / accumulation).backward()

                if boundary:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8)
                    torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.8)
                    optimizer.step()
                    reference_optimizer.step()
                    for actual, expected in zip(model.module.parameters(), reference.parameters()):
                        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
                        gathered = [torch.empty_like(actual) for _ in range(world_size)]
                        dist.all_gather(gathered, actual.detach())
                        for other in gathered:
                            torch.testing.assert_close(actual, other, rtol=1e-9, atol=1e-11)
                        for key in ("exp_avg", "exp_avg_sq"):
                            torch.testing.assert_close(
                                optimizer.state[actual][key],
                                reference_optimizer.state[expected][key],
                                rtol=1e-9, atol=1e-11,
                            )
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class DDPAccumulationTests(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is required")
    def test_updates_match_global_batch_reference(self):
        with tempfile.TemporaryDirectory(prefix="lact-ddp-") as directory:
            rendezvous = (Path(directory) / "rendezvous").as_uri()
            mp.spawn(_worker, args=(2, rendezvous), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
