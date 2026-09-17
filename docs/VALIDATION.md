# Validation Record

Validation date: 2026-09-16.

## Copies and Portability

The provenance manifest distinguishes original source hashes from hashes of locally adapted destination files. It records 84 copied source/data files, representing 95 source paths, with 48 local adaptations. Source projects remain unchanged. Newly written launchers, documentation, path helpers, and the regression are recorded separately.

All 31 model/evaluator/embedding files were checked against their source SHA-256 while copying: 12,707,163,359 bytes. HumanML3D copying verified every caption, motion, reasoning-text, and split file against its source: 145,844 files totaling 4,432,207,821 bytes. These are physical copies, not source links. Identical model-package metadata is retained where each model package requires it.

The reasoning JSON was selected by content: all 66,515 training rows match its reasoning sequences. The earlier unavailable UniMo reasoning path is not a runtime dependency. Shared processed train/validation/test JSON files preserve their source bytes and record counts. No training records were deduplicated or regenerated.

Repeat ordinary checks with:

```bash
python prepare/verify_copies.py --check-sources
```

Use `--include-resources` for a full model/raw-dataset hash scan. Original source directories are required only for `--check-sources` and are resolved relative to the repository. Ordinary training uses bundled local resources.

## Syntax, Imports, and Resource Loading

All 73 Python files present before the monitoring addition parsed successfully and all seven shell scripts passed `bash -n`. The SFT training, data preparation, evaluation, and demo root launchers passed `--help` checks in the existing `motionR1` environment. GRPO imports and CLI checks passed in the `trl` environment; that alone did not establish generation compatibility.

GPU resource checks loaded the local VQ-VAE, evaluator checkpoint (epoch 28), and GloVe vocabulary successfully. Custom rewards resolve to `third_party.trl_motion`. The absent optional skating reward produces the original warning; the default UniMo reward mode uses the successfully loaded local resources.

The initial four-GPU launch exposed an inherited temporary-directory setting on a filesystem that does not support Unix sockets. The background launcher now selects and probes a local temporary directory before starting dataset workers. A subsequent launch exposed the original wrapper's incompatibility with Transformers 5.5's cache entries. The existing `lucid` environment has PyTorch 2.8.0 and Transformers 4.57.3; imports and a real cache-slicing check passed there. Transformers 4.57.3 is pinned in the dependency file. No model computation was changed to work around this version mismatch.

## DDP Synchronization Regression

The GRPO accumulation flag is now set before the policy forward pass. The regression starts two CPU Gloo workers, runs different data on each rank, and checks both model parameters and AdamW state against a single-process global-batch reference after every update. Accumulation factors 1, 3, and 8 include incomplete final windows. Gradient clipping is included.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -m unittest discover -s tests/grpo -p test_ddp_accumulation.py -v
```

Result in the `trl` environment: one test passed, 5.604 seconds. The SFT synchronization behavior is unchanged. The historical SFT suite's missing `_compute_warmup_lr` import remains a pre-existing failure, also reproduced in the original project.

## Background Training

The requested four-process GRPO run uses physical GPUs 4, 5, 6, and 7 and starts from the included SFT `checkpoint-epoch6`. Each launch has its own timestamped YAML, log directory, and checkpoint destination. `outputs/grpo/active_run.txt` identifies the latest submission; its `launch.json` and `train.pid` record the process and paths.

The launcher uses a new session plus `nohup` and redirects stdin/stdout/stderr, so it does not depend on an open Codex session. The supervisor's parent becomes PID 1. The latest requested run enables online W&B logging and retains its local W&B files inside the log directory. All 66,515 training captions matched POS annotations, including 78 recovered from the bundled HumanML3D caption files. Before the logging change, the detached run completed its first optimizer update and subsequent iterations with finite loss/reward/KL values. It was stopped and restarted from the same SFT checkpoint to enable online W&B from step one; the earlier log is retained. A launch record by itself does not prove training progress; inspect iteration and optimizer-step messages in the corresponding log. Full training convergence and final evaluation are not established by startup checks.

The previous online run, `grpo_gpu4_5_6_7_20260916_184555`, used reward weights `[1, 0, 1]` on GPUs 4–7. W&B server verification at `2026-09-16T18:53:00.694722+08:00` confirmed uploaded step-one loss, reward, KL, gradient norm, and learning rate; all were finite. The verification snapshot is saved alongside its local log as `wandb_verified.json`. This run was subsequently stopped at the user's request to change the reward weights. Its logs and outputs are retained. W&B history can appear one update behind because the original logger supplies an explicit step without immediately committing its row.

Run page: https://wandb.ai/a274617269-zhejiang-university/latent-cot-grpo/runs/yfyopwe2

## Restart with All Reward Weights Enabled

The run `grpo_gpu4_5_6_7_20260916_185840` starts fresh from `checkpoints/sft/checkpoint-epoch6` with reward weights `[1, 1, 1]`. Its supervisor is PID 138730, and its four workers use physical GPUs 4–7. A configuration comparison confirmed that only the reward weights and the independent output/run names differ from the previous run. The canonical GRPO template and README now describe these weights.

All four reward instances report `format=1.0, motion_sim=1.0, text_sim=1.0`. Training iterations have finite losses and rewards. Generated-sample records are checked against the actual formula: `1 + motion_sim + text_sim` for valid format, and zero otherwise. Standard output/errors, generated samples, and local W&B records are saved in this run's own directories. The supervisor runs in a separate session with PID 1 as its parent.

Run page: https://wandb.ai/a274617269-zhejiang-university/latent-cot-grpo/runs/8hq2re31

## Detached Supervision and Evaluation Preflight

Training shuffling was already enabled by `DistributedSampler(train_dataset, shuffle=True)`, with `set_epoch(epoch)` called before each epoch. The running trainer was not restarted or changed for this request.

An independent monitor supervises `grpo_gpu4_5_6_7_20260916_185840`. It records progress every 60 seconds and schedules one final-checkpoint test evaluation on physical GPU 4 with batch size 32. The evaluation configuration inherits the model, latent, and sampling settings from the training run, with sampling explicitly enabled. The completion check requires exited training processes, the final-save marker, finite training metrics, and complete weight files. Evaluation results are stored locally and uploaded to the training W&B summary under `evaluation/*`.

The real evaluation preflight passed in the `lucid` environment on GPU 4 using the supplied SFT checkpoint. The HumanML3D test loader contained 4,646 samples after its existing filtering/segment handling. Two sampled captions produced 44 and 29 motion tokens. VQ-VAE decoding and both generated/ground-truth evaluator embeddings were finite, with embedding shape `[2, 512]`. The FID numerical function also passed a known identical-distribution check. These are pipeline checks, not final GRPO evaluation scores.

Three CPU monitor tests passed in 1.214 seconds. They cover the complete training-to-evaluation path, GPU environment isolation, result preservation, failure/non-finite/incomplete-checkpoint rejection, and a final log write racing with process exit. No production training process is used by these tests. The monitor PID, live progress, evaluation command, and preflight evidence are recorded under the run's `monitor/` directory.
