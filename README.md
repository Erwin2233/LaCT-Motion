# LaCT-Motion

LaCT-Motion combines latent chain-of-thought supervised fine-tuning (SFT) with Group Relative Policy Optimization (GRPO) for text-to-human motion generation. The model uses Qwen2.5-3B-Instruct, a curriculum that replaces explicit reasoning steps with latent states, and a vocabulary of 512 motion codes.

The project follows a UniMo-style layout with root training, evaluation, data preparation, and demo entry points. The Motion-R1 modules and custom TRL motion rewards used by the pipeline are included locally. Runtime code does not require a sibling project checkout or an external project-path environment variable. Python resolves bundled resources from the project directory; configuration and command examples use relative paths.

## Repository Layout

```text
LaCT-Motion/
├── train_sft.py                  # Curriculum SFT entry point
├── train_grpo.py                 # GRPO entry point
├── get_train_data.py             # Data preparation entry point
├── eval_t2m.py                   # Standard text-to-motion evaluation
├── demo.py                       # Motion generation and skeleton preview
├── _bootstrap.py                # Stage-specific import adapter
├── _paths.py                    # Project-local resource paths
├── requirements.txt
├── data/
│   ├── t2m_train.json            # Shared SFT/GRPO training data
│   ├── t2m_val.json
│   ├── t2m_test.json
│   ├── texts_think_steps.json    # Reasoning annotations
│   └── motion_train.json         # Captions, POS tokens, and motion codes
├── Qwen/Qwen2.5-3B-Instruct/     # Base model and tokenizer
├── checkpoints/
│   ├── sft/checkpoint-epoch6/    # SFT checkpoint used by the original GRPO runs
│   ├── sft/runs/                # New SFT outputs
│   ├── grpo/                    # New GRPO outputs
│   └── t2m/                     # Reward/evaluation weights and normalization
├── ckpt/vqvae.pth               # Motion decoder checkpoint
├── glove/                      # Evaluator vocabulary and embeddings
├── dataset/
│   ├── HumanML3D/              # Local raw motions, captions, and split files
│   └── {sft,grpo}/             # Training datasets and collators
├── models/{sft,grpo}/           # CoconutMotion implementations
├── training/{sft,grpo}/         # Training loops and DDP accumulation helper
├── rewards/grpo/               # GRPO losses and reward integration
├── evaluation/common/          # Shared evaluator modules
├── evaluation/{sft,grpo}/       # Stage-specific evaluation code
├── demo/{sft,grpo}/             # Inference implementations
├── utils/{sft,grpo}/
├── third_party/
│   ├── motion_r1/              # VQ-VAE, motion utilities, and license
│   └── trl_motion/             # Custom rewards, T2M dependencies, and license
├── options/{sft,grpo}/          # Training configurations
├── options/local/              # Local run configurations
├── scripts/{sft,grpo}/          # Shell workflows
├── prepare/                    # Data builder and integrity verifier
├── tests/{sft,grpo}/
├── debug/{sft,grpo}/
├── docs/                       # Provenance and retained historical documents
├── outputs/                    # Run logs, process records, and runtime files
└── results/                    # Generated motions and evaluation results
```

Shared code and processed datasets are stored once. Stage-specific implementations remain separate when they differ. Model packages retain the tokenizer/configuration files required to load each model independently. Standalone SMPL rendering, auxiliary physics/latent analysis, historical W&B runs, caches, and old experiment logs are excluded.

## Environment

Run commands from the project root:

```bash
cd LaCT-Motion
python -m pip install -r requirements.txt
```

Activate the Python environment you intend to use before running a launcher. Shell scripts use `python` and `torchrun` from the active environment, with no fixed Conda installation path. Transformers is pinned to 4.57.3 because the latent-generation cache interface is incompatible with the tested 5.5 release. Other dependencies specify minimum versions; this is not a complete environment lock.

SFT import and CLI checks use PyTorch 2.2.0 and Transformers 4.57.3. The GRPO launch environment uses PyTorch 2.8.0 and Transformers 4.57.3. The SFT template selects `flash_attention_2`; it requires a compatible FlashAttention installation. GRPO uses `eager` attention. FFmpeg must be available on `PATH` for skeleton video export.

The custom TRL reward implementation is bundled under `third_party/trl_motion`; a separately modified TRL checkout is not required. Optional mathematical or motion-to-text reward variants may require additional packages such as `math_verify` or CLIP. The default text-to-motion reward path uses the included VQ-VAE, evaluator, and GloVe resources.

## Included Data and Model Resources

| Resource | Project-relative location | Contents |
|---|---|---|
| Training split | `data/t2m_train.json` | 66,515 records |
| Validation split | `data/t2m_val.json` | 4,170 records |
| Test split | `data/t2m_test.json` | 12,503 records |
| Reasoning annotations | `data/texts_think_steps.json` | 87,245 records with `id` and `steps` |
| POS-tagged motion/caption data | `data/motion_train.json` | 66,490 records |
| Qwen base model | `Qwen/Qwen2.5-3B-Instruct/` | Model shards, configuration, and tokenizer |
| Initial SFT checkpoint for GRPO | `checkpoints/sft/checkpoint-epoch6/` | Model shards and tokenizer from the checkpoint referenced by the original GRPO configurations |
| Motion VQ-VAE | `ckpt/vqvae.pth` | Motion decoder weights |
| Reward/evaluation model | `checkpoints/t2m/text_mot_match/model/finest.tar` | Text, motion, and movement encoders |
| Evaluation configuration | `checkpoints/t2m/Comp_v6_KLD005/opt.txt` | Evaluator dimensions and options |
| Motion normalization | `checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/` | `mean.npy` and `std.npy` |
| Word embeddings | `glove/` | `our_vab_data.npy`, `our_vab_idx.pkl`, and `our_vab_words.pkl` |
| HumanML3D resources | `dataset/HumanML3D/` | Raw joint features, caption/POS text, reasoning text, and train/validation/test split files |

Each processed split is a JSON array with `question` (a ChatML prompt), `steps` (a list of reasoning strings), and `answer` (motion tokens wrapped in `<Motion>...</Motion>`). These three datasets are shared by SFT and GRPO and have not been filtered, regenerated, or rewritten.

The included reasoning JSON was recovered from an existing same-name copy after the former source path became unavailable. Its reasoning sequences match every nonempty `steps` sequence in the included training split. Original source hashes and the source of each retained file are recorded in `docs/COPY_MANIFEST.json`.

The supplied SFT checkpoint is used to initialize a new GRPO run. Its former SFT optimizer state is not needed for that purpose and is not copied. New SFT runs save under `checkpoints/sft/runs/` to keep the supplied initialization checkpoint separate.

## Configuration

The templates already point to resources inside this project:

- SFT: `options/sft/t2m_coconut.yaml`
- GRPO: `options/grpo/t2m_grpo.yaml`
- Historical GRPO-side SFT settings: `options/grpo/t2m_coconut.yaml`

Create a separate configuration for an experiment if needed:

```bash
mkdir -p options/local
cp options/sft/t2m_coconut.yaml options/local/sft.yaml
cp options/grpo/t2m_grpo.yaml options/local/grpo.yaml
```

Match `c_thought`, `max_latent_stage`, `nb_code`, and tokenizer/model settings to the chosen SFT checkpoint. The bundled GRPO configuration uses `checkpoints/sft/checkpoint-epoch6`, `loss_type: dapo`, `reward_mode: unimo`, reward weights `[1.0, 1.0, 1.0]` for format, motion-to-motion similarity, and motion-to-text similarity, and gradient accumulation of 8. A valid motion output receives `1 + motion_sim + text_sim`; an invalid format receives zero. Choose batch size and GPU count for the available resources.

Relative command-line and configuration paths are interpreted from the project root by the root launchers.

## Training

### Stage 1: Curriculum SFT

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  train_sft.py options/sft/t2m_coconut.yaml
```

SFT takes the YAML filename as a positional argument. It gradually replaces explicit reasoning with latent tokens and runs the configured consolidation phase. The supplied SFT checkpoint remains available separately from new SFT outputs.

### Stage 2: GRPO

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  torchrun --standalone --nproc_per_node=4 \
  train_grpo.py --config options/grpo/t2m_grpo.yaml
```

GRPO takes the configuration through `--config`. It generates groups of motion completions, computes group-relative rewards, and updates the latent policy. `save_path` controls the checkpoint directory. The implementation includes `grpo`, `dapo`, `dr_grpo`, `bnpo`, `cispo`, and `sapo` objectives; the default is DAPO.

Training uses `DistributedSampler(shuffle=True)` and calls `set_epoch(epoch)` before each epoch, so distributed sample order is reshuffled every epoch.

To keep a run alive after closing the terminal or Codex, use the background launcher:

```bash
python scripts/grpo/launch_background.py \
  --gpus 4,5,6,7 --config options/grpo/t2m_grpo.yaml --wandb
```

The launcher uses the active Python environment, creates a new session, applies `nohup`, and redirects all standard streams. Each run gets a timestamped configuration, checkpoint directory, and log directory. It preserves the template's training hyperparameters and uses a local temporary directory that supports the Unix sockets needed by dataset workers. Override `--temp-dir` if necessary.

`--wandb` enables online metric logging using the current W&B login and the configuration's project. Use `--wandb-project` and `--wandb-entity` to select another destination, or `--no-wandb` to disable uploads. W&B's local run files are stored inside the run's log directory. Standard output and errors are always saved to `train.log`; generated samples are saved to `generations.jsonl` in the checkpoint directory.

The printed process ID belongs to the detached `torchrun` supervisor. A successful launch submission does not imply that model loading or training has succeeded; inspect the log for iteration and optimizer-step output. Follow the latest submitted run from the project root:

```bash
tail -f "$(cat outputs/grpo/active_run.txt)/train.log"
```

Each run directory contains `train.pid` and `launch.json`, including its configuration and checkpoint paths. Historical logs are not copied from the source projects; these files are created by new runs.

### Background Monitoring and Final Evaluation

Attach an independent monitor to the latest training run:

```bash
python scripts/grpo/monitor_training.py --eval-gpu 4 --background
```

Use `--run-dir outputs/grpo/<run_id>` to select a particular run. The monitor also uses a detached session and `nohup`. It checks training progress every 60 seconds, records non-finite metrics and process failures, and flags a lack of progress for 30 minutes. Its process details, status, events, and log are saved under the training run's `monitor/` directory.

After all tracked training processes exit, the monitor requires the final-save completion message and verifies the final model metadata and safetensors payloads. Successful training then triggers one HumanML3D test evaluation on the selected GPU with batch size 32. Sampling uses the training configuration's temperature, top-p, top-k, maximum generation length, and latent settings; the current configuration uses temperature 1.0, top-p 0.9, and at most 80 new tokens. `--repeat` controls evaluation repetitions.

Evaluation metrics are saved to `results/grpo/<run_id>/final.json`, with the evaluator log under `outputs/grpo/<run_id>/evaluation/eval.log`. The monitor writes FID, R-Precision, matching score, diversity, and embedding similarities to the training W&B run under `evaluation/*`. Upload failures are recorded in the local monitor status. A failed or incomplete training run is recorded without starting evaluation. Existing result files are preserved.

Inspect the latest run's monitor status from the project root:

```bash
cat "$(cat outputs/grpo/active_run.txt)/monitor/status.json"
```

## Data Preparation

Training can use the supplied JSON files directly. To rebuild them from the included annotations, captions, raw motions, and VQ-VAE:

```bash
python get_train_data.py \
  --think-steps data/texts_think_steps.json \
  --texts-dir dataset/HumanML3D/texts \
  --motion-dir dataset/HumanML3D/new_joint_vecs \
  --splits-dir dataset/HumanML3D \
  --vqvae-path ckpt/vqvae.pth \
  --meta-dir checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta \
  --output-dir data/rebuilt \
  --device cuda:0
```

The default output directory is `data/rebuilt/`, so rebuilding does not replace the included splits. Update a local training configuration to use rebuilt files if desired.

## Evaluation and Generation

Evaluate a compatible full-model checkpoint with an SFT-style configuration matching its latent settings:

```bash
python eval_t2m.py options/sft/t2m_coconut.yaml \
  --checkpoint checkpoints/sft/checkpoint-epoch6 \
  --batch-size 32 --repeat 1 --device cuda:0 \
  --output results/t2m_metrics.json
```

Metrics include FID, R-Precision at Top-1/2/3, matching score, and diversity. Add `--multimodality` for multimodality evaluation or `--unimo-sampling` for the evaluator's UniMo sampling settings. Pass a GRPO checkpoint through `--checkpoint` to evaluate it with the same interface.

Generate a motion and skeleton preview:

```bash
python demo.py options/sft/t2m_coconut.yaml \
  --checkpoint checkpoints/sft/checkpoint-epoch6 \
  --text "A person walks forward slowly and then turns around." \
  --output results/demo --render-mode skeleton --device cuda:0
```

Use `--text-file prompts.txt` for multiple prompts. The demo saves motion features, joint coordinates, metadata, and a skeleton video when FFmpeg is available. Standalone SMPL rendering is excluded.

## Verification and Implementation Notes

Verify the retained source/data files and their recorded local adaptations:

```bash
python prepare/verify_copies.py
```

To verify model and raw-dataset resources as well, use `--include-resources`. The optional `--check-sources` flag additionally compares the original workspace sources when those source directories are available; normal execution does not use them.

The GRPO DDP synchronization bug has been fixed by selecting gradient synchronization before the policy forward pass. A two-process CPU regression checks model parameters and AdamW state against a single-process global-batch reference, including accumulation factors 1, 3, and 8 and incomplete final accumulation windows:

```bash
python -m unittest discover -s tests/grpo -p test_ddp_accumulation.py -v
```

SFT keeps synchronization enabled during each backward pass and updates parameters at the accumulation boundary. Training objectives and hyperparameters are otherwise preserved except for explicit run configuration choices.

The historical SFT regression suite still references `_compute_warmup_lr`, which is absent from the retained trainer. The older GRPO-specific inference/evaluation files also retain a missing legacy `train` import; use the root demo and evaluation entry points. GRPO validation/resume options are not a complete evaluation or optimizer-resume workflow; use the explicit evaluation command above. These separate issues are not covered by the DDP fix.

## Acknowledgements

The implementations build on Coconut-style latent reasoning, Qwen, UniMo, Motion-R1, HumanML3D, and TRL. Bundled third-party code retains its copyright notices and licenses. Qwen's license is included with the base model files.
