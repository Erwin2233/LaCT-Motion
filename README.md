# **[ECCV 2026]** LaCT-Motion

**Practice Makes Perfect: From Explicit Decomposition to Reinforced Latent Planning in Text-to-Motion Generation**

[![Paper](https://img.shields.io/badge/Paper-ECCV%202026-b31b1b.svg)](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/13122.pdf)

Ronghao Yu<sup>1,2\*‡</sup>, Xiyue Bai<sup>3\*</sup>, Yang Liu<sup>4\*</sup>, Juncheng Wang<sup>5</sup>, Chao Xu<sup>2,6</sup>, Yimo Shao<sup>7</sup>, Baigui Sun<sup>2,6</sup>, Yong Liu<sup>1,8†</sup>, Shan Luo<sup>4†</sup>

<sup>1</sup>Zhejiang University, <sup>2</sup>IROOTECH TECHNOLOGY, <sup>3</sup>Fudan University, <sup>4</sup>King's College London, <sup>5</sup>The Hong Kong Polytechnic University, <sup>6</sup>Wolf 1069 b Lab, Sany Group, <sup>7</sup>The University of Melbourne, <sup>8</sup>Huzhou Institute of Zhejiang University

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding authors. <sup>‡</sup>This work was conducted in collaboration with IROOTECH TECHNOLOGY.

This repository is the **official implementation** of our **ECCV 2026** paper. Paper PDF: <https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/13122.pdf>

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
├── prepare/                    # Data builder, integrity verifier, and download scripts
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

## Data and Model Resources

The project expects the following resources at these project-relative locations. The five JSON files under `data/` are tracked in this repository. Everything else is excluded from version control by `.gitignore` and must be downloaded or produced as described in [Pretrained Weight and Dataset Preparation](#pretrained-weight-and-dataset-preparation).

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
| HumanML3D resources | `dataset/HumanML3D/` | Raw joint features, caption/POS text, optional reasoning text, and train/validation/test split files |

Each processed split is a JSON array with `question` (a ChatML prompt), `steps` (a list of reasoning strings), and `answer` (motion tokens wrapped in `<Motion>...</Motion>`). These three datasets are shared by SFT and GRPO and have not been filtered, regenerated, or rewritten.

The included reasoning JSON was recovered from an existing same-name copy after the former source path became unavailable. Its reasoning sequences match every nonempty `steps` sequence in the included training split. Original source hashes and the source of each retained file are recorded in `docs/COPY_MANIFEST.json`.

The `checkpoint-epoch6` SFT checkpoint is used to initialize a new GRPO run. Its former SFT optimizer state is not needed for that purpose and is not kept. New SFT runs save under `checkpoints/sft/runs/` to keep the initialization checkpoint separate.

## Pretrained Weight and Dataset Preparation

The download steps mirror those of [UniMo](https://github.com/GuocunWang/UniMo); the VQ-VAE, GloVe, and evaluator files are byte-identical to the ones UniMo uses. The download scripts need `gdown` and `unzip`:

```bash
python -m pip install gdown
```

1. **VQ-VAE, GloVe, and evaluator weights**

   These come from the [Motion-Agent](https://github.com/szqwu/Motion-Agent) release. Run the scripts from the project root:

   ```bash
   bash prepare/download_ckpt.sh        # ckpt/vqvae.pth
   bash prepare/download_glove.sh       # glove/our_vab_{data.npy,idx.pkl,words.pkl}
   bash prepare/download_extractor.sh   # checkpoints/t2m/{Comp_v6_KLD005,text_mot_match,VQVAEV3_CB1024_CMT_H1024_NRES3}
   ```

   Each script downloads one Motion-Agent archive (`motion_agent.zip`, `glove.zip`, or `t2m.zip`) and keeps only the files listed in the table above; `motionllm.pth` and the unused evaluator models are discarded, and the KIT-ML archive is not downloaded. Unlike the upstream scripts, they never delete `checkpoints/`, so SFT and GRPO checkpoints stored there are preserved. If Google Drive is not reachable from the machine, download the archive in a browser, place it in the project root under the same name, and rerun the script; the download step is skipped when the archive is already present.

2. **Qwen2.5-3B-Instruct**

   Place the official [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) model in `Qwen/Qwen2.5-3B-Instruct/`:

   ```bash
   hf download Qwen/Qwen2.5-3B-Instruct --local-dir Qwen/Qwen2.5-3B-Instruct
   ```

   Older `huggingface_hub` releases use `huggingface-cli download` with the same arguments.

3. **HumanML3D dataset**

   Follow the [HumanML3D repository](https://github.com/EricGuo5513/HumanML3D) to obtain AMASS and HumanAct12 and run its processing notebooks, which produce the motion features, the captions with POS tags, and the split files. Copy them into:

   ```text
   dataset/HumanML3D/
   ├── new_joint_vecs/   # <id>.npy, 263-dim HumanML3D features (29,226 files)
   ├── texts/            # <id>.txt, one "caption#POS tokens#start#end" line per caption (29,232 files)
   ├── train.txt
   ├── val.txt
   └── test.txt
   ```

   `Mean.npy` and `Std.npy` from HumanML3D are not used; normalization comes from `checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/`.

4. **Reasoning text (optional)**

   `dataset/HumanML3D/final/` holds one `<thinking>...</thinking>` file per caption, named `<motion id>_<caption index>.txt` (87,383 files). The processed splits already contain the reasoning steps, and `get_train_data.py` rebuilds them from `data/texts_think_steps.json`, which is included in the repository, so this directory is not required for training. The evaluation loader reads it when present and otherwise uses an empty chain-of-thought and prints a warning per caption; the reported metrics do not depend on it. UniMo's `texts_think_qwen.zip` uses a different per-motion layout and is not a drop-in replacement.

5. **SFT checkpoint for GRPO**

   `checkpoints/sft/checkpoint-epoch6/` is not distributed. Produce an equivalent checkpoint with Stage 1 below (saved under `checkpoints/sft/runs/`) and point `sft_checkpoint` in the GRPO configuration to it.

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

## Acknowledgements

The implementations build on Coconut-style latent reasoning, Qwen, UniMo, Motion-R1, HumanML3D, and TRL. Bundled third-party code retains its copyright notices and licenses. Qwen's license is included with the base model files.

## Citation

If you find LaCT-Motion useful for your research, please cite our paper:

```bibtex
@inproceedings{yu2026lactmotion,
  title     = {Practice Makes Perfect: From Explicit Decomposition to Reinforced Latent Planning in Text-to-Motion Generation},
  author    = {Yu, Ronghao and Liu, Yang and Wang, Juncheng and Xu, Chao and Shao, Yimo and Sun, Baigui and Liu, Yong and Luo, Shan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
