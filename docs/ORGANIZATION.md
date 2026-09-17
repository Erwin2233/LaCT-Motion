# Organization and Source Preservation

The project combines the SFT implementation from `latent-cot-motion` and the GRPO implementation from `latent-cot-grpo`, using the directory layout of UniMo as a reference. Original source directories were not modified. The main README contains the setup and usage instructions without requiring another Markdown document.

## Mapping

| Original group | Local destination |
|---|---|
| SFT and GRPO model wrappers | `models/{sft,grpo}/` |
| Training loops and configuration | `training/{sft,grpo}/`, `options/{sft,grpo}/` |
| Training datasets and collators | `dataset/{sft,grpo}/` |
| Identical processed JSON splits | One shared copy under `data/` |
| Shared data builder | `prepare/build_data.py` |
| Standard evaluation and inference | `evaluation/{sft,grpo}/`, `demo/{sft,grpo}/` |
| Identical evaluator modules | `evaluation/common/eval_infra/` |
| GRPO rewards and losses | `rewards/grpo/` |
| Motion-R1 VQ-VAE and motion utilities | `third_party/motion_r1/` |
| Customized TRL rewards and T2M dependencies | `third_party/trl_motion/` |
| Shell workflows, tests, and debugging tools | `scripts/`, `tests/`, `debug/` |
| Historical technical documents | `docs/{sft,grpo}/` |

`_bootstrap.py` maps the original flat import names to the appropriate stage and makes root entry points interpret relative paths from the project root. `_paths.py` resolves bundled resources. Runtime imports do not depend on sibling source projects. The identical Motion-R1 skeleton parameters reuse the shared evaluator module.

## Included Data and Resources

The train/validation/test JSON files contain 66,515, 4,170, and 12,503 records, respectively, and match both source projects. They have not been transformed or filtered. The additional reasoning JSON contains 87,245 annotations and matches the reasoning sequences of every training row. The POS-tagged motion JSON contains 66,490 records.

Qwen2.5-3B-Instruct, the `checkpoint-epoch6` model referenced by the previous GRPO configuration, VQ-VAE weights, evaluator weights/options, normalization arrays, and GloVe files are copied locally. The original SFT optimizer state is unnecessary for initializing GRPO and is not included. HumanML3D includes 29,232 caption files, 29,226 motion arrays, 87,383 reasoning text files, and three split files.

`docs/COPY_MANIFEST.json` records 84 retained source/data files representing 95 original source paths, plus 145,875 model/dataset resource files totaling 17,139,371,180 bytes. Resource directory records use an aggregate digest over sorted names, sizes, and individual SHA-256 hashes. Model packages keep the configuration/tokenizer files required to load each package independently.

## Local Adaptations

Forty-eight retained files have local path, import, configuration, documentation, or dependency adaptations. The manifest keeps each original hash separately from the destination hash. The requested GRPO fix selects DDP gradient synchronization before the policy forward pass, including the final incomplete accumulation window. A two-process regression checks parameter and AdamW-state agreement against a single-process reference. The SFT training loop and model/reward mathematics are unchanged.

Transformers is pinned to 4.57.3 because the tested 5.5 cache interface is incompatible with the generation wrapper. The detached GRPO launcher creates a separate session, ignores terminal hangup, redirects standard streams, supports online W&B logging, and assigns independent configuration, log, and checkpoint locations. New W&B run files are kept alongside the local training log. Dataset-worker temporary files use a local filesystem that supports Unix sockets.

## Exclusions and Remaining Limitations

Standalone SMPL rendering, auxiliary physics/latent/statistical analysis, historical W&B runs, source logs, caches, and generated experiment artifacts remain excluded. Earlier cleanup affected only the destination. New training logs and checkpoints are outputs of the explicitly requested run. Bundled dependencies retain their licenses.

The historical SFT regression suite still imports an absent `_compute_warmup_lr`. The older GRPO-specific evaluation/inference files still depend on a missing legacy `train` module; root evaluation and demo entry points use the newer SFT implementations. GRPO validation/resume options are not a complete evaluation or optimizer-resume workflow. These independent limitations are outside the synchronization fix.
