"""Launch a detached GRPO run with its own configuration, log, and checkpoints."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="options/grpo/t2m_grpo.yaml")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable online W&B logging (default: use the configuration)")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--temp-dir", default="/tmp",
                        help="Local filesystem supporting Unix sockets for dataset workers")
    args = parser.parse_args()
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        parser.error("--gpus must contain distinct, nonnegative physical GPU indices")
    root = Path(__file__).resolve().parents[2]
    temp_dir = Path(args.temp_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="lact-socket-", dir=temp_dir) as check_dir:
        with socket.socket(socket.AF_UNIX) as check_socket:
            check_socket.bind(str(Path(check_dir) / "check.sock"))

    now = datetime.now().astimezone()
    run_id = "grpo_gpu" + "_".join(map(str, gpu_ids)) + "_" + now.strftime("%Y%m%d_%H%M%S")
    config = yaml.safe_load((root / args.config).read_text())
    use_wandb = config.get("use_wandb", True) if args.wandb is None else args.wandb
    config.update(save_path=f"./checkpoints/grpo/{run_id}",
                  use_wandb=use_wandb, show_progress_bar=False)
    if use_wandb:
        config.update(wandb_mode="online", wandb_run_name=run_id)
        if args.wandb_project:
            config["wandb_project"] = args.wandb_project
        if args.wandb_entity:
            config["wandb_entity"] = args.wandb_entity
    config_rel = Path("options/local") / f"{run_id}.yaml"
    (root / config_rel).parent.mkdir(parents=True, exist_ok=True)
    with (root / config_rel).open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    run_rel = Path("outputs/grpo") / run_id
    run_dir = root / run_rel
    run_dir.mkdir(parents=True, exist_ok=False)
    log_rel = run_rel / "train.log"
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, gpu_ids)),
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
        NCCL_DEBUG="WARN",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        TMPDIR=str(temp_dir), TMP=str(temp_dir), TEMP=str(temp_dir),
    )
    environment["WANDB_MODE"] = "online" if use_wandb else "disabled"
    environment["WANDB_DIR"] = str(run_dir)
    if use_wandb:
        environment.pop("WANDB_DISABLED", None)
        environment.pop("WANDB_RUN_ID", None)
        environment.pop("WANDB_RESUME", None)
    else:
        environment["WANDB_DISABLED"] = "true"
    arguments = ["-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                 f"--nproc_per_node={len(gpu_ids)}", "train_grpo.py",
                 "--config", str(config_rel)]
    with (root / log_rel).open("xb") as log:
        process = subprocess.Popen(
            ["nohup", sys.executable, *arguments], cwd=root, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
    metadata = dict(run_id=run_id, pid=process.pid, process_group=process.pid,
                    session=process.pid, gpu_ids=gpu_ids, config=str(config_rel),
                    log=str(log_rel), checkpoints=config["save_path"],
                    command=["python", *arguments], detached=True,
                    wandb_enabled=use_wandb, created_at=now.isoformat())
    (run_dir / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "train.pid").write_text(str(process.pid) + "\n")
    (root / "outputs/grpo/active_run.txt").write_text(str(run_rel) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Launch submitted; inspect train.log to confirm training progress.")


if __name__ == "__main__":
    main()
