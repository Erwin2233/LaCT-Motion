"""Supervise an existing GRPO run and evaluate its completed final checkpoint."""

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("fid", "top1", "top2", "top3", "matching_score", "diversity",
           "motion_emb_cos", "semantic_cos")


def now():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # Linux process start ticks; protects against PID reuse.
    except (FileNotFoundError, ProcessLookupError):
        return None


def checkpoint_ready(directory):
    """Check model metadata and complete safetensors payloads without loading a GPU."""
    try:
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            json.loads((directory / name).read_text())
        index = directory / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = set(weight_map.values())
        else:
            weight_map = None
            shards = {"model.safetensors"}
        if not shards:
            raise ValueError("No model weight shards")
        tensors = {}
        for name in shards:
            if Path(name).name != name:
                raise ValueError("Unexpected shard path")
            path = directory / name
            with path.open("rb") as stream:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                if header_size > 100_000_000:
                    raise ValueError("Invalid safetensors header size")
                header = json.loads(stream.read(header_size))
            entries = {k: v for k, v in header.items() if k != "__metadata__"}
            if not entries:
                raise ValueError("Empty weight shard")
            end = max(v["data_offsets"][1] for v in entries.values())
            if 8 + header_size + end != path.stat().st_size:
                raise ValueError(f"Incomplete weight shard: {name}")
            tensors.update({key: name for key in entries})
        if weight_map is not None and any(tensors.get(k) != v for k, v in weight_map.items()):
            raise ValueError("Weight index does not match shard tensors")
        return True, f"{len(shards)} complete shards, {len(tensors)} tensors"
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        return False, str(error)


class TrainingLog:
    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.pending = ""
        self.last_progress = path.stat().st_mtime if path.exists() else time.time()
        self.complete = False
        self.nonfinite = False
        self.last_error = None
        self.progress = {}

    def refresh(self):
        if not self.path.exists():
            return
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            data = stream.read()
            self.offset = stream.tell()
        content = self.pending + data.decode(errors="replace").replace("\r", "\n")
        lines = content.split("\n")
        self.pending = lines.pop()
        for line in lines:
            iteration = re.search(r"\[iter (\d+)/(\d+)\]", line)
            step = re.search(r"\[Epoch (\d+)/(\d+)\] Step (\d+)/(\d+)", line)
            if iteration:
                self.progress.update(iteration=int(iteration[1]), iterations_per_epoch=int(iteration[2]))
            if step:
                self.progress.update(epoch=int(step[1]), epochs=int(step[2]),
                                     step=int(step[3]), total_steps=int(step[4]))
            if iteration or step:
                self.last_progress = self.path.stat().st_mtime
                self.progress["last_metric_line"] = line.strip()
                if re.search(r"\b(?:loss|reward|kl|grad_norm)=[+-]?(?:nan|inf)\b", line, re.I):
                    self.nonfinite = True
            if "Traceback (most recent call last):" in line or "Error:" in line:
                self.last_error = line.strip()
            if "Training complete. Final model saved to:" in line:
                self.complete = True


def evaluation_environment(gpu):
    environment = os.environ.copy()
    for key in list(environment):
        if key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                   "GROUP_RANK", "ROLE_RANK"} or key.startswith("TORCHELASTIC_"):
            environment.pop(key)
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp",
                       PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       WANDB_MODE="disabled", WANDB_DISABLED="true",
                       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return environment


def worker(plan_path):
    plan = json.loads(plan_path.read_text())
    directory = ROOT / plan["monitor_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "monitor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("A monitor already owns this run.", flush=True)
        return 1
    write_json(directory / "process.json", {"pid": os.getpid(),
               "start_ticks": process_identity(os.getpid()), "started_at": now()})
    status = {"run_id": plan["run_id"], "monitor_pid": os.getpid(),
              "train_pid": plan["train_pid"], "eval_gpu": plan["eval_gpu"],
              "shuffle": True, "evaluation_output": plan["eval_output"],
              "started_at": now()}
    prior_state = None

    def save(state, **details):
        nonlocal prior_state
        status.update(state=state, updated_at=now(), **details)
        write_json(directory / "status.json", status)
        if state != prior_state:
            event = {"time": now(), "state": state, **details}
            with (directory / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)
            prior_state = state

    try:
        reader = TrainingLog(ROOT / plan["train_log"])
        while True:
            reader.refresh()
            alive = [record["pid"] for record in plan["processes"]
                     if process_identity(record["pid"]) == record["start_ticks"]]
            stalled = time.time() - reader.last_progress > plan["stall_seconds"]
            state = ("training_nonfinite" if reader.nonfinite else
                     "training_stalled" if stalled else "training")
            save(state, training_progress=reader.progress, live_training_pids=alive,
                 seconds_since_progress=round(time.time() - reader.last_progress),
                 nonfinite_detected=reader.nonfinite)
            if not alive:
                break
            time.sleep(plan["poll_seconds"])

        # A process can finish between reading its log and checking its PID.
        # Consume its final writes after all tracked processes have exited.
        reader.refresh()
        # Never evaluate a partially written checkpoint or a failed training run.
        if not reader.complete or reader.nonfinite:
            reason = ("Non-finite training metrics were detected." if reader.nonfinite else
                      reader.last_error or "Training exited without a final-save completion marker.")
            save("training_failed", error=reason)
            return 1
        ready, description = checkpoint_ready(ROOT / plan["checkpoint"])
        if not ready:
            save("checkpoint_incomplete", error=description)
            return 1

        result_path = ROOT / plan["eval_output"]
        if result_path.exists():
            save("evaluation_output_exists", error="Existing results are preserved; choose another output path.")
            return 1
        result_path.parent.mkdir(parents=True, exist_ok=True)
        eval_log = ROOT / plan["eval_log"]
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        with eval_log.open("ab") as stream:
            process = subprocess.Popen(
                [sys.executable, "-u", *plan["eval_args"]], cwd=ROOT,
                env=evaluation_environment(plan["eval_gpu"]), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            )
        save("evaluating", eval_pid=process.pid, checkpoint_validation=description,
             evaluation_started_at=now(), eval_log=plan["eval_log"])
        while process.poll() is None:
            time.sleep(plan["poll_seconds"])
            save("evaluating")
        if process.returncode != 0:
            save("evaluation_failed", error=f"Evaluator exited with code {process.returncode}")
            return 1
        results = json.loads(result_path.read_text())
        metrics = {key: results.get(key) for key in METRICS}
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in metrics.values()):
            save("evaluation_failed", error="Missing or non-finite evaluation metrics")
            return 1
        save("completed", metrics=metrics, completed_at=now(), wandb_upload="pending")
        if plan.get("wandb_url"):
            try:
                import wandb
                path = plan["wandb_url"].split("wandb.ai/", 1)[1].replace("/runs/", "/")
                run = wandb.Api(timeout=30).run(path)
                run.summary.update({**{f"evaluation/{k}": v for k, v in metrics.items()},
                                    "evaluation/status": "completed",
                                    "evaluation/result_path": plan["eval_output"],
                                    "evaluation/repeat": plan["repeat"],
                                    "evaluation/checkpoint": plan["checkpoint"]})
                save("completed", wandb_upload="completed")
            except Exception as error:
                save("completed", wandb_upload="failed", wandb_error=str(error))
                print(f"Evaluation saved locally; W&B upload failed: {error}", flush=True)
        else:
            save("completed", wandb_upload="disabled")
        return 0
    except Exception as error:
        save("monitor_failed", error=str(error))
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", help="Defaults to outputs/grpo/active_run.txt")
    parser.add_argument("--eval-gpu", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--stall-minutes", type=float, default=30)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--worker", metavar="PLAN", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(ROOT / args.worker)
    if min(args.repeat, args.poll_seconds, args.stall_minutes) <= 0:
        parser.error("Repeat and monitoring intervals must be positive")
    run_relative = args.run_dir or (ROOT / "outputs/grpo/active_run.txt").read_text().strip()
    run_directory = (ROOT / run_relative).resolve()
    run_relative = run_directory.relative_to(ROOT).as_posix()
    metadata = json.loads((run_directory / "launch.json").read_text())
    if args.eval_gpu not in metadata["gpu_ids"]:
        parser.error("Evaluation must use one of this run's authorized GPU indices")
    directory = run_directory / "monitor"
    directory.mkdir(exist_ok=True)
    existing = directory / "process.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        if previous.get("start_ticks") and process_identity(previous["pid"]) == previous["start_ticks"]:
            print(f"Monitor already running: PID {previous['pid']}")
            return 0
    status_path = directory / "status.json"
    if status_path.exists() and json.loads(status_path.read_text()).get("state") == "completed":
        print("Evaluation is already complete; results are preserved.")
        return 0
    training = yaml.safe_load((ROOT / metadata["config"]).read_text())
    evaluation = {**training, "cot": False, "no_cot": False, "do_sample": True,
                  "repetition_penalty": 1.0, "use_wandb": False}
    eval_config = Path("options/local") / f"{metadata['run_id']}_eval.yaml"
    if (ROOT / eval_config).exists():
        if yaml.safe_load((ROOT / eval_config).read_text()) != evaluation:
            parser.error("Existing evaluation configuration differs; preserve it and resolve the mismatch")
    else:
        (ROOT / eval_config).write_text(yaml.safe_dump(evaluation, sort_keys=False))

    train_pid = metadata["pid"]
    processes = []
    identity = process_identity(train_pid)
    if identity:
        command = Path(f"/proc/{train_pid}/cmdline").read_bytes().split(b"\0")
        if metadata["config"].encode() not in command or b"torch.distributed.run" not in command:
            parser.error("The recorded training PID belongs to another command")
        children = Path(f"/proc/{train_pid}/task/{train_pid}/children").read_text().split()
        for pid in [train_pid, *map(int, children)]:
            start_ticks = process_identity(pid)
            if start_ticks:
                processes.append({"pid": pid, "start_ticks": start_ticks})
    checkpoint = (Path(metadata["checkpoints"]) / "final").as_posix()
    output = (Path("results/grpo") / metadata["run_id"] / "final.json").as_posix()
    eval_log = (Path(run_relative) / "evaluation/eval.log").as_posix()
    eval_args = ["eval_t2m.py", str(eval_config), "--checkpoint", checkpoint,
                 "--batch-size", "32", "--repeat", str(args.repeat),
                 "--device", "cuda:0", "--seed", str(training.get("seed", 42)),
                 "--flops-mode", "off", "--output", output]
    plan = {"run_id": metadata["run_id"], "created_at": now(), "train_pid": train_pid,
            "processes": processes, "train_log": metadata["log"],
            "monitor_dir": directory.relative_to(ROOT).as_posix(),
            "checkpoint": checkpoint, "eval_config": str(eval_config),
            "eval_gpu": args.eval_gpu, "repeat": args.repeat, "eval_args": eval_args,
            "eval_output": output, "eval_log": eval_log,
            "poll_seconds": args.poll_seconds, "stall_seconds": args.stall_minutes * 60,
            "wandb_url": metadata.get("wandb_url"), "shuffle": True,
            "shuffle_source": "DistributedSampler(shuffle=True); set_epoch(epoch) each epoch",
            "multimodality_enabled": False}
    plan_path = directory / "plan.json"
    write_json(plan_path, plan)
    if not args.background:
        return worker(plan_path)
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       TMPDIR="/tmp", TMP="/tmp", TEMP="/tmp")
    with (directory / "monitor.log").open("ab") as stream:
        process = subprocess.Popen(
            ["nohup", sys.executable, "-u", str(Path(__file__).resolve()),
             "--worker", plan_path.relative_to(ROOT).as_posix()],
            cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    (directory / "monitor.pid").write_text(str(process.pid) + "\n")
    print(json.dumps({"monitor_pid": process.pid, "run_id": metadata["run_id"],
                      "status": (directory / "status.json").relative_to(ROOT).as_posix(),
                      "final_evaluation": output, "eval_gpu": args.eval_gpu}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
