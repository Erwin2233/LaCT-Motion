"""Exercise the completion-to-evaluation workflow without models or GPUs."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/grpo/monitor_training.py"
SPEC = importlib.util.spec_from_file_location("lact_monitor", SCRIPT)
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


class MonitorTests(unittest.TestCase):
    def prepare(self, root, log, truncate=False):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            (checkpoint / name).write_text("{}")
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        payload = struct.pack("<Q", len(header)) + header + b"\0" * 4
        (checkpoint / "model.safetensors").write_bytes(payload[:-1] if truncate else payload)
        (root / "training.log").write_text(log)
        (root / "fake_eval.py").write_text(
            "import json, os, pathlib, sys\n"
            f"result = {{key: 0.5 for key in {monitor.METRICS!r}}}\n"
            "result['visible_gpu'] = os.environ['CUDA_VISIBLE_DEVICES']\n"
            "result['distributed_rank'] = os.environ.get('RANK')\n"
            "pathlib.Path(sys.argv[1]).write_text(json.dumps(result))\n"
        )
        plan = {"run_id": "fixture", "train_pid": 0, "processes": [],
                "monitor_dir": "monitor", "train_log": "training.log", "checkpoint": "checkpoint",
                "eval_gpu": 4, "eval_output": "results/final.json", "eval_log": "eval.log",
                "eval_args": ["fake_eval.py", "results/final.json"], "poll_seconds": 0.02,
                "stall_seconds": 1800, "wandb_url": None, "repeat": 1}
        plan_path = root / "plan.json"
        plan_path.write_text(json.dumps(plan))
        return plan_path

    def test_completed_run_evaluates_once_with_isolated_gpu_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.prepare(root, "Training complete. Final model saved to: checkpoint\n")
            with patch.object(monitor, "ROOT", root), patch.dict("os.environ", {"RANK": "77"}):
                self.assertEqual(monitor.worker(plan), 0)
                output = root / "results/final.json"
                original = output.read_bytes()
                data = json.loads(original)
                self.assertEqual(data["visible_gpu"], "4")
                self.assertIsNone(data["distributed_rank"])
                status = json.loads((root / "monitor/status.json").read_text())
                self.assertEqual(status["state"], "completed")
                self.assertEqual(monitor.worker(plan), 1)
                self.assertEqual(output.read_bytes(), original)

    def test_failure_nonfinite_and_incomplete_weights_never_evaluate(self):
        complete = "Training complete. Final model saved to: checkpoint\n"
        cases = [("RuntimeError: training failed\n", False, "training_failed"),
                 (complete, True, "checkpoint_incomplete"),
                 ("[iter 1/2] loss=nan | reward=1.0 | kl=0.0\n" + complete,
                  False, "training_failed")]
        for log, truncated, expected in cases:
            with self.subTest(expected=expected, truncated=truncated), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = self.prepare(root, log, truncate=truncated)
                with patch.object(monitor, "ROOT", root):
                    self.assertEqual(monitor.worker(plan), 1)
                self.assertFalse((root / "results/final.json").exists())
                status = json.loads((root / "monitor/status.json").read_text())
                self.assertEqual(status["state"], expected)

    def test_completion_written_during_exit_check_is_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = self.prepare(root, "[iter 1/1] loss=0.1 | reward=2.0 | kl=0.0\n")
            plan = json.loads(plan_path.read_text())
            plan["processes"] = [{"pid": 987654321, "start_ticks": "old"}]
            plan_path.write_text(json.dumps(plan))

            def finish_during_pid_check(pid):
                if pid == 987654321:
                    with (root / "training.log").open("a") as stream:
                        stream.write("Training complete. Final model saved to: checkpoint\n")
                    return None
                return "monitor-start"

            with patch.object(monitor, "ROOT", root), patch.object(monitor, "process_identity", finish_during_pid_check):
                self.assertEqual(monitor.worker(plan_path), 0)
            self.assertTrue((root / "results/final.json").exists())


if __name__ == "__main__":
    unittest.main()
