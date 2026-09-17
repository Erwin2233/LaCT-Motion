"""Resolve original flat imports after copying files into functional directories.

Each process uses one stage so the SFT and GRPO implementations keep their own
module identities. External project dependencies are bundled under third_party.
"""

import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
_active_stage = None


def module_paths(stage):
    """Return the preserved modules used by a stage, without importing them."""
    if stage not in {"sft", "grpo"}:
        raise ValueError("stage must be 'sft' or 'grpo'")
    paths = {}
    for directory in (
        "models", "dataset", "utils", "prepare", "training", "rewards",
        "evaluation", "demo", "debug",
    ):
        for path in sorted((ROOT / directory / stage).glob("*.py")):
            if path.stem in paths:
                raise RuntimeError(f"Duplicate module for {stage}: {path.stem}")
            paths[path.stem] = path
    paths["build_data"] = ROOT / "prepare" / "build_data.py"
    paths["eval_infra"] = ROOT / "evaluation" / "common" / "eval_infra" / "__init__.py"
    paths["eval_infra.dataset_TM_eval"] = (
        ROOT / "evaluation" / stage / "eval_infra" / "dataset_TM_eval.py"
    )
    return paths


class _StageImports(importlib.abc.MetaPathFinder):
    def __init__(self, paths):
        self.paths = paths

    def find_spec(self, fullname, path=None, target=None):
        source = self.paths.get(fullname)
        if source is None:
            return None
        return importlib.util.spec_from_file_location(fullname, source)


def activate(stage):
    """Select the stage-specific modules stored in this checkout."""
    global _active_stage
    if _active_stage is not None:
        if _active_stage != stage:
            raise RuntimeError("Run SFT and GRPO commands in separate processes.")
        return
    paths = module_paths(stage)
    conflicts = sorted(name for name in paths if name in sys.modules)
    if conflicts:
        raise RuntimeError(
            "Activate LaCT-Motion before importing stage modules: "
            + ", ".join(conflicts)
        )
    sys.meta_path.insert(0, _StageImports(paths))
    _active_stage = stage


def run(stage, relative_script):
    """Execute a copied entry point with its original command-line interface."""
    activate(stage)
    os.chdir(ROOT)
    script = ROOT / relative_script
    runpy.run_path(str(script), run_name="__main__")
