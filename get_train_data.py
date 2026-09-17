"""Rebuild training JSON separately from the included dataset files."""
import sys
from _bootstrap import ROOT, run

if __name__ == "__main__":
    if not any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in sys.argv[1:]):
        sys.argv.extend(["--output-dir", str(ROOT / "data" / "rebuilt")])
    run("sft", "prepare/build_data.py")
