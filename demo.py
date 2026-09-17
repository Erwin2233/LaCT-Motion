"""Run text-to-motion inference using the supported skeleton output mode."""
import sys
from _bootstrap import run

if __name__ == "__main__":
    modes = []
    for index, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith("--render-mode="):
            modes.append(arg.split("=", 1)[1])
        elif arg == "--render-mode" and index + 1 < len(sys.argv):
            modes.append(sys.argv[index + 1])
    if "smpl" in modes:
        raise SystemExit("The standalone SMPL renderer was removed. Use --render-mode skeleton.")
    if not any(arg == "--render-mode" or arg.startswith("--render-mode=") for arg in sys.argv[1:]):
        sys.argv.extend(["--render-mode", "skeleton"])
    run("sft", "demo/sft/inference.py")
