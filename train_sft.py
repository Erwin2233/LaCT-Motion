"""Run the preserved Coconut curriculum SFT trainer."""
from _bootstrap import run

if __name__ == "__main__":
    run("sft", "training/sft/train.py")
