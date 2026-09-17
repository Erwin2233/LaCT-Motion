"""Run the SFT project's T2M evaluator, including multimodality support."""
from _bootstrap import run

if __name__ == "__main__":
    run("sft", "evaluation/sft/evaluate.py")
