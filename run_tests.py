"""Run the copied SFT regression tests without loading model checkpoints."""
import unittest
from _bootstrap import ROOT, activate

if __name__ == "__main__":
    activate("sft")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests" / "sft"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
