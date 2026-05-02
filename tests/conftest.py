from pathlib import Path
import sys


# Ensure tests can import top-level project modules in CI.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
