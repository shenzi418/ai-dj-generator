import os
import sys

# Keep the test suite light: no Demucs / torch model downloads.
os.environ.setdefault("AI_DJ_ENABLE_DEMUCS_ANALYSIS", "0")
os.environ.setdefault("AI_DJ_ENABLE_DEMUCS_TRANSITION", "0")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
