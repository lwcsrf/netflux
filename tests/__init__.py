import pathlib
import sys

# Ensure tests can import the main package modules even when
# running directly without editable installation
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
