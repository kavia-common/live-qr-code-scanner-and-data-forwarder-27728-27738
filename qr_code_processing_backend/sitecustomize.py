import os
import sys

# Ensure the local 'src' directory is importable regardless of environment PYTHONPATH.
here = os.path.dirname(__file__)
src = os.path.join(here, 'src')
if src not in sys.path:
    sys.path.insert(0, src)
