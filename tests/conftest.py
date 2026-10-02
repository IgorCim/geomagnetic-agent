"""Shared setup for the Phase 1 test package.

The repository keeps its modules at the top level (``geomag_math.py``,
``geomag_coords.py``, ``agent_core.py``) rather than installing them as a
package, so pytest's rootdir insertion does not put the repository root on
``sys.path`` for tests living in ``tests/``. Adding it here keeps these tests
importable without requiring an editable install, which the Colab notebook does
not perform.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))