"""Make `import promptfidelity` resolve to src/promptfidelity without
requiring an editable install -- keeps `python -m pytest promptfidelity/tests/
-q` working from a bare checkout with only pytest installed."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
