"""Enable ``python -m eval_gate``."""

from __future__ import annotations

import sys

from eval_gate.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
