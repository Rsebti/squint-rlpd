"""Pure-syntax check for train_rlpd.py and rlpd_utils.py.

Runs on any Python install — does NOT import maniskill, torch, tensordict, etc.
Use this on the laptop (no NVIDIA GPU) to catch syntax / typo errors before
pushing to the 5090 machine where the real run happens.

Usage:
    python parse_check_rlpd.py
"""

from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["train_rlpd.py", "rlpd_utils.py"]


def main() -> int:
    fail = 0
    for fname in FILES:
        path = os.path.join(HERE, fname)
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError as exc:
            print(f"[FAIL] {fname}: SyntaxError at line {exc.lineno}: {exc.msg}")
            fail += 1
            continue

        # Quick structural checks specific to train_rlpd.
        if fname == "train_rlpd.py":
            class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
            for required in ("Args", "CNNEncoder", "Projection", "Actor", "Critic"):
                if required not in class_names:
                    print(f"[FAIL] {fname}: missing required class {required!r}")
                    fail += 1
            # Verify the Critic no longer takes num_atoms / v_min / v_max (C51 stripped).
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "Critic":
                    init = next((b for b in node.body
                                 if isinstance(b, ast.FunctionDef) and b.name == "__init__"), None)
                    if init is not None:
                        params = {a.arg for a in init.args.args}
                        bad = params & {"num_atoms", "v_min", "v_max"}
                        if bad:
                            print(f"[FAIL] {fname}: Critic.__init__ still has C51 params {bad!r}")
                            fail += 1
        print(f"[ OK ] {fname}")
    if fail:
        print(f"\n{fail} file(s) failed parse check.")
        return 1
    print("\nAll files parse cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
