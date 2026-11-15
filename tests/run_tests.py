"""Run every test_* function in tests/test_*.py without needing pytest.

Usage (from the repository root):  python tests/run_tests.py
If pytest is installed, `python -m pytest tests` runs the same tests.
"""
import importlib.util
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main() -> int:
    tests_dir = os.path.join(ROOT, "tests")
    failed = passed = 0
    for fname in sorted(os.listdir(tests_dir)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        spec = importlib.util.spec_from_file_location(fname[:-3], os.path.join(tests_dir, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {fname}::{name}")
            except Exception:
                failed += 1
                print(f"FAIL {fname}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
