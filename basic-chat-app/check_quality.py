"""Fail if protected app logic or frozen evaluation inputs change.

This is a source-integrity guard, NOT an accuracy measurement. main() is excluded
so deployment host/port configuration can change without altering retrieval.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "quality-baseline.json"


def fingerprints():
    tree = ast.parse((ROOT / "app.py").read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name != "main"]
    result = {"app_logic_except_server_startup": hashlib.sha256(ast.dump(tree).encode()).hexdigest()}
    for filename in ("eval_questions.jsonl", "embedding-eval/frozen-corpus.json", "embedding-eval/new-model.index.npy", "embedding-eval/calibration.json"):
        result[filename] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    return result


def check():
    expected = json.loads(MANIFEST.read_text())["sha256"]
    actual = fingerprints()
    changed = [key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)]
    if changed:
        raise SystemExit("Protected logic/data changed; investigate and evaluate before updating the baseline: " + ", ".join(changed))
    print("PASS: protected application logic and frozen evaluation inputs match the baseline.")
    print("This checks source integrity; it does not measure answer accuracy or external model behavior.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="Explicitly replace the source baseline after review.")
    args = parser.parse_args()
    if args.capture:
        MANIFEST.write_text(json.dumps({"scope": "Source integrity, not measured accuracy", "sha256": fingerprints()}, indent=2) + "\n")
    else:
        check()
