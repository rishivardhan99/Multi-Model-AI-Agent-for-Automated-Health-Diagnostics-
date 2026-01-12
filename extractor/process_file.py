# extractor/process_file.py
#!/usr/bin/env python3
"""
Lightweight single-file entry-point for your extractor.

Behavior:
 - Accepts a path to a single input file and invokes the in-repo batch runner's process_file(path)
   if available (keeps all core extractor logic unchanged).
 - If the batch runner module cannot be imported or doesn't expose process_file, falls back to
   calling the most likely batch script(s) as a subprocess with that single file path.
 - Returns non-zero exit on failure (suitable for Streamlit's run_subprocess checks).
"""
import argparse
import sys
import subprocess
from pathlib import Path
import importlib.util
import os
import traceback

ROOT = Path(__file__).resolve().parent.parent  # repo root (same assumption as your other scripts)
EX = ROOT / "extractor"

# candidate batch runner files to try importing or exec-ing
CANDIDATES = [
    EX / "run_batch_model1.py",
    EX / "process_samples.py",
    EX / "process_file.py",
    EX / "run_extractor.py",
    EX / "batch_runner.py",
    EX / "step1_ingest.py",
    ROOT / "run_batch_model1.py",
    ROOT / "run_extractor.py",
]

def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if not spec or not spec.loader:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None

def call_subprocess(script_path: Path, arg_path: Path):
    # run script (prefer python interpreter explicitly)
    cmd = [sys.executable, str(script_path), str(arg_path)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT))
    out, err = proc.communicate()
    return proc.returncode, out, err

def try_import_and_call(script_path: Path, file_path: Path) -> (int, str, str):
    name = f"ext_runner_{script_path.stem}"
    mod = import_module_from_path(name, script_path)
    if mod is None:
        return 2, "", f"Could not import {script_path}"
    # prefer a process_file function
    if hasattr(mod, "process_file") and callable(getattr(mod, "process_file")):
        try:
            # call and return success
            mod.process_file(file_path)
            return 0, f"Invoked {script_path.name}.process_file({file_path})", ""
        except Exception as e:
            tb = traceback.format_exc()
            return 3, "", f"Error while calling process_file: {e}\n{tb}"
    # try main() if present
    if hasattr(mod, "main") and callable(getattr(mod, "main")):
        try:
            # Some batch runners accept a path arg via global var or environment; easiest is to call main()
            # but we need to set argv
            saved_argv = sys.argv[:]
            try:
                sys.argv = [str(script_path), str(file_path)]
                mod.main()
                return 0, f"Invoked {script_path.name}.main() with arg {file_path}", ""
            finally:
                sys.argv = saved_argv
        except Exception as e:
            tb = traceback.format_exc()
            return 4, "", f"Error while calling main(): {e}\n{tb}"
    return 5, "", f"No process_file() or main() callable found in {script_path}"

def main():
    p = argparse.ArgumentParser(description="Process a single file using the extractor core (safe wrapper).")
    p.add_argument("input", help="Path to input file (pdf/image/json)")
    p.add_argument("--fallback-subprocess", action="store_true", help="Prefer subprocess fallback even if import works (debug)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(10)

    # 1) Try to import a known batch runner module and call its process_file(path)
    for cand in CANDIDATES:
        if not cand.exists():
            continue
        # skip current script to avoid recursion
        if cand.resolve() == Path(__file__).resolve():
            continue
        # try import & call unless user forced subprocess fallback
        if not args.fallback_subprocess:
            rc, out, err = try_import_and_call(cand, input_path)
            if rc == 0:
                print(out)
                sys.exit(0)
            else:
                # print debug info and continue to next candidate
                print(f"[DEBUG] Import/call candidate {cand.name} failed: rc={rc}, err={err}", file=sys.stderr)

        # 2) fallback: call as subprocess with the single-file arg
        rc2, sout, serr = call_subprocess(cand, input_path)
        if rc2 == 0:
            print(f"Subprocess {cand.name} succeeded.")
            if sout:
                print(sout)
            sys.exit(0)
        else:
            print(f"[DEBUG] Subprocess {cand.name} returned rc={rc2}. stdout:\n{sout}\nstderr:\n{serr}", file=sys.stderr)

    # if nothing worked
    print("No extractor entrypoint succeeded. Looked for:", file=sys.stderr)
    for c in CANDIDATES:
        print(" - " + str(c), file=sys.stderr)
    sys.exit(20)

if __name__ == "__main__":
    main()
