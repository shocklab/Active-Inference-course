#!/usr/bin/env python3
"""check_all.py — the whole pipeline. Run this before committing anything.

    python3 build/check_all.py            # everything offline
    python3 build/check_all.py --full     # also DOIs (network) and notebooks (slow)
    python3 build/check_all.py --fast     # skip the prose census

Stages, in dependency order. Each exists because something specific went wrong:

  build        content/ -> docs/, substituting computed numbers
  site         unresolved [eq:] refs, unregistered widgets, dead links, tag
               imbalance, stray maths delimiters
  numbers      every derived number computed rather than typed  (four wrong
               values reached a draft of Week 1 by being worked out by hand)
  definitions  every symbol defined where it is first read  (a reader caught
               $o$ named "sensory states" before that distinction existed)
  outline      every week names a source, a derivation target, widgets, a link
               (one probe found a topic mislabelled and uncommitted)
  prose        AI-writing tells, per zone, per 1,000 words, on the built pages
  references   every DOI resolves and matches its stored metadata     [--full]
  notebooks    every notebook executes end to end                     [--full]

Exit code is the number of failed stages. Advisory stages (outline, prose) are
reported but never fail the run, because their output is a queue to read rather
than a verdict to obey.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

STAGES = [
    ("build",       ["build/build_site.py"],       "blocking", True),
    ("site",        ["build/check_site.py"],       "blocking", True),
    ("numbers",     ["build/check_numbers.py"],    "blocking", True),
    ("definitions", ["build/check_definitions.py"], "advisory", True),
    ("outline",     ["build/check_outline.py"],    "advisory", True),
    ("prose",       ["build/check_prose.py"],      "advisory", "not fast"),
    ("references",  ["build/check_references.py"], "blocking", "full"),
]


def notebooks():
    import glob
    nbs = sorted(glob.glob(os.path.join(ROOT, "notebooks", "*", "*.ipynb")))
    if not nbs:
        return 0, "no notebooks found"
    r = subprocess.run([PY, "-m", "jupyter", "nbconvert", "--to", "notebook",
                        "--execute", "--inplace",
                        "--ExecutePreprocessor.timeout=600"] + nbs,
                       capture_output=True, text=True, cwd=ROOT)
    return r.returncode, f"{len(nbs)} notebooks executed" if r.returncode == 0 else r.stderr[-500:]


def main():
    full = "--full" in sys.argv
    fast = "--fast" in sys.argv
    results, t0 = [], time.time()

    for name, cmd, severity, when in STAGES:
        if when == "full" and not full:
            results.append((name, "skip", "use --full", 0.0)); continue
        if when == "not fast" and fast:
            results.append((name, "skip", "--fast", 0.0)); continue

        # Flush before handing stdout to a child, or the parent's buffered
        # headers land after the child's output and every section is
        # misattributed. A pipeline that lies about which stage spoke is worse
        # than no pipeline.
        print(f"\n{'='*74}\n  {name.upper()}\n{'='*74}", flush=True)
        t = time.time()
        r = subprocess.run([PY] + cmd, cwd=ROOT)
        dt = time.time() - t
        ok = r.returncode == 0
        results.append((name, "pass" if ok else ("FAIL" if severity == "blocking" else "look"),
                        severity, dt))

    if full:
        print(f"\n{'='*74}\n  NOTEBOOKS\n{'='*74}", flush=True)
        t = time.time()
        rc, msg = notebooks()
        print(" ", msg)
        results.append(("notebooks", "pass" if rc == 0 else "FAIL", "blocking", time.time() - t))

    print(f"\n{'='*74}\n  SUMMARY   ({time.time()-t0:.1f}s)\n{'='*74}", flush=True)
    failed = 0
    for name, status, severity, dt in results:
        mark = {"pass": "  ok  ", "FAIL": " FAIL ", "look": " look ", "skip": " skip "}[status]
        print(f"{mark} {name:<14} {dt:>6.1f}s   {severity if status!='skip' else severity}")
        if status == "FAIL":
            failed += 1

    if failed:
        print(f"\n{failed} blocking stage(s) failed. Do not commit.")
    else:
        looks = sum(1 for _, s, _, _ in results if s == "look")
        print("\nAll blocking stages pass."
              + (f" {looks} advisory stage(s) have a queue to read." if looks else ""))
        if not full:
            print("Run with --full before publishing: DOIs and notebooks are not checked here.")
    return failed


if __name__ == "__main__":
    sys.exit(main())
