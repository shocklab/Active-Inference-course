#!/usr/bin/env python3
"""check_outline.py — is each week planned, or merely named?

    python3 build/check_outline.py

Background: on 2026-08-26 a single probe ("where is variational message passing?")
found a topic mislabelled, uncommitted and uncross-linked. Measuring all twelve
weeks showed a clean separation: every week naming zero things to *derive* had a
gap; every week naming one or more had none. See notes/outline-audit-2026-08-26.md.

So this checks, per planned week:
  source   a source we have actually read is named
  derive   at least one thing to be DERIVED, not merely covered
  widget   concrete widget specs
  xlink    at least one pointer to another week

THIS IS A PROXY, and it has now failed in both directions:

  false positive  wiring a citation into a week satisfied `source` without anyone
                  having read the paper. See the reading-status table in OUTLINE.md.
  false negative  Week 7 was flagged as missing `derive` while containing the best
                  derivation in the outline, because the bullet opened "The week's
                  main result" rather than the word "Derive". Matching vocabulary
                  is not reading substance.

Use it to catch a week nobody has thought about. Do not use it to conclude that a
week is ready. The real check is reading the source.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTLINE = os.path.join(ROOT, "OUTLINE.md")

SOURCE = re.compile(r"main\.tex|notebook|Sources?:|diagnosis|Jonathan|Smith|Parr|Bruineberg")
DERIVE = re.compile(r"\*\*Derive|\bderive\b|\bderived\b|\bderivation\b|prove[dn]?\b|proof|"
                    r"in full|line by line", re.I)
WIDGET = re.compile(r"Widgets?:")
XLINK = re.compile(r"→ (?:back|forward)|Week ?\d+|Weeks ?\d+")


def main():
    txt = open(OUTLINE, encoding="utf-8").read()
    blocks = re.split(r"^### Week ", txt, flags=re.M)[1:]
    rows, incomplete = [], []

    for b in blocks:
        head = b.split("\n")[0]
        num = int(head.split(" ")[0])
        built = "built" in head.lower()
        checks = {
            "source": bool(SOURCE.search(b)),
            "derive": bool(DERIVE.search(b)),
            "widget": bool(WIDGET.search(b)),
            "xlink":  bool(XLINK.search(b)),
        }
        # A built week's lessons are the commitment; it is exempt from `derive`
        # and `xlink` here because check_site.py verifies the real pages instead.
        required = ["source", "widget"] if built else list(checks)
        missing = [k for k in required if not checks[k]]
        rows.append((num, built, checks, missing))
        if missing:
            incomplete.append((num, missing))

    print(f"{'wk':<4}{'state':<9}{'source':>8}{'derive':>8}{'widget':>8}{'xlink':>7}   missing")
    print("-" * 72)
    for num, built, c, missing in rows:
        tick = lambda k: "  yes" if c[k] else "   NO"
        state = "built" if built else "planned"
        print(f"{num:<4}{state:<9}{tick('source'):>8}{tick('derive'):>8}"
              f"{tick('widget'):>8}{tick('xlink'):>7}   {', '.join(missing) or '-'}")

    print()
    if incomplete:
        print(f"{len(incomplete)} of {len(rows)} weeks are named but not yet planned:")
        for num, missing in incomplete:
            print(f"  Week {num:>2}: missing {', '.join(missing)}")
        print("\nA week with no derivation target has not been thought about, only listed.")
        print("Ground it by reading the source, not by adding words that match the regex.")
    else:
        print(f"all {len(rows)} weeks carry a source, a derivation target, "
              f"widget specs and a cross-link")
    return 0 if not incomplete else 2      # 2 = advisory, not a build failure


if __name__ == "__main__":
    sys.exit(main())
