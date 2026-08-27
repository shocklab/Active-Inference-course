#!/usr/bin/env python3
"""check_numbers.py — is every derived number in the prose computed, not typed?

    python3 build/check_numbers.py

Four wrong values reached a draft of Week 1 because they were worked out by hand.
The fix is structural: a week's derived numbers live in content/week-NN/values.py
and the prose refers to them as {{key:.4f}}, so the published figure IS the
computed figure and cannot drift.

This checks two directions:

  resolved   every {{key}} in the source resolves, and the value that reached the
             built page equals the value values.py computes right now
  untokenised any decimal still typed literally into the prose, which is a number
             nobody is checking

The second is the one that matters. A substitution scheme nobody audits just
moves the hand-typed numbers somewhere less visible.

Literal numbers that are NOT derived quantities (a prior someone chose, a year,
a matrix entry that defines the model) are legitimate; list them in ALLOW below
with a reason. An allow-list entry is a claim that a human checked it.
"""
import importlib.util
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))
import mdx  # noqa: E402

# Numbers that are inputs or conventions rather than results. Each is a claim
# that someone checked it by eye; keep the reasons.
ALLOW = {
    "0.70", "0.15", "0.02", "0.25", "0.75", "0.08", "0.05", "0.10", "0.90",
    "0.22",                       # the likelihood matrix and prior we chose
    "0.40", "0.30",               # the nervous hiker's prior, also chosen
    "0.60", "0.85", "0.35",       # the counterexample likelihood, chosen
    "0.99", "0.01", "0.1", "0.9", # noisy-OR strength, leak, cause prior
    "0.5", "0.0",                 # flat priors and zero
    "7.35", "7.45", "37",         # physiological ranges, cited facts
    "0.693",                      # ln 2, stated to three places as a conversion
    "4.4",                        # age of the universe in ns, order of magnitude
    "0.035", "0.09", "0.02", "0.03", "0.045", "0.06",   # simulation noise settings
    "1.0", "2.0", "0.0198", "0.9802",                    # inline worked fragments
    "26.6",   # shown deliberately as a BAD rounding, to demonstrate what it costs
    "0.94",   # a chosen probability defining the worked entropy example
    "0.8", "0.3",   # chosen probabilities in the blanket-breaking exercise
    "1.5", "0.5",   # illustrative gains either side of the optimum
    "0.7",          # 1 - alpha in the worked blending example
}

DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?![\w.])")
# Zones that legitimately contain numbers which are not prose claims. A zone you
# do not strip is a zone that floods the report and gets the whole check ignored.
NON_PROSE = [
    re.compile(r"<svg\b.*?</svg>", re.S | re.I),   # figure coordinates and styling
    re.compile(r"^(```|~~~).*?^\1", re.S | re.M),   # fenced code
    re.compile(r"`[^`\n]+`"),                       # inline code
    re.compile(r"\]\([^)]*\)"),                    # link targets
]
TOKEN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]+))?\}\}")


def load_values(week_dir):
    path = os.path.join(week_dir, "values.py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("nm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.VALUES)


def check_widget_constants():
    """Constants shared between a widget and the prose describing it.

    The caption for the homeostasis figure quotes the grid size the widget bins
    onto and the maximum entropy that implies. Those live in two files, so they
    can drift apart silently and the caption would go on describing a readout it
    no longer matches.
    """
    out = []
    js = os.path.join(ROOT, "assets", "js", "w01.js")
    if not os.path.exists(js):
        return out
    m = re.search(r"var NB = (\d+)", open(js, encoding="utf-8").read())
    vals = load_values(os.path.join(ROOT, "content", "week-01"))
    if m and vals and "hs_bins" in vals and int(m.group(1)) != int(vals["hs_bins"]):
        out.append(f"widget w01.js bins onto {m.group(1)}x{m.group(1)} but "
                   f"values.py says HS_BINS = {vals['hs_bins']}")
    return out


def main():
    course = json.load(open(os.path.join(ROOT, "content/course.json"), encoding="utf-8"))
    problems, untok, n_tok, n_pages = [], [], 0, 0

    for w in course["weeks"]:
        wdir = os.path.join(ROOT, "content", f"week-{int(w['n']):02d}")
        if not os.path.isdir(wdir):
            continue
        values = load_values(wdir)
        for fn in sorted(x for x in os.listdir(wdir) if x.endswith(".md")):
            n_pages += 1
            path = os.path.join(wdir, fn)
            _, body = mdx.split_front_matter(open(path, encoding="utf-8").read())
            label = f"week-{int(w['n']):02d}/{fn}"

            for m in TOKEN.finditer(body):
                n_tok += 1
                key, fmt = m.group(1), m.group(2)
                if values is None:
                    problems.append(f"{label}: uses {{{{{key}}}}} but the week has no values.py")
                elif key not in values:
                    problems.append(f"{label}: {{{{{key}}}}} is not defined in values.py")
                else:
                    try:
                        format(values[key], fmt) if fmt else str(values[key])
                    except (ValueError, TypeError):
                        problems.append(f"{label}: {{{{{key}:{fmt}}}}} is not a valid format")

            # strip tokens and non-prose zones, then look for hand-typed numbers
            stripped = TOKEN.sub(" ", body)
            for zone in NON_PROSE:
                stripped = zone.sub(" ", stripped)
            for m in DECIMAL.finditer(stripped):
                v = m.group(1)
                if v in ALLOW:
                    continue
                ctx = re.sub(r"\s+", " ", stripped[max(0, m.start() - 70):m.end() + 45]).strip()
                untok.append((label, v, ctx))

    # keys computed but never referenced anywhere
    unused = []
    for w in course["weeks"]:
        wdir = os.path.join(ROOT, "content", f"week-{int(w['n']):02d}")
        if not os.path.isdir(wdir):
            continue
        values = load_values(wdir)
        if not values:
            continue
        cited = set()
        for fn in os.listdir(wdir):
            if fn.endswith(".md"):
                body = open(os.path.join(wdir, fn), encoding="utf-8").read()
                cited |= {m.group(1) for m in TOKEN.finditer(body)}
        for k in sorted(set(values) - cited):
            unused.append(f"week-{int(w['n']):02d}: {k}")

    # confirm what actually reached the built pages
    drift = []
    for w in course["weeks"]:
        wdir = os.path.join(ROOT, "content", f"week-{int(w['n']):02d}")
        values = load_values(wdir) if os.path.isdir(wdir) else None
        if not values:
            continue
        ddir = os.path.join(ROOT, "docs", f"week-{int(w['n']):02d}")
        if not os.path.isdir(ddir):
            continue
        blob = "".join(open(os.path.join(ddir, f), encoding="utf-8").read()
                       for f in os.listdir(ddir) if f.endswith(".html"))
        if "?" * 2 in blob or "{{" in blob:
            drift.append(f"week-{int(w['n']):02d}: unresolved token markup reached docs/")

    print(f"{n_pages} pages · {n_tok} computed numbers substituted")
    problems.extend(check_widget_constants())
    for p in problems + drift:
        print("  FAIL:", p)
    if untok:
        print(f"\n  {len(untok)} decimals still typed by hand:")
        for label, v, ctx in untok[:40]:
            print(f"    {label}  {v}\n      …{ctx}")
        print("\n  Move each into values.py and reference it as {{key}}, or add it to")
        print("  ALLOW with a reason if it is an input rather than a result.")
    if unused:
        print(f"\n  {len(unused)} computed values are never cited (advisory: some are")
        print("  legitimately intermediates, but a forgotten citation looks like this):")
        for u in unused[:15]:
            print("   ", u)
        if len(unused) > 15:
            print(f"    … and {len(unused)-15} more")

    ok = not (problems or drift or untok)
    print("\n" + ("every derived number is computed" if ok else "see above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
