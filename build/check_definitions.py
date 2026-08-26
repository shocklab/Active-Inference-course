#!/usr/bin/env python3
"""check_definitions.py — is every symbol and term defined where it is first used?

    python3 build/check_definitions.py

Prompted by a reader's catch: Lesson 1 wrote "sensory states" but named them $o$,
in a lesson that came before the states/observations distinction existed. The
letter was unmotivated and the terminology was inconsistent with the rest of the
course. Auditing for the class rather than the instance turned up five more:
"nats" used from Lesson 1 and never defined anywhere, $t$ and $T$ unnamed,
$\\mathrm{H}$'s square brackets unexplained, $s$ used in a margin note before its
definition, and $\\pi$ spent on a prior when it is the policy symbol from Week 9.

Two checks, both in reading order across the whole course:

  first-use   every maths symbol's first appearance must sit near a definitional
              cue ("write", "call", "let", "denote", "is the", "where", ...)
  double-bold a term bolded as a first-use definition on two different pages is
              defined twice, so at least one of them is wrong

BOTH ARE FUZZY. A symbol can pass by sitting near the word "where" and still be
undefined, and prose can define something perfectly without using a cue word.
Read the report; do not just check the exit code.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))
import mdx  # noqa: E402

MATH = re.compile(r"\$\$(.+?)\$\$|(?<!\$)\$([^\$\n]+?)\$", re.S)
# {{key:.4f}} substitution tokens are not maths; their format specifiers were
# being scanned as symbols and reporting "f" as an undefined variable.
NUMBER_TOKEN = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*(?::[^}]+)?\}\}")
MACRO = re.compile(r"\\([a-zA-Z]+)")
LETTER = re.compile(r"(?<![\\a-zA-Z])([A-Za-z])(?![a-zA-Z])")
BOLD = re.compile(r"\*\*([^*\n]{3,45}?)\*\*")

CUES = ("write", "writing", "written", "call", "calling", "called", "denote",
        "denoting", "let ", "stands for", "defined", "define", "is the", "are the",
        "for the", "where", "with ", "meaning", "that is", "read as", "count",
        "here ", "we use", "given by", "measures", "which is",
        "the mutual", "the kullback", "parameter", "slider")

# Universal notation nobody needs told, plus structural LaTeX.
EXEMPT = {
    "\\sum", "\\prod", "\\int", "\\ln", "\\log", "\\exp", "\\lim", "\\frac", "\\tfrac",
    "\\dfrac", "\\sqrt", "\\infty", "\\partial", "\\nabla", "\\cdot", "\\times",
    "\\mid", "\\to", "\\in", "\\le", "\\ge", "\\neq", "\\approx", "\\sim", "\\pm",
    "\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\quad", "\\qquad", "\\text",
    "\\mathrm", "\\mathbb", "\\mathcal", "\\mathbf", "\\boldsymbol", "\\begin",
    "\\end", "\\aligned", "\\nonumber", "\\dots", "\\ldots", "\\cdots", "\\vdots",
    "\\hat", "\\tilde", "\\bar", "\\vec", "\\underbrace", "\\overbrace", "\\binom",
    "\\Leftarrow", "\\Rightarrow", "\\Longrightarrow", "\\Longleftarrow",
    "\\xrightarrow", "\\iff", "\\implies", "\\forall", "\\exists", "\\lVert",
    "\\rVert", "\\underbrace", "\\qquad", "\\ \\", "\\sim",
    "\\min", "\\max", "\\inf", "\\sup", "\\arg",
    "\\langle", "\\rangle", "\\lVert", "\\rVert", "\\colon", "\\operatorname",
    "e", "i", "j", "k", "n", "m", "c", "x", "y", "z",   # generic dummies and indices
}

# Names of things in worked examples are not technical terms, so bolding one in
# two different examples is not a double definition. Do not edit good prose to
# silence a proxy; exempt the case and say why.
EXAMPLE_NAMES = {"leopard", "baboon", "nothing", "gust", "quiet",
                 "tawny flash", "branch shakes"}


def pages_in_reading_order():
    course = json.load(open(os.path.join(ROOT, "content/course.json"), encoding="utf-8"))
    out = []
    for w in course["weeks"]:
        d = os.path.join(ROOT, "content", f"week-{int(w['n']):02d}")
        if os.path.isdir(d):
            for fn in sorted(x for x in os.listdir(d) if x.endswith(".md")):
                out.append((f"Week {w['n']} · {fn}", os.path.join(d, fn)))
    return out


def main():
    first, bolds, problems = {}, {}, []

    for label, path in pages_in_reading_order():
        _, body = mdx.split_front_matter(open(path, encoding="utf-8").read())
        body = NUMBER_TOKEN.sub(" ", body)

        for m in BOLD.finditer(body):
            raw_term = m.group(1).strip()
            # A run-in heading ("**Surprise.** From ...") is a label, not a
            # definition. Test before stripping, or the guard never fires.
            if raw_term.endswith((".", ":")) or len(raw_term.split()) > 5:
                continue
            term = raw_term.lower()
            bolds.setdefault(term, []).append(label)

        for m in MATH.finditer(body):
            expr = m.group(1) or m.group(2)
            pos = m.start()
            ctx = re.sub(r"\s+", " ", body[max(0, pos - 220):pos + len(expr) + 90])
            toks = {"\\" + t for t in MACRO.findall(expr)}
            toks |= set(LETTER.findall(re.sub(r"\\[a-zA-Z]+", " ", expr)))
            for t in toks - EXEMPT:
                if t not in first:
                    first[t] = (label, ctx.strip())

    print("FIRST USE OF EACH SYMBOL\n" + "-" * 74)
    for t, (label, ctx) in sorted(first.items(), key=lambda kv: kv[1][0]):
        ok = any(c in ctx.lower() for c in CUES)
        if not ok:
            problems.append(f"{t} first used in {label} with no definitional cue nearby")
        print(f"{'   ' if ok else ' ! '}{t:<12} {label}")
        if not ok:
            print(f"      …{ctx[-150:]}")

    dupes = {t: p for t, p in bolds.items()
             if len(set(p)) > 1 and t not in EXAMPLE_NAMES}
    if dupes:
        print("\nTERMS BOLDED AS A FIRST DEFINITION ON MORE THAN ONE PAGE\n" + "-" * 74)
        for t, p in sorted(dupes.items()):
            print(f" ! {t}  —  {', '.join(sorted(set(p)))}")
            problems.append(f"'{t}' is bolded as a definition on {len(set(p))} pages")

    print(f"\n{len(first)} symbols · {len(bolds)} bolded terms · {len(problems)} to look at")
    if problems:
        print("\nThese are candidates, not verdicts. Read each and decide.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
