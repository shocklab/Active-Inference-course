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

# Cues that a definition is being given. Deliberately NARROW.
#
# This list once contained "parameter" and "slider". They were added to silence a
# false positive on lambda, and they promptly created a false negative: "the
# parameter kappa sets how hard it pushes back" passed for weeks while telling a
# reader nothing about kappa's kind, range or units. Loosening a check to quiet a
# false positive is how you manufacture a false negative. If a cue is too broad,
# fix the prose, not the cue.
CUES = ("write", "writing", "written", "call", "calling", "called", "denote",
        "denoting", "let ", "stands for", "defined", "define", "is the", "are the",
        "meaning", "read as", "given by", "we use",
        "with probability", "for some function", "for some", "function",
        "the mutual", "the kullback")

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
    "\\min", "\\max", "\\inf", "\\sup", "\\arg", "\\propto", "\\mathbb",
    "E",   # expectation, standard for this audience and listed in notation.md
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


# ── symbol collisions against the notation page ──────────────────────────
# `2\pi` is the circle constant in a Gaussian normaliser, never the policy that
# notation.md reserves for Week 10. Exempted here in that exact form only: a bare
# `\pi`, or `\pi` in any other company, still collides. Kept this narrow on
# purpose. The kappa miss in Week 1 happened because a cue list was widened to
# silence one false positive, and the widening swallowed a real symbol with it.
CONSTANT_PI = re.compile(r"2\s*\\pi\b")

NOTE_ROW = re.compile(r"^\s*(\$[^:]+?\$)\s*::\s*(.+?)\s*$", re.M)
WEEK_TAG = re.compile(r"Week&nbsp;(\d+)|Week (\d+)")



# Single Latin letters carry reservations too, and until Week 2 nothing checked
# them: the collision scan read only `\macro` tokens, so `g` reserved for the
# observation function in Week 5 could be spent in Week 2 in silence. Same class
# of miss the scan exists to catch, one letter shorter.
STRIP_MACRO = re.compile(r"\\[a-zA-Z]+\s*(?:\{[^{}]*\})?")
BARE_LETTER = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")

# Letters so common in generic algebra that reserving them course-wide would be
# meaningless: bound indices, dummy variables, and the constant e.
GENERIC_LETTERS = set("ijklmnpqre")


def _bare_letters(src):
    """Single Latin letters used as symbols, with macro names removed first."""
    return {c for c in BARE_LETTER.findall(STRIP_MACRO.sub(" ", src))
            if c not in GENERIC_LETTERS}


# Operators and decorations that can sit in front of the symbol a row is about.
ROW_PREFIX = re.compile(
    r"^[\s$\-+]*(?:\\(?:ln|log|exp|left|mathrm|mathbf|mathbb|mathcal|bar|hat|dot|tilde)\b\s*)*")
ROW_HEAD = re.compile(r"\\[a-zA-Z]+|[A-Za-z]")


def _row_symbols(sym):
    """The one symbol a notation.md row defines: its head, not its arguments.

    `$Q(s)$` defines Q and merely mentions s; `$P(o \mid s)$` defines the letter
    P. Reading every letter in the row made each argument look like a
    reservation, which turned the collision scan into noise the moment it
    learned to see bare letters at all.
    """
    core = ROW_PREFIX.sub("", sym.strip())
    core = re.sub(r"^\{+", "", core)
    m = ROW_HEAD.search(core)
    if not m:
        return set()
    head = m.group(0)
    return set() if head in EXEMPT else {head}


def check_collisions():
    """A symbol used in one week that notation.md assigns to a later one.

    This is how sigma went unnoticed: the notation page reserved it for the
    softmax in Week 9 while Lesson 1 used it for a noise standard deviation, and
    nothing compared the two. A reader looking sigma up mid-course would have
    found only the wrong meaning.
    """
    note = os.path.join(ROOT, "content", "notation.md")
    if not os.path.exists(note):
        return []
    owner = {}
    for m in NOTE_ROW.finditer(open(note, encoding="utf-8").read()):
        sym, desc = m.group(1), m.group(2)
        w = WEEK_TAG.search(desc)
        home = int(w.group(1) or w.group(2)) if w else None
        for tok in _row_symbols(sym):
            owner.setdefault(tok, []).append((home, desc))

    out = []
    course = json.load(open(os.path.join(ROOT, "content/course.json"), encoding="utf-8"))
    for wk in course["weeks"]:
        d = os.path.join(ROOT, "content", f"week-{int(wk['n']):02d}")
        if not os.path.isdir(d):
            continue
        used = set()
        for fn in os.listdir(d):
            if fn.endswith(".md"):
                body = NUMBER_TOKEN.sub(" ", open(os.path.join(d, fn), encoding="utf-8").read())
                body = CONSTANT_PI.sub("2", body)
                for mm in MATH.finditer(body):
                    src = mm.group(1) or mm.group(2)
                    used |= {"\\" + t for t in MACRO.findall(src)}
                    used |= _bare_letters(src)
        for tok in sorted(used):
            for home, desc in owner.get(tok, []):
                if home and home > int(wk["n"]):
                    out.append(f"{tok} is used in Week {wk['n']} but notation.md files it "
                               f"under Week {home}: \"{desc[:70]}...\"")
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

    for c in check_collisions():
        problems.append(c)
        print(f"\n COLLISION: {c}")

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
