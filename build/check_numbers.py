#!/usr/bin/env python3
"""check_numbers.py — is every derived number in the prose computed, not typed?

    python3 build/check_numbers.py

Four wrong values reached a draft of Week 1 because they were worked out by hand.
The fix is structural: a week's derived numbers live in content/week-NN/values.py
and the prose refers to them as {{key:.4f}}, so the published figure IS the
computed figure and cannot drift.

This checks two directions:

  resolved   every {{key}} in the source resolves, and each lesson's built page
             is what the builder makes of the current content and the values
             values.py computes right now: its article body is compared with a
             fresh render, line for line (and no <mark>?key?</mark> remains)
  untokenised any decimal still typed literally into the prose, which is a number
             nobody is checking: lesson pages, the words inside SVG figures, and
             the standalone pages

The second is the one that matters. A substitution scheme nobody audits just
moves the hand-typed numbers somewhere less visible.

Literal numbers that are NOT derived quantities (a prior someone chose, a year,
a matrix entry that defines the model) are legitimate where they are first
stated; list that sentence in ALLOW below, exactly, with its file, its count and
a reason. An allow-list entry is a claim that a human checked it, and one whose
sentence no longer occurs as declared fails.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "build"))
import mdx  # noqa: E402

# Numbers that are inputs or conventions rather than results. Each entry is a
# claim that someone read the sentence it names and checked the digits in it
# are inputs.
#
# An entry is (file, fragment, count, reason). The fragment is copied exactly
# from the source, long enough to be unique, and holds the sentence where the
# input is stated; count is how many times that fragment occurs in the file.
# Only decimals inside an occurrence of the fragment are admitted. The same
# digit string anywhere else in the file fails, and so does a fragment that no
# longer occurs exactly `count` times (a stale allowance is a hole waiting for
# a number to fall into it).
#
# An input is stated once. Where the prose restates it later (a worked line, a
# table column, an exercise that reuses a lesson's model), the restatement cites
# the value from values.py as a token, so it cannot drift from the input.
#
# History: this was one set of digit strings per week, which admitted every
# typed 0.7 in Week 1; then a set of digits per file with a regex for the words
# round each occurrence, whose broad alternatives ("becomes", a table row of any
# state) admitted a computed posterior typed into a sentence or a table
# (2026-11-12). A fixed fragment and count admit nothing they were not written
# for.
W1, W2, W4 = "week-01/", "week-02/", "week-04/"
ALLOW = [
    # Week 1, Lesson 1
    (W1 + "01-soft-body.md", "A gain of $0.15$ removes", 1,
     "the illustrative gain chosen for the drift example"),
    (W1 + "01-soft-body.md", "A gain of $0.5$ and one\nof $1.5$ therefore", 1,
     "illustrative gains; 1.5 mirrors 0.5 about the optimum"),
    (W1 + "01-soft-body.md", "For the chosen distribution $(0.94,0.02,0.02,0.02)$", 1,
     "the chosen distribution of the worked entropy example"),
    # Week 1, Lesson 2
    (W1 + "02-process-and-model.md",
     "copy it with probability\n$0.9$, flipping it otherwise. Let $\\mu$ copy $o$ with probability $0.8$,", 1,
     "the chosen copy probabilities of the blanket example"),
    (W1 + "02-process-and-model.md", "with probability $0.3$ copy $\\vartheta$", 1,
     "the chosen bypass probability in the blanket-breaking variant"),
    (W1 + "02-process-and-model.md",
     "Let $P(\\mathrm{bright}\\mid\\mathrm{off})=0.20$ and write $\\theta$ for\n"
     "$P(\\mathrm{bright}\\mid\\mathrm{on})$, initially $0.80$.", 1,
     "the lamp sensor probabilities, chosen"),
    # Week 1, Lesson 3
    (W1 + "03-bayes.md",
     "| tawny flash | 0.70 | 0.15 | 0.02 |\n| branch shakes | 0.25 | 0.75 | 0.08 |\n"
     "| quiet | 0.05 | 0.10 | 0.90 |", 1,
     "the likelihood matrix we chose"),
    (W1 + "03-bayes.md", "is\n$P(s) = (0.08,\\ 0.22,\\ 0.70)$ over (leopard", 1,
     "the chosen mountain prior"),
    (W1 + "03-bayes.md", "At\n$\\alpha = 0.3$ the leopard column", 1,
     "the chosen ambiguity of the blending example"),
    (W1 + "03-bayes.md", "nervous hiker with prior\n$P(s) = (0.40,\\ 0.30,\\ 0.30)$", 1,
     "the nervous hiker's prior, chosen for the exercise"),
    (W1 + "03-bayes.md", "a flat prior $P(s) = (0.5,\\ 0.5)$", 1,
     "the flat prior of the counterexample"),
    (W1 + "03-bayes.md",
     "\\begin{pmatrix} 0.60 & 0.10 \\\\ 0.35 & 0.85 \\\\ 0.05 & 0.05 \\end{pmatrix}", 1,
     "the counterexample likelihood, chosen"),
    # Week 1, Lesson 4
    (W1 + "04-inverse-problem.md", "P(s_{\\text{gust}} = 1) = P(s_{\\text{bab}} = 1) = 0.1,", 1,
     "the chosen cause prior of the noisy-OR model"),
    (W1 + "04-inverse-problem.md", "with $w = 0.9$ and $\\lambda = 0.01$:", 1,
     "the chosen noisy-OR strength and leak"),
    (W1 + "04-inverse-problem.md", "since the Big Bang, roughly $4.4 \\times 10^{26}$.", 1,
     "the age of the universe in nanoseconds, an order of magnitude from outside the course"),
    # Week 1, Lesson 5
    (W1 + "05-the-denominator.md",
     "P(s_1)=(0.40,0.60),\\qquad\nT=\\begin{pmatrix}0.70&0.20\\\\0.30&0.80\\end{pmatrix},\n\\qquad\n"
     "P(o=1\\mid s_3=0)=0.10,\\quad P(o=1\\mid s_3=1)=0.90.", 1,
     "the chain's chosen initial distribution, transition matrix and sensor"),
    # Week 1, exercises
    (W1 + "06-exercises.md",
     "overheating is $0.02$. The sensor raises the alarm with probability $0.90$ when\n"
     "the body is overheated and with probability $0.05$ when it is normal.", 1,
     "the alarm exercise's chosen prior and sensor"),
    (W1 + "06-exercises.md", "$\\alpha = 0.25$, $0.5$ and $0.75$, and compare", 1,
     "the ambiguity values the exercise asks for"),
    (W1 + "06-exercises.md", "$\\sigma = 0.09$ and $\\sigma = 0.035$ before you run it", 1,
     "the simulation noise settings the exercise asks about"),
    (W1 + "06-exercises.md", "small gain, $\\kappa = 0.01$ and $t = 10$ give", 1,
     "a chosen small gain for the check"),
    (W1 + "06-exercises.md", "$v_t \\ge 0.95\\,v$ exactly when", 1,
     "the chosen 95% settling threshold"),
    # Week 2
    (W2 + "03-climbing.md", "edge of the range $0.05$ to $8$", 1,
     "the lower end of the chosen search range"),
    (W2 + "05-exercises.md", "Plot the log joint on $0.05 \\le d \\le 10$", 1,
     "the lower end of a plotting range the exercise asks for"),
    (W2 + "05-exercises.md", "$P(\\mathrm{hazard}\\mid o)=0.30$", 1,
     "the chosen posterior hazard probability in the decision example"),
    # Week 4: confirmed an input. The same 0.5 is the cut in twocause.py's
    # corr_gt05, whose result the sentence quotes as tc_ex_corr_pct.
    (W4 + "05-two-causes.md", "exceeds $0.5$ in magnitude", 1,
     "the correlation threshold chosen for the count, the cut in twocause.py"),
    # Standalone pages have no values.py, so anything numeric there must be a convention.
    ("orientation.md", "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)", 1,
     "the version of the CC BY 4.0 licence, a name"),
    ("ai-use.md", "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)", 1,
     "the version of the CC BY 4.0 licence, a name"),
]
STANDALONE = ("index", "notation", "orientation", "ai-use")

# A full stop after a number ends a sentence and does not make it part of a
# dotted name. This read (?![\w.]) until 2026-11-12, so every decimal typed at
# the end of a sentence ("The computed derivative is 0.7.") was invisible.
DECIMAL = re.compile(r"(?<![\w.])(\d+\.\d+)(?!\w|\.\d)")
# Zones that legitimately contain numbers which are not prose claims. A zone you
# do not strip is a zone that floods the report and gets the whole check ignored.
NON_PROSE = [
    # Figure coordinates and styling are not claims, but the words an SVG prints
    # are: a label reading 0.450 is as much a number in the prose as any other.
    # The whole SVG is replaced by the text of its <text> elements.
    (re.compile(r"<svg\b.*?</svg>", re.S | re.I), lambda m: " ".join(
        re.sub(r"<[^>]+>", " ", t) for t in
        re.findall(r"<text\b[^>]*>(.*?)</text>", m.group(0), re.S | re.I))),
    (re.compile(r"^(```|~~~).*?^\1", re.S | re.M), " "),   # fenced code
    (re.compile(r"`[^`\n]+`"), " "),                       # inline code
    (re.compile(r"\]\([^)]*\)"), " "),                    # link targets
    # `[lesson:1.5]` is a cross-reference, and its week.lesson pair is not a
    # decimal anyone typed as a result. Adding the syntax without adding this
    # put 61 false positives into the report at once, which is how a check
    # stops being read.
    (re.compile(r"\[(?:lesson|week):[^\]]*\]"), " "),
]
TOKEN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]+))?\}\}")


_VALUES_CACHE = {}


def load_values(week_dir):
    """The week's VALUES dict, executed once per run.

    Week 3's values.py takes several seconds to run its simulations, and this
    script used to execute it three times (once per pass) and build_site.py
    once per lesson. The result is the same each time, so keep it."""
    path = os.path.join(week_dir, "values.py")
    if not os.path.exists(path):
        return None
    if path not in _VALUES_CACHE:
        spec = importlib.util.spec_from_file_location("nm", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VALUES_CACHE[path] = dict(mod.VALUES)
    return _VALUES_CACHE[path]


def mask_allowed(body, label):
    """The body with every decimal inside an ALLOW fragment for this file masked,
    and a failure for each entry whose fragment does not occur `count` times.
    Masking keeps every other character where it was, so what remains is
    scanned exactly as before."""
    chars, errors = list(body), []
    for where, frag, count, _ in ALLOW:
        if where != label:
            continue
        if not DECIMAL.search(frag):
            errors.append(f"ALLOW entry for {where} names a fragment with no decimal in it: {frag!r}")
            continue
        starts, k = [], body.find(frag)
        while k >= 0:
            starts.append(k)
            k = body.find(frag, k + len(frag))
        if len(starts) != count:
            errors.append(f"ALLOW entry for {where} expects {count} occurrence(s) of {frag!r}, "
                          f"found {len(starts)}; correct the entry, or the source")
        for k in starts:
            for m in DECIMAL.finditer(frag):
                for x in range(k + m.start(), k + m.end()):
                    if chars[x].isdigit():
                        chars[x] = "#"
    return "".join(chars), errors


def untokenised(body, label):
    """Decimals typed by hand in a page's source that no ALLOW fragment holds,
    and the ALLOW entries for this file that do not match it as declared."""
    out = []
    masked, errors = mask_allowed(body, label)
    stripped = TOKEN.sub(" ", masked)
    for zone, repl in NON_PROSE:
        stripped = zone.sub(repl, stripped)
    for m in DECIMAL.finditer(stripped):
        ctx = re.sub(r"\s+", " ", stripped[max(0, m.start() - 70):m.end() + 45]).strip()
        out.append((label, m.group(1), ctx))
    return out, errors



def fresh_render_diffs():
    """Is docs/ a faithful render of the current content and values?

    Every lesson is rendered again with build_site's own functions (the same
    substitution, Markdown, equation numbering and link resolution), from the
    content and the values.py computed in this run, and the whole document is
    compared with the built page, which build_site writes verbatim. Any
    difference fails, with the first line that differs. This replaced a search
    for each substituted number in short runs of the text round it, which a
    repeated phrase could satisfy with the wrong occurrence (2026-11-12):
    comparing the whole page leaves no occurrence to mistake for another. A
    first version compared only the text between the first
    <article class="abody"> and the first </article>, so a corrupted value
    after a nested </article> or one inside an HTML comment passed; the whole
    document has no such boundary to fool."""
    import build_site as bs
    saved = bs.CONTENT, bs.COURSE, bs.OUT, bs._NUMBERS
    bs.CONTENT = os.path.join(ROOT, "content")
    bs.COURSE = os.path.join(bs.CONTENT, "course.json")
    bs.OUT = os.path.join(ROOT, "docs")
    bs._NUMBERS = _VALUES_CACHE        # the same values, executed once
    out = []
    try:
        _, weeks = bs.load_course()
        flat = [ls for w in weeks for ls in w.lessons]
        for w in weeks:
            for ls in w.lessons:
                label = os.path.relpath(ls.path, bs.CONTENT)
                built = os.path.join(bs.OUT, ls.href)
                if not os.path.exists(built):
                    out.append(f"{label}: no built page at docs/{ls.href}")
                    continue
                log = io.StringIO()
                with contextlib.redirect_stdout(log):
                    fresh = bs.render_lesson(ls, w, weeks, flat)
                for line in log.getvalue().splitlines():
                    out.append(f"{label}: the builder reports {line.strip()}")
                with open(built, encoding="utf-8", newline="") as f:
                    page = f.read()
                if fresh != page:
                    fl, bl = fresh.split("\n"), page.split("\n")
                    k = next((k for k, (x, y) in enumerate(zip(fl, bl)) if x != y),
                             min(len(fl), len(bl)))
                    want = fl[k] if k < len(fl) else "(end of page)"
                    got = bl[k] if k < len(bl) else "(end of page)"
                    out.append(f"{label}: docs/{ls.href} differs from a fresh render at "
                               f"line {k + 1}: built {got[:160]!r}, current "
                               f"{want[:160]!r} (stale or corrupted build?)")
    finally:
        bs.CONTENT, bs.COURSE, bs.OUT, bs._NUMBERS = saved
    return out


# The builder marks a missing key as <mark>?key?</mark>. An earlier version of
# this check looked for "??", which that markup never contains, so a missing key
# reached the page and the check stayed green.
UNRESOLVED = re.compile(r"<mark>\?[^<]*\?</mark>|\{\{")


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
    unused, drift, labels = [], [], set()

    for w in course["weeks"]:
        n = int(w["n"])
        wdir = os.path.join(ROOT, "content", f"week-{n:02d}")
        if not os.path.isdir(wdir):
            continue
        values = load_values(wdir)
        cited = set()
        for fn in sorted(x for x in os.listdir(wdir) if x.endswith(".md")):
            n_pages += 1
            path = os.path.join(wdir, fn)
            meta, body = mdx.split_front_matter(open(path, encoding="utf-8").read())
            label = f"week-{n:02d}/{fn}"

            for m in TOKEN.finditer(body):
                n_tok += 1
                key, fmt = m.group(1), m.group(2)
                cited.add(key)
                if values is None:
                    problems.append(f"{label}: uses {{{{{key}}}}} but the week has no values.py")
                elif key not in values:
                    problems.append(f"{label}: {{{{{key}}}}} is not defined in values.py")
                elif re.fullmatch(r"sci\d*", fmt or ""):
                    pass   # mdx renders LaTeX scientific notation, not a Python format
                else:
                    try:
                        format(values[key], fmt) if fmt else str(values[key])
                    except (ValueError, TypeError):
                        problems.append(f"{label}: {{{{{key}:{fmt}}}}} is not a valid format")

            found, errs = untokenised(body, label)
            untok.extend(found)
            problems.extend(errs)
            labels.add(label)

            slug = re.sub(r"^\d+[-_]", "", fn[:-3])
            built = os.path.join(ROOT, "docs", f"week-{n:02d}", f"{slug}.html")
            if os.path.exists(built) and UNRESOLVED.search(open(built, encoding="utf-8").read()):
                drift.append(f"{label}: unresolved token markup reached the built page")

        if values:
            for k in sorted(set(values) - cited):
                unused.append(f"week-{n:02d}: {k}")

    # Standalone pages: no values.py, so every decimal must be a convention.
    for name in STANDALONE:
        path = os.path.join(ROOT, "content", f"{name}.md")
        if not os.path.exists(path):
            continue
        n_pages += 1
        _, body = mdx.split_front_matter(open(path, encoding="utf-8").read())
        if TOKEN.search(body):
            problems.append(f"{name}.md: a standalone page has no values.py to fill {{{{...}}}}")
        found, errs = untokenised(body, f"{name}.md")
        untok.extend(found)
        problems.extend(errs)
        labels.add(f"{name}.md")
        built = os.path.join(ROOT, "docs", f"{name}.html")
        if os.path.exists(built) and UNRESOLVED.search(open(built, encoding="utf-8").read()):
            drift.append(f"{name}.md: unresolved token markup reached the built page")

    # Confirm what actually reached the built pages: each lesson's article body
    # must be what the builder makes of the current content and values now.
    drift.extend(fresh_render_diffs())

    for where in sorted({e[0] for e in ALLOW} - labels):
        problems.append(f"ALLOW entries name {where}, which this run did not read")

    print(f"{n_pages} pages · {n_tok} computed numbers substituted")
    problems.extend(check_widget_constants())
    for p in problems + drift:
        print("  FAIL:", p)
    if untok:
        print(f"\n  {len(untok)} decimals still typed by hand:")
        for label, v, ctx in untok[:40]:
            print(f"    {label}  {v}\n      …{ctx}")
        print("\n  Move each into values.py and reference it as {{key}}, or add it to")
        print("  ALLOW, as the exact sentence that states it, with its count and a reason,")
        print("  if it is an input stated there for the first time. A restated input is a")
        print("  token too.")
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
