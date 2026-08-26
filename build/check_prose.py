#!/usr/bin/env python3
"""check_prose.py — does the writing read as human?

    python3 build/check_prose.py

Wraps the calibrated census at ~/.claude/skills/writing-review/census.py rather
than reinventing it: that script already measures the tell families per zone,
per 1,000 words, on the BUILT pages, and knows that per-paragraph counting
reports a false clean. Adds the rules specific to this course on top.

Measured on the built site, never on content/. A tell introduced by the renderer
is still a tell, and a budget measured on source misses the zones a reader meets.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
CENSUS = os.path.expanduser("~/.claude/skills/writing-review/census.py")

# From CLAUDE.md. The plainer word wins; a flagged word is fine when it is
# genuinely the right one, so these are candidates, not automatic failures.
BANNED = [
    "delve", "dive into", "leverage", "underscore", "bolster", "foster",
    "seamless", "intricate", "multifaceted", "holistic", "pivotal",
    "groundbreaking", "transformative", "testament", "cutting-edge",
    "game-changing", "unlock", "empower", "elevate", "streamline",
    "in today's", "it's worth noting", "worth noting", "at its core",
    "in essence", "fundamentally,", "simply put", "the short version",
    "bottom line", "needless to say",
    # flagged in review: self-labelling a thing's own value instead of showing it
    "earns its keep", "earn its keep", "pays for itself", "does a lot of work",
    "doing a lot of work", "load-bearing", "the real question", "that matters",
]
# Hard rules: these are never right in this course's prose.
HARD = [
    ("em-dash", re.compile(r"—")),
    ("en-dash as sentence punctuation", re.compile(r"\s–\s")),
    ("spaced hyphen as sentence punctuation", re.compile(r"\S\s-\s\S")),
]


def prose_of(path):
    """Body prose only: drop nav, code, maths, figures and the footer."""
    h = open(path, encoding="utf-8").read()
    m = re.search(r'<article class="abody">(.*?)</article>', h, re.S)
    if not m:
        return ""
    body = m.group(1)
    for pat in (r"<pre.*?</pre>", r"<code.*?</code>", r"<svg.*?</svg>",
                r'<div class="eqn".*?</div>\s*</div>', r"<figcaption.*?</figcaption>"):
        body = re.sub(pat, " ", body, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    # KaTeX renders in the browser, so the static HTML still holds raw LaTeX.
    # A minus sign inside maths is not a spaced hyphen; strip maths before
    # measuring punctuation, or every equation reports as a style failure.
    text = re.sub(r"\$\$.+?\$\$", " ", text, flags=re.S)
    text = re.sub(r"(?<!\$)\$[^\$\n]+?\$", " ", text)
    return re.sub(r"\s+", " ", text)


def main():
    pages = [os.path.join(dp, f) for dp, _, fs in os.walk(DOCS)
             for f in fs if f.endswith(".html")]
    if not pages:
        print("no built pages; run build_site.py first")
        return 1

    print("── calibrated census " + "─" * 52)
    if os.path.exists(CENSUS):
        r = subprocess.run([sys.executable, CENSUS] + sorted(pages),
                           capture_output=True, text=True)
        print(r.stdout.rstrip() or r.stderr.rstrip())
    else:
        print(f"  census.py not found at {CENSUS}; skipping the calibrated pass")

    print("\n── course-specific rules " + "─" * 48)
    fails, cands, words = [], [], 0
    for p in sorted(pages):
        text = prose_of(p)
        words += len(text.split())
        rel = os.path.relpath(p, ROOT)
        for name, pat in HARD:
            for m in pat.finditer(text):
                fails.append(f"{rel}: {name} — …{text[max(0,m.start()-45):m.end()+45].strip()}…")
        low = text.lower()
        for w in BANNED:
            for m in re.finditer(r"\b" + re.escape(w), low):
                cands.append(f"{rel}: '{w}' — …{text[max(0,m.start()-45):m.start()+60].strip()}…")

    print(f"  {len(pages)} pages · {words:,} words of body prose")
    for f in fails:
        print("  FAIL:", f)
    for c in cands[:25]:
        print("  candidate:", c)
    if len(cands) > 25:
        print(f"  … and {len(cands)-25} more candidates")
    if not fails and not cands:
        print("  no em-dashes, no dash evasions, no banned vocabulary")
    print("\n  Census triage queues are candidates, not verdicts. This course "
          "teaches\n  distinctions, so expect a high keep-rate on the X-not-Y family.")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
