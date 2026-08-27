#!/usr/bin/env python3
"""check_site.py — verify the built site before publishing.

    python3 build/check_site.py

Checks, in order of how much trouble they have already caused:
  1. every [eq:name] reference resolves to a labelled equation on the same page
  2. every widget named in content is registered in a JS file that page loads
  3. every internal href points at a file that exists
  4. HTML tags balance on every page (counting every tag the corpus uses,
     not only the ones we thought of)
  5. no stray LaTeX delimiters left outside a maths span
  6. no markup the builder does not implement, reaching a page verbatim
Exit code is non-zero if anything fails, so it can gate a commit.
"""
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
CONTENT = os.path.join(ROOT, "content")
ASSETS = os.path.join(ROOT, "assets", "js")

fails, warns = [], []


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


def html_files():
    for dp, _, fns in os.walk(DOCS):
        for fn in fns:
            if fn.endswith(".html"):
                yield os.path.join(dp, fn)


# ── 1. equation references ───────────────────────────────────────────────
def check_eqrefs():
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        defined = set(re.findall(r'<div class="eqn" id="eq-([^"]+)"', h))
        used = set(re.findall(r'<a class="eqref" href="#eq-([^"]+)"', h))
        for u in used - defined:
            fail(f"eqref -> nothing: [eq:{u}] on {os.path.relpath(path, ROOT)}")
        if '>(?)<' in h:
            fail(f"unresolved equation number rendered as (?) on {os.path.relpath(path, ROOT)}")


# ── 2. widgets ───────────────────────────────────────────────────────────
def check_widgets():
    registered = {}
    for fn in os.listdir(ASSETS):
        if fn.endswith(".js"):
            src = open(os.path.join(ASSETS, fn), encoding="utf-8").read()
            for name in re.findall(r"""(?:AIF|A)\.register\(\s*['"]([^'"]+)['"]""", src):
                registered.setdefault(name, set()).add("assets/js/" + fn)
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        loaded = set(re.findall(r'<script defer src="[^"]*?(assets/js/[^"]+)"', h))
        for name in re.findall(r'data-widget="([^"]+)"', h):
            where = registered.get(name)
            rel = os.path.relpath(path, ROOT)
            if not where:
                fail(f"widget '{name}' is used on {rel} but registered nowhere")
            elif not (where & loaded):
                fail(f"widget '{name}' on {rel} is defined in {sorted(where)} "
                     f"but that file is not loaded (loaded: {sorted(loaded)})")
    used_anywhere = set()
    for path in html_files():
        used_anywhere |= set(re.findall(r'data-widget="([^"]+)"',
                                        open(path, encoding="utf-8").read()))
    for name in set(registered) - used_anywhere:
        warn(f"widget '{name}' is registered but used on no page")


# ── 3. internal links ────────────────────────────────────────────────────
def check_links():
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        base = os.path.dirname(path)
        for href in re.findall(r'href="([^"]+)"', h):
            if href.startswith(("http", "mailto:", "#", "data:")):
                continue
            target = os.path.normpath(os.path.join(base, href.split("#")[0]))
            if not os.path.exists(target):
                fail(f"dead link {href} on {os.path.relpath(path, ROOT)}")


# ── 4. tag balance ───────────────────────────────────────────────────────
VOID = {"br", "hr", "img", "input", "link", "meta", "source", "col", "area",
        "base", "embed", "param", "track", "wbr", "path", "circle", "rect",
        "line", "polyline", "polygon", "use", "stop", "ellipse"}


def check_tags():
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        h = re.sub(r"<script.*?</script>", "", h, flags=re.S)
        h = re.sub(r"<style.*?</style>", "", h, flags=re.S)
        h = re.sub(r"<!--.*?-->", "", h, flags=re.S)
        opens = Counter(t.lower() for t in re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)(?:\s[^>]*?)?(?<!/)>", h))
        closes = Counter(t.lower() for t in re.findall(r"</([a-zA-Z][a-zA-Z0-9]*)\s*>", h))
        for tag in set(opens) | set(closes):
            if tag in VOID or tag in ("html", "body", "head"):
                continue
            if opens[tag] != closes[tag]:
                fail(f"tag imbalance <{tag}>: {opens[tag]} open vs {closes[tag]} close "
                     f"on {os.path.relpath(path, ROOT)}")


# ── 4b. figure integrity ─────────────────────────────────────────────────
def check_svg():
    """Markdown ends a raw-HTML block at a blank line, injecting </p><p> into an
    SVG. Every element survives, so the tag-balance check passes while the figure
    is visibly broken. Caught in review, not by a script; now by a script."""
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        for m in re.finditer(r"<svg\b.*?</svg>", h, re.S | re.I):
            svg = m.group(0)
            for bad in ("<p>", "</p>", "<br", "<h2", "<h3"):
                if bad in svg:
                    fail(f"'{bad}' injected inside an <svg> on "
                         f"{os.path.relpath(path, ROOT)}: the figure will not render")
                    break
            if svg.count("<text") and "</text>" not in svg:
                fail(f"unterminated <text> in an <svg> on {os.path.relpath(path, ROOT)}")


# ── 5. stray maths delimiters ────────────────────────────────────────────
def check_math():
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        body = re.sub(r"<script.*?</script>", "", h, flags=re.S)
        body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
        n = body.count("$")
        if n % 2:
            fail(f"odd number of $ ({n}) on {os.path.relpath(path, ROOT)}: "
                 f"an unclosed maths span")
        for stray in re.findall(r"\\(?:begin|end)\{(align|equation|gather)\*?\}", body):
            warn(f"raw LaTeX environment {stray} on {os.path.relpath(path, ROOT)}; "
                 f"KaTeX wants 'aligned' inside $$...$$")


# ── 6. markup the builder does not implement ─────────────────────────────
# Written after four `[@key]` citations reached a built page verbatim. The
# builder has no citation syntax and never did; the habit came from Pandoc.
# Nothing here looked wrong to any other check, because a leaked `[@key]` is
# balanced, well-formed, and contains no maths.
#
# Each pattern is markup some other tool would resolve and this one will not.
UNSUPPORTED = [
    (r"\[@[A-Za-z][\w:.-]*\]", "Pandoc-style citation; write the reference inline "
                                "with a DOI link, as REFERENCES.md does"),
    (r"\{%[^}]*%\}", "Jinja/Liquid tag"),
    (r"\[\[[^\]|]+\]\]", "wiki-style link"),
    (r":::\s*\w+", "an unclosed or unknown block directive, rendered as text"),
]


def check_unsupported():
    for path in html_files():
        h = open(path, encoding="utf-8").read()
        body = re.sub(r"<script.*?</script>", "", h, flags=re.S)
        body = re.sub(r"<style.*?</style>", "", body, flags=re.S)
        for pat, what in UNSUPPORTED:
            for hit in sorted(set(re.findall(pat, body)))[:4]:
                fail(f"unrendered markup {hit!r} on "
                     f"{os.path.relpath(path, ROOT)}: {what}")


def main():
    if not os.path.isdir(DOCS):
        print("docs/ does not exist; run build_site.py first")
        return 1
    check_eqrefs(); check_widgets(); check_links(); check_tags()
    check_svg(); check_math(); check_unsupported()

    pages = len(list(html_files()))
    for w in warns:
        print("  warn:", w)
    for f in fails:
        print("  FAIL:", f)
    print(f"\n{pages} pages checked · {len(fails)} failures · {len(warns)} warnings")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
