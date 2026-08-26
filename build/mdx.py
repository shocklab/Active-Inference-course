#!/usr/bin/env python3
"""mdx.py — Markdown + LaTeX + block directives for the Active Inference course.

Why not plain python-markdown: it eats maths. `$P(x_1|x_2)$` loses its
underscores to emphasis and `\\{` to escaping. So the pipeline is

    stash code  ->  stash maths  ->  restore code  ->  markdown  ->  restore maths

which keeps every character between dollar signs untouched all the way to the
browser, where KaTeX renders it.

Authoring syntax on top of CommonMark
-------------------------------------
  $x$ , $$...$$              inline and display maths
  $$...$$ {#vfe}             a *labelled* display equation: numbered by section,
                             so Week 3 lesson 2 gives (3.2.1), (3.2.2), ...
  [eq:vfe]                   cross-reference, renders as a link "(3.2.1)"
  {{ev_tawny:.4f}}           a COMPUTED number, substituted at build time from
                             the week's numbers.py. Never type a derived number.
  ::: type Title ... :::     block directives (see BOXES below)
  ::: mn Label ... :::       a margin note, floats into the right column
  ::: widget name | caption  mounts an interactive widget
  ---solution---             inside an exercise, splits problem from solution

Display equations are numbered only when labelled. Numbering every line of a
derivation is noise; numbering the three or four lines you actually refer back
to is navigation.
"""
import re
import markdown as _md

# ── placeholder tokens ───────────────────────────────────────────────────
# Alphanumeric so no Markdown rule (emphasis, escaping, autolinks) can touch them.
CODE_T = "zqCODEqz{}zqz"
RAW_T  = "zqRAWqz{}zqz"
MATH_T = "zqMATHqz{}zqz"
BLOK_T = "zqBLOKqz{}zqz"

_MD_EXT = ["tables", "fenced_code", "attr_list", "sane_lists", "footnotes", "md_in_html"]

BOXES = {
    #  key            css class      default header label        numbered?
    "definition":   ("definition",   "Definition",                True),
    "theorem":      ("theorem",      "Theorem",                   True),
    "proposition":  ("proposition",  "Proposition",               True),
    "lemma":        ("lemma",        "Lemma",                     True),
    "remark":       ("remark",       "Remark",                    False),
    "aside":        ("aside",        "Aside",                     False),
    "keyidea":      ("keyidea",      "The idea",                  False),
    "warning":      ("warning",      "Careful",                   False),
    "derivation":   ("derivation",   "Derivation",                False),
    "exercise":     ("exercise",     "Exercise",                  True),
    "checkpoint":   ("checkpoint",   "Check yourself",            False),
}


class Doc:
    """Per-page rendering state: counters, stashes, and the collected TOC.

    `prefix` is the section number this page sits at, e.g. "3.2" for Week 3
    lesson 2, so its figures number 3.2.1, 3.2.2 and its equations (3.2.1)...
    Numbering that carries the section survives being quoted somewhere else in
    the course; a bare "(1)" does not.
    """

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.code, self.math, self.blok, self.raw = [], [], [], []
        self.eqn_no = 0
        self.fig_no = 0
        self.sec_no = 0
        self.labels = {}        # label -> equation number
        self.pending_refs = []  # labels referenced before definition
        self.missing_numbers = []
        self.counters = {}
        self.toc = []           # (level, id, text)

    # -- stash helpers ----------------------------------------------------
    def put_code(self, s):
        self.code.append(s)
        return CODE_T.format(len(self.code) - 1)

    def put_raw(self, s):
        self.raw.append(s)
        return RAW_T.format(len(self.raw) - 1)

    def put_math(self, s):
        self.math.append(s)
        return MATH_T.format(len(self.math) - 1)

    def put_blok(self, s):
        self.blok.append(s)
        return BLOK_T.format(len(self.blok) - 1)

    def bump(self, key):
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]


# ── 1. code stashing ─────────────────────────────────────────────────────
_FENCE = re.compile(r"^(```|~~~)[^\n]*\n.*?^\1[ \t]*$", re.S | re.M)
_INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)", re.S)


def _stash_code(text, doc):
    text = _FENCE.sub(lambda m: doc.put_code(m.group(0)), text)
    text = _INLINE_CODE.sub(lambda m: doc.put_code(m.group(0)), text)
    return text


def _restore_code(text, doc):
    for i, s in enumerate(doc.code):
        text = text.replace(CODE_T.format(i), s)
    return text


# ── 1b. verbatim block HTML ──────────────────────────────────────────────
# Markdown ends a raw-HTML block at the first blank line, so an <svg> written
# with blank lines between its logical groups gets </p><p> injected into the
# middle of it and stops rendering. Every element survives, so a tag-balance
# check passes while the figure is broken on the page. Stash the whole block.
_RAW_BLOCK = re.compile(r"^<(svg|table|figure)\b.*?^</\1>\s*$", re.S | re.M | re.I)


def _stash_raw(text, doc):
    return _RAW_BLOCK.sub(lambda m: doc.put_raw(m.group(0)), text)


def _restore_raw(html, doc):
    for i, s in enumerate(doc.raw):
        tok = RAW_T.format(i)
        html = re.sub(r"<p>\s*" + re.escape(tok) + r"\s*</p>", lambda _m, s=s: s, html)
        html = html.replace(tok, s)
    return html


# ── 2. maths stashing ────────────────────────────────────────────────────
# Display first (longest match wins), then inline. A labelled display equation
# is  $$ ... $$ {#label}  with the label on the closing line.
_DISPLAY = re.compile(r"\$\$(.+?)\$\$[ \t]*(?:\{#([A-Za-z0-9_:.-]+)\})?", re.S)
_BRACKET_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)
# Inline $...$: no newline-newline inside, not preceded/followed by another $.
_INLINE = re.compile(r"(?<![\$\\])\$(?!\$)((?:[^\$\\\n]|\\.|\n(?!\n))+?)(?<!\\)\$(?!\$)")
_PAREN = re.compile(r"\\\((.+?)\\\)", re.S)


def _display_html(body, label, doc):
    """Render one display equation, numbered only if it carries a label."""
    body = body.strip()
    if label:
        doc.eqn_no += 1
        n = f"{doc.prefix}.{doc.eqn_no}" if doc.prefix else str(doc.eqn_no)
        doc.labels[label] = n
        eid = f"eq-{label}"
        no = f'<div class="eqno"><a href="#{eid}" title="Link to equation {n}">({n})</a></div>'
        return (f'<div class="eqn" id="{eid}"><div class="eqbody">$${body}$$</div>{no}</div>')
    return f'<div class="eqn"><div class="eqbody">$${body}$$</div></div>'


def _stash_math(text, doc):
    def disp(m):
        return doc.put_math(_display_html(m.group(1), m.group(2), doc))

    text = _DISPLAY.sub(disp, text)
    text = _BRACKET_DISPLAY.sub(lambda m: doc.put_math(_display_html(m.group(1), None, doc)), text)
    text = _INLINE.sub(lambda m: doc.put_math("$" + m.group(1) + "$"), text)
    text = _PAREN.sub(lambda m: doc.put_math("$" + m.group(1) + "$"), text)
    return text


def _restore_math(html, doc):
    # Block-level equations sitting alone in a <p> must break out of it, or the
    # markup is invalid (a div inside a p closes the p at parse time).
    for i, s in enumerate(doc.math):
        tok = MATH_T.format(i)
        if s.startswith('<div class="eqn"'):
            html = re.sub(r"<p>\s*" + re.escape(tok) + r"\s*</p>", lambda _m, s=s: s, html)
        html = html.replace(tok, s)
    return html


# ── 3. cross-references ──────────────────────────────────────────────────
_EQREF = re.compile(r"\[eq:([A-Za-z0-9_:.-]+)\]")


def _resolve_refs(html, doc):
    def rep(m):
        lab = m.group(1)
        n = doc.labels.get(lab)
        if n is None:
            doc.pending_refs.append(lab)
            return '<a class="eqref" href="#eq-%s">(?)</a>' % lab
        return f'<a class="eqref" href="#eq-{lab}">({n})</a>'

    return _EQREF.sub(rep, html)


# ── 4. block directives ──────────────────────────────────────────────────
_OPEN = re.compile(r"^:::[ \t]*([A-Za-z][A-Za-z0-9_-]*)[ \t]*(.*?)[ \t]*$")


def _split_directives(text):
    """Yield ('text', s) and ('dir', type, args, inner) with depth-counted fences."""
    out, lines, i = [], text.split("\n"), 0
    buf = []
    while i < len(lines):
        m = _OPEN.match(lines[i])
        if not m:
            buf.append(lines[i])
            i += 1
            continue
        if buf:
            out.append(("text", "\n".join(buf)))
            buf = []
        dtype, dargs = m.group(1), m.group(2)
        depth, inner, i = 1, [], i + 1
        while i < len(lines):
            if _OPEN.match(lines[i]):
                depth += 1
            elif lines[i].strip() == ":::":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            inner.append(lines[i])
            i += 1
        out.append(("dir", dtype, dargs, "\n".join(inner)))
    if buf:
        out.append(("text", "\n".join(buf)))
    return out


_SOLUTION_SPLIT = re.compile(r"^[ \t]*-{3,}\s*solution\s*-{3,}[ \t]*$", re.M | re.I)


def _render_directive(dtype, dargs, inner, doc):
    dtype = dtype.lower()

    # margin note -------------------------------------------------------
    if dtype in ("mn", "margin"):
        lab = f'<span class="mnl">{dargs}</span>' if dargs else ""
        return f'<aside class="mn">{lab}{_render_inner(inner, doc)}</aside>'

    # interactive widget ------------------------------------------------
    if dtype == "widget":
        name, _, cap = dargs.partition("|")
        name, cap = name.strip(), cap.strip()
        doc.fig_no += 1
        fno = f"{doc.prefix}.{doc.fig_no}" if doc.prefix else str(doc.fig_no)
        capm = _render_inner(cap, doc) if cap else ""
        capm = re.sub(r"^<p>|</p>$", "", capm.strip())
        body = _render_inner(inner, doc) if inner.strip() else ""
        return (
            f'<figure class="widget" id="fig-{fno.replace(".", "-")}">'
            f'<div class="wmount" data-widget="{name}">'
            f'<div class="wfallback">This figure is interactive and needs JavaScript.</div></div>'
            f'<figcaption><span class="fno">Figure {fno}.</span> {capm}{body}</figcaption>'
            f"</figure>"
        )

    # static figure ------------------------------------------------------
    if dtype in ("fig", "figure"):
        doc.fig_no += 1
        fno = f"{doc.prefix}.{doc.fig_no}" if doc.prefix else str(doc.fig_no)
        cap = _render_inner(dargs, doc) if dargs else ""
        cap = re.sub(r"^<p>|</p>$", "", cap.strip())
        return (
            f'<figure id="fig-{fno.replace(".", "-")}">{_render_inner(inner, doc)}'
            f'<figcaption><span class="fno">Figure {fno}.</span> {cap}</figcaption></figure>'
        )

    # notation table -----------------------------------------------------
    if dtype == "notation":
        rows = []
        for line in inner.split("\n"):
            if not line.strip():
                continue
            term, _, desc = line.partition("::")
            if not desc:
                continue
            t = _inline_only(term.strip(), doc)
            d = _inline_only(desc.strip(), doc)
            rows.append(f"<tr><th>{t}</th><td>{d}</td></tr>")
        return ('<table class="notation"><tbody>' + "".join(rows)
                + "</tbody></table>")

    # standard boxes -----------------------------------------------------
    if dtype in BOXES:
        css, label, numbered = BOXES[dtype]
        head = label
        if numbered:
            head = f"{label} {doc.bump(dtype)}"
        title = f'<span class="bt">{_inline_only(dargs, doc)}</span>' if dargs else ""
        if dtype == "exercise" and _SOLUTION_SPLIT.search(inner):
            prob, sol = _SOLUTION_SPLIT.split(inner, 1)
            bodyh = (
                _render_inner(prob, doc)
                + "<details><summary>Solution</summary>"
                + _render_inner(sol, doc)
                + "</details>"
            )
        else:
            bodyh = _render_inner(inner, doc)
        return (
            f'<div class="box {css}"><div class="boxhd">{head}{title}</div>{bodyh}</div>'
        )

    # unknown directive: render inner, keep the class so it is visible ----
    return f'<div class="box {dtype}">{_render_inner(inner, doc)}</div>'


def _render_inner(text, doc):
    """Render directive contents through the same pipeline (recursively)."""
    return _to_html(text, doc)


def _inline_only(text, doc):
    """Render a short fragment without wrapping it in <p>."""
    h = _to_html(text, doc).strip()
    if h.startswith("<p>") and h.endswith("</p>") and h.count("<p>") == 1:
        h = h[3:-4]
    return h


# ── 5. headings & the page TOC ───────────────────────────────────────────
_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def slug(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = _SLUG_BAD.sub("-", s.lower()).strip("-")
    return s or "section"


def _number_headings(html, doc, number_sections):
    seen = {}

    def rep(m):
        lvl, attrs, inner = m.group(1), m.group(2), m.group(3)
        if 'id="' in attrs:
            hid = re.search(r'id="([^"]+)"', attrs).group(1)
        else:
            base = slug(inner)
            seen[base] = seen.get(base, 0) + 1
            hid = base if seen[base] == 1 else f"{base}-{seen[base]}"
            attrs = f' id="{hid}"' + attrs
        prefix = ""
        if lvl == "2" and number_sections:
            doc.sec_no += 1
            prefix = f'<span class="secno">{doc.sec_no}</span>'
        doc.toc.append((int(lvl), hid, re.sub(r"<[^>]+>", "", inner).strip()))
        return f"<h{lvl}{attrs}>{prefix}{inner}</h{lvl}>"

    return re.sub(r"<h([23])([^>]*)>(.*?)</h\1>", rep, html, flags=re.S)


# ── 6. the pipeline ──────────────────────────────────────────────────────
def _to_html(text, doc):
    parts = _split_directives(text)
    chunks = []
    for p in parts:
        if p[0] == "text":
            chunks.append(p[1])
        else:
            _, dtype, dargs, inner = p
            html = _render_directive(dtype, dargs, inner, doc)
            chunks.append("\n\n" + doc.put_blok(html) + "\n\n")
    text = "".join(chunks)

    text = _stash_raw(text, doc)
    text = _stash_code(text, doc)
    text = _stash_math(text, doc)
    text = _restore_code(text, doc)

    html = _md.markdown(text, extensions=_MD_EXT, output_format="html5")

    # blocks alone in a <p> must break out of it
    for i in range(len(doc.blok)):
        tok = BLOK_T.format(i)
        html = re.sub(r"<p>\s*" + re.escape(tok) + r"\s*</p>", lambda _m, i=i: doc.blok[i], html)
        html = html.replace(tok, doc.blok[i])
    return html


def _wrap_tables(html):
    """Give every table a horizontal-scroll wrapper, exactly once."""
    def rep(m):
        return '<div class="tablewrap">' + m.group(0) + "</div>"
    return re.sub(r"<table\b[^>]*>.*?</table>", rep, html, flags=re.S)


# ── computed numbers ─────────────────────────────────────────────────────
# Every derived number in the prose is substituted from the week's numbers.py at
# build time, so it cannot drift from the computation that produced it. Typing a
# number by hand is how four wrong values reached a draft of Week 1.
NUMBER_TOKEN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]+))?\}\}")


def substitute_numbers(text, values, missing):
    """Replace {{key}} / {{key:fmt}} with computed values. Records unknown keys."""
    if values is None:
        values = {}

    def rep(m):
        key, fmt = m.group(1), m.group(2)
        if key not in values:
            missing.append(key)
            return f"<mark>?{key}?</mark>"
        v = values[key]
        try:
            return format(v, fmt) if fmt else str(v)
        except (ValueError, TypeError):
            missing.append(f"{key} (bad format {fmt!r})")
            return f"<mark>?{key}?</mark>"

    return NUMBER_TOKEN.sub(rep, text)


# ── front matter ─────────────────────────────────────────────────────────
def split_front_matter(raw):
    """Parse a leading `---` block of `key: value` lines. Lists via `[a, b]`."""
    meta = {}
    if not raw.startswith("---"):
        return meta, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return meta, raw
    head, body = raw[3:end], raw[end + 4:]
    for line in head.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            v = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        elif v.lower() in ("true", "false"):
            v = v.lower() == "true"
        else:
            v = v.strip("'\"")
        meta[k.strip()] = v
    return meta, body.lstrip("\n")


def render(raw, *, number_sections=True, numbers=None, prefix=""):
    """Markdown source -> (html, meta, doc). `doc` carries the TOC and counters."""
    meta, body = split_front_matter(raw)
    doc = Doc(prefix=prefix)
    doc.missing_numbers = []
    body = substitute_numbers(body, numbers, doc.missing_numbers)
    if "number_sections" in meta:
        number_sections = bool(meta["number_sections"])
    html = _to_html(body, doc)
    html = _restore_raw(html, doc)
    html = _restore_math(html, doc)
    html = _resolve_refs(html, doc)
    html = _number_headings(html, doc, number_sections)
    html = _wrap_tables(html)
    return html, meta, doc
