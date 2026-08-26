#!/usr/bin/env python3
"""build_site.py — render content/ into docs/ for GitHub Pages.

    python3 build/build_site.py            # whole site
    python3 build/build_site.py week-01     # one week (still rewrites nav)

content/course.json is the single source of truth for the week list, the part
structure and the ordering. Lesson pages are discovered by scanning
content/week-NN/*.md; their order comes from the numeric filename prefix.
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import mdx      # noqa: E402
import theme    # noqa: E402

CONTENT = os.path.join(ROOT, "content")
OUT = os.path.join(ROOT, "docs")
ASSETS = os.path.join(ROOT, "assets")
COURSE = os.path.join(CONTENT, "course.json")

SITE_TITLE = "Active Inference"
SITE_CODE = "Active Inference"


# ── model ────────────────────────────────────────────────────────────────
class Lesson:
    def __init__(self, week, path, meta, order):
        self.week = week
        self.path = path
        self.meta = meta
        self.order = order
        base = os.path.basename(path)[:-3]
        self.slug = re.sub(r"^\d+[-_]", "", base)
        self.title = meta.get("title") or self.slug.replace("-", " ").capitalize()
        self.deck = meta.get("deck", "")
        self.href = f"week-{week:02d}/{self.slug}.html"

    def __repr__(self):
        return f"<Lesson {self.week}.{self.order} {self.slug}>"


class Week:
    def __init__(self, d):
        self.n = int(d["n"])
        self.title = d["title"]
        self.part = d.get("part", "")
        self.summary = d.get("summary", "")
        self.status = d.get("status", "planned")
        self.lessons = []

    @property
    def href(self):
        return f"week-{self.n:02d}/index.html"

    @property
    def dirname(self):
        return f"week-{self.n:02d}"


def load_course():
    with open(COURSE, encoding="utf-8") as f:
        data = json.load(f)
    weeks = [Week(w) for w in data["weeks"]]
    by_n = {w.n: w for w in weeks}
    for w in weeks:
        d = os.path.join(CONTENT, w.dirname)
        if not os.path.isdir(d):
            continue
        files = sorted(x for x in os.listdir(d) if x.endswith(".md"))
        for i, fn in enumerate(files):
            raw = open(os.path.join(d, fn), encoding="utf-8").read()
            meta, _ = mdx.split_front_matter(raw)
            by_n[w.n].lessons.append(Lesson(w.n, os.path.join(d, fn), meta, i + 1))
    return data, weeks


# ── chrome ───────────────────────────────────────────────────────────────
def topbar(root, label="Contents"):
    return (
        '<div class="topbar"><div class="tin">'
        f'<a class="home" href="{root}index.html">&#8592; {SITE_TITLE}</a>'
        f'<button class="tocbtn" type="button">{label}</button>'
        f'<span class="tcode">Open course</span>'
        "</div></div>"
    )


def sidenav(weeks, root, cur_week=None, cur_href=None, page_toc=None):
    parts, seen = [], None
    parts.append('<nav class="sidenav"><div class="navhd">Contents</div>')
    for w in weeks:
        if w.part != seen:
            seen = w.part
            parts.append(f'<div class="part">{w.part}</div>')
        cur = " cur" if cur_week == w.n else ""
        parts.append(
            f'<a class="wk{cur}" href="{root}{w.href}">'
            f'<span class="n">{w.n:02d}</span>{w.title}</a>'
        )
        if cur_week == w.n and w.lessons:
            parts.append('<div class="lessons">')
            for ls in w.lessons:
                lc = " cur" if cur_href == ls.href else ""
                parts.append(f'<a class="{lc.strip()}" href="{root}{ls.href}">{ls.title}</a>')
            parts.append("</div>")
    parts.append(
        f'<div class="part">Reference</div>'
        f'<a class="wk" href="{root}notation.html"><span class="n">&#167;</span>Notation</a>'
        f'<a class="wk" href="{root}orientation.html"><span class="n">&#167;</span>How to use this course</a>'
    )
    parts.append("</nav>")
    return "".join(parts)


FOOT = (
    '<footer class="sitefoot">'
    "<p>Active Inference, an open course by "
    '<a href="https://www.shocklab.net">Jonathan Shock</a>, '
    "Department of Mathematics &amp; Applied Mathematics, University of Cape Town.</p>"
    '<p>Released under <a href="https://creativecommons.org/licenses/by/4.0/">CC&nbsp;BY&nbsp;4.0</a>. '
    "Derivations, figures, code and examples are the author&rsquo;s own. "
    'Corrections and contributions welcome on <a href="https://github.com/shocklab/Active-Inference-course">GitHub</a>.</p>'
    "</footer>"
)


def pagenav(prev, nxt, root):
    if not prev and not nxt:
        return ""
    bits = ['<nav class="pagenav">']
    if prev:
        bits.append(
            f'<a class="pv" href="{root}{prev.href}"><span class="dir">Previous</span>'
            f'<span class="ttl">{prev.title}</span></a>'
        )
    if nxt:
        bits.append(
            f'<a class="nx" href="{root}{nxt.href}"><span class="dir">Next</span>'
            f'<span class="ttl">{nxt.title}</span></a>'
        )
    bits.append("</nav>")
    return "".join(bits)


# ── page renderers ───────────────────────────────────────────────────────
def render_lesson(lesson, week, weeks, flat):
    raw = open(lesson.path, encoding="utf-8").read()
    html, meta, doc = mdx.render(raw)
    if doc.pending_refs:
        missing = sorted(set(doc.pending_refs) - set(doc.labels))
        if missing:
            print(f"    ! unresolved [eq:...] refs in {lesson.slug}: {', '.join(missing)}")

    root = "../"
    i = flat.index(lesson)
    prev = flat[i - 1] if i > 0 else None
    nxt = flat[i + 1] if i + 1 < len(flat) else None

    meta_bits = [f"Week {week.n}", f"&#167;{lesson.order}"]
    if meta.get("time"):
        meta_bits.append(meta["time"])
    deck = f'<p class="deck">{lesson.deck}</p>' if lesson.deck else ""

    body = (
        topbar(root)
        + '<div class="shell">'
        + sidenav(weeks, root, week.n, lesson.href)
        + '<main class="maincol"><header class="ahead">'
        + f'<div class="eyebrow">{week.title}</div>'
        + f"<h1>{lesson.title}</h1>{deck}"
        + '<div class="meta">' + "".join(f"<span>{b}</span>" for b in meta_bits) + "</div>"
        + "</header>"
        + f'<article class="abody">{html}</article>'
        + pagenav(prev, nxt, root)
        + FOOT
        + "</main></div>"
    )

    scripts = ["assets/js/aif-core.js"]
    for s in meta.get("scripts", []) or []:
        scripts.append(f"assets/js/{s}")
    return theme.page_shell(
        title=lesson.title, site_title=SITE_TITLE, body=body, root=root,
        scripts=scripts, description=lesson.deck,
    )


def render_week_index(week, weeks, flat):
    root = "../"
    rows = []
    for ls in week.lessons:
        deck = f"<p>{ls.deck}</p>" if ls.deck else ""
        rows.append(
            '<div class="weekrow"><div class="wn">'
            f'<div class="l">&#167;</div><div class="n">{ls.order}</div></div>'
            f'<div class="wc"><h3><a href="{root}{ls.href}">{ls.title}</a></h3>{deck}</div></div>'
        )
    if not rows:
        rows.append(
            '<div class="box remark"><div class="boxhd">Not yet written</div>'
            "<p>This week is planned but not drafted. The outline lives in "
            "<code>OUTLINE.md</code> in the repository.</p></div>"
        )

    idx = [w for w in weeks if w.lessons or True]
    pos = [w.n for w in idx].index(week.n)
    prevw = idx[pos - 1] if pos > 0 else None
    nextw = idx[pos + 1] if pos + 1 < len(idx) else None

    nav_bits = ['<nav class="pagenav">']
    if prevw:
        nav_bits.append(
            f'<a class="pv" href="{root}{prevw.href}"><span class="dir">Previous week</span>'
            f'<span class="ttl">{prevw.title}</span></a>'
        )
    if nextw:
        nav_bits.append(
            f'<a class="nx" href="{root}{nextw.href}"><span class="dir">Next week</span>'
            f'<span class="ttl">{nextw.title}</span></a>'
        )
    nav_bits.append("</nav>")

    body = (
        topbar(root)
        + '<div class="shell">'
        + sidenav(weeks, root, week.n, None)
        + '<main class="maincol"><header class="ahead">'
        + f'<div class="eyebrow">Week {week.n} &#183; {week.part}</div>'
        + f"<h1>{week.title}</h1>"
        + (f'<p class="deck">{week.summary}</p>' if week.summary else "")
        + "</header>"
        + f'<article class="abody"><div class="weeklist">{"".join(rows)}</div></article>'
        + "".join(nav_bits)
        + FOOT
        + "</main></div>"
    )
    return theme.page_shell(
        title=f"Week {week.n}: {week.title}", site_title=SITE_TITLE, body=body,
        root=root, scripts=["assets/js/aif-core.js"], description=week.summary,
    )


def render_index(data, weeks):
    parts, seen = [], None
    for w in weeks:
        if w.part != seen:
            seen = w.part
            blurb = data.get("parts", {}).get(w.part, "")
            parts.append(f'<div class="partband">{w.part}<span>{blurb}</span></div>')
        links = ""
        if w.lessons:
            links = (
                "<ul>"
                + "".join(f'<li><a href="{ls.href}">{ls.title}</a></li>' for ls in w.lessons)
                + "</ul>"
            )
        else:
            links = '<ul><li style="color:#90909a">In preparation</li></ul>'
        parts.append(
            '<div class="weekrow"><div class="wn"><div class="l">Week</div>'
            f'<div class="n">{w.n}</div></div>'
            f'<div class="wc"><h3><a href="{w.href}">{w.title}</a></h3>'
            f"<p>{w.summary}</p>{links}</div></div>"
        )

    intro = ""
    idx_md = os.path.join(CONTENT, "index.md")
    if os.path.exists(idx_md):
        intro, _, _ = mdx.render(open(idx_md, encoding="utf-8").read(), number_sections=False)
        intro = f'<div class="introcopy">{intro}</div>'
    body = (
        '<div class="cover"><div class="cin">'
        f'<div class="eyebrow">{data.get("eyebrow", "Open course")}</div>'
        '<div class="rule"></div>'
        f'<h1>{data["title"]}</h1>'
        f'<p class="sub">{data["subtitle"]}</p>'
        "</div></div>"
        '<div class="wide"><article class="abody" style="max-width:none;padding-top:2.4rem">'
        f"{intro}"
        f'<div class="weeklist" style="margin-top:1.5rem">{"".join(parts)}</div>'
        f"{FOOT}"
        "</article></div>"
    )
    return theme.page_shell(
        title="Home", site_title=SITE_TITLE, body=body, root="",
        scripts=["assets/js/aif-core.js"], description=data["subtitle"],
        body_class="landing",
    )


def render_standalone(name, weeks):
    src = os.path.join(CONTENT, f"{name}.md")
    if not os.path.exists(src):
        return None
    raw = open(src, encoding="utf-8").read()
    html, meta, doc = mdx.render(raw)
    root = ""
    body = (
        topbar(root)
        + '<div class="shell">'
        + sidenav(weeks, root)
        + '<main class="maincol"><header class="ahead">'
        + f'<div class="eyebrow">{meta.get("eyebrow", "Reference")}</div>'
        + f'<h1>{meta.get("title", name)}</h1>'
        + (f'<p class="deck">{meta["deck"]}</p>' if meta.get("deck") else "")
        + "</header>"
        + f'<article class="abody">{html}</article>{FOOT}</main></div>'
    )
    scripts = ["assets/js/aif-core.js"] + [f"assets/js/{s}" for s in (meta.get("scripts") or [])]
    return theme.page_shell(
        title=meta.get("title", name), site_title=SITE_TITLE, body=body, root=root,
        scripts=scripts, description=meta.get("deck", ""),
    )


# ── driver ───────────────────────────────────────────────────────────────
def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    data, weeks = load_course()
    flat = [ls for w in weeks for ls in w.lessons]

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, ".nojekyll"), "w").close()

    dst_assets = os.path.join(OUT, "assets")
    if os.path.isdir(dst_assets):
        shutil.rmtree(dst_assets)
    shutil.copytree(ASSETS, dst_assets)

    n = 0
    for w in weeks:
        if only and only != w.dirname:
            continue
        write(os.path.join(OUT, w.href), render_week_index(w, weeks, flat))
        n += 1
        for ls in w.lessons:
            write(os.path.join(OUT, ls.href), render_lesson(ls, w, weeks, flat))
            n += 1

    write(os.path.join(OUT, "index.html"), render_index(data, weeks))
    n += 1
    for name in ("notation", "orientation"):
        page = render_standalone(name, weeks)
        if page:
            write(os.path.join(OUT, f"{name}.html"), page)
            n += 1

    built = sum(1 for w in weeks if w.lessons)
    print(f"built {n} pages  ({built}/{len(weeks)} weeks have content)")


if __name__ == "__main__":
    main()
