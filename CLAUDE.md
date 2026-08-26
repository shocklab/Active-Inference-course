# Active Inference — open course

## What this is

A twelve-week open course deriving active inference in full, by Jonathan Shock,
Department of Mathematics & Applied Mathematics, University of Cape Town.

- **Repo:** https://github.com/shocklab/Active-Inference-course
- **Pages:** https://shocklab.github.io/Active-Inference-course/ (served from `/docs` on `main`)
- **Licence:** CC BY 4.0. Authorised under clauses 8.2 and 9.2.1 of the UCT
  Intellectual Property Policy (2011). Suggested citation:
  `Shock, J. (2026). Active Inference [Open course]. University of Cape Town.`
- **Audience:** postgraduate mathematics / applied mathematics. Assume multivariable
  calculus, linear algebra and probability. Assume nothing about neuroscience or ML.
- **Reason it exists:** the standard texts are good on ideas and thin on working.
  This course does the algebra they skip. If a step does not follow, that is a
  defect in the course.

## Source material — READ THIS FIRST

`source-material/` is **gitignored and must never be published**.

| File | Licence | What we may do |
|---|---|---|
| `namjoshi-2024-...pdf` | **Private draft of a copyrighted MIT Press book.** Strictest. | Orientation only. Do not quote, paraphrase closely, reproduce examples, or mirror its derivation structure. Its broad ordering idea (inference → continuous → discrete) is a general pedagogical choice and is justified on our own grounds in `content/orientation.md`. Nothing else leaves the folder. Recommend it in further reading. |
| `smith-2022-...pdf` | **CC BY-NC-ND 4.0** (verified in the PDF) | Read, cite, link, brief quotation with attribution under fair dealing. |
| `shock-notes-latex/` | Jonathan's own | **Free to use in full.** Seeds Weeks 4 and 10. |
| `shock-active-inference-notebook.nb` | Jonathan's own | **Free to use in full.** See `notes/mathematica-notebook-diagnosis.md`. |

Parr, Pezzulo & Friston (MIT Press, 2022) is also **CC BY-NC-ND 4.0**, not permissive.

**What NoDerivatives means here.** Citation is not licensing, so citing, linking,
referring to equation numbers, and brief quotation with attribution are all fine
and unaffected. Mathematical results are facts and are not copyrightable;
explaining them in our own words is fine. What is not fine: reproducing figures,
tables, code or substantial passages, and adapting any of it into this course,
because a CC BY-NC-ND work cannot be relicensed into our CC BY 4.0.

**Everything published must be our own derivations, examples and figures.** That
is the premise of the course, not merely a licence constraint.

## Outline discipline — why Week 9 went wrong

Audited 2026-08-26 after one probe ("where is variational message passing?")
found the topic mislabelled, uncommitted and uncross-linked.

**Root cause: outline quality tracked whether the source was read or only its
table of contents.** Weeks 4 and 10, seeded by Jonathan's own notes read line by
line, named 6 and 4 sources and 3 derivation targets each. Weeks 5, 7, 8, 9 and
11, built from chapter headings, named zero sources and zero derivation targets.
A bullet made of a heading looks like a plan and commits to nothing.

Three failure modes it produced, all three of which hit message passing:
1. **Topic named, nothing committed to** — no derivation target, no source, no real widget.
2. **Term placed by association** — right family of words, wrong member. "Variational
   message passing" was written into Week 9 because Namjoshi has a section with that
   title; the scheme that actually belongs there is *marginal* message passing.
3. **Missing cross-links** — connections real in the mathematics, absent from the plan.

**Rules that follow.** Every week in `OUTLINE.md` must name (i) a source we have
actually read, (ii) at least one thing to be *derived*, not merely covered,
(iii) concrete widget specs, (iv) a backward and a forward cross-link.
`build/check_outline.py` enforces this mechanically, but it is a proxy: it
measures whether a bullet looks grounded, not whether it is. The real check is
reading the source.

## Build

```
python3 build/build_site.py            # whole site
python3 build/build_site.py week-01    # one week (nav is still rewritten)
```

Local preview: `.claude/launch.json` defines `aif-site` on port 8811.

```
build/theme.py       design system: CSS, fonts, page shell. One :root block controls the palette.
build/mdx.py         Markdown + LaTeX + block directives. See its docstring for the syntax.
build/build_site.py  walks content/, reads course.json, writes docs/.
content/course.json  single source of truth for weeks, parts and ordering.
content/week-NN/*.md lessons, ordered by numeric filename prefix.
assets/js/aif-core.js widget substrate: Plot, sliders, maths helpers, mounting.
assets/js/wNN.js      per-week widgets, loaded via a lesson's `scripts:` front matter.
```

**Never edit `docs/` by hand.** It is generated. Edit `content/` and rebuild.

## Authoring rules

### Maths
Display equations are numbered **only if labelled**: `$$ ... $$ {#name}`, referenced
as `[eq:name]`. Numbering every intermediate line of a derivation is noise.
Never write maths inside a code fence expecting it to render; fences are excluded
from maths scanning by design.

### Numbers are regenerated, never typed
Every numerical claim in prose must be computed and pasted from an actual run.
Four wrong values reached a draft of Week 1 by being worked out by hand. Before
writing any number, run it. Before writing "always", "never" or "only", enumerate
the set the claim covers.

### Define everything at first use, in reading order

A reader caught Lesson 1 writing "sensory states" while naming them $o$, one
lesson before the states/observations distinction existed. Auditing for the class
rather than the instance found five more: "nats" used from Lesson 1 and never
defined anywhere, $t$ and $T$ unnamed, the square brackets in $\mathrm{H}[\cdot]$
unexplained, $s$ used in a margin note before its definition, and $\pi$ spent on
a prior when it is the policy symbol from Week 9.

Rules:
- **Introduce the symbol with the word.** If a letter is chosen for a reason, give
  the reason ($o$ for observation). A reader who cannot see why the letter is that
  letter has to memorise instead of understand.
- **One name per thing per stretch of the course.** "Sensory states" and
  "observations" are both standard; using both before saying they are the same is
  not. Where a second name is genuinely needed (the Markov blanket partition in
  Lesson 2), introduce it explicitly as a second name.
- **Nothing is defined only in `notation.md`.** That page is a reference, not part
  of the reading path. Units, conventions and symbols must be defined where they
  are first read.
- **Nothing is defined only in a widget.** Slider labels live in JavaScript and a
  reader of the prose never sees them.
- **Do not spend a reserved symbol early.** `notation.md` reserves $\pi$, $\gamma$,
  $\tau$, $\mathbf{A}$–$\mathbf{E}$, $\mathcal{D}$ and $\Pi$ for later weeks.
- **Bold means first definition**, plus table labels and run-in headings. A term
  bolded on two pages is defined twice, so one of them is wrong.

`build/check_definitions.py` reports every symbol's first use in reading order and
flags double-bolded terms. It is fuzzy in both directions and says so; read the
report rather than the exit code. It has already produced one false positive of
its own (worked-example entity names) and contained the same class of bug it
exists to catch (it stripped a trailing full stop before testing for one, so the
run-in-heading exemption never fired).

### Widgets
A widget referenced in content must be registered in a JS file that lesson loads,
or the reader sees a "not loaded" box. Two failure modes already found and fixed,
both worth remembering:
- **Init order.** `Plot.attach()` sizes and draws synchronously, so widget state
  must exist *before* `attach` is called. Build state, then attach.
- **Mount timing.** Deferred scripts run at `readyState === 'interactive'`, so
  the core must mount on `DOMContentLoaded`, never immediately, or it races the
  per-week widget file.
Canvas sizing must not depend on `requestAnimationFrame` alone: a hidden or
throttled tab never fires it and the figure stays blank.

### Layout arithmetic
The column budget must add up: sidebar 14 + gap 2.5 + measure 44 + gap 2.5 +
margin 13 + padding 4 = 80rem = `--shell`. Change one, change the others.
`.abody` carries `max-width: var(--measure)` so floats anchor correctly; figures
and tables opt out with an explicit width. `.abody > *` beats a bare element
selector on specificity, so wide-content rules need `.abody figure`, not `figure`.

## Prose style

The global `human-prose` output style and `~/.claude/writing-tells.md` apply.
Additionally, for this course specifically:

- **British spelling.** No em-dashes, ever, and no en-dash or spaced-hyphen substitutes.
- **Derivations are the product.** Where the books say "it can be shown", show it.
  Never compress a derivation to save space; the space is the point.
- **State the assumption when it is made.** Where the standard treatment is doing
  something questionable, mark it with `::: warning` at that spot. Week 12
  collects them and answers them. Do not save up scepticism for the end.
- **No self-labelling.** Do not tell the reader which of your own points is
  important, honest, or clever. Make the point.
- **No bold for emphasis.** Bold is for first use of a technical term being
  defined, and for table labels. Nothing else.
- Run the `writing-review` skill on anything substantial before publishing.

## Code streams

NumPy is canonical and every week has it. `pymdp` appears from Week 9 only, where
it actually applies; it has nothing to say about Weeks 5 to 8 and must not be
padded in. JAX is an advanced stream, present only where vectorising buys
something real. Where a variant would be a wrapper around nothing, omit it and
say so.

## Current state

Week 1 is built. Weeks 2 to 12 are outlined in `OUTLINE.md` and not yet drafted.
`OUTLINE.md` names, per week, which of Jonathan's own notes seed it.
