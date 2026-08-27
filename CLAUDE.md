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
| `papers/` | arXiv e-prints, licences vary per paper. Check `metadata.json` and the DOI before leaning on one. | Read and cite freely. Adapting requires checking the individual licence, which is usually not CC BY. |
| `papers/2001.07203/` | **CC BY 4.0** for the version of record, confirmed against Crossref for doi:10.1016/j.jmp.2020.102447 | The one permissive source we have. Adaptation is licensed, and CC BY 4.0 is compatible with this course's own licence. Attribution is still required. Even so, prefer our own derivations: that is the premise of the course, not a licence constraint. Spine of Weeks 9 to 11. |

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

## The verification pipeline

```bash
python3 build/check_all.py            # everything offline, ~2 seconds
python3 build/check_all.py --full     # + DOI resolution and notebook execution
```

**Run it before every commit; run `--full` before publishing.** Each stage exists
because something specific went wrong, and the reasons are in each script's
docstring so nobody deletes a check without knowing what it was for.

| Stage | Catches | Why it exists |
|---|---|---|
| `build_site.py` | — | renders `content/` to `docs/`, substituting computed numbers |
| `check_site.py` | dead `[eq:]` refs, unregistered widgets, dead links, tag imbalance, stray `$` | a widget silently failed to mount; a figure was capped at the wrong width |
| `check_values.py` | any derived number typed by hand | **four wrong values reached a draft of Week 1** because they were worked out by hand |
| `check_definitions.py` | a symbol used before it is defined; a term bolded as a definition twice | a reader caught `$o$` called "sensory states" one lesson before that distinction existed |
| `check_outline.py` | a week named but not planned | one probe found a topic mislabelled, uncommitted and uncross-linked |
| `check_prose.py` | AI-writing tells, per zone, per 1,000 words | wraps the calibrated census at `~/.claude/skills/writing-review/census.py` |
| `check_references.py` | a DOI that does not resolve or whose metadata drifted | two citations were wrong from memory, including an author list |
| notebooks | a notebook that no longer runs | the published Colab links are only as good as the last execution |

Blocking stages fail the run. Advisory stages (`outline`, `prose`, `definitions`)
print a queue to read, never a verdict to obey: this course teaches distinctions,
so a high keep-rate on the "X, not Y" family is correct, not a defect.

### Numbers are computed, never typed

A week's derived numbers live in `content/week-NN/values.py`, which exports a
`VALUES` dict. The prose refers to them as `{{key:.4f}}` and `build_site.py`
substitutes at render time, so the published figure **is** the computed figure and
cannot drift from it. Week 1 has 148 such substitutions.

`check_values.py` scans for any decimal still typed literally and reports it.
Numbers that are genuinely *inputs* rather than results (a prior we chose, a
matrix entry that defines a model, a cited physiological range) go in its `ALLOW`
list with a reason; an entry there is a claim that a human checked it.

Add a number to a lesson by adding it to `values.py` first. If you find yourself
typing a digit into prose, stop.

### These checks have all failed in both directions

Do not treat a green pipeline as proof. Recorded so far:

- `check_outline.py` passed a week whose citation had been wired in without anyone
  reading the paper, and separately flagged the week containing the best derivation
  in the course as having none, because the bullet said "main result" rather than
  "Derive".
- `check_definitions.py` stripped a trailing full stop *before* testing for one, so
  its own run-in-heading exemption never fired.
- `check_values.py` initially reported 73 failures that were all SVG coordinates.
- `check_prose.py` reported every equation as a punctuation failure, because KaTeX
  renders client-side and the static HTML still holds raw LaTeX minus signs.
- `check_all.py` misattributed every section, because the parent's buffered headers
  landed after each child's output.
- `check_definitions.py` read only `\macro` tokens when scanning for symbols spent
  before their reserved week, so any bare letter could collide in silence: $g$ was
  reserved for Week 5 and used in Week 2 with nothing to say so.
- Taught to read bare letters, it then filed every *argument* of a notation row as a
  reservation, so `$Q(s)$` looked like a definition of $s$ and the report drowned.
- Taught to take only a row's head symbol, it still skipped every row whose
  description had no `Week N` in it, which was 33 of them, because the page is
  sectioned by week and the rows say "Lesson 1". $\Pi$ hid there.
- Stripping `\mathbf` and `\mathcal` to find the letter underneath made
  $\mathbf{D}$, $\mathcal{D}$ and $D_{\mathrm{KL}}$ one symbol, and it reported
  collisions between three things no reader would confuse.

Read the reports. The exit code is a prompt to look, not an answer.

### A green pipeline is not a rendered page

Week 2's central figure, the predictive coding circuit, read $g'(d)\cdot\Pi€_u$ for
a working day. Entity `&#8364;` where `&#7524;` was meant. Tag balance passed, SVG
integrity passed, maths checking passed, because none of them look at what the glyphs
say. Reading `svg text` content in the browser found it in one call.

The same page's widget note claimed a skew "drops to zero" where the running widget
showed 0.191, because the note had been written from the algebra rather than from the
thing the reader will actually see.

**So: before committing a page with a figure or a widget, open it and read what it
says.** Not a screenshot glance. Pull the text content, drive the controls, and check
the readout against the Python. Three independent implementations of Week 2 (the
`values.py` module, the widget's JavaScript, and the notebook) agree to four figures,
and that agreement is worth more than any single check, because the ways each could
be wrong are not the same ways.

### A hand-drawn curve is a typed number in disguise

Week 2's first figure was hand-authored SVG. It put the posterior mode at x=252 where
the computation puts it at 209.8, and the mean at 316 against 262: a picture
disagreeing with the prose beside it, in a project whose whole numbers discipline
exists to stop exactly that. Path data is now emitted from the same arrays the
statistics come from, via `_svg_path` in `content/week-02/values.py`, and cited as
`{{fig_post_path}}`. Schematics with no data in them (the circuit diagram) may still
be drawn by hand.

### What the collision check cannot see

It catches a symbol **spent early**: used in Week $N$ when `notation.md` files it
under a later week. It does not catch a symbol **spent twice**, because that
needs meaning and it only has letters.

Every one of these got through it, and every one is real:

- $\mu$ for a generic Gaussian mean in Week 2, when Week 1 owns it for the
  agent's internal states.
- $a$ for a unit's activity, when Week 1 owns it for actions.
- $w$ for a synaptic weight, against Week 1's noisy-OR probability.
- $z$ for an out-of-reach quantity, against the sensory noise of Week 5.
- $\alpha$ for a learning rate, against Week 1's channel ambiguity.

Four of the five were written in a single edit, by the author of the check, ten
minutes after extending it. **The check licenses nothing. Before spending a
letter, grep `notation.md` and Week 1 for it.**

Two related traps in the notation page itself, both found by reading rather than
by running anything: a row that mentions a later week in passing was refiled
under that week until `home` became the earliest of all evidence rather than the
first match; and $\sigma$ still meant softmax in one row long after the rename
that was supposed to remove it, because it sat inside another row's description
where no head-symbol scan reaches.

### A deck lost its opening quotation mark for a week

`split_front_matter` stripped quote characters off both ends of every value with
a bare `.strip("'\"")`. A deck that *began* with a quoted phrase lost the opening
mark and kept the closing one, so the published page read
`Exact inference is intractable" is four claims`. Balanced HTML, valid maths,
correct numbers, and a stray quotation mark in the subtitle of the page.

Nothing could have caught it: the checkers see the rendered string and have no
idea what it was supposed to be. It was found by dumping every title and deck
into a list to review the wording, which is the same act that finds bad headings.
Quotes are now stripped only as a matched pair.

### The builder has no citation syntax, and nothing said so

Four `[@key]` citations reached a built page verbatim, carried over from Pandoc
habit. Every existing check passed them: a leaked `[@key]` is balanced,
well-formed, contains no maths and breaks no tag. `check_site.py` now has an
`UNSUPPORTED` list for markup some other tool would resolve and this one will
not, and it is verified by planting three kinds and confirming all three fire.

Cite as REFERENCES.md does: the author and year as link text, the DOI as the
href.

### Check the check by breaking it

`check_definitions.py` reported zero collisions after being rewritten, which is also
what a broken check reports. Planting `\gamma`, `\mathbf{C}`, `\tilde{s}` and
`\mathcal{D}` in Week 2 and confirming all four fire is what made the zero mean
something. The first run of that test said `\tilde{s}` passed; the cause was
`printf` turning the `\t` into a tab, not the checker. Verify the harness too.

## Authoring rules

### Maths
Display equations are numbered **only if labelled**: `$$ ... $$ {#name}`, referenced
as `[eq:name]`. Numbering every intermediate line of a derivation is noise.
Never write maths inside a code fence expecting it to render; fences are excluded
from maths scanning by design.

### A float cannot carry twenty digits, and will not say so

`V["x"] = 1.4011053080505894180` stores a double, which holds about 17
significant digits. Formatting it with `{{x:.20f}}` does not fail or warn: it
prints the double's own expansion, so the page published
`1.40110530805058952630` for a number whose true value is
`1.4011053080505894180`. Three invented digits, and every check green, because
the number matched its key exactly and the key held what it had been given.

Anything quoted beyond about 15 digits must be carried as `Decimal` or as text
from the source to the page. High-precision results out of Mathematica are the
usual case.

### Cross-references are directives, never paths

Point at another lesson with `[lesson:1.5]`, and at a week with `[week:4]`.
`build_site.py` resolves them against the real lesson list, so a renamed file
fails the build rather than rotting into a dead link. Never hand-write
`../week-01/the-denominator.html`.

- **Link text is automatic**: "Lesson&nbsp;5" when the target is in the citing
  page's own week, "Week&nbsp;1, Lesson&nbsp;5" when it is not. Override with a
  pipe, `[lesson:1.5|the denominator lesson]`, and put no directive inside that
  label. A blanket regex over already-edited text nested one inside another and
  produced `[lesson:1.5|Week 1's [lesson:2.5]]`, which markdown then tore in half.
- **Mark every occurrence.** Only the first reference to a given target on a page
  renders as a link; the rest come out as plain text. Week&nbsp;1's Lesson&nbsp;2
  refers back to Lesson&nbsp;1 six times, and six identical links is a rash, not
  navigation. Thinning them at build time means the rule survives a paragraph
  being moved.
- A lesson never links to itself, and a week with no lessons yet renders as plain
  text rather than a link to an empty index.
- `check_site.py` fails on any `[lesson:` or `[week:` left in the output.

**When you add a syntax, tell the other checks about it.** `[lesson:1.5]`
contains `1.5`, so `check_numbers.py` read 61 cross-references as hand-typed
decimals the moment the syntax existed. A report that floods is a report nobody
reads.

### Numbers are regenerated, never typed
Every numerical claim in prose must be computed and pasted from an actual run.
Four wrong values reached a draft of Week 1 by being worked out by hand. Before
writing any number, run it. Before writing "always", "never" or "only", enumerate
the set the claim covers.

### Show the working: this is the whole point of the course

Audited 2026-08-26. Week 1 contained **zero `::: derivation` blocks and zero
`::: definition` blocks** across five lessons. A course whose stated premise is
that the textbooks compress a page of algebra into "it can be shown that" had, in
its first week, not one worked derivation and not one numbered definition. The
machinery existed and had never been used.

Rules:
- **If a claim is not obvious, show it.** Not a sketch, not "one can check that":
  the intermediate lines, with a sentence after any step whose justification is not
  visible from the line itself. This is the product, not the packaging.
- **Length is not a cost here.** A lesson that doubles in length because it now
  derives what it previously asserted has improved. Do not compress to hit a
  reading-time estimate; change the estimate.
- **Number the definitions.** A term defined only in running prose cannot be
  referred back to. `::: definition` numbers by section, as figures and equations do.
- **Do not hide a derivation behind a disclosure widget.** Making the reader click
  for the algebra is a softer version of the sin the course exists to correct. The
  `::: derivation` block marks it visually; that is enough.

### Two kinds of audit, and they need different briefs

A reader agent returns what its brief asks for. Briefing one to find *errors*
produced a list of errors, and none of the pedagogical thinness that was actually
wanted. Run both, and keep them separate:

**Correctness audit.** "What is wrong, undefined, contradictory or unsupported?"
Catches wrong numbers, symbols used before definition, contradictions between
pages. Found the mutual-information value quoted against the wrong model and the
forward/reverse KL mix-up.

**Depth audit.** "Where could you not close the page and reproduce this from the
definitions, and why not?" Categories: compressed step, asserted not shown,
missing intuition, needs a worked example, needs a second example, needs a
picture, needs a counterexample or limit, wants a check, too fast. Tell the agent
explicitly that **length is not a constraint** and that recommending three more
paragraphs is a good recommendation, or it will default to being economical and
report nothing.

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
- **Headings name the section; they do not perform.** Flagged twice now, so it
  is a rule rather than a correction. The failures found, in order of how often
  they recurred:
  - **An explainer appended to the name.** "Explaining away, and why it ruins the
    obvious shortcut" is a heading plus a defence of the heading. Six of these.
    The name alone was better every time: "Explaining away".
  - **Chattiness.** "Nothing else can do it either" belongs in speech, not a
    contents list.
  - **Signposting.** "An aside on brains and sea squirts" → "Brains and sea
    squirts". The reader can see it is an aside.
  - **The second person.** "What the ambiguity slider is showing you" → "shows".
  - **A fragment.** "How large a step" is not a phrase.
  - **Repeating the lesson's own title.** A section called "Climbing" inside a
    lesson called "Climbing the log joint".
  - **An ordinal with nothing to count from.** "The fourth reason is locality",
    where sections one to three were never numbered.

  Declarative claims are good headings when the claim is the section's point:
  "Drift is the default", "Structure is the reason", "Surprise is not a property
  of the world". A negation stays when the negation *is* the finding.

  **Judge them extracted, never in place.** In situ every heading looks
  reasonable, because the paragraph beneath it has already explained it. Pull
  every title, deck and heading into one list and read it top to bottom; the
  patterns are invisible until they are adjacent.
- Run the `writing-review` skill on anything substantial before publishing.

## Code streams

NumPy is canonical and every week has it. `pymdp` appears from Week 9 only, where
it actually applies; it has nothing to say about Weeks 5 to 8 and must not be
padded in. JAX is an advanced stream, present only where vectorising buys
something real. Where a variant would be a wrapper around nothing, omit it and
say so.

## Current state

Weeks 1 and 2 are built. Week 1 has six lessons since 2026-08-27, the new one
being **What makes the denominator hard**: the standard "the sum is intractable"
is four separate claims, and the lesson separates them, because the reason
usually given is the weakest of the four. A sum of $4^{40}$ terms is computed
exactly in 624 operations when the model is a chain, so the count proves almost
nothing; treewidth is what sits in the exponent; the complexity results rule out
cleverer algorithms rather than just slower ones; the continuous case fails for
an unrelated analytic reason visible in one variable; and locality is a fifth
obstacle that applies to bodies rather than computers. Weeks 3 to 12 are outlined in `OUTLINE.md` and not yet
drafted. `OUTLINE.md` names, per week, which of Jonathan's own notes seed it.

Week 2 derives predictive coding from gradient ascent on the log joint, so the
circuit arrives in Week 2 and variational free energy not until Week 4. It carries
two widgets (`ascent-errors`, `precision-posterior`) and two notebooks; the JAX one
is not a translation of the NumPy one but does what needs many runs at once.

Known outstanding, in order:
1. Read Da Costa et al. (2020), arXiv:2001.07203, before drafting Weeks 9 to 11.
   Downloaded 2026-08-27 and sitting in `source-material/papers/2001.07203/`, PDF
   and LaTeX source both, but **not yet read**. The outline calls it the spine of
   those three weeks, and Week 9 went wrong once already by being built from a
   table of contents rather than a source.
2. Week 1's exercises have gaps: no exercise computes an entropy, mutual information
   is never independently exercised, the regulated $\kappa > 0$ case is tested only
   by the widget, there is no rung-zero warm-up, and the reader is never asked to
   implement anything. Week 2's problem 2 is the first that asks for code.
3. A glossary built from `::: notation` blocks.
4. Whether Weeks 6 to 8 need a shared worked continuous example.
