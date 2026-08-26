# Active Inference

A twelve-week open course that does the algebra the textbooks leave out, and lets
you turn every knob.

**Read it at [shocklab.github.io/active-inference-course](https://shocklab.github.io/active-inference-course/)**

Active inference is a principled account of how a thing that persists must model
the world it persists in, and how perception and action turn out to be the same
operation applied to opposite sides of one boundary. The ideas are well covered
elsewhere. The working is not: steps get skipped, indices go unexplained, and
derivations that need a page get compressed into "it can be shown that".

This course does the working. Every result is derived in full, every figure is
interactive, and every week ships a Python notebook.

## What's here

| Part | Weeks | |
|---|---|---|
| I. Inference | 1–4 | Bayes, the inverse problem, point estimation, learning, and variational free energy |
| II. Continuous time | 5–8 | Predictive coding, generalised coordinates, generalised filtering, action |
| III. Discrete time | 9–11 | POMDPs, expected free energy, policy selection and learning |
| IV. Perspective | 12 | Extensions, the relation to reinforcement learning, and the objections |

Prerequisites: multivariable calculus, linear algebra and probability at roughly
final-year undergraduate level. No neuroscience or machine learning assumed.

## Building it

```bash
python3 build/build_site.py
```

Renders `content/` into `docs/`, which GitHub Pages serves. `content/course.json`
holds the week structure; lessons are Markdown with LaTeX and a few block
directives, documented in `build/mdx.py`.

## Contributing

Corrections are genuinely welcome, especially to the derivations. If a step does
not follow, that is a defect in the course, not in the reader. Open an issue.

## Licence

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Shock, J. (2026). *Active Inference* [Open course]. University of Cape Town.
