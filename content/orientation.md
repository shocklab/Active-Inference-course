---
title: How to use this course
eyebrow: Reference
deck: What is here, what it assumes, how the twelve weeks fit together, and what it deliberately leaves out.
number_sections: false
---

## What this is

A full derivation of active inference, from Bayes' theorem to policy selection,
with the intermediate steps written out. It exists because the standard
references are good on ideas and thin on working, and because the gap between
"I follow the argument" and "I could rederive this" is where most people
stall.

Every result is proved rather than asserted. Every figure is live. Every week
has a Python notebook that rebuilds its results from scratch.

## What it assumes

Multivariable calculus, linear algebra, and probability at roughly the level of
a good undergraduate degree in a mathematical subject. Specifically: you should
be comfortable with expectations and conditional distributions, with gradients
and the chain rule, with matrix algebra, and with reading a derivation that
moves several steps at a time.

It assumes nothing about neuroscience, nothing about machine learning, and
nothing about the free energy principle. Every piece of domain vocabulary is
defined where it is first used.

You do not need to be able to program to follow the arguments. The notebooks
are there to be run and broken, not to be a prerequisite.

## The shape of the argument

This course takes the discrete formulation last. That is the reverse of the
commonest presentation, and the reason is worth understanding before you start.

Parr, Pezzulo and Friston introduce the discrete, partially observable Markov
decision process early, because that is where active inference is most often
applied. The trouble is that the discrete formulation compresses several
distinct ideas into one set of matrices, so a reader meets the expected free
energy before they have a firm grip on the ordinary free energy it generalises.

Here, Part&nbsp;I builds classical inference until the variational bound is
unavoidable. Part&nbsp;II then does continuous time, where free energy under a
Laplace approximation turns into precision-weighted prediction error and the
connection to the ordinary calculus you already know stays visible. Part&nbsp;III
does the discrete case last, by which point the matrices are a repackaging of
things you have already derived rather than a new formalism to absorb.

| Part | Weeks | What gets built |
|---|---|---|
| I. Inference | 1&ndash;4 | Bayes, the inverse problem, point estimation, learning, and the variational free energy |
| II. Continuous time | 5&ndash;8 | Predictive coding, generalised coordinates, generalised filtering, and action |
| III. Discrete time | 9&ndash;11 | POMDPs, expected free energy, policy selection and learning |
| IV. Perspective | 12 | Extensions, the relation to reinforcement learning, and the objections |

## How each week is built

Every week is four or five short lessons and one problems page. The lessons are
meant to be read in order and take twenty to thirty minutes each. The problems
page takes considerably longer and is where the learning happens.

Inside a lesson you will meet a few recurring blocks:

::: definition A worked term
Definitions are numbered and are the things you are expected to be able to state
back precisely.
:::

::: derivation
Derivations show every line. Where a step is not obvious, the reason is given in
prose immediately after it rather than left as an exercise. These are the parts
of the course that exist because the books skip them.
:::

::: keyidea
Marks the two or three sentences per lesson that the rest of the argument leans
on. If you remember nothing else from a page, remember these.
:::

::: warning
Marks a place where the standard treatment is doing something questionable, or
where an assumption is quietly carrying weight. These accumulate, and Week&nbsp;12
collects them.
:::

::: exercise A problem with its solution folded away
Exercises appear inline as well as on the problems page. The solution is always
there, one click away.
---solution---
Open it after you have tried, not before. The solutions are written out in full,
including the arithmetic, so you can find the exact step where you diverged.
:::

Margin notes carry asides, alternative notation, and pointers forwards and
backwards. Nothing essential is ever only in a margin note.

## The code

Three streams, and they do not all run every week.

**NumPy** is canonical. Every week has a NumPy notebook, everything is built
from primitives, and nothing is hidden inside a library call. This is the one to
work through if you are only going to do one.

**pymdp** appears from Week&nbsp;9. It is the standard Python library for
discrete active inference, and once we have built the discrete machinery
ourselves it is worth seeing what the community actually uses, and worth
checking our implementation against it. It has nothing to say about
Weeks&nbsp;5 to 8, so it does not appear there.

**JAX** is an advanced stream, present where vectorisation buys something real:
evaluating many policies at once, learning across thousands of trials, running
a simulation ten thousand times to get a distribution rather than a trajectory.
Where a JAX version would be a transliteration of the NumPy one, it is omitted.

## What this course does not do

It does not cover Bayesian mechanics, the physics-of-self-organisation programme
that grew out of the same work. That is a different subject with a different
mathematical apparatus and it deserves its own course.

It does not survey applications. There is a large literature applying active
inference to psychiatry, robotics, and cognitive science; Week&nbsp;12 points at
it but does not attempt a review.

It does not tell you that active inference is correct. The framework makes
strong claims, some of which are contested by serious people on serious grounds.
Week&nbsp;12 puts the main objections properly rather than as a footnote, and
several lessons flag assumptions as they are made. You should finish the course
able to state both the case for the framework and the strongest case against it.

## Licence and corrections

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Use it,
adapt it, teach from it; attribution is all that is asked.

Corrections are genuinely welcome, particularly to the derivations. If a step
does not follow, that is a defect in the course, not in the reader. Open an
issue on [GitHub](https://github.com/shocklab/Active-Inference-course).
