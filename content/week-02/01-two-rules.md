---
title: Two rules a brain has to obey
deck: Before asking how an agent should infer, ask what an agent is physically able to compute. The answer narrows the field sharply.
week: 2
time: 20 min
---

Week&nbsp;1 ended with a negative result and three responses to it. The exact
posterior is not available, because evaluating it means summing or integrating
over every state the model admits. Of the three ways out, only one offers a
partial answer at every instant and improves with more time, and that is to give
up on the exact posterior and optimise towards it instead.

That was an argument about mathematics. Before turning it into an algorithm it is
worth asking a different question, and asking it first changes what the rest of
the course looks like. Not *what should an agent compute*, but *what is an agent
physically able to compute at all?*

## The constraints

Two rules, both about locality. Neither is a theorem. They are modelling
assumptions, chosen because they are the minimum that a device built out of cells
and connections could plausibly satisfy, and everything in Weeks&nbsp;2, 3 and 5
is an attempt to meet them.

::: definition Local computation
A unit's output depends only on the signals arriving at it and on the strengths of
the connections carrying them. It cannot read a quantity held somewhere else in
the network.
:::

::: definition Local plasticity
A connection's strength changes using only quantities available at its two ends:
the activity of the unit sending and the activity of the unit receiving.
:::

Stated so flatly they sound almost empty. They are not. Between them they rule
out most of the obvious ways to do inference.

**Normalising is ruled out.** Bayes' theorem divides by $P(o) = \sum_s P(o\mid s)P(s)$.
That sum ranges over every state in the model, so computing it requires a device
that can see all of them at once. No unit obeying the first rule can, which is a
sharper version of Week&nbsp;1's complaint: the trouble is not only that the sum
is long, but that there is nobody to do it.

**Carrying a whole distribution is ruled out**, or at least made expensive. A
distribution over a continuous variable is an infinite object. A unit that
represents it must either restrict it to a family with a few parameters, or store
values at a grid of points and pay for the grid.

**Routing a global error backwards is ruled out.** The obvious way to fit a model
is to compute how wrong its output was and send that error back through the
network, adjusting each connection by its contribution. That is what
backpropagation does. But the error is a global quantity, and delivering it to a
particular synapse means routing information from far away to a place that,
under the second rule, can see only its own two ends.

::: warning These are assumptions, not findings
It is not settled that real nervous systems obey either rule. There is evidence
for local learning at synapses, and there are proposals for how something
functionally like backpropagation might be implemented biologically. The
constraints are adopted here because a framework that respects them is more
interesting than one that does not: it has to earn its results rather than
helping itself to arbitrary computation. Where the course later leans on them,
it will say so.
:::

## Why begin here

Because it turns what follows into a search with a specification.

Without the constraints, the next three weeks would be a series of results about
probability that happen to be described in the language of brains. With them,
every derivation has a question attached: could a thing made of local units do
this? When the answer comes out yes, that is a real finding rather than a
coincidence of notation. When it comes out no, the assumption that broke is
visible and can be argued about.

The pattern recurs. In Lesson&nbsp;3 an inference rule will turn out to be a
weighted difference between two locally available quantities. In Week&nbsp;3 a
learning rule will turn out to be a product of the activity at a connection's two
ends. Neither was aimed at; both fall out of differentiating the same quantity.
That is the sort of thing that makes a framework worth taking seriously, and it
is only visible if the constraints were stated in advance.

::: checkpoint
- State both rules in your own words, without using the word "local".
- Bayes' theorem has four terms. Which one violates the first rule, and why is
  the objection stronger than "that sum is long"?
- Give an example of a computation that satisfies local computation but violates
  local plasticity.
:::
