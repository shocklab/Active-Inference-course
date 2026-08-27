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

## What a device is made of

The constraints are about locality, and locality is about parts, so the parts
need naming first. Two words, used in their stylised sense throughout the course
and nowhere assuming any neuroscience.

::: definition Unit and connection
A **unit** is the smallest thing that holds a number and does arithmetic on it.
Read it as a stylised neuron, or a small population of them acting together. At
any moment a unit has an **activity**, written $r$ for *rate*, one real number
which is what it is currently reporting. The letter $a$ is not available: Week&nbsp;1
spent it on the agent's actions.

A **connection** carries the activity of one unit to another, multiplied by a
number called its **weight**. Read it as a synapse, and the weight as how
strongly that synapse drives the cell it lands on. A unit's inputs are exactly
the activities arriving along the connections that reach it, each scaled by that
connection's weight. Weights are written $w$, and they are the only thing in the
course that ever changes slowly.
:::

::: fig A unit, its incoming connections, and what it can and cannot see. Everything inside the shaded region is available to the shaded unit. The quantity on the right is not, and no connection reaches it.
<svg viewBox="0 0 620 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Three units feed one central unit along weighted connections; a fourth quantity sits unconnected and out of reach.">
  <rect x="1" y="1" width="618" height="208" fill="#fdfcf9" stroke="#e4e1d9"/>
  <ellipse cx="228" cy="112" rx="150" ry="76" fill="#e9f2f0" stroke="#14837a" stroke-width="1" stroke-dasharray="4 3"/>
  <text x="228" y="200" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#0f5f57">what this unit can see</text>
  <defs>
    <marker id="u1" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 z" fill="#3c3c42"/>
    </marker>
  </defs>
  <circle cx="112" cy="58" r="19" fill="#ffffff" stroke="#3c3c42" stroke-width="1.6"/>
  <text x="112" y="63" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="14" fill="#3c3c42">r&#8321;</text>
  <circle cx="112" cy="112" r="19" fill="#ffffff" stroke="#3c3c42" stroke-width="1.6"/>
  <text x="112" y="117" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="14" fill="#3c3c42">r&#8322;</text>
  <circle cx="112" cy="166" r="19" fill="#ffffff" stroke="#3c3c42" stroke-width="1.6"/>
  <text x="112" y="171" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="14" fill="#3c3c42">r&#8323;</text>
  <path d="M132 64 L286 104" fill="none" stroke="#3c3c42" stroke-width="1.5" marker-end="url(#u1)"/>
  <path d="M132 112 L286 112" fill="none" stroke="#3c3c42" stroke-width="1.5" marker-end="url(#u1)"/>
  <path d="M132 160 L286 120" fill="none" stroke="#3c3c42" stroke-width="1.5" marker-end="url(#u1)"/>
  <text x="205" y="76" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="12" fill="#8a6d1f">w&#8321;</text>
  <text x="205" y="106" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="12" fill="#8a6d1f">w&#8322;</text>
  <text x="205" y="152" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="12" fill="#8a6d1f">w&#8323;</text>
  <circle cx="310" cy="112" r="24" fill="#0f5f57"/>
  <text x="310" y="118" text-anchor="middle" font-family="'Source Serif 4',serif" font-size="15" fill="#ffffff">r</text>
  <text x="310" y="60" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#3c3c42">r = w&#8321;r&#8321; + w&#8322;r&#8322; + w&#8323;r&#8323;</text>
  <rect x="452" y="34" width="132" height="42" rx="5" fill="#f9efe9" stroke="#9c4d2f" stroke-dasharray="3 3"/>
  <text x="518" y="59" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#9c4d2f">a quantity held elsewhere</text>
  <path d="M452 55 L360 74" fill="none" stroke="#9c4d2f" stroke-width="1.4" stroke-dasharray="3 3"/>
  <line x1="396" y1="56" x2="416" y2="76" stroke="#9c4d2f" stroke-width="2"/>
  <line x1="416" y1="56" x2="396" y2="76" stroke="#9c4d2f" stroke-width="2"/>
  <text x="518" y="120" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">no connection,</text>
  <text x="518" y="136" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#6d6d75">therefore no access</text>
</svg>
:::

Why should a physical device be limited this way? Because a cell has no way of
reading a quantity except by having something delivered to it. A neuron's soma
sees the currents its own synapses inject and nothing else; there is no shared
memory it can consult and no address it can name. Whatever it is going to use in
its arithmetic, some connection has to have brought.

## The constraints

Two rules, both about locality. Neither is a theorem. They are modelling
assumptions, chosen because they are the minimum that a device built out of units
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

## What the constraints buy

Because it turns what follows into a search with a specification.

Without the constraints, the next three weeks would be a series of results about
probability that happen to be described in the language of brains. With them,
every derivation has a question attached: could a thing made of local units do
this? When the answer comes out yes, that is a real finding rather than a
coincidence of notation. When it comes out no, the assumption that broke is
visible and can be argued about.

The pattern recurs. In [lesson:2.3] an inference rule will turn out to be a
weighted difference between two locally available quantities. In Week&nbsp;3 a
learning rule will turn out to be a product of the activity at a connection's two
ends. Neither was aimed at; both fall out of differentiating the same quantity.
That is the sort of thing that makes a framework worth taking seriously, and it
is only visible if the constraints were stated in advance.

## A rule that satisfies both

Every example so far has been of a rule breaking. Here is the smallest one that
does not, so there is something to compare against.

A unit receives $r_1$ and $r_2$ along connections of weight $w_1$ and $w_2$, and
reports $r = w_1 r_1 + w_2 r_2$. Local computation holds: the only quantities in
that expression are the two activities delivered to it and the two weights on the
connections that delivered them. Now let each weight change by a small multiple
of $r_i \, r$, the product of the activities at that connection's two ends. Local
plasticity holds too, since a synapse can in principle register how active its
sender and its receiver are.

That rule is not chosen at random. Something with exactly that shape falls out of
a derivative in Week&nbsp;3, and the fact that it is a product of the two ends is
what will make it implementable. The point for now is only that the constraints
are satisfiable, so ruling things out with them is not vacuous.

::: checkpoint
- State both rules in your own words, without using the word "local".
- Bayes' theorem has four terms. Which one violates the first rule, and why is
  the objection stronger than "that sum is long"?
- Take the unit above and change its weight update so that each weight moves by a
  multiple of $r_i\,(r - \bar r)$, where $\bar r$ is the average activity of every
  unit in the network. Which rule does this break, and which does it still
  satisfy?
:::
