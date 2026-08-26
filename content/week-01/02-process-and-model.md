---
title: Two models, not one
deck: The world's causal machinery and the agent's picture of it are separate objects. Almost every confusion in this field starts by conflating them.
week: 1
time: 20 min
scripts: [w01.js]
---

An agent is coupled to a world. Both of them generate things. The world
generates the sense data that arrive at the agent's surface. The agent generates
predictions about what those data are going to be. These are two different
generative structures, they live in two different places, and they are written
with two different letters for a reason.

Getting this distinction wrong is the single most common way to become confused
about active inference, so we will fix it now and then keep the notation honest
for twelve weeks.

## The generative process

The **generative process** is what the world actually does. There are states of
affairs out there, evolving according to whatever physics, chemistry and other
agents dictate, and some of what they do impinges on the agent's sensory
surface.

Write $\vartheta$ for the true state of the world and $o$ for what arrives at the
sensory surface. The generative process specifies how $\vartheta$ evolves, and how
$\vartheta$ gives rise to $o$. Write $a$ for the agent's **actions**. The process also specifies how $a$ feeds
back into $\vartheta$, and it has to: an agent that could only watch would have no
way of staying in the small region of Lesson&nbsp;1, since staying there is
something a body does rather than something that happens to it.

The agent does not have access to $\vartheta$. Not partially, not noisily: not at
all. Everything the agent will ever know about the world it must extract from
$o$, and later from the consequences of its own $a$.

::: mn On the notation
Much of the free energy principle literature writes $\eta$, or sometimes
$\vartheta$, for external states of the generative process, keeping the letter
$s$ for states inside the agent's model. We follow that; $s$ is defined in the
next section. When you meet a paper that writes $s$ for both, read carefully:
the ambiguity is not always harmless.
:::

## The generative model

The **generative model** is the agent's account of where its observations come
from. It is a joint probability distribution over observations and a set of
quantities the agent treats as their causes:

$$
P(o, s) \;=\; \underbrace{P(o \mid s)}_{\text{likelihood}} \;
             \underbrace{P(s)}_{\text{prior}} .
$$ {#gen-model}

Here $s$ is a **hidden state** *of the model*. It is a variable in the agent's
head. It is not $\vartheta$, it need not correspond to anything in $\vartheta$,
and there is no requirement that the two even have the same dimension or the
same type.

The likelihood $P(o \mid s)$ says what observations the agent expects when the
model-state is $s$. Note the phrasing: not *when the world is in state $s$*,
because the world is not in any model-state at all. The model-state is a variable
inside the agent. The prior $P(s)$ says which model-states the agent
expects before looking. Together they specify a joint distribution, and from a
joint distribution every marginal and every conditional follows in principle.

::: fig The two structures and the surface between them. Everything to the left of the sensory and active states belongs to the world. Everything to the right belongs to the agent. The two never touch directly.
<svg viewBox="0 0 720 268" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Diagram: the generative process on the left, the agent's generative model on the right, coupled only through sensory states o and active states a.">
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M1 1 L9 5 L1 9" fill="none" stroke="context-stroke" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <rect x="1" y="1" width="718" height="266" fill="#fdfcf9" stroke="#e4e1d9"/>
  <rect x="18" y="26" width="238" height="216" fill="#f4f2ec" stroke="#d3cfc4"/>
  <rect x="464" y="26" width="238" height="216" fill="#e9f2f0" stroke="#0f5f57" stroke-opacity=".35"/>
  <text x="137" y="47" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="11.5" font-weight="600" letter-spacing="1.6" fill="#6d6d75">THE WORLD</text>
  <text x="583" y="47" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="11.5" font-weight="600" letter-spacing="1.6" fill="#0f5f57">THE AGENT</text>
  <text x="137" y="64" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="12.5" font-style="italic" fill="#90909a">generative process</text>
  <text x="583" y="64" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="12.5" font-style="italic" fill="#14837a">generative model</text>

  <circle cx="137" cy="134" r="35" fill="#ffffff" stroke="#6d6d75" stroke-width="1.5"/>
  <text x="137" y="140" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="21" fill="#3c3c42">&#977;</text>
  <text x="137" y="188" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#90909a">true external state</text>

  <circle cx="583" cy="134" r="35" fill="#ffffff" stroke="#0f5f57" stroke-width="1.5"/>
  <text x="583" y="140" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="21" fill="#0f5f57">s</text>
  <text x="583" y="188" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#14837a">believed state</text>

  <rect x="322" y="60" width="76" height="42" fill="#ffffff" stroke="#8a6d1f" stroke-width="1.5"/>
  <text x="360" y="87" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="19" fill="#8a6d1f">o</text>
  <text x="360" y="118" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#8a6d1f">sensory states</text>

  <rect x="322" y="166" width="76" height="42" fill="#ffffff" stroke="#9c4d2f" stroke-width="1.5"/>
  <text x="360" y="193" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="19" fill="#9c4d2f">a</text>
  <text x="360" y="224" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10" fill="#9c4d2f">active states</text>

  <path d="M170 118 C 240 92, 270 84, 318 81" fill="none" stroke="#8a6d1f" stroke-width="1.8" marker-end="url(#ar)"/>
  <path d="M402 81 C 452 84, 486 96, 552 116" fill="none" stroke="#8a6d1f" stroke-width="1.8" marker-end="url(#ar)"/>
  <path d="M552 152 C 486 172, 452 184, 402 187" fill="none" stroke="#9c4d2f" stroke-width="1.8" marker-end="url(#ar)"/>
  <path d="M318 187 C 270 184, 240 176, 170 150" fill="none" stroke="#9c4d2f" stroke-width="1.8" marker-end="url(#ar)"/>

  <text x="245" y="72" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#8a6d1f">causes</text>
  <text x="478" y="80" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#8a6d1f">informs</text>
  <text x="478" y="196" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#9c4d2f">selects</text>
  <text x="245" y="200" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="9.5" fill="#9c4d2f">changes</text>

  <line x1="300" y1="26" x2="300" y2="242" stroke="#d3cfc4" stroke-dasharray="3 4"/>
  <line x1="420" y1="26" x2="420" y2="242" stroke="#d3cfc4" stroke-dasharray="3 4"/>
</svg>
:::

The two dashed lines mark the only place the world and the agent meet. The
variables sitting on them have names. The **sensory states** are the observations
$o$ of Lesson&nbsp;1, seen now as one half of an interface rather than as a bare
measurement; they carry information inward. The **active states** $a$ carry
influence outward. Nothing crosses except through the two. That interface is
called a **Markov blanket**. The name is borrowed from graphical models, and it
carries the usual meaning: conditioned on the blanket, what is inside is
independent of what is outside. Here that reads

$$
P(\vartheta, \mu \mid o, a) \;=\; P(\vartheta \mid o, a)\; P(\mu \mid o, a),
$$ {#blanket}

writing $\mu$ for the agent's internal states. Knowing the sensory and active
states renders the world and the inside of the agent conditionally independent:
any influence one has on the other has already passed through the blanket. That
is the formal version of the claim that you never perceive the world, only your
own surface.

Whether real organisms have blankets in this sense, and whether the ones drawn in
the literature are found or imposed, is a live argument. Week&nbsp;12 takes it up.

The four-way partition into external, sensory, active and internal states is
standard in this literature, and you will meet all four names again in
Week&nbsp;5, where the internal states acquire a letter of their own.

::: warning A model can be wrong and still be a model
Nothing above requires $P(o \mid s)$ to be the true likelihood, or $P(s)$ to be
the true distribution over anything. The agent's model is a hypothesis. Much of
what makes active inference interesting comes from watching what happens when the
model and the process disagree in structured ways, which is also the route by
which the framework has been applied to psychiatric conditions.
:::

## What follows from keeping them apart

Three things become clear once you keep the two apart, and stay muddy if you do not.

**Where the intractability lives.** To use the model you need the probability of
an observation on its own, $P(o) = \sum_s P(o \mid s) P(s)$, and that sum runs
over the agent's $s$, not the world's $\vartheta$. Avoiding it occupies the next
three weeks. The world
does not compute anything. It has no marginalisation problem. The difficulty is
entirely on the agent's side of the blanket. That is a useful place for it to be:
it makes the problem one of algorithm rather than physics, so it can be attacked
by choosing a better approximation scheme, which is what Week&nbsp;4 does.

**What "surprise" refers to.** An observation is surprising relative to
$P(o)$ computed from the agent's model. The world does not find anything
surprising. Two agents receiving identical sense data can be surprised to
completely different degrees, and that is not a paradox: they have different
models.

**What action does.** Action does not change $P(o \mid s)$ or $P(s)$; those are
the agent's own commitments. Action changes $\vartheta$, and therefore changes
which $o$ shows up next. Perception moves the model towards the data; action
moves the data towards the model. They act on opposite sides of the blanket. That is what lets a single quantity,
the free energy of Week&nbsp;4, govern both: perception cannot cheat by simply
deciding the data are whatever the model wanted, because the data are on the far
side of the interface and only action can reach them.

::: remark A last confusion worth heading off
The agent's states $s$ are usually described as its "beliefs". This is a
technical use of the word. A belief here is a probability distribution held by
some part of a system, of the sort a Kalman filter has. It carries no
implication of awareness. A Kalman filter has beliefs in exactly this sense; an
ordinary bimetallic thermostat does not, since it holds no distribution over
anything, which is worth noticing because it means the word is not vacuous.
:::

## Notation to carry forward

::: notation
$\vartheta$ :: True states of the world. The generative *process*. The agent never sees these.
$s$ :: Hidden states in the agent's generative *model*. Latent variables the agent infers.
$o$ :: Observations, the sensory states on the blanket. Shared by both sides.
$a$ :: Actions, the active states on the blanket. Chosen by the agent, felt by the world.
$P(o \mid s)$ :: The likelihood. What the model says each state predicts.
$P(s)$ :: The prior over states, before any observation.
$P(o)$ :: The model evidence, or marginal likelihood. The quantity whose logarithm is minus the surprise.
$P(s \mid o)$ :: The posterior. What the agent should believe after seeing $o$. The object of the next two lessons.
:::

::: checkpoint
- If an agent's model contains a hidden state called "predator" and there is no predator in the world, is the model's likelihood $P(o \mid s)$ ill-defined? Why not?
- Which of $\vartheta$, $s$, $o$, $a$ appear on both sides of the Markov blanket?
- Explain in one sentence why the marginalisation problem belongs to the agent rather than the world.
:::
