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
$\vartheta$ gives rise to $o$. It also specifies how the agent's actions $a$ feed
back into $\vartheta$, because an agent that could not change the world would not
be much of an agent.

The agent does not have access to $\vartheta$. Not partially, not noisily: not at
all. Everything the agent will ever know about the world it must extract from
$o$, and later from the consequences of its own $a$.

::: mn On the notation
Namjoshi and much of the literature write $\vartheta$ or $\eta$ for external
states of the generative process, reserving $s$ for states *in the model*. We
follow that. When you meet a paper that writes $s$ for both, read carefully.
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

The likelihood $P(o \mid s)$ says what observations the agent expects if the
world is in model-state $s$. The prior $P(s)$ says which model-states the agent
expects before looking. Together they specify a joint distribution, and from a
joint distribution you can in principle compute anything.

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

The two dashed lines mark the only place the world and the agent meet. Sensory
states carry information inward; active states carry influence outward. Nothing
crosses except through them. That interface is called a **Markov blanket**, and
it is the formal statement of the idea that you never perceive the world, only
your own surface.

::: warning A model can be wrong and still be a model
Nothing above requires $P(o \mid s)$ to be the true likelihood, or $P(s)$ to be
the true distribution over anything. The agent's model is a hypothesis. Much of
what makes active inference interesting, and all of what makes it useful for
modelling psychopathology, comes from watching what happens when the model and
the process disagree in structured ways.
:::

## Why the distinction earns its keep

Three things become clear once you keep the two apart, and stay muddy if you do not.

**Where the intractability lives.** The sum we will spend the next three weeks
trying to avoid is over the agent's $s$, not the world's $\vartheta$. The world
does not compute anything. It has no marginalisation problem. The difficulty is
entirely on the agent's side of the blanket, which is why it can be solved by
building a better agent.

**What "surprise" refers to.** An observation is surprising relative to
$P(o)$ computed from the agent's model. The world does not find anything
surprising. Two agents receiving identical sense data can be surprised to
completely different degrees, and that is not a paradox: they have different
models.

**What action does.** Action does not change $P(o \mid s)$ or $P(s)$; those are
the agent's own commitments. Action changes $\vartheta$, and therefore changes
which $o$ shows up next. Perception moves the model towards the data; action
moves the data towards the model. They act on opposite sides of the blanket, and
this is why one quantity can govern both without the whole thing collapsing into
wishful thinking.

::: remark A last confusion worth heading off
The agent's states $s$ are usually described as its "beliefs". This is a
technical use of the word. A belief here is a probability distribution held by
some part of a system, of the sort a Kalman filter has. It carries no
implication of awareness, and thermostats have them.
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
