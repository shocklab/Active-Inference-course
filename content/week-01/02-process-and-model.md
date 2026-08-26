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

Getting this distinction wrong is the commonest way to become confused
about active inference, so we will fix it now and then hold to that notation
for twelve weeks.

## The generative process

The **generative process** is what the world actually does. There are states of
affairs out there, evolving according to whatever physics, chemistry and other
agents dictate, and some of what they do impinges on the agent's sensory
surface.

Write $\vartheta$ for the true state of the world and $o$ for what arrives at the
sensory surface. The generative process specifies how $\vartheta$ evolves, and how
$\vartheta$ gives rise to $o$. Write $a$ for the agent's **actions**, drawn from whatever set of things the agent can do: for a thermostat $a \in \{\text{heat}, \text{idle}\}$, for an animal a continuous space of muscle commands. Week&nbsp;9 makes the discrete case concrete; the lamp example below has no actions at all, being about perception alone. The process also specifies how $a$ feeds
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

writing $\mu$ for the agent's **internal states**: the physical variables that
carry the model, such as the firing rates or synaptic weights that encode
$P(s)$. So $\mu$ and $s$ are not the same kind of thing. $s$ is a variable *in*
the model, a label for a possible state of affairs; $\mu$ is the stuff the model
is made of. Week&nbsp;5 gives $\mu$ a job of its own; until then $s$ does the
work and $\mu$ appears only here, to state the blanket. Knowing the sensory and active
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

::: remark One more confusion
The agent is often said to hold "beliefs", and the word is technical here. The belief is the *distribution*, $P(s)$ or later $Q(s)$, not the state $s$ itself: $s$ is a possible state of affairs, and the belief is how much weight the agent puts on it. Calling $s$ a belief, as people often do and as the diagram's label does, is shorthand for that. A Kalman filter holds beliefs in exactly this sense, its mean and covariance being a Gaussian over states. It carries no
implication of awareness. A Kalman filter has beliefs in exactly this sense; an
ordinary bimetallic thermostat does not, since it holds no distribution over
anything, which means the word is not vacuous.
:::

## A worked case: two agents and a lamp

Everything above has been structural. Here is the smallest example that makes it
concrete, and it will do more work than its size suggests.

A lamp sits behind a frosted screen. The true state $\vartheta$ is **on** or
**off**, each equally likely, and a light sensor reports **bright** or **dim**.
The sensor is imperfect: when the lamp is on it reads bright with probability
${{lamp_true_on}}$, and when the lamp is off it still reads bright with
probability ${{lamp_true_off}}$. That is the generative process, and it is a fact
about the world, not about anybody's beliefs.

Now two agents watch the same screen.

**Agent A** has a model that matches: states $s \in \{\text{on}, \text{off}\}$,
prior $P(s) = ({{lamp_prior}}, {{lamp_prior}})$, and a likelihood equal to the
true one. Its evidence for a bright reading is

$$
P(\text{bright}) = {{lamp_prior}} \times {{lamp_true_on}} + {{lamp_prior}} \times {{lamp_true_off}} = {{lamp_ev_bright_true:.2f}},
$$ {#lamp-evidence}

so a bright reading carries $-\ln {{lamp_ev_bright_true:.2f}} = {{lamp_surprise_bright_true:.4f}}$
nats of surprise, in the natural-log units of Lesson&nbsp;1,
and by Bayes the posterior on the lamp being on is
${{lamp_prior}} \times {{lamp_true_on}} / {{lamp_ev_bright_true:.2f}} = {{lamp_post_on_true:.4f}}$.

**Agent B** has the same states and the same prior but believes its sensor is far
better than it is: ${{lamp_b_on}}$ and ${{lamp_b_off}}$ in place of
${{lamp_true_on}}$ and ${{lamp_true_off}}$. Its evidence for bright is
${{lamp_ev_bright_b:.2f}}$, its surprise ${{lamp_surprise_bright_b:.4f}}$ nats,
and its posterior ${{lamp_post_on_b:.4f}}$.

Three things follow, and each one makes something earlier on this page checkable.

**Surprise belongs to the model.** The two agents received the same reading from
the same lamp and were surprised by different amounts,
${{lamp_surprise_bright_true:.4f}}$ against ${{lamp_surprise_bright_b:.4f}}$
nats. Nothing about the world distinguishes them. This is what it meant to say
that $P(o)$ is computed from the model.

**A wrong model is still a model.** Agent B's likelihood is false, but nothing
about it is ill-formed: its columns still sum to one, Bayes still applies, its
posterior is still a distribution. The machinery does not detect the error. That
is the answer to the checkpoint question below about a state called "predator"
with no predator in the world.

**Mismodelling has a price, and the price is exactly a divergence.** Two pieces
of notation first. Write $P^{*}$ with a star for a distribution belonging to the
*process*, the frequencies the world actually produces, and $P_A$, $P_B$ with an
agent's letter for the same distribution computed under that agent's *model*. The
star is the world; the subscript is whose head it is in.

Over a long run the readings arrive with the true frequencies, bright
${{lamp_ev_bright_true:.2f}}$ of the time. Write $\bar{S}_X$ for the **average surprise** of an
agent $X$: its per-observation surprise, weighted by how often each observation
really turns up,

$$
\bar{S}_X \;=\; \sum_{o} P^{*}(o)\,\big[-\ln P_X(o)\big].
$$ {#avg-surprise}

For agent A this is ${{lamp_entropy_true:.6f}}$ nats, which is exactly
$\mathrm{H}[P^{*}(o)]$: the entropy of Lesson&nbsp;1 applied to the world's own
frequencies. That is the floor, because an agent whose $P_X$ already equals
$P^{*}$ has nothing left to get wrong. Agent B averages
${{lamp_avg_surprise_b:.6f}}$ nats. The excess is
${{lamp_excess:.6f}}$ nats, and

$$
D_{\mathrm{KL}}\big[P^{*}(o) \,\|\, P_B(o)\big] = {{lamp_kl:.6f}}\ \text{nats},
$$ {#lamp-kl}

the same number. The extra surprise a mismodelling agent pays, per observation
and on average, *is* the divergence between the world's statistics and its own.
That is the precise version of Lesson&nbsp;1's remark that the two meanings of
$P(o)$ coincide when the model is good and part company when it is not, and it
is why the framework can get away with using one as a stand-in for the other.

::: exercise Confirm the price
Verify [eq:lamp-kl] by hand: compute $P^{*}(\text{dim})$ and
$P_B(\text{dim})$, then evaluate
$\sum_o P^{*}(o)\ln\big[P^{*}(o)/P_B(o)\big]$.
---solution---
The true marginals are ${{lamp_ev_bright_true:.2f}}$ and
${{lamp_ev_dim_true:.2f}}$. Agent B's are ${{lamp_ev_bright_b:.2f}}$ and
${{lamp_ev_bright_b:.2f}}$, since its assumed sensor is symmetric and its prior
is flat. So

$$
D_{\mathrm{KL}} = {{lamp_ev_bright_true:.2f}}\ln\frac{ {{lamp_ev_bright_true:.2f}} }{ {{lamp_ev_bright_b:.2f}} }
+ {{lamp_ev_dim_true:.2f}}\ln\frac{ {{lamp_ev_dim_true:.2f}} }{ {{lamp_ev_bright_b:.2f}} }
= {{lamp_kl:.6f}}.
$$

The first term is negative and the second positive; the sum is positive, as any
KL divergence must be. Note how small it is. Agent B is badly wrong about its
sensor, yet pays only about five thousandths of a nat per reading, because the
error happens to leave its *marginal* over observations nearly right. Being wrong
about the world in a way that does not show up in your predictions is cheap, and
Week&nbsp;12 asks whether that should worry us.
:::

::: exercise Break the blanket
Take four binary variables $\vartheta, o, a, \mu$ and build a joint distribution
in which $\vartheta$ influences $\mu$ only through $o$. Verify [eq:blanket]
numerically. Then add a direct dependence of $\mu$ on $\vartheta$ and show the
equality fails.
---solution---
Let $\vartheta$ be a fair coin, let $o = \vartheta$ with probability $0.9$ and
$o = 1 - \vartheta$ otherwise, and let $\mu$ depend on $o$ alone, say
$\mu = o$ with probability $0.8$. Fix $a$ constant so it plays no role. Then for
each value of $o$,

$$
P(\vartheta, \mu \mid o) = P(\vartheta \mid o)\,P(\mu \mid o)
$$

holds by construction, because $\mu$ was generated from $o$ without ever
consulting $\vartheta$: conditioning on $o$ leaves nothing for the two to share.

Now let $\mu$ peek: with probability $0.3$ set $\mu = \vartheta$ directly,
ignoring $o$. Condition on $o = 1$ and the two are no longer independent, because
learning $\mu = 1$ now raises the probability that $\vartheta = 1$ by a route
that does not pass through $o$. The blanket has a hole in it, and [eq:blanket]
reports the hole.

The general point: a Markov blanket is not a wall, it is a claim about which
paths exist. Whether real organisms have one in this sense is the argument
Week&nbsp;12 takes up.
:::

## Notation to carry forward

::: notation
$\vartheta$ :: True states of the world. The generative *process*. The agent never sees these.
$s$ :: Hidden states in the agent's generative *model*. Latent variables the agent infers.
$o$ :: Observations, the sensory states on the blanket. Shared by both sides.
$a$ :: Actions, the active states on the blanket. Chosen by the agent, felt by the world.
$\mu$ :: Internal states: the physical variables carrying the model. Not the same as $s$, which is a variable inside it. Returns in Week&nbsp;5.
$P(o \mid s)$ :: The likelihood. What the model says each state predicts.
$P(s)$ :: The prior over states, before any observation.
$P(o)$ :: The model evidence, or marginal likelihood. The quantity whose logarithm is minus the surprise.
$P(s \mid o)$ :: The posterior. What the agent should believe after seeing $o$. The object of the next two lessons.
$P(o, s)$ :: The joint distribution over observations and model states. The generative model itself, [eq:gen-model].
$P^{*}(\cdot)$ :: A star marks a distribution belonging to the *process*: the frequencies the world actually produces. No agent can evaluate it.
$P_A(\cdot),\ P_B(\cdot)$ :: A subscript marks the same distribution computed under a named agent's *model*.
$D_{\mathrm{KL}}[Q \,\|\, P]$ :: Kullback&ndash;Leibler divergence, $\sum_x Q(x)\ln[Q(x)/P(x)]$. Non-negative, zero only when $Q = P$, and not symmetric. Lesson&nbsp;3.
$\mathrm{H}[P]$ :: Shannon entropy, $-\sum_x P(x)\ln P(x)$, in nats. Lesson&nbsp;1.
:::

::: checkpoint
- If an agent's model contains a hidden state called "predator" and there is no predator in the world, is the model's likelihood $P(o \mid s)$ ill-defined? Why not?
- Which of $\vartheta$, $s$, $o$, $a$ appear on both sides of the Markov blanket?
- Explain in one sentence why the marginalisation problem belongs to the agent rather than the world.
:::
