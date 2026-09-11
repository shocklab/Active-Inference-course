---
title: Generative process and generative model
deck: Distinguishing the process that produces observations from the model used to interpret them.
week: 1
time: 20 min
scripts: [w01.js, review-widgets.js]
---

An agent receives observations and uses them to infer something about its surroundings. We need to distinguish the process producing the observations from the model used to interpret them. The model may omit variables, use different categories, or describe the observation process incorrectly.

## The generative process

The **generative process** is what the world actually does. There are states of
affairs out there, evolving according to whatever physics, chemistry and other
agents dictate, and some of what they do impinges on the agent's sensory
surface.

Write $\vartheta$ for the true state of the world and $o$ for what arrives at the
sensory surface. The generative process specifies how $\vartheta$ evolves, and how
$\vartheta$ gives rise to $o$. Write $a$ for the agent's **actions**, drawn from whatever set of things the agent can do: for a thermostat $a \in \{\text{heat}, \text{idle}\}$, for an animal a continuous space of muscle commands. Week&nbsp;9 makes the discrete case concrete; the lamp example below has no actions at all, being about perception alone. If actions affect the environment, the process must specify that effect too.
For example, moving a sensor changes the observations it receives. The lamp
example below holds actions fixed so that we can examine perception alone.

The agent observes $o$, not $\vartheta$ directly. This does not mean $o$ carries
no information about $\vartheta$: that information is what inference uses.
It means the agent must infer the hidden state through its observations.

::: mn On the notation
Much of the free energy principle literature writes $\eta$, or sometimes
$\vartheta$, for external states of the generative process, keeping the letter
$s$ for states inside the agent's model. We follow that; $s$ is defined in the
next section. When you meet a paper that writes $s$ for both, read carefully:
the ambiguity is not always harmless.

This course uses $\vartheta$ for external process states. Later, $\eta$ denotes
an inference rate; check the convention when moving between sources.

:::

## The generative model

The **generative model** is the agent's account of where its observations come
from. It is a joint probability distribution over observations and a set of
quantities the agent treats as their causes:

$$
P(o, s) \;=\; \underbrace{P(o \mid s)}_{\text{likelihood}} \;
             \underbrace{P(s)}_{\text{prior}} .
$$ {#gen-model}

Here $s$ is a **hidden state** *of the model*. It labels a possible state of affairs in the model. It is not $\vartheta$, it need not correspond to anything in $\vartheta$,
and there is no requirement that the two even have the same dimension or the
same type.

The likelihood $P(o \mid s)$ says what observations the agent expects when the
model-state is $s$. The likelihood is a conditional prediction within the model. When $s$ describes
the lamp as on, it predicts what the sensor would report under that hypothesis.
The prior $P(s)$ assigns probabilities before the new observation. Their product
specifies the joint, from which marginal and conditional distributions follow.

::: fig The two structures and the surface between them. Everything to the left of the sensory and active states belongs to the world. The right-hand node denotes the physical internal state carrying beliefs about model states. The arrows are a schematic of interaction, not a proof of conditional independence.
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
  <text x="583" y="140" text-anchor="middle" font-family="'Source Serif 4',Georgia,serif" font-size="21" fill="#0f5f57">&#956;</text>
  <text x="583" y="188" text-anchor="middle" font-family="'IBM Plex Sans',sans-serif" font-size="10.5" fill="#14837a">internal state encoding beliefs</text>

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

The **sensory states** are the observations $o$, described here as part of the
agent's interface. The **active states** $a$ describe how it influences the
surroundings. Write $\mu$ for its **internal states**, such as physical variables
encoding beliefs about $s$. Thus $s$ labels a hypothesis; $\mu$ is a physical
state of a system representing beliefs about that hypothesis.

::: definition Markov blanket
A set of variables is a Markov blanket separating internal from external states
when conditioning on that set makes the two groups independent. For the proposed
partition, the required statement is

$$
P^*(\vartheta,\mu\mid o,a)=P^*(\vartheta\mid o,a)P^*(\mu\mid o,a).
$$ {#blanket}

The star marks the joint distribution of the physical process. This is an
additional statistical condition to establish in a model. Drawing a sensory
surface and action arrows does not establish it. The equality says that, once
$o,a$ are known, learning $\mu$ adds no information about $\vartheta$ under
this joint distribution.
:::

For a simple example, suppose $\vartheta$ generates $o$, which generates $\mu$,
and $a$ is fixed. The joint factorises as
$P^*(\vartheta)P^*(o\mid\vartheta)P^*(\mu\mid o)$. Conditioning on $o$
and normalising the first two factors gives
$P^*(\vartheta\mid o)P^*(\mu\mid o)$, which has the required independence.
Adding feedback or other dependencies requires checking the condition again;
a static factorisation does not settle a dynamical model.

[week:12] examines when such partitions are appropriate for organisms.

::: warning A model can be wrong and still be a model
Nothing above requires $P(o \mid s)$ to be the true likelihood, or $P(s)$ to be
the true distribution over anything. The agent's model is a hypothesis. Much of
what makes active inference interesting comes from watching what happens when the
model and the process disagree in structured ways, which is also the route by
which the framework has been applied to psychiatric conditions.
:::

## Using and changing the model

The model evidence is $P(o)=\sum_sP(o\mid s)P(s)$. [lesson:1.3] derives its
role in Bayes' theorem, and [lesson:1.5] examines algorithms for computing it.
The sum runs over the model's hypotheses. Physically producing a sensor reading
and calculating its probability are different operations.

An observation's surprise is $-\ln P(o)$ under the chosen model. Two agents
can therefore assign different surprise to the same reading.

Inference holds the generative model fixed and changes beliefs about hidden
states. Learning changes the model's parameters or structure. Action affects the
process that generates later observations. Some models let action also change
the observation channel, for example by turning a sensor. The consequences have
to be specified rather than inferred from a slogan about perception and action.

A **belief** here is a probability distribution over possible states. It need
not involve awareness. A Kalman filter's state estimate and covariance specify a
Gaussian belief. A bimetallic thermostat need not explicitly represent any such
distribution, even though we can model its feedback mathematically.

## Two agents and a lamp

Consider a lamp behind a frosted screen and two agents interpreting its sensor.

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
nats of surprise, in the natural-log units of [lesson:1.1],
and by Bayes the posterior on the lamp being on is
${{lamp_prior}} \times {{lamp_true_on}} / {{lamp_ev_bright_true:.2f}} = {{lamp_post_on_true:.4f}}$.

**Agent B** has the same states and the same prior but believes its sensor is far
better than it is: ${{lamp_b_on}}$ and ${{lamp_b_off}}$ in place of
${{lamp_true_on}}$ and ${{lamp_true_off}}$. Its evidence for bright is
${{lamp_ev_bright_b:.2f}}$, its surprise ${{lamp_surprise_bright_b:.4f}}$ nats,
and its posterior ${{lamp_post_on_b:.4f}}$.

We can now compare their predictions and beliefs.

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
$\mathrm{H}[P^{*}(o)]$: the entropy of [lesson:1.1] applied to the world's own
frequencies. The non-negativity calculation below shows why this is a lower bound on average surprise. Agent B averages
${{lamp_avg_surprise_b:.6f}}$ nats. The excess is
${{lamp_excess:.6f}}$ nats.

The **Kullback&ndash;Leibler divergence** of a finite distribution $p$ relative to $q$
is $D_{\mathrm{KL}}[p\|q]=\sum_o p(o)\ln[p(o)/q(o)]$.
Terms with $p(o)=0$ contribute zero; if $p(o)>0$ but $q(o)=0$, the divergence
is infinite. Here both model probabilities are positive.

$$
D_{\mathrm{KL}}\big[P^{*}(o) \,\|\, P_B(o)\big] = {{lamp_kl:.6f}}\ \text{nats},
$$ {#lamp-kl}

Subtracting the actual
entropy from the average surprise gives

$$
\bar S_X-\mathrm H[P^*]
=\sum_o P^*(o)[-\ln P_X(o)+\ln P^*(o)]
=D_{\mathrm{KL}}[P^*\|P_X].
$$

Why is this nonnegative? The elementary inequality $\ln z\le z-1$ for $z>0$
gives

$$
-\sum_o p(o)\ln\frac{q(o)}{p(o)}
\ge\sum_{o:p(o)>0}[p(o)-q(o)]\ge0.
$$

The final inequality allows $q$ to put mass where $p$ is zero. Equality requires
$p=q$. Thus a model matching the observation frequencies minimises average
surprise, but matching those frequencies alone need not identify the correct
hidden causes.

::: exercise Verify the excess surprise
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
Let $\vartheta$ be a fair coin and let the sensor copy it with probability
$0.9$, flipping it otherwise. Let $\mu$ copy $o$ with probability $0.8$,
flipping it otherwise, independently of $\vartheta$ conditional on $o$.
Fix $a$ constant. Given $o=1$, the conditional probabilities are
$P(\vartheta=1\mid o=1)=0.9$ and $P(\mu=1\mid o=1)=0.8$.
The joint table is

| $\vartheta$ | $\mu$ | $P(\vartheta,\mu\mid o=1)$ |
|---|---|---|
| 0 | 0 | {{blanket_00:.4f}} |
| 0 | 1 | {{blanket_01:.4f}} |
| 1 | 0 | {{blanket_10:.4f}} |
| 1 | 1 | {{blanket_11:.4f}} |

Each entry is the corresponding product of marginals. Complementing both
binary labels gives the same verification for $o=0$.

Now change the generation of $\mu$: with probability $0.3$ copy $\vartheta$
directly; otherwise use the previous noisy copy of $o$. Given $o=1$,

$$
P(\mu=1\mid\vartheta=1,o=1)=0.3+0.7(0.8)={{blanket_direct:.4f}},
$$

whereas for $\vartheta=0$ it is $0.7(0.8)={{blanket_other:.4f}}$.
Consequently the joint entry and the product of marginals differ:

$$
P(\vartheta=1,\mu=1\mid o=1)={{blanket_joint:.4f}},
\qquad
P(\vartheta=1\mid o=1)P(\mu=1\mid o=1)={{blanket_product:.4f}}.
$$

This explicit distribution fails conditional independence. A direct edge can
permit such a failure; its presence alone would not prove failure for every
possible parameter setting.
:::

## Changing beliefs while holding evidence fixed

Take a binary lamp model with prior probability $P(s=\mathrm{on})=1/2$.
Let $P(\mathrm{bright}\mid\mathrm{off})=0.20$ and write $\theta$ for
$P(\mathrm{bright}\mid\mathrm{on})$, initially $0.80$. The parameter $\theta$
is a probability in $[0,1]$ describing sensor reliability. A bright reading
arrives. Its evidence is ${{lab_evidence:.3f}}$ and the exact posterior probability
of on is ${{lab_posterior:.3f}}$.

Represent an approximate belief by $Q(\mathrm{on})=q$ and
$Q(\mathrm{off})=1-q$. Here $q\in[0,1]$ is a probability, not a state or a
sensor parameter. Define

$$
F(q)=\sum_s Q(s)\ln\frac{Q(s)}{P(s,\mathrm{bright})}.
$$ {#belief-free-energy}

This is the finite-state **variational free energy**. Terms with $Q(s)=0$ are
interpreted by their limit. Substituting
$P(s,\mathrm{bright})=P(s\mid\mathrm{bright})P(\mathrm{bright})$ gives

$$
F(q)=D_{\mathrm{KL}}[Q\|P(s\mid\mathrm{bright})]-\ln P(\mathrm{bright}).
$$ {#belief-gap}

The logarithm splits a product into a sum, and the evidence term comes outside
the expectation because $\sum_sQ(s)=1$. KL divergence is non-negative and is
zero when the beliefs equal the exact posterior. [lesson:1.3] examines it in detail.

At $q=1/2$, free energy is ${{lab_F_initial:.4f}}$ nats. Setting
$q={{lab_posterior:.3f}}$ reduces it to ${{lab_surprise:.4f}}$ nats. The evidence
has stayed at ${{lab_evidence:.3f}}$ throughout: inference has reduced the gap,
not the surprise of this observation under the model.

::: widget belief-evidence | Move the belief probability q while holding the sensor parameter fixed. The evidence and surprise remain constant, while the free-energy gap changes. Changing the sensor parameter changes the generative model and can change the evidence. The parameter slider demonstrates a model change; it does not implement a learning algorithm or change the physical lamp.
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
$D_{\mathrm{KL}}[Q \,\|\, P]$ :: Kullback&ndash;Leibler divergence, $\sum_x Q(x)\ln[Q(x)/P(x)]$. Non-negative, zero only when $Q = P$, and not symmetric. [lesson:1.3].
$\mathrm{H}[P]$ :: Shannon entropy, $-\sum_x P(x)\ln P(x)$, in nats. [lesson:1.1].
:::

::: checkpoint
- If an agent's model contains a hidden state called "predator" and there is no predator in the world, is the model's likelihood $P(o \mid s)$ ill-defined? Why not?
- Which of $\vartheta$, $s$, $o$, $a$ appear on both sides of the Markov blanket?
- Explain in one sentence why the marginalisation problem belongs to the agent rather than the world.
:::
