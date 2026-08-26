---
title: Bayes with states and observations
deck: The posterior is the belief an ideal agent would hold. Writing it down is easy; the trouble is entirely in one denominator.
week: 1
time: 30 min
scripts: [w01.js]
---

The agent has a generative model $P(o,s) = P(o \mid s)P(s)$. An observation
arrives. What should it now believe about $s$?

There is only one answer consistent with the axioms of probability, and it takes
two lines to get. The joint probability of a state and an observation can be
factored either way round,

$$
P(s \mid o)\,P(o) \;=\; P(o, s) \;=\; P(o \mid s)\,P(s),
$$ {#product-rule}

since both sides are the definition of conditional probability applied in the
two possible orders. Divide [eq:product-rule] by $P(o)$, which is legitimate
whenever the observation was possible at all, and the answer falls out:

$$
P(s \mid o) \;=\; \frac{P(o \mid s)\,P(s)}{P(o)}, \qquad
P(o) \;=\; \sum_{s} P(o \mid s)\,P(s).
$$ {#bayes}

Everything in this course is a consequence of not being able to compute the
denominator. This page is mostly about why, because if you leave
this lesson thinking Bayes' theorem is hard, you have taken away the wrong
lesson. Bayes' theorem is trivial. The sum underneath it is not.

## The four quantities, named

::: notation
$P(s)$, the prior :: What the agent expects before it looks. In this framework a prior is read as the agent's own commitment rather than as a measured frequency in the world. That reading is a choice, and Week&nbsp;12 returns to what it costs.
$P(o \mid s)$, the likelihood :: Read as a function of $s$ for fixed $o$, this scores how well each state would account for the data you actually got. It is not a distribution over $s$ and does not sum to one over $s$.
$P(o)$, the evidence :: The probability the model assigns to seeing $o$ at all, having averaged over every state it thinks possible. Its negative logarithm is the surprise from Lesson&nbsp;1.
$P(s \mid o)$, the posterior :: The updated belief. Prior reweighted by how well each state explains the data, then renormalised.
:::

Two of these deserve a moment.

The likelihood is a **column** of a matrix, not a row. If you tabulate
$P(o \mid s)$ with observations down the side and states across the top, then
each column is a probability distribution over observations and sums to one.
When an observation arrives you pick out the corresponding row and read it
across the states. That row is not a distribution and there is no reason at all
for it to sum to one. Students trip over this constantly; the matrix does not
care which way you read it, but the normalisation does.

::: mn Rows and columns
From Week&nbsp;9 this matrix is written $\mathbf{A}$, with
$A_{ij} = P(o = i \mid s = j)$, and the problems at the end of this week already
use that name. Columns sum to one. Remember it now and the
discrete half of the course will cost you much less pain.
:::

The evidence is the odd one out, because it is the only term that requires you
to know about states you did not observe and are not asking about. To evaluate
$P(o)$ you must visit every state the model admits, ask what each would have
predicted, weight by how plausible each was, and add. Nothing about the question
"was that a leopard?" suggests that answering it should require an inventory of
everything else that exists. And yet.

## A worked example, with all the arithmetic

You are on a mountain path at dusk. Three states of the world concern you:
there is a **leopard** in the scrub, there is a **baboon** in the scrub, or
there is **nothing** in the scrub. Three things you might sense: a **tawny
flash**, a **branch shaking**, or **quiet**.

Your model is this likelihood matrix. Each column is one state's prediction
about what you will sense, and so each column sums to one.

| $P(o \mid s)$ | leopard | baboon | nothing |
|---|---|---|---|
| tawny flash | 0.70 | 0.15 | 0.02 |
| branch shakes | 0.25 | 0.75 | 0.08 |
| quiet | 0.05 | 0.10 | 0.90 |

Your prior, from a lifetime on this mountain, is
$P(s) = (0.08,\ 0.22,\ 0.70)$ over (leopard, baboon, nothing). Leopards are rare.

You see a tawny flash. Take the tawny-flash row and multiply it, elementwise,
by the prior:

$$
\begin{aligned}
P(o \mid s)P(s) \;=\; \big(\,
&0.70 \times 0.08,\;\;
0.15 \times 0.22,\;\;
0.02 \times 0.70 \,\big) \\[2pt]
\;=\; \big(\,& {{num_tawny_leopard:.3f}},\;\; {{num_tawny_baboon:.3f}},\;\; {{num_tawny_nothing:.3f}} \,\big).
\end{aligned}
$$

These are the three numerators in [eq:bayes]. Their sum is the denominator:

$$
P(o = \text{tawny flash}) \;=\; {{num_tawny_leopard:.3f}} + {{num_tawny_baboon:.3f}} + {{num_tawny_nothing:.3f}} \;=\; {{ev_tawny:.3f}} .
$$ {#worked-evidence}

Divide through:

$$
P(s \mid o) \;=\; \frac{({{num_tawny_leopard:.3f}},\ {{num_tawny_baboon:.3f}},\ {{num_tawny_nothing:.3f}})}{{{ev_tawny:.3f}}}
\;=\; ({{post_tawny_leopard:.4f}},\ {{post_tawny_baboon:.4f}},\ {{post_tawny_nothing:.4f}}).
$$ {#worked-posterior}

::: mn Sanity check
${{post_tawny_leopard:.4f}} + {{post_tawny_baboon:.4f}} + {{post_tawny_nothing:.4f}} = 1$. If your posterior does not sum to one you have
divided by the wrong thing, which in practice means you forgot a state.
:::

Read what happened. The leopard hypothesis started at {{prior_leopard_pct:.0f}}% and finished at {{post_tawny_leopard_pct:.0f}}%,
a factor of nearly seven. It did not get there by being likely; it got there by
being the only state that predicts tawny flashes well. Meanwhile "nothing"
started as the overwhelming favourite at {{prior_nothing_pct:.0f}}% and was demoted to {{post_tawny_nothing_pct:.0f}}%, because
whatever else is true, empty scrub does not flash tawny.

Two derived quantities, while the numbers are in front of us.

The section promised all the arithmetic, so here are the other two observations
worked the same way. Multiply the relevant row elementwise by the prior, add:

| observed | numerators $P(o\mid s)P(s)$ | $P(o)$ | surprise, nats |
|---|---|---|---|
| tawny flash | {{num_tawny_leopard:.3f}}, {{num_tawny_baboon:.3f}}, {{num_tawny_nothing:.3f}} | {{ev_tawny:.3f}} | {{surprise_tawny:.3f}} |
| branch shakes | {{num_branch_leopard:.3f}}, {{num_branch_baboon:.3f}}, {{num_branch_nothing:.3f}} | {{ev_branch:.3f}} | {{surprise_branch:.3f}} |
| quiet | {{num_quiet_leopard:.3f}}, {{num_quiet_baboon:.3f}}, {{num_quiet_nothing:.3f}} | {{ev_quiet:.3f}} | {{surprise_quiet:.3f}} |

The three evidence values sum to one, as they must: every observation has to be
one of the three.

**Surprise.** From [eq:worked-evidence], $-\ln P(o) = -\ln {{ev_tawny:.3f}} = {{surprise_tawny:.3f}}$ nats.
Quiet is what this model expects, and it is duly unsurprised by it.

**Information gained.** How far did the observation move you? The natural
measure is the Kullback&ndash;Leibler divergence from the prior to the posterior,

$$
D_{\mathrm{KL}}\big[P(s\mid o)\,\|\,P(s)\big]
\;=\; \sum_{s} P(s \mid o)\,\ln \frac{P(s\mid o)}{P(s)} .
$$ {#info-gain}

Read the notation carefully, because it runs backwards from the name. In
$D_{\mathrm{KL}}[Q\,\|\,P]$ the distribution written *first* is the one the
expectation is taken under, and the convention is to say this out loud as "the
divergence from $P$ to $Q$". So the posterior appears first above even though we
describe it as measuring how far the prior moved. It is not symmetric,
$D_{\mathrm{KL}}[Q\,\|\,P] \neq D_{\mathrm{KL}}[P\,\|\,Q]$, so the order is
never decorative; Week&nbsp;4 turns on getting it right.

Worked out term by term for the tawny flash, using the posterior
[eq:worked-posterior] and the prior:

| state | $P(s\mid o)$ | $P(s)$ | ratio | $P(s\mid o)\ln(\text{ratio})$ |
|---|---|---|---|---|
| leopard | {{post_tawny_leopard:.4f}} | 0.08 | {{ratio_tawny_leopard:.3f}} | $+${{klterm_tawny_leopard:.4f}} |
| baboon | {{post_tawny_baboon:.4f}} | 0.22 | {{ratio_tawny_baboon:.3f}} | $+${{klterm_tawny_baboon:.4f}} |
| nothing | {{post_tawny_nothing:.4f}} | 0.70 | {{ratio_tawny_nothing:.3f}} | {{klterm_tawny_nothing:.4f}} |

summing to ${{info_tawny:.3f}}$ nats. Notice that the last term is **negative**:
the observation pushed probability mass *off* the "nothing" hypothesis, and a
state whose probability falls contributes negatively. Individual terms of a KL
divergence carry either sign. Only the total is guaranteed non-negative, and that
guarantee is a theorem about the sum, not about its parts.

For the shaking branch the same computation gives ${{info_branch:.3f}}$ nats, and
for quiet ${{info_quiet:.3f}}$. The observation that surprised you most also taught you most,
which is intuitive and, as the exercises will show, not a theorem.

::: widget bayes-discrete | The same calculation, live. The middle bars are the likelihood row for whichever observation you select. Move the priors and watch the posterior swing; move the ambiguity slider and watch it stop responding to data at all.
:::

## What the ambiguity slider is showing you

The figure's third slider is a parameter we will write $\lambda$, running from
$0$ to $1$, which blends the likelihood towards a uniform one: at $\lambda = 0$
the matrix is as printed above, and at $\lambda = 1$ every state predicts every
observation with equal probability. It is a dial for how informative the senses
are.

Push $\lambda$ towards 1 in that figure. The likelihood bars flatten until every
state predicts every observation equally well. Watch the posterior: it slides
back onto the prior and stays there. The information-gain readout goes to zero.

Name it, because it returns as a term in an equation in
Week&nbsp;10. When $P(o \mid s)$ is close to uniform in $o$ for every $s$, the
sensory channel is **ambiguous**: observations do not discriminate between
states, so observing costs you effort and buys you nothing. When $P(o \mid s)$ is
close to a permutation matrix, an observation pins the state down exactly.

An agent that can choose where to look has an interest in looking where the
likelihood is sharp. Hold that thought for nine weeks.

## Where the difficulty is

In the example above the denominator was a sum of three terms and you did it in
your head. Now count the terms for something less contrived.

An animal foraging tracks its own position, its energy reserves, the time since
it last saw a predator, the wind direction, whether each of several patches has
been depleted, and what its companions are doing. Each of these is a state
*factor*. The joint state space is the product of all of them, so if there are
$n$ factors with $k$ values each, the sum in [eq:bayes] has $k^n$ terms.

::: widget evidence-blowup | The number of terms in the evidence sum, on a logarithmic vertical axis. The vertical axis is logarithmic. The number of factors sits in the exponent.
:::

Nothing about this is fixable by better hardware, and here is why rather than on faith. Hardware buys you a constant factor: twice the
speed, twice the sums per second. The cost here is $k^n$, so a doubling of speed
buys exactly one more state factor before you are back where you started. The
gap between what improves multiplicatively and what grows exponentially does not
close.

::: warning Where this argument does not apply
A mathematician will already have objected, and correctly. Exact inference is
perfectly tractable in several important cases: conjugate models, where the
posterior stays in a family closed under updating; linear Gaussian systems, where
the Kalman filter gives exact answers in closed form; and graphical models whose
dependency structure is a tree or close to one, where belief propagation is
exact and cheap. The blow-up above is the *generic* case, densely coupled and
without exploitable structure.

So the claim has to be narrower than "exact inference is impossible". It is that
the structure which rescues those special cases cannot be assumed for an agent
modelling an open-ended world, and that a framework claiming to describe such
agents cannot rest on it. Week&nbsp;2 uses conjugacy where it is available and
says exactly where it runs out. Adding one more thing to keep
track of multiplies the work by $k$, and brains manage this in a few tens of
milliseconds using about twenty watts.

So the exact posterior is not available. This is not a temporary engineering
inconvenience; it is a structural feature of being a finite thing embedded in a
large world. Everything from Week&nbsp;4 onwards is a response to it: rather
than computing $P(s \mid o)$, we will posit an approximation $Q(s)$ that we
*can* compute with, and then work out how to make it as close to the posterior
as possible without ever evaluating the posterior.

::: exercise Bayes with a hostile prior
Using the same likelihood matrix, suppose you are a nervous hiker with prior
$P(s) = (0.40,\ 0.30,\ 0.30)$. You observe **quiet**. Compute the evidence, the
posterior, and the surprise, and compare the surprise with the ${{surprise_quiet:.3f}}$ nats
obtained under the original prior.
---solution---
The quiet row is $(0.05,\ 0.10,\ 0.90)$. Numerators:

$$
(0.05 \times 0.40,\ 0.10 \times 0.30,\ 0.90 \times 0.30)
= ({{num_quiet_nervous_leopard:.3f}},\ {{num_quiet_nervous_baboon:.3f}},\ {{num_quiet_nervous_nothing:.3f}}).
$$

Evidence $P(o) = {{num_quiet_nervous_leopard:.3f}} + {{num_quiet_nervous_baboon:.3f}} + {{num_quiet_nervous_nothing:.3f}} = {{ev_quiet_nervous:.3f}}$, so the surprise is
$-\ln {{ev_quiet_nervous:.3f}} = {{surprise_quiet_nervous:.3f}}$ nats, nearly three times the ${{surprise_quiet:.3f}}$ nats of the calm
hiker. Posterior:

$$
\frac{({{num_quiet_nervous_leopard:.3f}},\ {{num_quiet_nervous_baboon:.3f}},\ {{num_quiet_nervous_nothing:.3f}})}{{{ev_quiet_nervous:.3f}}} = ({{post_quiet_nervous_leopard:.4f}},\ {{post_quiet_nervous_baboon:.5f}},\ {{post_quiet_nervous_nothing:.5f}}).
$$

Quiet still argues for an empty path, and it does so strongly enough to take
"nothing" from {{prior_nervous_nothing_pct:.0f}}% to {{post_quiet_nervous_nothing_pct:.0f}}%. But the nervous hiker is more surprised by quiet than
the calm one is, because their model expected trouble. Two agents, one
observation, different surprise. This is the concrete version of the claim in
Lesson&nbsp;2 that surprise is a property of the model and not of the world.
:::

::: exercise Surprise and information are not the same quantity
In the worked example the most surprising observation was also the most
informative. Show that this cannot be a general law, by constructing a model in
which the *rarest* observation carries *no* information at all.
---solution---
Take two states with a flat prior $P(s) = (0.5,\ 0.5)$, and three observations
with likelihood

$$
P(o \mid s) \;=\; \begin{pmatrix} 0.60 & 0.10 \\ 0.35 & 0.85 \\ 0.05 & 0.05 \end{pmatrix},
$$

rows indexed $o_1, o_2, o_3$ and columns by state. Both columns sum to one, so
this is a legitimate likelihood. The third row is deliberately flat: whatever the
state, $o_3$ turns up 5% of the time.

Working through [eq:bayes] for each observation:

| observed | $P(o)$ | surprise, nats | posterior | information gain, nats |
|---|---|---|---|---|
| $o_1$ | {{ctr_ev_1:.3f}} | {{ctr_surprise_1:.3f}} | ({{ctr_post_1_a:.3f}}, {{ctr_post_1_b:.3f}}) | {{ctr_info_1:.3f}} |
| $o_2$ | {{ctr_ev_2:.3f}} | {{ctr_surprise_2:.3f}} | ({{ctr_post_2_a:.3f}}, {{ctr_post_2_b:.3f}}) | {{ctr_info_2:.3f}} |
| $o_3$ | {{ctr_ev_3:.3f}} | {{ctr_surprise_3:.3f}} | ({{ctr_post_3_a:.3f}}, {{ctr_post_3_b:.3f}}) | {{ctr_info_3:.3f}} |

The rarest observation, $o_3$, is by some distance the most surprising, and it
leaves the posterior exactly where the prior was. The most informative
observation, $o_1$, is {{ctr_prob_ratio:.0f}} times more probable.

The structural reason is that the two quantities are averages of different
things. Averaging surprise over $P(o)$ gives the entropy $\mathrm{H}[P(o)]$;
averaging information gain over $P(o)$ gives the **mutual information** between
states and observations,

$$
I(s;o) \;=\; \sum_{o} P(o) \, D_{\mathrm{KL}}\big[P(s \mid o) \,\|\, P(s)\big],
$$

which measures how much, on average, an observation tells you about the state.
For the three-observation model above it is {{mi_counterexample:.3f}} nats: the average of the three
information gains, weighted by how often each observation actually occurs.
Nothing links them pointwise. An observation is surprising when the model
thought it unlikely; it is informative when the states disagree about it. A
sensory channel can easily be rare and useless at once, and an agent choosing
where to point its eyes had better be able to tell the difference. That
distinction becomes the two terms of expected free energy in Week&nbsp;10.
:::

::: checkpoint
- Why does a row of the likelihood matrix not sum to one, while a column does?
- Which single term in Bayes' theorem is responsible for the intractability, and why does the intractability grow multiplicatively rather than additively?
- If the likelihood is uniform in $o$ for every $s$, what is the posterior, and what is the information gain?
:::
