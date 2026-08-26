---
title: Bayes with states and observations
deck: The posterior is the belief an ideal agent would hold. Writing it down is easy; the trouble is entirely in one denominator.
week: 1
time: 30 min
scripts: [w01.js]
---

The agent has a generative model $P(o,s) = P(o \mid s)P(s)$. An observation
arrives. What should it now believe about $s$?

There is only one answer consistent with the axioms of probability, and it is
the same one you have seen since your first statistics course, dressed for this
particular occasion:

$$
P(s \mid o) \;=\; \frac{P(o \mid s)\,P(s)}{P(o)}, \qquad
P(o) \;=\; \sum_{s} P(o \mid s)\,P(s).
$$ {#bayes}

Everything in this course is a consequence of not being able to compute the
denominator. It is worth spending a page on why that is, because if you leave
this lesson thinking Bayes' theorem is hard, you have taken away the wrong
lesson. Bayes' theorem is trivial. The sum underneath it is not.

## The four quantities, named

::: notation
$P(s)$, the prior :: What the agent expects before it looks. Not a statement about the world's frequencies; a statement about the agent's commitments.
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
We will write this matrix $\mathbf{A}$ from Week&nbsp;9 onwards, with
$A_{ij} = P(o = i \mid s = j)$. Columns sum to one. Remember it now and the
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
\;=\; \big(\,& 0.056,\;\; 0.033,\;\; 0.014 \,\big).
\end{aligned}
$$

These are the three numerators in [eq:bayes]. Their sum is the denominator:

$$
P(o = \text{tawny flash}) \;=\; 0.056 + 0.033 + 0.014 \;=\; 0.103 .
$$ {#worked-evidence}

Divide through:

$$
P(s \mid o) \;=\; \frac{(0.056,\ 0.033,\ 0.014)}{0.103}
\;=\; (0.5437,\ 0.3204,\ 0.1359).
$$ {#worked-posterior}

::: mn Sanity check
$0.5437 + 0.3204 + 0.1359 = 1$. If your posterior does not sum to one you have
divided by the wrong thing, which in practice means you forgot a state.
:::

Read what happened. The leopard hypothesis started at 8% and finished at 54%,
a factor of nearly seven. It did not get there by being likely; it got there by
being the only state that predicts tawny flashes well. Meanwhile "nothing"
started as the overwhelming favourite at 70% and was demoted to 14%, because
whatever else is true, empty scrub does not flash tawny.

Two derived quantities are worth computing while the numbers are in front of us.

**Surprise.** From [eq:worked-evidence], $-\ln P(o) = -\ln 0.103 = 2.273$ nats.
Compare the other two observations: a shaking branch has $P(o) = 0.241$ and
surprise $1.423$ nats; quiet has $P(o) = 0.656$ and surprise $0.422$ nats.
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

For the tawny flash this is $0.940$ nats. For the shaking branch, $0.524$. For
quiet, $0.225$. The observation that surprised you most also taught you most,
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

This is worth naming, because it will return as a term in an equation in
Week&nbsp;10. When $P(o \mid s)$ is close to uniform in $o$ for every $s$, the
sensory channel is **ambiguous**: observations do not discriminate between
states, so observing costs you effort and buys you nothing. When $P(o \mid s)$ is
close to a permutation matrix, an observation pins the state down exactly.

An agent that can choose where to look has an interest in looking where the
likelihood is sharp. Hold that thought for nine weeks.

## Where the difficulty actually is

In the example above the denominator was a sum of three terms and you did it in
your head. Now count the terms for something less contrived.

An animal foraging tracks its own position, its energy reserves, the time since
it last saw a predator, the wind direction, whether each of several patches has
been depleted, and what its companions are doing. Each of these is a state
*factor*. The joint state space is the product of all of them, so if there are
$n$ factors with $k$ values each, the sum in [eq:bayes] has $k^n$ terms.

::: widget evidence-blowup | The number of terms in the evidence sum, on a logarithmic vertical axis. The point is not that the numbers are large. The point is which way the exponent sits.
:::

Nothing about this is fixable by better hardware. Adding one more thing to keep
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
posterior, and the surprise, and compare the surprise with the $0.422$ nats
obtained under the original prior.
---solution---
The quiet row is $(0.05,\ 0.10,\ 0.90)$. Numerators:

$$
(0.05 \times 0.40,\ 0.10 \times 0.30,\ 0.90 \times 0.30)
= (0.020,\ 0.030,\ 0.270).
$$

Evidence $P(o) = 0.020 + 0.030 + 0.270 = 0.320$, so the surprise is
$-\ln 0.320 = 1.139$ nats, nearly three times the $0.422$ nats of the calm
hiker. Posterior:

$$
\frac{(0.020,\ 0.030,\ 0.270)}{0.320} = (0.0625,\ 0.09375,\ 0.84375).
$$

Quiet still argues for an empty path, and it does so strongly enough to take
"nothing" from 30% to 84%. But the nervous hiker is more surprised by quiet than
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
| $o_1$ | 0.350 | 1.050 | (0.857, 0.143) | 0.283 |
| $o_2$ | 0.600 | 0.511 | (0.292, 0.708) | 0.089 |
| $o_3$ | 0.050 | 2.996 | (0.500, 0.500) | 0.000 |

The rarest observation, $o_3$, is by some distance the most surprising, and it
leaves the posterior exactly where the prior was. The most informative
observation, $o_1$, is six times more probable.

The structural reason is that the two quantities are averages of different
things. Averaging surprise over $P(o)$ gives the entropy $\mathrm{H}[P(o)]$;
averaging information gain over $P(o)$ gives the **mutual information** between
states and observations,

$$
I(s;o) \;=\; \sum_{o} P(o) \, D_{\mathrm{KL}}\big[P(s \mid o) \,\|\, P(s)\big],
$$

which measures how much, on average, an observation tells you about the state.
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
