---
title: Problems and code
deck: Eleven problems with full solutions, and the notebooks that let you rebuild this week's figures yourself.
week: 1
time: 180 min
scripts: [w01.js]
number_sections: false
---

Work the problems before opening the solutions. The ones marked with a dagger
are the ones that will be used again later in the course.

## The notebooks

Each week ships a notebook that rebuilds the week's figures from scratch. The
NumPy version is the canonical one and every week has it. Where a topic is
covered by `pymdp`, the standard active inference library, there is a second
notebook showing the same computation in library form so you can check your own
implementation against it. Where vectorising buys something real, there is a
JAX version too.

| Notebook | What it does | Run it |
|---|---|---|
| `week01_numpy.ipynb` | Bayes with states and observations, the evidence sum, explaining away, and the homeostasis simulation | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-01/week01_numpy.ipynb) |
| `week01_jax.ipynb` | The same, vectorised: posteriors over a grid of priors at once, and 10,000 homeostasis runs in parallel to get the survival curve | [Colab](https://colab.research.google.com/github/shocklab/Active-Inference-course/blob/main/notebooks/week-01/week01_jax.ipynb) |

There is no `pymdp` notebook this week. The library operates on the discrete
generative models we meet in Week&nbsp;9; there is nothing here for it to do yet.

## Problems

::: exercise A single alarm
An internal temperature sensor reports either an alarm or no alarm. The body is
in one of two states, overheated or normal. The prior probability of
overheating is $0.02$. The sensor raises the alarm with probability $0.90$ when
the body is overheated and with probability $0.05$ when it is normal.

(a) Write down the likelihood matrix $\mathbf{A}$, with observations down the
side and states across the top as in [lesson:1.3], and check which of its
rows and columns sum to one.

(b) The alarm sounds. Compute the evidence, the posterior probability of
overheating, and the surprise.

(c) The **odds** of an event with probability $p$ are $p/(1-p)$. Show that, for
two states $s_1$ and $s_2$ and an observation with $P(o \mid s_2) > 0$, Bayes'
theorem can be written as

$$
\frac{P(s_1 \mid o)}{P(s_2 \mid o)}
= \frac{P(o \mid s_1)}{P(o \mid s_2)} \cdot \frac{P(s_1)}{P(s_2)},
$$

posterior odds equal the likelihood ratio times the prior odds, and redo (b)
this way.

(d) A second alarm sounds, conditionally independent of the first given the
state. Find the posterior probability of overheating using the odds form.
---solution---
(a) With rows (alarm, no alarm) and columns (overheated, normal),

$$
\mathbf{A} = \begin{pmatrix} 0.90 & 0.05 \\ 0.10 & 0.95 \end{pmatrix}.
$$

The columns sum to one, since each is the distribution of the report given one
state. The rows do not: the alarm row sums to $0.90 + 0.05$ and the no-alarm row
to $0.10 + 0.95$, and neither has any reason to sum to one.

(b) The numerators $P(o \mid s)P(s)$ of Bayes' theorem are

$$
0.90 \times 0.02 = {{wu_num_hot:.3f}}, \qquad
0.05 \times (1 - 0.02) = {{wu_num_normal:.3f}} .
$$

Their sum is the evidence, $P(\text{alarm}) = {{wu_ev:.3f}}$, so the posterior
probability of overheating is ${{wu_num_hot:.3f}}/{{wu_ev:.3f}} = {{wu_post:.4f}}$,
and the surprise is $-\ln {{wu_ev:.3f}} = {{wu_surprise:.3f}}$ nats.

A sensor that catches nine overheatings in ten leaves overheating less likely
than not after it fires, at about {{wu_post_pct:.0f}}%. The reason is the
prior: normal states are so common that their occasional false alarms
outnumber the true ones, ${{wu_num_normal:.3f}}$ against ${{wu_num_hot:.3f}}$.
In whole numbers: of {{wu_N:,}} bodies, {{wu_n_hot}} are overheated and
{{wu_n_hot_alarm}} of those raise the alarm; {{wu_n_normal:,}} are normal and
{{wu_n_normal_alarm}} of those raise it too. Of the {{wu_n_alarm}} alarms,
{{wu_n_hot_alarm}} come from overheated bodies, and
${{wu_n_hot_alarm}}/{{wu_n_alarm}} = {{wu_post:.4f}}$.

(c) Write Bayes' theorem for each state and divide:

$$
\frac{P(s_1 \mid o)}{P(s_2 \mid o)}
= \frac{P(o \mid s_1)P(s_1)/P(o)}{P(o \mid s_2)P(s_2)/P(o)}
= \frac{P(o \mid s_1)}{P(o \mid s_2)} \cdot \frac{P(s_1)}{P(s_2)} .
$$

The evidence cancels. With only two states, $P(s_2) = 1 - P(s_1)$ and
$P(s_2 \mid o) = 1 - P(s_1 \mid o)$, so the two ratios of state probabilities
are odds in the sense just defined. Here the prior odds are
$0.02/(1-0.02) = 1/{{wu_prior_odds_inv}}$ and the likelihood ratio is
$0.90/0.05 = {{wu_lr:.0f}}$, so the posterior odds are
${{wu_lr:.0f}}/{{wu_prior_odds_inv}} = {{wu_post_odds:.4f}}$. To convert
odds back to a probability, solve $\text{odds} = p/(1-p)$ for $p$, which gives
$p = \text{odds}/(1+\text{odds})$. Keeping the fraction,
$p = ({{wu_lr:.0f}}/{{wu_prior_odds_inv}})/(1 + {{wu_lr:.0f}}/{{wu_prior_odds_inv}}) = {{wu_lr:.0f}}/({{wu_prior_odds_inv}} + {{wu_lr:.0f}}) = {{wu_lr:.0f}}/{{wu_post_den}} = {{wu_post:.4f}}$,
as in (b).

(d) The posterior after the first alarm is the prior for the second, so the
odds are multiplied by the same likelihood ratio again:

$$
\frac{ {{wu_lr:.0f}}^2 }{ {{wu_prior_odds_inv}} } = \frac{ {{wu_lr_sq}} }{ {{wu_prior_odds_inv}} } = {{wu_two_odds:.3f}},
\qquad
P(\text{overheated} \mid \text{two alarms}) = \frac{{{wu_two_odds:.3f}}}{1 + {{wu_two_odds:.3f}}} = {{wu_two_post:.3f}} .
$$

Bayes' theorem applied directly to both alarms gives the same answer. The
likelihoods of two alarms are $0.90^2$ and $0.05^2$, so the numerators are
${{wu_two_num_hot:.4f}}$ and ${{wu_two_num_normal:.5f}}$, the evidence is
${{wu_two_ev:.5f}}$, and the posterior is
${{wu_two_num_hot:.4f}}/{{wu_two_ev:.5f}} = {{wu_two_post:.3f}}$.

The odds form never needs the evidence, and each conditionally independent
observation contributes one multiplicative factor. Taking logarithms turns the
product into a sum, so in log odds each observation adds its log likelihood
ratio. Here the prior log odds are $\ln(1/{{wu_prior_odds_inv}}) = {{wu_logodds_prior:.4f}}$
and each alarm adds $\ln {{wu_lr:.0f}} = {{wu_log_lr:.4f}}$, giving
${{wu_logodds_one:.4f}}$ after one alarm and ${{wu_logodds_two:.4f}}$ after two.
The sign changes between them, which is the moment overheating becomes the
more probable state. The problem on what the prior decides, below, uses a
variant of this form, in which the second hypothesis is "not leopard" and its
likelihood is a prior-weighted mixture of the baboon and nothing columns.
:::

::: exercise Normalisation, both ways &dagger;
Let $\mathbf{A}$ be the likelihood matrix with $A_{ij} = P(o = i \mid s = j)$.

(a) State which of the rows and columns of $\mathbf{A}$ sum to one, and why.

(b) Show that $\sum_i P(o = i) = 1$ follows from that, for any prior $P(s)$.

(c) A colleague builds a model in which the *rows* of $\mathbf{A}$ have been
normalised by mistake. Describe, without computing anything, what goes wrong.
---solution---
(a) The columns. Column $j$ is the distribution $P(o \mid s = j)$ over all
possible observations given that the state is $j$. It is a probability
distribution over $o$ and so must sum to one. A row is $P(o = i \mid s)$ read as
a function of $s$: a likelihood, not a distribution, and it has no reason to sum
to anything in particular.

(b) Write $P(o = i) = \sum_j A_{ij} P(s = j)$. Then

$$
\sum_i P(o = i) = \sum_i \sum_j A_{ij} P(s=j)
= \sum_j P(s=j) \sum_i A_{ij}
= \sum_j P(s=j) \cdot 1 = 1,
$$

where the interchange of the two finite sums is unproblematic and the inner sum
is one by (a).

(c) Row-normalising makes $\sum_i A_{ij} \neq 1$ in general, so the columns are
no longer distributions over observations. The immediate symptom is that
$\sum_i P(o=i) \neq 1$: the evidence no longer defines a probability
distribution, so the surprise $-\ln P(o)$ is not comparable across observations
and can even be negative. Bayes' theorem still returns a normalised posterior,
because the division by $P(o)$ hides the error, which is exactly what makes this
mistake hard to find. Check the column sums.
:::

::: exercise What the prior decides
Using the leopard, baboon and nothing model from [lesson:1.3], write $\rho$ for the prior
probability that a leopard is present, and find the value of $\rho$ at which observing a tawny flash leaves you exactly
undecided between leopard and not-leopard, that is,
$P(s = \text{leopard} \mid o) = 1/2$. Keep the baboon and nothing priors in the
ratio $22 : 70$ as $\rho$ varies.
---solution---
Let the leopard prior be $\rho$. The other two share $1 - \rho$ in the ratio
$22:70$, so they are $\tfrac{22}{92}(1-\rho)$ and $\tfrac{70}{92}(1-\rho)$. The
tawny-flash row is $(0.70, 0.15, 0.02)$. The posterior odds of leopard against
everything else are

$$
\frac{0.70\,\rho}{\left(0.15 \cdot \tfrac{22}{92} + 0.02 \cdot \tfrac{70}{92}\right)(1-\rho)}
= \frac{0.70\,\rho}{{{tip_coef:.6f}}\,(1-\rho)} .
$$

Setting this to 1 gives $0.70\rho = {{tip_coef:.6f}}(1-\rho)$, so
$\rho = {{tip_coef:.6f}} / {{tip_denominator:.6f}} = {{tip_q:.4f}}$.

So a leopard prior of about {{tip_q_pct:.1f}}% is the tipping point: the model in
[lesson:1.3] used $q = 0.08$, which is just above it, which is why the
posterior came out at ${{post_tawny_leopard:.3f}}$ rather than exactly a half. The observation is
strong, but it is only just strong enough to overturn a prior of that size, and
a slightly more sceptical hiker would have remained unconvinced by the same
evidence.
:::

::: exercise Two observations in a row &dagger;
You observe a branch shaking, and then, a moment later, a tawny flash. Assume
the two observations are conditionally independent given the state, and that the
state has not changed.

(a) Compute the posterior after both observations, starting from the
[lesson:1.3] prior.

(b) Show that you get the same answer whichever order you process them in.

(c) What feature of the model makes (b) true, and give a physical situation in
which it would fail.
---solution---
(a) After the shaking branch the posterior was $({{post_branch_leopard:.6f}}, {{post_branch_baboon:.6f}}, {{post_branch_nothing:.6f}})$.
Use this as the prior and apply the tawny-flash row $(0.70, 0.15, 0.02)$:

$$
(0.70 \times {{post_branch_leopard:.6f}},\ 0.15 \times {{post_branch_baboon:.6f}},\ 0.02 \times {{post_branch_nothing:.6f}})
= ({{seq_num_leopard:.6f}},\ {{seq_num_baboon:.6f}},\ {{seq_num_nothing:.6f}}).
$$

The sum is ${{seq_ev:.6f}}$, giving a posterior of
$({{seq_post_leopard:.5f}},\ {{seq_post_baboon:.5f}},\ {{seq_post_nothing:.5f}})$.

Baboon is now the leading hypothesis at 62%. The shaking branch was strong
evidence for a baboon, and although the tawny flash pulls back towards leopard,
it does not pull far enough.

(b) Processing the flash first gives posterior
$({{post_tawny_leopard:.6f}}, {{post_tawny_baboon:.6f}}, {{post_tawny_nothing:.6f}})$, then applying the branch row
$(0.25, 0.75, 0.08)$:

$$
({{alt_num_leopard:.6f}},\ {{alt_num_baboon:.6f}},\ {{alt_num_nothing:.6f}}), \quad \text{sum } {{alt_ev:.6f}},
$$

which normalises to $({{seq_post_leopard:.5f}},\ {{seq_post_baboon:.5f}},\ {{seq_post_nothing:.5f}})$: identical.

(c) Conditional independence given the state. Under it, the joint likelihood
factorises as $P(o_1, o_2 \mid s) = P(o_1 \mid s) P(o_2 \mid s)$, so the
posterior is proportional to $P(s) P(o_1\mid s) P(o_2 \mid s)$, and
multiplication commutes. It fails whenever one observation influences the next
other than through the state. A concrete case: you shout after seeing the branch
move. Whatever is in the scrub reacts to the shout, so the second observation
depends on the first directly, not only through what the animal is. Sequential
active inference agents create exactly this kind of dependence on purpose, which
is why Week&nbsp;9 has to be careful about the difference between the time index
on an observation and the time index on a belief.
:::

::: exercise When does an observation tell you nothing?
Prove that the posterior equals the prior for every prior if and only if
$P(o \mid s)$ does not depend on $s$.
---solution---
($\Leftarrow$) Suppose $P(o\mid s) = c$ for all $s$, for the observed $o$. Then
$P(o) = \sum_s c\,P(s) = c$, and

$$
P(s \mid o) = \frac{c\,P(s)}{c} = P(s).
$$

($\Rightarrow$) Suppose $P(s\mid o) = P(s)$ for every prior. Bayes gives
$P(o\mid s)P(s) = P(o)P(s)$ for every $s$, so wherever $P(s) > 0$ we may divide
to get $P(o\mid s) = P(o)$, a quantity that does not depend on $s$. Taking a
prior with full support forces this at every $s$.

This is the formal content of the ambiguity slider. A sensory channel is useful
exactly to the extent that different states disagree about what it will report.
:::

::: exercise Posterior entropy
Use the leopard, baboon and nothing model of [lesson:1.3], with prior
$(0.08, 0.22, 0.70)$.

(a) Compute the entropy of the prior, and the entropy of the posterior after
each of the three observations. Which observations increase the entropy of the
agent's beliefs about the state?

(b) Show that for any finite model

$$
\sum_o P(o)\,\mathrm{H}[P(s \mid o)] \;=\; \mathrm{H}[P(s)] \;-\; I(s;o),
$$

where $I(s;o)$ is the mutual information defined in [lesson:1.3].

(c) Check (b) numerically for this model, and explain why your answer to (a)
does not contradict it.
---solution---
(a) The prior entropy is

$$
\mathrm{H}[P(s)] = -(0.08 \ln 0.08 + 0.22 \ln 0.22 + 0.70 \ln 0.70) = {{pe_prior:.5f}} \text{ nats}.
$$

For comparison, the largest entropy on three states is
$\ln 3 = {{pe_max3:.4f}}$ nats, reached by the uniform distribution.

Each posterior is the numerators of Bayes' theorem divided by the evidence, as
in [lesson:1.3], which gives the tawny-flash case in full and lists the
numerators and evidence for the other two:

$$
\frac{({{num_branch_leopard:.3f}},\ {{num_branch_baboon:.3f}},\ {{num_branch_nothing:.3f}})}{{{ev_branch:.3f}}},
\qquad
\frac{({{num_quiet_leopard:.3f}},\ {{num_quiet_baboon:.3f}},\ {{num_quiet_nothing:.3f}})}{{{ev_quiet:.3f}}}
$$

for the shaking branch and for quiet. For the tawny flash the entropy is

$$
-({{post_tawny_leopard:.4f}} \ln {{post_tawny_leopard:.4f}} + {{post_tawny_baboon:.4f}} \ln {{post_tawny_baboon:.4f}} + {{post_tawny_nothing:.4f}} \ln {{post_tawny_nothing:.4f}}) = {{pe_tawny:.5f}} \text{ nats},
$$

and the same formula gives the other two:

| observed | $P(o)$ | posterior | entropy, nats | change, nats |
|---|---|---|---|---|
| tawny flash | {{ev_tawny:.3f}} | ({{post_tawny_leopard:.3f}}, {{post_tawny_baboon:.3f}}, {{post_tawny_nothing:.3f}}) | {{pe_tawny:.5f}} | ${{pe_change_tawny:+.4f}}$ |
| branch shakes | {{ev_branch:.3f}} | ({{post_branch_leopard:.3f}}, {{post_branch_baboon:.3f}}, {{post_branch_nothing:.3f}}) | {{pe_branch:.5f}} | ${{pe_change_branch:+.4f}}$ |
| quiet | {{ev_quiet:.3f}} | ({{post_quiet_leopard:.3f}}, {{post_quiet_baboon:.3f}}, {{post_quiet_nothing:.3f}}) | {{pe_quiet:.5f}} | ${{pe_change_quiet:+.4f}}$ |

The tawny flash and the shaking branch both increase the entropy. The prior
puts most of its mass on "nothing". The tawny flash moves mass from there
onto leopard and baboon, and the shaking branch moves it almost entirely onto
baboon; either way the belief ends up less concentrated. After a tawny flash
the entropy is close to the three-state maximum of ${{pe_max3:.4f}}$. The agent
is less certain of the state than before, although the observation moved its
beliefs further than any other: its information gain was ${{info_tawny:.3f}}$
nats. Entropy measures how spread a belief is, and information gain measures
how far it moved, so a large movement away from a concentrated belief can
raise both.

(b) Start from the definition of $I(s;o)$ and split the logarithm of the ratio:

$$
\begin{aligned}
I(s;o) &= \sum_o P(o) \sum_s P(s \mid o) \ln \frac{P(s \mid o)}{P(s)} \\
&= \sum_o P(o) \sum_s P(s \mid o) \ln P(s \mid o) \;-\; \sum_o P(o) \sum_s P(s \mid o) \ln P(s) .
\end{aligned}
$$

The inner sum of the first term is $-\mathrm{H}[P(s \mid o)]$, so the first term
is $-\sum_o P(o)\,\mathrm{H}[P(s \mid o)]$. In the second term, exchange the two
finite sums and use the product rule:

$$
\begin{aligned}
\sum_o P(o) \sum_s P(s \mid o) \ln P(s)
&= \sum_s \ln P(s) \sum_o P(o)\,P(s \mid o) \\
&= \sum_s \ln P(s) \sum_o P(s, o) \\
&= \sum_s P(s) \ln P(s) = -\mathrm{H}[P(s)] .
\end{aligned}
$$

The sums over $o$ run over observations with $P(o) > 0$, for which the posterior
is defined. Terms with $P(s \mid o) = 0$ contribute zero under the convention
$0 \ln 0 = 0$ of [lesson:1.1], and $P(s) = 0$ forces $P(s \mid o) = 0$, so no
logarithm of zero survives. Together,

$$
I(s;o) = \mathrm{H}[P(s)] - \sum_o P(o)\,\mathrm{H}[P(s \mid o)],
$$

which rearranges to the identity asked for. In words: the amount by which an
observation lowers the entropy of beliefs, averaged over the observations, is
the mutual information.

(c) The weighted average of the posterior entropies is

$$
\begin{aligned}
&{{ev_tawny:.3f}} \times {{pe_tawny:.4f}} + {{ev_branch:.3f}} \times {{pe_branch:.4f}} + {{ev_quiet:.3f}} \times {{pe_quiet:.4f}} \\
&\quad= {{pe_wterm_tawny:.4f}} + {{pe_wterm_branch:.4f}} + {{pe_wterm_quiet:.4f}} = {{pe_avg:.4f}} \text{ nats},
\end{aligned}
$$

and ${{pe_prior:.4f}} - {{pe_avg:.4f}} = {{mi_leopard_model:.4f}}$. Computing the
mutual information directly, as the average of the three information gains of
[lesson:1.3],

$$
{{ev_tawny:.3f}} \times {{info_tawny:.4f}} + {{ev_branch:.3f}} \times {{info_branch:.4f}} + {{ev_quiet:.3f}} \times {{info_quiet:.4f}} = {{mi_leopard_model:.4f}} \text{ nats},
$$

gives the same value. In the unrounded stored values the two routes differ
by less than $10^{-12}$.

Mutual information is an average of KL divergences, each non-negative by the
argument in [lesson:1.2], so $I(s;o) \ge 0$ and the identity gives

$$
\sum_o P(o)\,\mathrm{H}[P(s \mid o)] \le \mathrm{H}[P(s)] .
$$

On average over the observations the model expects, an observation cannot
increase the entropy of beliefs about the state. Nothing constrains a single
observation, and (a) has two of three increasing it. They are outweighed by
quiet, which the model expects with probability ${{ev_quiet:.3f}}$ and which
lowers the entropy by ${{pe_drop_quiet:.4f}}$ nats. The average in this
statement is taken under the model's own $P(o)$. If the world produced
observations with different frequencies, the average under those frequencies
could rise: in a world that produced a tawny flash every time, each update
from this prior would raise it.
:::

::: exercise The two forms of mutual information &dagger;
(a) Show that

$$
I(s;o) = \sum_{s,o} P(s,o) \ln \frac{P(s,o)}{P(s)\,P(o)}
$$

and

$$
I(s;o) = \mathrm{H}[P(o)] - \sum_s P(s)\,\mathrm{H}[P(o \mid s)] .
$$

Explain why the first form shows that $I(s;o)$ is symmetric in $s$ and $o$,
although each information gain $D_{\mathrm{KL}}[P(s \mid o)\,\|\,P(s)]$ is a
statement about states alone.

(b) Evaluate the second form for the leopard model and compare it with the
previous problem.

(c) Show that $0 \le I(s;o) \le \min\big(\mathrm{H}[P(s)],\, \mathrm{H}[P(o)]\big)$.

(d) The channel-noise weight $\alpha$ of [lesson:1.3] replaces $\mathbf{A}$ by
$\mathbf{A}(\alpha) = (1-\alpha)\mathbf{A} + \alpha U$, where every entry of $U$
is $1/3$. Compute $I(s;o)$ for the leopard prior at
$\alpha = 0.25$, $0.5$ and $0.75$, and compare it with the straight line
$(1-\alpha)\,I_0$, where $I_0$ is the value at $\alpha = 0$. Prove that
$I(s;o) \le (1-\alpha)\,I_0$ for every $\alpha \in [0,1]$.
---solution---
(a) By the product rule, $P(o)\,P(s \mid o) = P(s,o)$ and
$P(s \mid o)/P(s) = P(s,o)/[P(s)P(o)]$. Substituting both into the definition,

$$
I(s;o) = \sum_o \sum_s P(o)\,P(s \mid o) \ln \frac{P(s \mid o)}{P(s)}
= \sum_{s,o} P(s,o) \ln \frac{P(s,o)}{P(s)\,P(o)} .
$$

Define $I(o;s)$ by the same recipe with the roles exchanged,
$\sum_s P(s)\,D_{\mathrm{KL}}[P(o \mid s)\,\|\,P(o)]$. The same two substitutions
turn it into the same double sum, because the joint, the product of marginals
and the double sum are all unchanged when the names $s$ and $o$ are swapped.
So $I(s;o) = I(o;s)$. The symmetry belongs to the average: a single
term $D_{\mathrm{KL}}[P(s \mid o)\,\|\,P(s)]$ has no partner with the roles
reversed that it must equal.

For the second form, write the ratio the other way round,
$P(s,o)/[P(s)P(o)] = P(o \mid s)/P(o)$, write $P(s,o) = P(s)P(o \mid s)$ by the
product rule in the first term, and split the logarithm:

$$
\begin{aligned}
I(s;o) &= \sum_s P(s) \sum_o P(o \mid s) \ln P(o \mid s) \;-\; \sum_o \ln P(o) \sum_s P(s,o) \\
&= -\sum_s P(s)\,\mathrm{H}[P(o \mid s)] + \mathrm{H}[P(o)] ,
\end{aligned}
$$

using $\sum_s P(s,o) = P(o)$ in the second term. This is the computation of the
previous problem with the roles of $s$ and $o$ exchanged.

(b) The evidence vector is $P(o) = ({{ev_tawny:.3f}}, {{ev_branch:.3f}}, {{ev_quiet:.3f}})$,
with entropy ${{mi_H_o:.4f}}$ nats. The columns of $\mathbf{A}$ have entropies
${{mi_Hcol_leopard:.4f}}$ (leopard), ${{mi_Hcol_baboon:.4f}}$ (baboon) and
${{mi_Hcol_nothing:.4f}}$ (nothing). Weighting by the prior,

$$
0.08 \times {{mi_Hcol_leopard:.4f}} + 0.22 \times {{mi_Hcol_baboon:.4f}} + 0.70 \times {{mi_Hcol_nothing:.4f}} = {{mi_Hcond:.4f}},
$$

so $I(s;o) = {{mi_H_o:.4f}} - {{mi_Hcond:.4f}} = {{mi_obs_side:.4f}}$ nats, the
value found from the state side in the previous problem. No posterior and no
entropy appears in both routes; they share only the joint distribution and
the evidence vector computed from it.

(c) The lower bound holds because $I(s;o)$ is a $P(o)$-weighted average of KL
divergences, each non-negative by [lesson:1.2]. Every entropy is non-negative,
since each term $-p \ln p$ is non-negative for $p \in [0,1]$. The previous
problem's identity therefore gives
$I(s;o) = \mathrm{H}[P(s)] - \sum_o P(o)\,\mathrm{H}[P(s\mid o)] \le \mathrm{H}[P(s)]$,
and the second form in (a) gives $I(s;o) \le \mathrm{H}[P(o)]$ in the same way.
For the leopard model, ${{mi_obs_side:.4f}}$ sits below both
${{pe_prior:.4f}}$ and ${{mi_H_o:.4f}}$. An observation cannot tell you more
about the state, on average, than you were uncertain about to begin with, nor
more than the observation itself varies.

(d) Computed from the definition with $\mathbf{A}(\alpha)$ in place of $\mathbf{A}$:

| $\alpha$ | $I(s;o)$, nats | $(1-\alpha)\,I_0$, nats | $I(s;o)/I_0$ |
|---|---|---|---|
| 0.25 | {{mi_alpha_25:.4f}} | {{mi_line_25:.4f}} | {{mi_frac_25:.3f}} |
| 0.5 | {{mi_alpha_50:.4f}} | {{mi_line_50:.4f}} | {{mi_frac_50:.3f}} |
| 0.75 | {{mi_alpha_75:.4f}} | {{mi_line_75:.4f}} | {{mi_frac_75:.3f}} |

with $I_0 = {{mi_leopard_model:.4f}}$. For this model the information falls
faster than the line: mixed halfway towards uniform, the channel keeps
{{mi_frac_50_pct:.0f}}% of it. The bound proved below guarantees only that it
keeps at most half, and other models come closer to that bound.

The idea of the proof is a chord. At $\alpha = 1$ every column of
$\mathbf{A}(\alpha)$ is the same, so every observation has the same probability
under every state and $I = 0$. If $I$ is a convex function of $\alpha$ along the
segment, its graph lies below the straight line joining $(0, I_0)$ to $(1, 0)$,
which is $(1-\alpha)I_0$. The log-sum inequality is the tool that supplies the
convexity of each KL divergence.

For the proof, use the form of (a) with $P(o \mid s)/P(o)$ in the logarithm and
$P(s,o) = P(s)P(o \mid s)$:

$$
I(s;o) = \sum_s P(s) \sum_o P(o \mid s) \ln \frac{P(o \mid s)}{P(o)} ,
$$

whose inner sum is a KL divergence, so

$$
I(s;o) = \sum_s P(s)\, D_{\mathrm{KL}}\big[P(o \mid s) \,\|\, P(o)\big] .
$$

Write $p_s$ for column $s$ of $\mathbf{A}$, $q = \sum_s P(s)\,p_s$ for the
evidence vector, and $\bar p = (\tfrac13, \tfrac13, \tfrac13)$. Under
$\mathbf{A}(\alpha)$ the columns are $(1-\alpha)p_s + \alpha \bar p$, and because the
prior sums to one the evidence vector is
$\sum_s P(s)[(1-\alpha)p_s + \alpha \bar p] = (1-\alpha)q + \alpha \bar p$. So it is
enough to show that, for distributions $p$, $q$ and $\bar p$ on the observations,

$$
\begin{aligned}
D_{\mathrm{KL}}\big[(1-\alpha)p + \alpha \bar p \,\big\|\, (1-\alpha)q + \alpha \bar p\big]
&\le (1-\alpha)\,D_{\mathrm{KL}}[p \,\|\, q] + \alpha\,D_{\mathrm{KL}}[\bar p \,\|\, \bar p] \\
&= (1-\alpha)\,D_{\mathrm{KL}}[p \,\|\, q] ,
\end{aligned}
$$

and then average over $s$ with weights $P(s)$.

The inequality follows from the **log-sum inequality**: for non-negative
$c_1, c_2$ and positive $\phi_1, \phi_2$,

$$
(c_1 + c_2) \ln \frac{c_1 + c_2}{\phi_1 + \phi_2} \le c_1 \ln \frac{c_1}{\phi_1} + c_2 \ln \frac{c_2}{\phi_2} .
$$

To prove it, let $\zeta(y) = y \ln y$, with $\zeta(0) = 0$. It is convex on
$y > 0$ because $\zeta''(y) = 1/y > 0$ there, and since it is continuous on
$y \ge 0$ the convexity inequality extends to $y = 0$ by taking limits. Set
$\omega = \phi_1/(\phi_1 + \phi_2)$, so that $1 - \omega = \phi_2/(\phi_1+\phi_2)$ and

$$
\frac{c_1 + c_2}{\phi_1 + \phi_2} = \omega \frac{c_1}{\phi_1} + (1-\omega)\frac{c_2}{\phi_2} .
$$

Convexity gives $\zeta$ of this point at most
$\omega\, \zeta(c_1/\phi_1) + (1-\omega)\, \zeta(c_2/\phi_2)$.
Multiplying through by $\phi_1 + \phi_2$,

$$
(\phi_1 + \phi_2)\, \zeta\!\left(\frac{c_1 + c_2}{\phi_1 + \phi_2}\right) \le \phi_1\, \zeta\!\left(\frac{c_1}{\phi_1}\right) + \phi_2\, \zeta\!\left(\frac{c_2}{\phi_2}\right),
$$

and writing out $\zeta$ gives the log-sum inequality. Now take $0 < \alpha < 1$
and assume $q(o) > 0$ for every $o$, which holds for the leopard model because
no row of $\mathbf{A}$ is zero. Apply the inequality for each observation $o$
with $c_1 = (1-\alpha)p(o)$, $c_2 = \alpha \bar p(o)$, $\phi_1 = (1-\alpha)q(o)$
and $\phi_2 = \alpha \bar p(o)$. The left side is

$$
\big[(1-\alpha)p(o) + \alpha \bar p(o)\big] \ln \frac{(1-\alpha)p(o) + \alpha \bar p(o)}{(1-\alpha)q(o) + \alpha \bar p(o)} ,
$$

and summing over $o$ turns it into the divergence between the two mixtures. On the right, the factors $(1-\alpha)$ and $\alpha$
cancel inside each logarithm, leaving
$(1-\alpha)\sum_o p(o)\ln[p(o)/q(o)] + \alpha \sum_o \bar p(o) \ln 1$. That is the
inequality required. At $\alpha = 0$ or $\alpha = 1$ one of the terms is absent
and the result holds with equality.

What has been proved is the chord bound along the segment from $\mathbf{A}$ to
the uniform channel. Replacing $\bar p$ by the columns of a second arbitrary
channel, the same argument shows that mutual information is convex in the
channel for a fixed prior, but the problem needs only the special case. The widget's information-gain readout is a single observation's
KL divergence, not this average, and is not covered by the result.
:::

::: exercise Explaining away with three causes
Extend the noisy-OR example of [lesson:1.4] to three independent causes, each
with prior $0.1$ and each firing the observation with probability $0.9$, keeping
the leak at $0.01$. Compute $P(s_1 = 1 \mid o = 1)$ and compare it with the
two-cause answer of ${{noisy_marg_gust:.3f}}$. Explain the direction of the change.
---solution---
There are eight joint states. Write $k$ for the number of causes present. The
likelihood depends only on $k$:

$$
P(o=1 \mid k) = 1 - (0.1)^k (0.99),
$$

giving ${{noisy_lik_k0:.2f}}, {{noisy_lik_k1:.3f}}, {{noisy_lik_k2:.4f}}, {{noisy_lik_k3:.5f}}$ for $k = 0,1,2,3$. The prior for a state
with $k$ causes present is $0.1^k \, 0.9^{3-k}$, and there are $\binom{3}{k}$
such states. So

$$
P(o=1) = \sum_{k=0}^{3} \binom{3}{k} 0.1^k 0.9^{3-k} \left[1 - 0.99 \times 0.1^k\right]
= {{noisy3_ev:.5f}} .
$$

For the marginal, sum the unnormalised mass over the four states with $s_1 = 1$:

$$
P(s_1=1, o=1) = 0.1 \sum_{k'=0}^{2} \binom{2}{k'} 0.1^{k'} 0.9^{2-k'}
\left[1 - 0.99 \times 0.1^{\,k'+1}\right] = {{noisy3_joint_mass:.5f}} ,
$$

so $P(s_1 = 1 \mid o = 1) = {{noisy3_joint_mass:.5f}} / {{noisy3_ev:.5f}} = {{noisy3_marg:.4f}}$.

The answer *falls* from ${{noisy_marg_gust:.3f}}$ to ${{noisy3_marg:.3f}}$. With more candidate explanations
available, the evidence is spread more thinly across them: the observation still
says "something did this", but there are now three somethings to choose from and
each individually gets less of the credit. This comparison concerns these two specified models. It does not establish a general monotonic relation between cause count and approximation error.
:::

::: exercise The survival curve &dagger;
Consider the homeostasis model from [lesson:1.1] with the action strength set
to zero, $\kappa = 0$. It is then a two-dimensional random walk with independent increments of standard deviation
$\sigma$ per step in each coordinate, starting at the origin, absorbed when
$\|x\| > 1$.

(a) Use a small-step diffusion approximation to motivate an expected absorption time proportional to $\sigma^{-2}$. Why need this not be exact for a discrete walk?

(b) The notebook measures this by simulation. Predict the ratio of survival
times at $\sigma = 0.09$ and $\sigma = 0.035$ before you run it.
---solution---
(a) After $n$ steps the walk's displacement has variance $n\sigma^2$ in each
coordinate, so the typical distance from the origin is of order
$\sigma\sqrt{n}$. Absorption happens when this reaches the boundary at radius
$1$, so $\sigma\sqrt{n} \sim 1$ and $n \sim \sigma^{-2}$. This estimates the scaling, not an exact discrete first-passage time. A finite
step can overshoot the boundary; the approximation improves as the step size
becomes small relative to the radius.

(b) The ratio is $({{sigma_hi}} / {{sigma_lo}})^{-2} = ({{sigma_lo}}/{{sigma_hi}})^2 = {{lifetime_ratio:.3f}}$. Halving the
noise should quadruple the lifetime, and the roughly {{sigma_ratio:.1f}}-fold increase in
$\sigma$ here should cut it by a factor of about ${{lifetime_factor:.1f}}$.

For stable regulation the unabsorbed process has a stationary variance, but an
absorbing boundary can still be crossed. Compare survival distributions at
fixed observation horizons rather than inferring indefinite survival from
bounded variance.
:::

::: exercise The regulated walk
Take the homeostat $x_{t+1} = (1-\kappa)x_t + \sigma\xi_t$ of [lesson:1.1] with
$0 < \kappa < 2$ and no absorbing boundary, starting from $x_0 = 0$. Write $v_t$ for the variance of one
coordinate of $x_t$, and $v = \sigma^2/[\kappa(2-\kappa)]$ for its stationary variance.

(a) Solve the recursion $v_{t+1} = (1-\kappa)^2 v_t + \sigma^2$ from [lesson:1.1]
exactly, and show that at fixed $t$ the result tends to $t\sigma^2$ as
$\kappa \to 0$.

(b) At the lesson's gain $\kappa = {{hs_kappa}}$, after how many steps does
$v_t$ first reach 95% of $v$?

(c) Show that the covariance of one coordinate at times $t$ and $t + j$ is
$(1-\kappa)^j v_t$ for every whole number $j \ge 0$, so that it tends to
$(1-\kappa)^j v$ once $t$ is large. What does a gain above one do to it?

(d) For large $t$, when $v_t$ has settled at $v$, find the probability that
$x_t$ lies outside the unit circle. Evaluate it at $\sigma = {{hs_sigma}}$ and
$\kappa = {{hs_kappa}}$, and say what it does and does not tell you about
survival.
---solution---
(a) Measure the variance from its stationary value, $v_t = v + \delta_t$.
Substituting into the recursion,

$$
v + \delta_{t+1} = (1-\kappa)^2 (v + \delta_t) + \sigma^2 .
$$

The stationary value satisfies $v = (1-\kappa)^2 v + \sigma^2$, and subtracting
that equation leaves $\delta_{t+1} = (1-\kappa)^2 \delta_t$. So
$\delta_t = (1-\kappa)^{2t} \delta_0$, with $\delta_0 = v_0 - v = -v$ because the walk starts
at the origin with no spread. Hence

$$
v_t = v\left[1 - (1-\kappa)^{2t}\right] = \frac{\sigma^2\left[1 - (1-\kappa)^{2t}\right]}{\kappa(2-\kappa)} .
$$

For the limit, expand the numerator by the binomial theorem,
$(1-\kappa)^{2t} = 1 - 2t\kappa + O(\kappa^2)$, so
$1 - (1-\kappa)^{2t} = 2t\kappa + O(\kappa^2)$. The denominator is
$2\kappa - \kappa^2$. Dividing top and bottom by $\kappa$,

$$
v_t = \sigma^2\,\frac{2t + O(\kappa)}{2 - \kappa} \;\longrightarrow\; t\sigma^2
\quad\text{as } \kappa \to 0 ,
$$

which is the variance of the unregulated walk in [lesson:1.1]. As a check at a
small gain, $\kappa = 0.01$ and $t = 10$ give $v_t/\sigma^2 = {{reg_check_ratio:.2f}}$
against $t = 10$. The limit is taken at fixed $t$. For any fixed $\kappa > 0$ the factor $(1-\kappa)^{2t}$
eventually becomes negligible, and the variance stops growing and settles at $v$.

(b) From (a), $v_t \ge 0.95\,v$ exactly when $(1-\kappa)^{2t} \le 0.05$. Taking
logarithms, and noting that $\ln(1-\kappa) < 0$ reverses the inequality on
division,

$$
t \ge \frac{\ln 0.05}{2\ln(1-\kappa)} = {{reg_settle_95:.2f}} ,
$$

so the first such step is $t = {{reg_settle_step}}$: after that many steps the
variance is within 5% of its stationary value. The next problem relies on this
when it discards the start of a simulated run.

(c) Applying the homeostat recursion $j$ times, by induction on $j$,

$$
x_{t+j} = (1-\kappa)^j x_t + \sigma \sum_{i=0}^{j-1} (1-\kappa)^{j-1-i}\,\xi_{t+i} .
$$

The case $j = 1$ is the recursion itself, and substituting the case $j$ into
$x_{t+j+1} = (1-\kappa)x_{t+j} + \sigma\xi_{t+j}$ gives the case $j+1$. Now
$x_t$ depends only on $\xi_0, \dots, \xi_{t-1}$, so it is independent of every
$\xi_{t+i}$ with $i \ge 0$, and all of these have mean zero. Multiply one
coordinate of each side by the same coordinate of $x_t$ and take expectations.
Each noise term gives $\mathbb{E}[x_t \xi_{t+i}] = \mathbb{E}[x_t]\,\mathbb{E}[\xi_{t+i}] = 0$
by independence, so

$$
\mathrm{Cov}(x_t, x_{t+j}) = (1-\kappa)^j\,\mathbb{E}[x_t^2] = (1-\kappa)^j v_t ,
$$

using that every $x_t$ has mean zero. As $t$ grows this tends to
$(1-\kappa)^j v$, and the correlation between the two times tends to
$(1-\kappa)^j$. For $1 < \kappa < 2$ the base $1-\kappa$ is negative, so the
correlation alternates in sign with the lag. Each step overcorrects and the
displacement changes side. At lags 1, 2 and 3, a gain of $0.5$ gives correlations
${{reg_corr_k05_1}}$, ${{reg_corr_k05_2}}$ and ${{reg_corr_k05_3}}$, and a gain of
$1.5$ gives ${{reg_corr_k15_1}}$, ${{reg_corr_k15_2}}$ and ${{reg_corr_k15_3}}$.
Gains $\kappa$ and $2-\kappa$ give correlations of
the same magnitude and opposite sign at odd lags, which is the difference
between the two equally broad trajectories described in [lesson:1.1].

(d) Each coordinate of $x_t$ is, by repeated substitution, a fixed linear
combination of the independent Gaussian variables $\xi_0, \dots, \xi_{t-1}$, and
so is Gaussian, with mean zero and variance $v_t \to v$. The two coordinates are
built from separate, independent noise and so are independent. Their joint density
is $(2\pi v)^{-1}\exp[-\lVert x\rVert^2/(2v)]$, which depends on $x$ only through
its distance from the origin. Write $R$ for that distance. In polar coordinates
the area element is $R\,\mathrm{d}R$ times the element of angle, and integrating
over the angle gives a factor $2\pi$, so

$$
\begin{aligned}
P(\lVert x_t\rVert > 1) &= \int_1^\infty \frac{R}{v}\, e^{-R^2/(2v)}\,\mathrm{d}R
= \Big[-e^{-R^2/(2v)}\Big]_1^\infty \\
&= e^{-1/(2v)} = \exp\!\left(-\frac{\kappa(2-\kappa)}{2\sigma^2}\right) .
\end{aligned}
$$

At the lesson's settings the exponent is $1/(2v) = {{reg_outside_exponent:.1f}}$
and the probability is ${{reg_outside:sci2}}$. At the larger noise
$\sigma = {{sigma_hi}}$ of the survival-curve problem, with the same gain, it is
${{reg_outside_hi:sci2}}$.

The circle lies about {{hs_boundary_in_sd:.0f}} standard deviations from the
origin in each coordinate, so this is a statement about the far tail of a
Gaussian in an idealised model, and nobody could check a number like
${{reg_outside:sci2}}$ against a real body. It is the probability that the
unabsorbed process is outside the circle at one late time. It is not the probability of having died by that time. Survival requires that
the path never left, which is a statement about the whole path, and (c) shows
successive positions are correlated, so the per-step probabilities cannot be
multiplied as if they were independent trials. The waiting time needs the
first-passage calculation mentioned in the warning in [lesson:1.1]. What the
formula does show is how the noise enters. Without feedback, the survival
timescale of the previous problem varies as $\sigma^{-2}$. With feedback, the
probability of being outside varies as
$\exp[-\kappa(2-\kappa)/(2\sigma^2)]$, exponentially in $\sigma^{-2}$, so a change of noise by a factor of
{{sigma_ratio:.1f}} moves it by {{reg_outside_orders:.0f}} orders of magnitude.
:::

::: exercise An implementation
Work in a new cell of `week01_numpy.ipynb`, or in any Python session with NumPy.

(a) Write functions `bayes(A, prior, o)`, returning the posterior and the
evidence, `entropy(p)`, `kl(q, p)` and `mutual_information(A, prior)`. Make
`bayes` refuse a likelihood matrix whose columns do not sum to one. Check your
functions against the tawny-flash posterior of [lesson:1.3], its
information gain ${{info_tawny:.3f}}$ nats, and the mutual information of the
leopard model computed by both forms in the problems above.
Then confirm that the refusal fires on the leopard matrix with its rows
normalised instead of its columns, the mistake of the normalisation problem.

(b) Simulate the unabsorbed homeostat at $\sigma = {{hs_sigma}}$ and
$\kappa = {{hs_kappa}}$ for {{impl_T:,}} steps from the origin, discard the first
{{impl_burn}}, and estimate the stationary variance per coordinate, pooling
the two coordinates, and the correlation between successive steps. Compare them with the regulated-walk
problem. Repeat over {{impl_seeds}} seeds. How large is the spread between
seeds, and is the difference between your average and the formula within it?
---solution---
(a) One implementation:

```python
import numpy as np

def bayes(A, prior, o):
    if not np.allclose(A.sum(axis=0), 1):
        raise ValueError("columns of A must sum to one")
    joint = A[o] * prior
    evidence = joint.sum()
    return joint / evidence, evidence

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def kl(q, p):
    keep = q > 0
    return np.sum(q[keep] * np.log(q[keep] / p[keep]))

def mutual_information(A, prior):
    evidence = A @ prior
    return sum(evidence[o] * kl(bayes(A, prior, o)[0], prior)
               for o in range(A.shape[0]))

A = np.array([[0.70, 0.15, 0.02],
              [0.25, 0.75, 0.08],
              [0.05, 0.10, 0.90]])
prior = np.array([0.08, 0.22, 0.70])

post, ev = bayes(A, prior, 0)
print(post, ev, kl(post, prior))

state_side = mutual_information(A, prior)
obs_side = entropy(A @ prior) - sum(prior[s] * entropy(A[:, s]) for s in range(3))
assert abs(state_side - obs_side) < 1e-12

A_rows = A / A.sum(axis=1, keepdims=True)
try:
    bayes(A_rows, prior, 0)
except ValueError as err:
    print("refused:", err)
```

The posterior is $({{post_tawny_leopard:.4f}}, {{post_tawny_baboon:.4f}}, {{post_tawny_nothing:.4f}})$
with evidence ${{ev_tawny:.3f}}$ and information gain ${{info_tawny:.3f}}$ nats,
and both forms of the mutual information give ${{mi_leopard_model:.4f}}$ nats.
The refusal fires on `A_rows`. The guard is necessary but not sufficient: a
matrix with a negative entry and unit column sums would pass it. Without it, `bayes` would return a normalised
posterior from the wrong matrix, which is the failure the normalisation problem
describes, where the division by the evidence hides the error.

(b) Draw all the noise for a run at once, then apply the homeostat recursion step by step:

```python
def run(seed, kappa=0.15, sigma=0.035, T=50_000):
    xi = np.random.default_rng(seed).standard_normal((T, 2))
    x = np.zeros(2)
    path = np.empty((T, 2))
    for t in range(T):
        x = (1 - kappa) * x + sigma * xi[t]
        path[t] = x
    return path

v = 0.035**2 / (0.15 * (2 - 0.15))
variances, correlations = [], []
for seed in range(20):
    tail = run(seed)[100:]
    var = np.mean(tail**2)
    variances.append(var)
    correlations.append(np.mean(tail[1:] * tail[:-1]) / var)

print(v, np.mean(variances), np.std(variances, ddof=1))
print(np.mean(correlations), np.std(correlations, ddof=1))
```

The variance is estimated as the mean square, pooled over both coordinates and
all steps kept, because the mean is known to be zero. Discarding
{{impl_burn}} steps is {{impl_burn_over_settle:.0f}} times the settling time
found in the regulated-walk problem; by then the variance deficit
$(1-\kappa)^{2t}$ is negligible, so the start contributes no visible bias.

Seed 0 gives a variance of ${{impl_var_seed0:.6f}}$ against the formula's
$v = {{hs_var_per_coord:.6f}}$, and a successive-step correlation of
${{impl_rho_seed0:.4f}}$ against $1 - \kappa = {{reg_rho1:.2f}}$. Over
{{impl_seeds}} seeds the variance estimates have mean ${{impl_var_mean:.6f}}$
and standard deviation ${{impl_var_sd:sci2}}$, which is
{{impl_var_relsd_pct:.1f}}% of $v$. The mean of {{impl_seeds}} independent
estimates has a standard error smaller by $\sqrt{ {{impl_seeds}} }$, about
{{impl_var_se_pct:.1f}}% of $v$, and the mean lies
{{impl_var_below_pct:.2f}}% below $v$, within one standard error. The correlations have mean
${{impl_rho_mean:.4f}}$ and standard deviation ${{impl_rho_sd:.4f}}$.

Each run holds {{impl_n_values:,}} numbers, yet one run's variance is
uncertain at the level of {{impl_var_relsd_pct:.1f}}%. If the $N = {{impl_n_values:,}}$
squares were independent, each would have variance
$\mathbb{E}[x^4] - v^2 = 3v^2 - v^2 = 2v^2$ (the fourth moment of a zero-mean
Gaussian is $3v^2$), and their mean would have relative standard deviation
$\sqrt{2/N} = {{impl_naive_relsd_pct:.2f}}\%$, smaller than what is observed by a factor of
{{impl_obs_over_naive:.1f}}.

The squares are not independent. Write $\rho = 1 - \kappa$ for the correlation
between successive positions found in (c) of the regulated-walk problem. For
two zero-mean jointly Gaussian variables $X$ and $Y$,
$\mathrm{Cov}(X^2, Y^2) = 2\,\mathrm{Cov}(X, Y)^2$. To see it, write
$Y = cX + W$ with $c = \mathrm{Cov}(X,Y)/\mathbb{E}[X^2]$ and $W$ Gaussian and
independent of $X$. Then
$\mathbb{E}[X^2Y^2] = c^2\,\mathbb{E}[X^4] + \mathbb{E}[X^2]\,\mathbb{E}[W^2]$,
the cross term vanishing because $\mathbb{E}[W] = 0$, while
$\mathbb{E}[X^2]\,\mathbb{E}[Y^2] = \mathbb{E}[X^2](c^2\,\mathbb{E}[X^2] + \mathbb{E}[W^2])$.
Subtracting, and using $\mathbb{E}[X^4] = 3\,\mathbb{E}[X^2]^2$, leaves
$2c^2\,\mathbb{E}[X^2]^2 = 2\,\mathrm{Cov}(X,Y)^2$. So squares $j$ steps apart
have covariance $2v^2\rho^{2j}$. The variance of a mean of $N$ terms is
$N^{-2}$ times the sum of all their covariances, which for large $N$ is close
to $N^{-1}$ times the sum over every lag, positive and negative:

$$
\frac{2v^2}{N}\Big(1 + 2\sum_{j=1}^{\infty}\rho^{2j}\Big)
= \frac{2v^2}{N}\Big(1 + \frac{2\rho^2}{1-\rho^2}\Big)
= \frac{2v^2}{N}\cdot\frac{1+\rho^2}{1-\rho^2} .
$$

At $\rho = {{reg_rho1:.2f}}$ the factor $(1+\rho^2)/(1-\rho^2)$ is
${{impl_inflation:.2f}}$, so the predicted relative spread is
${{impl_naive_relsd_pct:.2f}}\% \times \sqrt{ {{impl_inflation:.2f}} } = {{impl_pred_relsd_pct:.2f}}\%$.
The observed {{impl_var_relsd_pct:.2f}}% is {{impl_obs_over_pred:.1f}} times
that. A standard deviation estimated from {{impl_seeds}} samples is itself
uncertain by roughly $1/\sqrt{2({{impl_seeds}}-1)}$, about
{{impl_sd_unc_pct:.0f}}%, so the two agree within that uncertainty. The
correlation between successive positions makes a run worth about
${{impl_inflation:.1f}}$ times fewer independent samples than it has entries. A simulation that
reports an average over correlated steps should report its spread over
independent runs, as here, rather than a standard error that treats every step
as independent.
:::

## Further reading

Everything here is freely accessible.

- Parr, Pezzulo and Friston, *Active Inference: The Free Energy Principle in
  Mind, Brain, and Behavior* (MIT Press, 2022), Chapters 1 and 2. Open access.
  Read it for the framing; we will do the algebra it skips.
- Smith, Friston and Whyte, "A step-by-step tutorial on active inference and its
  application to empirical data", *Journal of Mathematical Psychology* 107
  (2022). The most practical published introduction to the discrete case, with
  MATLAB code.
- Bruineberg, Dolega, Dewhurst and Baltieri, "The Emperor's New Markov Blankets",
  *Behavioral and Brain Sciences* (2022). A careful criticism of exactly the
  interface picture drawn in [lesson:1.2]. Read it now, and again after Week&nbsp;12.
