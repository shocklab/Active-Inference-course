---
title: Problems and code
deck: Six problems with full solutions, and the notebooks that let you rebuild this week's figures yourself.
week: 1
time: 90 min
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

::: exercise The prior does the work
Using the leopard, baboon and nothing model from Lesson&nbsp;3, find the value
of the leopard prior $q$ at which observing a tawny flash leaves you exactly
undecided between leopard and not-leopard, that is,
$P(s = \text{leopard} \mid o) = 1/2$. Keep the baboon and nothing priors in the
ratio $22 : 70$ as $q$ varies.
---solution---
Let the leopard prior be $q$. The other two share $1 - q$ in the ratio
$22:70$, so they are $\tfrac{22}{92}(1-q)$ and $\tfrac{70}{92}(1-q)$. The
tawny-flash row is $(0.70, 0.15, 0.02)$. The posterior odds of leopard against
everything else are

$$
\frac{0.70\,q}{\left(0.15 \cdot \tfrac{22}{92} + 0.02 \cdot \tfrac{70}{92}\right)(1-q)}
= \frac{0.70\,q}{0.051\,087\,(1-q)} .
$$

Setting this to 1 gives $0.70q = 0.051\,087(1-q)$, so
$q = 0.051\,087 / 0.751\,087 = 0.0680$.

So a leopard prior of about 6.8% is the tipping point: the model in
Lesson&nbsp;3 used $q = 0.08$, which is just above it, which is why the
posterior came out at $0.544$ rather than exactly a half. The observation is
strong, but it is only just strong enough to overturn a prior of that size, and
a slightly more sceptical hiker would have remained unconvinced by the same
evidence.
:::

::: exercise Two observations in a row &dagger;
You observe a branch shaking, and then, a moment later, a tawny flash. Assume
the two observations are conditionally independent given the state, and that the
state has not changed.

(a) Compute the posterior after both observations, starting from the
Lesson&nbsp;3 prior.

(b) Show that you get the same answer whichever order you process them in.

(c) What feature of the model makes (b) true, and give a physical situation in
which it would fail.
---solution---
(a) After the shaking branch the posterior was $(0.082988, 0.684647, 0.232365)$.
Use this as the prior and apply the tawny-flash row $(0.70, 0.15, 0.02)$:

$$
(0.70 \times 0.082988,\ 0.15 \times 0.684647,\ 0.02 \times 0.232365)
= (0.058091,\ 0.102697,\ 0.004647).
$$

The sum is $0.165436$, giving a posterior of
$(0.35114,\ 0.62077,\ 0.02809)$.

Baboon is now the leading hypothesis at 62%. The shaking branch was strong
evidence for a baboon, and although the tawny flash pulls back towards leopard,
it does not pull far enough.

(b) Processing the flash first gives posterior
$(0.543689, 0.320388, 0.135922)$, then applying the branch row
$(0.25, 0.75, 0.08)$:

$$
(0.135922,\ 0.240291,\ 0.010874), \quad \text{sum } 0.387087,
$$

which normalises to $(0.35114,\ 0.62077,\ 0.02809)$: identical.

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

::: exercise Explaining away with three causes
Extend the noisy-OR example of Lesson&nbsp;4 to three independent causes, each
with prior $0.1$ and each firing the observation with probability $0.9$, keeping
the leak at $0.01$. Compute $P(s_1 = 1 \mid o = 1)$ and compare it with the
two-cause answer of $0.505$. Explain the direction of the change.
---solution---
There are eight joint states. Write $k$ for the number of causes present. The
likelihood depends only on $k$:

$$
P(o=1 \mid k) = 1 - (0.1)^k (0.99),
$$

giving $0.01, 0.901, 0.9901, 0.99901$ for $k = 0,1,2,3$. The prior for a state
with $k$ causes present is $0.1^k \, 0.9^{3-k}$, and there are $\binom{3}{k}$
such states. So

$$
P(o=1) = \sum_{k=0}^{3} \binom{3}{k} 0.1^k 0.9^{3-k} \left[1 - 0.99 \times 0.1^k\right]
= 0.253\,97 .
$$

For the marginal, sum the unnormalised mass over the four states with $s_1 = 1$:

$$
P(s_1=1, o=1) = 0.1 \sum_{k'=0}^{2} \binom{2}{k'} 0.1^{k'} 0.9^{2-k'}
\left[1 - 0.99 \times 0.1^{\,k'+1}\right] = 0.091\,80 ,
$$

so $P(s_1 = 1 \mid o = 1) = 0.091\,80 / 0.253\,97 = 0.3615$.

The answer *falls* from $0.505$ to $0.361$. With more candidate explanations
available, the evidence is spread more thinly across them: the observation still
says "something did this", but there are now three somethings to choose from and
each individually gets less of the credit. Explaining away gets stronger as the
number of competing causes grows, which is precisely why a factorised
approximation gets worse in exactly the regime where you most want to use it.
:::

::: exercise The survival curve &dagger;
Consider the homeostasis model from Lesson&nbsp;1 with no action: a
two-dimensional random walk with independent increments of standard deviation
$\sigma$ per step in each coordinate, starting at the origin, absorbed when
$\|x\| > 1$.

(a) Argue from dimensional analysis alone that the expected number of steps
before absorption scales as $\sigma^{-2}$.

(b) The notebook measures this by simulation. Predict the ratio of survival
times at $\sigma = 0.09$ and $\sigma = 0.035$ before you run it.
---solution---
(a) After $n$ steps the walk's displacement has variance $n\sigma^2$ in each
coordinate, so the typical distance from the origin is of order
$\sigma\sqrt{n}$. Absorption happens when this reaches the boundary at radius
$1$, so $\sigma\sqrt{n} \sim 1$ and $n \sim \sigma^{-2}$. There is no other
length or time scale in the problem: the radius is fixed at one and the only
parameter is $\sigma$, so the scaling is forced.

(b) The ratio is $(0.09 / 0.035)^{-2} = (0.035/0.09)^2 = 0.151$. Halving the
noise should quadruple the lifetime, and the roughly 2.6-fold increase in
$\sigma$ here should cut it by a factor of about $6.6$.

This is worth registering as a piece of intuition about what regulation is
buying. Without action, lifetime scales as the inverse square of the noise: a
punishing dependence. With action, as the figure in Lesson&nbsp;1 shows, the
walk does not diffuse at all, and lifetime is no longer set by $\sigma$ in the
same way. That is a qualitative change in the character of the dynamics, not
merely a better constant.
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
  interface picture drawn in Lesson&nbsp;2. Worth reading now and again after
  Week&nbsp;12.
