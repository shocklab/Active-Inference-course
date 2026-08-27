---
title: What makes the denominator hard
deck: The claim that exact inference is intractable is four claims, not one. They fail in different places, are escaped by different methods, and only one of them is about the size of a sum.
week: 1
time: 45 min
scripts: [w01.js]
---

[lesson:1.3] counted the terms in

$$
P(o) \;=\; \sum_s P(o \mid s)\,P(s)
$$ {#denom}

and found $k^n$ of them for a model with $n$ state factors of $k$ values each.
[lesson:1.4] took that as settled and moved on. Almost every treatment of active
inference does the same, in a sentence or two, and the sentence is usually some
version of "the sum is intractable".

That sentence is doing more work than it can carry. There are at least four
distinct reasons the denominator is out of reach, they are not the same reason,
they fail at different points, and the methods that escape one often do nothing
about another. Worse, the reason most often given is the weakest of the four.
This lesson separates them.

## The count is not the reason

Start by demolishing the standard explanation, because it is wrong as stated.

Here is a model with $n = {{den_n}}$ binary-ish factors, $k = {{den_k}}$ values
each. Its state space has

$$
k^n \;=\; {{den_k}}^{{{den_n}}} \;\approx\; {{den_terms:sci3}}
$$

states, so [eq:denom] has that many terms. More states than there are atoms in a
mountain. Surely nothing can add them.

Now suppose the factors form a **chain**: factor 1 influences factor 2, which
influences factor 3, and so on, with no other dependencies. The joint is

$$
P(s_1, \dots, s_n) \;=\; P(s_1)\prod_{i=1}^{n-1} P(s_{i+1} \mid s_i),
$$

and the observation attaches to the chain somewhere. The sum still has
${{den_terms:sci3}}$ terms. Nothing about the count has changed.

::: derivation Summing a trillion trillion terms in linear time
Write the sum out and push each factor as far right as it will go:

$$
\sum_{s_1}\sum_{s_2}\cdots\sum_{s_n} P(s_1)\,P(s_2\mid s_1)\,P(s_3 \mid s_2)\cdots P(s_n \mid s_{n-1}).
$$

Only the last factor contains $s_n$, so the innermost sum can be done on its own:

$$
\sum_{s_n} P(s_n \mid s_{n-1}) \;=\; m_{n}(s_{n-1}),
$$

a function of $s_{n-1}$ alone, and computing it for every value of $s_{n-1}$
costs $k^2$ multiply-adds. Substitute it back and the expression is a sum over
$n-1$ variables of the same shape as before. Repeat.

After $n-1$ eliminations, each costing $k^2$, one sum over $s_1$ remains. Total
cost:

$$
(n-1)\,k^2 \;=\; {{den_chain_ops}} \text{ operations},
$$ {#chain-cost}

against ${{den_terms:sci3}}$ terms in the sum being computed. The answer is exact.
Not approximated, not sampled: the same number brute force would have produced.
:::

The distributive law did that, and nothing else. $ab + ac = a(b+c)$ turns two
multiplications into one, and applied recursively it turns an exponential sum
into a linear one.

Run both and compare. At $n = {{den_measured_n}}$, where brute force is still
just possible, the sum has ${{den_measured_terms:,}}$ terms. Brute force takes
about {{den_brute_s_n12}} seconds; elimination takes a fraction of a
millisecond, some five orders of magnitude less, and the two agree to the last
bit. Those are wall-clock times on one machine and yours will differ; the
notebook runs the comparison so you can watch it on your own. What does not vary
is the operation count, and that is what the argument rests on.

At $n = {{den_n}}$ brute force would take longer than the universe has existed.
Elimination still takes {{den_chain_ops}} operations.

::: keyidea
**A sum having $k^n$ terms tells you almost nothing about whether it can be
computed.** The count is a property of how the sum is *written*; the cost is a
property of how it *factorises*.

Every treatment that stops at "the sum has exponentially many terms" has stopped
one step before the argument starts. It is the step this course exists to take.
:::

## Structure is the reason

So what did the chain have that a general model does not?

Elimination worked because at each step there was a variable appearing in only
one remaining factor, so summing it out produced a function of one other
variable. In a general model, summing out $s_i$ produces a function of *every
variable that shared a factor with it*, and that function has to be tabulated
over all of them. If $s_i$ was coupled to $w$ others, that table has $k^w$
entries and the step costs $k^{w+1}$.

::: definition Treewidth
Eliminate the variables in some order. At each step, record how many variables
the newly created function depends on. The largest such number, minimised over
all elimination orders, is the **treewidth** $w$ of the model.

Exact inference by elimination costs about $n\,k^{\,w+1}$. The treewidth, not the
number of variables, is what sits in the exponent.
:::

::: mn Why "tree"
A tree has treewidth 1: at every stage there is a leaf whose elimination touches
only its parent. The definition measures how far a graph is from being a tree,
and elimination is exactly as expensive as that distance.
:::

Three models, same $k = {{den_k}}$, and the difference is entirely in the wiring.

<table>
<thead><tr><th>Structure</th><th>Variables</th><th>Treewidth</th><th>Elimination cost</th><th>Brute force</th></tr></thead>
<tbody>
<tr><td>chain or any tree</td><td>{{den_n}}</td><td>1</td><td>{{den_chain_ops}}</td><td>${{den_terms:sci1}}$</td></tr>
<tr><td>square grid</td><td>{{den_grid_n}}</td><td>{{den_grid_w}}</td><td>{{den_grid_ops:,}}</td><td>${{den_grid_brute:sci1}}$</td></tr>
<tr><td>densely coupled</td><td>{{den_n}}</td><td>{{den_dense_w}}</td><td>no better than brute force</td><td>${{den_terms:sci1}}$</td></tr>
</tbody>
</table>

The grid is the interesting row. Its treewidth grows like $\sqrt{n}$, so exact
inference is still exponential, but in $\sqrt{n}$ rather than $n$, and that is
the difference between hopeless and merely expensive.

The last row is the honest one for an agent. Nothing entitles a creature
modelling an open world to a sparse dependency graph, and [lesson:1.4] showed a
mechanism that actively destroys sparsity: **explaining away** couples causes
that were independent in the prior the moment a shared effect is observed. A
model may be built out of independent parts and still have a densely coupled
posterior. Conditioning is what fills the graph in.

::: keyidea
The obstacle is not $n$. It is $w$. Restate the claim of [lesson:1.3] accordingly:
exact inference is cheap when the model is nearly a tree, expensive when it is
not, and the world does not supply trees.
:::

## No algorithm does better

Everything so far is about one algorithm. Elimination is defeated by high
treewidth, but that leaves the obvious question: might some cleverer method
succeed where elimination fails? Most treatments never ask. The answer is known,
and it is stronger than "not so far".

Computing [eq:denom] for a general model is at least as hard as counting the
satisfying assignments of a Boolean formula, the canonical
$\#\mathrm{P}$-complete problem
([Valiant 1979](https://doi.org/10.1016/0304-3975(79)90044-6)). Deciding a much
weaker question, whether a particular posterior probability exceeds a threshold,
is already NP-hard
([Cooper 1990](https://doi.org/10.1016/0004-3702(90)90060-D)). So a
polynomial-time exact algorithm would give $\mathrm{P} = \mathrm{NP}$.

The result that actually closes the door is about approximation. It is tempting
to concede exactness and hope for a good estimate instead. But approximating the
denominator to within any fixed relative factor is *also* NP-hard
([Dagum & Luby 1993](https://doi.org/10.1016/0004-3702(93)90036-B)), and the
hardness survives when you ask only for a crude answer with high probability
([Roth 1996](https://doi.org/10.1016/0004-3702(94)00092-1)). There is no
polynomial-time scheme, deterministic or randomised, exact or approximate, unless
P and NP coincide.

::: warning What these results do and do not say
They are worst-case statements about the general problem. They do not say your
particular model is hard, and real models are often far from worst case, which
is why approximate inference works as well as it does in practice.

What they do establish is that no *general-purpose* method can exist. Any
working scheme must be exploiting something about the particular model it is
applied to, and is therefore obliged to say what. This is the standard the rest
of the course holds itself to: every time an approximation is made, the structure
being assumed gets named.
:::

Note also what these results are not. They are not the hardware argument of
[lesson:1.3]. That argument showed a faster machine cannot help, because a
constant factor cannot beat an exponent. This one shows a cleverer *algorithm*
cannot help either. Two different claims, and the second is the one that matters.

## The continuous case fails differently

Everything above concerns counting and structure. Now take the smallest possible
continuous problem: **one** hidden variable, no dimension to speak of, no
structure to exploit, nothing to count.

Write $x$ for the hidden variable and $u$ for what is measured. Let the prior over
$x$ be Gaussian with mean $x_p$ and variance $\Sigma_p$; let a **link function**
$g$ say what value of $u$ each $x$ predicts; and let the measurement carry
Gaussian noise of variance $\Sigma_u$ about that prediction. Then

$$
P(u) \;=\; \int_{-\infty}^{\infty} \mathcal{N}\big(u;\ g(x),\ \Sigma_u\big)\,\mathcal{N}\big(x;\ x_p,\ \Sigma_p\big)\,\mathrm{d}x .
$$ {#cont-denom}

That is the notation Week&nbsp;2 uses throughout, and it works this exact model
through with a concrete $g$. Here only the shape of the integral matters.

It is tempting to say this is hard because an integral is a sum with uncountably
many terms. That explanation is worthless: $\int_0^1 x\,\mathrm{d}x$ has
uncountably many terms too and is not hard. Something else is going on, and it
can be seen exactly.

::: derivation Why one link function integrates and another does not
Multiply the two Gaussians and collect the exponent. Constants that do not
contain $x$ come outside the integral and may be ignored.

**Take $g(x) = x$ first.** The exponent is

$$
-\frac{(x - x_p)^2}{2\Sigma_p} - \frac{(u - x)^2}{2\Sigma_u},
$$

and expanding it gives terms in $x^2$, $x^1$ and $x^0$ and nothing else. It is a
quadratic. Complete the square, as in Week&nbsp;2, and the integral becomes
$\int e^{-a(x-b)^2}\mathrm{d}x = \sqrt{\pi/a}$, the one nontrivial integral
everyone knows.

**Now take $g(x) = 1/x^2$.** The exponent is

$$
-\frac{(x - x_p)^2}{2\Sigma_p} - \frac{\big(u - x^{-2}\big)^2}{2\Sigma_u},
$$

and expanding the second bracket produces $u^2$, $-2u\,x^{-2}$ and $x^{-4}$. The
powers of $x$ present are now

$$
\{\,-4,\ -2,\ 0,\ 1,\ 2\,\}.
$$

That is not a quadratic and no substitution makes it one, because a substitution
that clears $x^{-4}$ reintroduces trouble at the other end. There is no square to
complete, so the one closed form available does not apply.
:::

The point generalises, and it is worth stating plainly because the literature
rarely does. **The catalogue of integrals we can do in closed form is tiny.** The
Gaussian is essentially the only continuous marginal that comes out, and
conjugacy is the study of exactly which prior-likelihood pairs land back inside
that catalogue. A nonlinear $g$ takes you outside it immediately, and once
outside there is nothing to fall back on. This has nothing to do with dimension,
nothing to do with treewidth, and nothing to do with complexity classes. One
variable is already enough.

::: mn A caution about "no closed form"
Failing to find an antiderivative is not proof that none exists, and definite
integrals sometimes have closed forms when the indefinite ones do not:
$\int e^{-x^2}\mathrm{d}x$ is not elementary, yet
$\int_{-\infty}^{\infty} e^{-x^2}\mathrm{d}x = \sqrt{\pi}$ exactly. The claim
here is the narrower and safer one: the technique that works is completing the
square, it applies only to quadratic exponents, and [eq:cont-denom] with a
nonlinear $g$ has a non-quadratic exponent. No general method replaces it.

For what it is worth as evidence rather than proof, a computer algebra system
returns [eq:cont-denom] unevaluated for [lesson:2.2]'s inverse-square link, both
as a definite integral and as an antiderivative, while returning
$\sqrt{2\pi/11}\,e^{-45/44}$ for the linear case in a second. Its value to
twenty digits is ${{nl_marginal_20:.19f}}$, and an integer-relation search finds
nothing among $\pi$, $\sqrt{\pi}$, $e$, $\sqrt{2}$ and $\ln 2$: the coefficients
it returns grow from ${{pslq_coeff_20:,}}$ to ${{pslq_coeff_36:,}}$ as the
precision fed to it rises from 20 digits to 36, which is what a search returns
when there is no relation to find.
:::

## Where quadrature and sampling break

Two obvious escapes from [eq:cont-denom], and each fails in a way that is worth
knowing, because both reappear in later weeks.

**Put down a grid.** With $m$ sample points and an integrand having $\rho$ bounded
derivatives in $n$ dimensions, the best error any deterministic rule can achieve
is of order $m^{-\rho/n}$. Holding the error at $\delta$ therefore needs about
$\delta^{-n/\rho}$ points, with $\delta$ the error we are willing to tolerate and
$\rho$ the count of derivatives. Neither letter is used elsewhere in the course.
The obvious letter for an error tolerance is reserved from Week&nbsp;2 for a
prediction error, which this is not. With $\rho = {{quad_r}}$ and a one per cent
target:

<table>
<thead><tr><th>Dimensions $n$</th><th>Grid points for {{quad_eps:.0%}} error</th></tr></thead>
<tbody>
<tr><td>1</td><td>{{quad_pts_n1:.0f}}</td></tr>
<tr><td>2</td><td>{{quad_pts_n2:.0f}}</td></tr>
<tr><td>5</td><td>${{quad_pts_n5:sci0}}$</td></tr>
<tr><td>10</td><td>${{quad_pts_n10:sci0}}$</td></tr>
<tr><td>20</td><td>${{quad_pts_n20:sci0}}$</td></tr>
</tbody>
</table>

The exponent $n$ is back, and it is in the same place it was for the discrete
sum. Grids convert the counting problem into a different counting problem of the
same size. Week&nbsp;2 uses one anyway, because with a single variable the first
row of that table is the one that applies.

**Sample instead.** Draw $x^{(1)}, \dots, x^{(N)}$ from some proposal $q$ and
average the ratio $P(u\mid x)P(x)/q(x)$. The error of a Monte Carlo estimate
falls like $N^{-1/2}$ *regardless of dimension*, which looks like an escape and
is genuinely one, but only from the rate.

The constant hidden in that rate is the estimator's variance, and it degrades as
the proposal mismatches the target. A serviceable measure is the effective sample
size, which behaves like $N\exp(-D_{\mathrm{KL}}[P\,\|\,q])$. Divergences add over
independent coordinates, so a per-coordinate mismatch of
{{is_kl_per_coord}} nats compounds:

<table>
<thead><tr><th>Coordinates $n$</th><th>Divergence, nats</th><th>Useful fraction of samples</th><th>Draws for {{is_target_ess:,}} effective</th></tr></thead>
<tbody>
<tr><td>1</td><td>{{is_kl_n1}}</td><td>{{is_ess_frac_n1:.3f}}</td><td>{{is_samples_n1:,.0f}}</td></tr>
<tr><td>10</td><td>{{is_kl_n10}}</td><td>${{is_ess_frac_n10:sci2}}$</td><td>{{is_samples_n10:,.0f}}</td></tr>
<tr><td>20</td><td>{{is_kl_n20}}</td><td>${{is_ess_frac_n20:sci2}}$</td><td>{{is_samples_n20:,.0f}}</td></tr>
<tr><td>50</td><td>{{is_kl_n50}}</td><td>${{is_ess_frac_n50:sci2}}$</td><td>${{is_samples_n50:sci2}}$</td></tr>
</tbody>
</table>

The exponential did not go away; it moved from the number of terms into the
number of samples. This is the honest reading of "Monte Carlo beats the curse of
dimensionality": the *rate* is dimension-free, and the *constant* is not.

## Locality

Three reasons so far, all of them about difficulty of computation. The fourth is
different in kind, and for this course it is the one that bites first.

Suppose the sum were short. Suppose the model had ten states and [eq:denom] had
ten terms, addable by hand. An agent still could not evaluate it, because
evaluating it requires something that can see all ten states at once, and a
device built from units that read only their own inputs contains no such thing.

Week&nbsp;2 states this as a constraint and builds on it. Note that it is
logically independent of everything above: a short sum can be unavailable for
want of anyone to do it, and a long sum can be available to a machine that
happens to have global memory. The complexity results apply to computers; this
one applies to bodies, and an account of biological inference needs both.

## Four reasons, four escapes

They come apart, and seeing which method dodges which is the fastest way to see
that they are genuinely different claims.

<table>
<thead><tr><th>Reason</th><th>Bites when</th><th>Escaped by</th><th>Not helped by</th></tr></thead>
<tbody>
<tr><td>Too many terms to enumerate</td><td>always, on paper</td><td>the distributive law: elimination, belief propagation</td><td>faster hardware</td></tr>
<tr><td>High treewidth</td><td>densely coupled models, and after explaining away</td><td>assuming structure that is not there, which is what mean-field does</td><td>any exact method</td></tr>
<tr><td>No closed form</td><td>any nonlinear link, even in one variable</td><td>conjugacy where available; otherwise approximation</td><td>more dimensions being small</td></tr>
<tr><td>No unit can see the state space</td><td>always, for an embodied agent</td><td>local message passing</td><td>every method above</td></tr>
</tbody>
</table>

Read the second row again, because it is the one the rest of the course turns on.
The escape from high treewidth is to *pretend the coupling is not there*: to
replace the true posterior with a product of independent factors and optimise
within that restricted family. That is the mean-field approximation of
Week&nbsp;4, and the table says plainly what it is buying and with what. It buys
tractability, and it pays in exactly the dependencies that explaining away
created.

Every subsequent week is an attempt on one or more of these four rows. It is
worth knowing which.

::: checkpoint
- A colleague says the denominator is intractable because the sum has $2^{100}$
  terms. Give the one-sentence counterexample.
- What quantity actually sits in the exponent of the cost of exact inference,
  and what does it measure?
- The hardware argument of [lesson:1.3] and the complexity results of this
  lesson rule out two different things. Which two?
- Give a problem that is hard for reason three but trivial for reasons one and
  two.
- Which of the four reasons would remain if an agent had unlimited time and a
  perfect computer, and why?
:::
