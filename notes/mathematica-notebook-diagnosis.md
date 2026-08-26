# Why the notebook's final learning loop mislearns

Diagnosis of the last simulation cell in `source-material/shock-active-inference-notebook.nb`
(Section 1.2, the three-state-factor example). Verified numerically 2026-08-26;
reproduction scripts described at the bottom.

## The setup

```
statesoftheworld: s1 ~ Exp(rate 0.9), s2 ~ Exp(rate 0.7), 1000 i.i.d. draws
observations:     oy ~ N(s2 * s1^2, 1)
                  oi ~ N(s2^2 * s1, 1)
loop:             priortable = {{2, 0.5}}
                  each step: theta* = argmin F(theta; oy, oi, mu)
                             mu <- 0.99 mu + 0.01 theta*
```

Target: recover the rates (0.9, 0.7). Observed: converges to about **(3.0, 2.3)**.

## What is correct

The free energy is right. With `q(s) = Exp(rate theta)` the moments are
`E[s]=1/th`, `E[s^2]=2/th^2`, `E[s^4]=24/th^4`, and

    E[(oy - s2 s1^2)^2] = oy^2 - 4 oy/(a^2 b) + 48/(a^4 b^2)
    E[(oi - s2^2 s1)^2] = oi^2 - 4 oi/(a b^2) + 48/(a^2 b^4)

reproducing every coefficient in the cell, both 48s included. The complexity term
`mu/theta + ln theta` is the correct `-H[q] - E_q[ln p]` for an exponential prior
of rate `mu`, up to a constant. The `oy`/`oi` column assignment is right.

Exact posterior inference by quadrature recovers the truth: implied rates
**0.902 and 0.690**. So the model and the data are sound; the approximation is not.

## Cause 1 (dominant): the variational family is too rigid

`Exp(theta)` has one parameter, so its shape is locked. Coefficient of variation
is exactly 1; the true posterior's is **0.71**, more concentrated than any
exponential can be. Because the likelihood involves `s1^2` and `s2^2`, the free
energy carries `E[s2^2]E[s1^4] = 48 m2^2 m1^4`, twelve times the square of the
predicted mean `2 m2 m1^2`. With one parameter the only way to shrink that
implied predictive variance is to shrink the mean.

| | E[s1] | E[s2] |
|---|---|---|
| exact posterior | 1.165 | 1.357 |
| mean-field exponential q | 0.498 | 0.564 |

Under by ~2.3x, which inverts into rates too high by the same factor.

## Cause 2: the M-step averages the wrong quantity

`mu <- 0.99 mu + 0.01 theta*` is an arithmetic mean of *rate* parameters. The
correct M-step for an exponential prior averages posterior **means** and inverts,
`lambda <- 1 / mean(E_q[s])`. Arithmetic mean of rates >= harmonic mean by
Jensen, so this biases upward too. Measured at about 14%.

## Cause 3: positive feedback

Higher `mu` pushes `E_q[s]` lower, which raises `theta*`, which raises `mu`.
With the prior pinned at the truth the one-step estimate is 2.29; the closed
loop runs on to 3.0.

## Fix

| variational family | implied rates | true |
|---|---|---|
| exact (quadrature) | 0.858, 0.737 | 0.9, 0.7 |
| Exponential (as written) | 2.009, 1.774 | |
| Gamma (two parameters) | 1.005, 0.832 | |

A Gamma `q` recovers most of it. The residual is the mean-field assumption:
the exact posterior over `(s1, s2)` has mean correlation **-0.54**, with a third
of observations beyond |0.5|, because the observations depend on products and a
larger `s1` trades against a smaller `s2`. No `q(s1)q(s2)` can represent that.

## Minor

- Loop bound still reads `i <= 1000 + 0 Length[observations]`, a toggle left mid-edit.
- `NMinimize` is called cold each iteration on a non-convex objective, so
  consecutive estimates carry optimiser noise unrelated to the data. Warm-starting
  from the previous `theta*` would remove it.

## Use in the course

This belongs in **Week 4** as the worked demonstration that mean-field
underestimates spread and fails when causes compete, with the exact posterior
computed alongside so the gap is visible rather than asserted. It connects
directly to the explaining-away material already written into Week 1 Lesson 4.
