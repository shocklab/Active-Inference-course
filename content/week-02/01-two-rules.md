---
title: Local computation and learning
deck: Modelling inference with connected units, and specifying what information each update requires.
week: 2
time: 20 min
---

An inference algorithm specifies computations. A proposed neural implementation
must also say where information is stored and how it reaches the places that
use it. We will derive an estimation rule and then ask whether connected units
could implement its operations.

## Units and connections

::: definition Unit and connection
A **unit** is a component of our model which holds an activity $r$, a real
number, and computes from incoming signals. The letter suggests a firing rate,
although the mathematical variable may also stand for a population-level signal.
Allowing signed values is a modelling convenience; a literal firing rate cannot
be negative.

A **connection** carries a signal from one unit to another. Its **weight** $w$
sets the signal's multiplicative gain. Here $w$ is a connection weight, whereas
[lesson:1.4] used the same letter locally for a noisy-OR success probability.
A connection weight need not lie between zero and one.
:::

A simple unit receives $r_1,r_2$ and reports $r=w_1r_1+w_2r_2$. This is a
stylised arithmetic component, not a full neuron model. Real cells have internal
dynamics and signals beyond those included here.

## Information available to an update

::: definition Local computation
A unit updates using its own state, incoming signals and associated connection
parameters. Information originating elsewhere must arrive through a specified
communication path.
:::

::: definition Local plasticity
A connection updates using quantities available at its endpoints. In the simplest
model these are the sending and receiving activities; a richer model must specify
any additional signals it makes available there.
:::

These are assumptions about an implementation. They do not prove that a nervous
system uses a particular inference algorithm, and they do not select gradient
ascent uniquely.

For example, the weight update $\Delta w_i=\beta r_i r$, with positive step size
$\beta$, uses the activities at its endpoints. This has the multiplicative form
called **Hebbian learning**. Whether such an update learns a useful model also
depends on its objective, dynamics and any stabilising terms. Endpoint locality
alone does not establish those properties.

## Normalisation through local messages

A sum can be calculated by a network of local units. To sum four values, one
unit adds the first pair and another adds the second; a third adds their outputs.
No unit needs all four original values. The chain example in [lesson:1.5]
uses the same principle for marginalisation.

The relevant questions are how many messages are needed, how large each is, how
long they take to arrive, and whether the connections required are available.
Local computation does not itself rule out normalisation. Likewise, a signal
originating at a distant output can become locally available if a mechanism
transmits it. For backpropagation, the biological questions concern that mechanism,
the required derivatives and coordination, rather than a prohibition on distant
information as such.

## Representing uncertainty

A general continuous density cannot be specified by a fixed finite list of
parameters without restricting the class of densities. There are nevertheless
several useful finite representations. A Gaussian requires a mean and covariance;
a grid stores values at chosen locations; samples represent an empirical
distribution. Each makes different demands on memory and computation.

This week begins with a point estimate because it lets us derive a compact
update and examine what uncertainty it omits. It is a methodological choice.
[week:4] returns to distributions and makes their approximation error explicit.

## A prediction-error implementation

For a Gaussian prior and Gaussian observation noise, differentiating the log
joint produces a sum of weighted residuals. [lesson:2.3] derives this result;
[lesson:2.4] identifies the signals and connections needed to compute it.
The observation function's derivative is among those requirements. Giving a
symbol to a connection does not by itself explain how that connection is learned.

[week:3] then distinguishes estimating a state from learning model parameters.
Some parameter updates have a product of activities that can be evaluated at
connection endpoints. We will check the required information for each update,
rather than assume locality from the resemblance of a diagram to a neural circuit.

These questions follow the pedagogical approach of
[Bogacz (2017)](https://doi.org/10.1016/j.jmp.2015.11.003), while keeping the
implementation assumptions separate from the probability identities.

::: exercise A communicated average
Suppose a weight update uses $r_i(r-\bar r)$, where $\bar r$ is the average
activity across a network. Is the rule local?
---solution---
The formula alone does not decide. If $\bar r$ is unavailable at the connection,
the proposed implementation is incomplete. If a specified aggregation mechanism
computes the average and sends it to the receiving unit, the update can use that
locally available signal. The communication cost and delay still need accounting
for. Naming a global statistic neither provides such a mechanism nor proves it
impossible.
:::

::: checkpoint
- Give a local circuit that sums several inputs without a shared memory.
- What information must be available to evaluate a Hebbian update?
- Name different finite representations of an uncertain continuous state.
- Which claims above are implementation assumptions, and which are algebraic facts?
:::
