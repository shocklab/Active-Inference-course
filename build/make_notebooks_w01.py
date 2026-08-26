#!/usr/bin/env python3
"""Generate the Week 1 notebooks. Run from the repo root:  python3 build/make_notebooks_w01.py"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "week-01")
os.makedirs(OUT, exist_ok=True)


def build(cells, path, kernel="python3"):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1])
                for c in cells]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
        "language_info": {"name": "python", "version": "3.10"},
    }
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", os.path.relpath(path, ROOT))


# ── NumPy ────────────────────────────────────────────────────────────────
numpy_cells = [
("md", """# Week 1 — The hypothesis-testing agent (NumPy)

Everything in Week 1, built from `numpy` primitives. Nothing is hidden inside a
library call.

Run the cells in order. The exercises at the bottom are the ones from the
lesson pages; the solutions are on the website, so try them here first.

**Contents**
1. Bayes with states and observations
2. The evidence sum, and what it costs
3. Explaining away, and what mean-field loses
4. Homeostasis: a body that acts, and one that does not"""),

("code", """import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)"""),

("md", """## 1. Bayes with states and observations

The likelihood array `A` has observations down the **rows** and states across the
**columns**, so `A[i, j] = P(o = i | s = j)`. Each column is a distribution over
observations and therefore sums to one. Each row is a likelihood read as a
function of the state and has no reason to sum to anything."""),

("code", """STATES = ['leopard', 'baboon', 'nothing']
OBS    = ['tawny flash', 'branch shakes', 'quiet']

A = np.array([
    [0.70, 0.15, 0.02],   # tawny flash
    [0.25, 0.75, 0.08],   # branch shakes
    [0.05, 0.10, 0.90],   # quiet
])

prior = np.array([0.08, 0.22, 0.70])

print('column sums (must all be 1):', A.sum(axis=0))
print('row sums    (no reason to be 1):', A.sum(axis=1))"""),

("code", """def posterior(A, prior, o):
    \"\"\"Bayes for a discrete model. Returns the posterior and the evidence P(o).\"\"\"
    unnormalised = A[o] * prior          # elementwise: P(o|s) P(s), one per state
    evidence = unnormalised.sum()        # the sum over every state in the model
    return unnormalised / evidence, evidence


for o, name in enumerate(OBS):
    post, ev = posterior(A, prior, o)
    surprise = -np.log(ev)
    info = np.sum(post * np.log(post / prior))     # D_KL[posterior || prior]
    print(f'{name:>14} | P(o)={ev:.4f}  surprise={surprise:.3f} nats'
          f'  info gained={info:.3f} nats')
    shown = '  '.join(f'{n}={v:.4f}' for n, v in zip(STATES, post))
    print(f'{"":>14} | posterior  {shown}')"""),

("md", """Those are the numbers quoted in Lesson 3. A tawny flash takes the leopard
hypothesis from 8% to 54%: not because leopards became likely, but because
nothing else explains tawny flashes.

### The ambiguity slider, in code

Blend the likelihood towards uniform and watch the observation stop telling you
anything."""),

("code", """def blur(A, lam):
    \"\"\"lam=0 leaves A alone; lam=1 makes every state predict every observation equally.\"\"\"
    B = (1 - lam) * A + lam / A.shape[0]
    return B / B.sum(axis=0, keepdims=True)      # keep the columns normalised

lams = np.linspace(0, 0.98, 40)
info = [np.sum(posterior(blur(A, l), prior, 0)[0]
               * np.log(posterior(blur(A, l), prior, 0)[0] / prior)) for l in lams]

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.plot(lams, info, color='#0f5f57', lw=2)
ax.set_xlabel('ambiguity $\\\\lambda$'); ax.set_ylabel('information gained, nats')
ax.set_title('An ambiguous channel carries no information', loc='left')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()"""),

("md", """## 2. The evidence sum, and what it costs

`P(o)` requires visiting every state the model admits. With `n` factors of `k`
values each, that is `k**n` terms."""),

("code", """for n in (5, 10, 20, 40, 89):
    terms = 2.0 ** n
    secs = terms / 1e9                      # one addition per nanosecond, generous
    print(f'{n:>3} binary factors -> {terms:.3e} terms, {secs:.3e} s at 1e9 add/s')

print()
print('age of the universe in seconds ~ 4.35e17')
print('so 89 binary factors already exceeds it.')"""),

("md", """## 3. Explaining away, and what mean-field loses

Two independent causes, either of which can shake a branch. Independent priors,
a shared observation, and a posterior that is anything but independent."""),

("code", """import itertools

p_cause, strength, leak = 0.1, 0.9, 0.01

def noisy_or(s):
    \"\"\"P(o=1 | s) for a vector of binary causes.\"\"\"
    return 1 - np.prod([1 - strength * x for x in s]) * (1 - leak)

def joint_posterior(n_causes):
    states = list(itertools.product([0, 1], repeat=n_causes))
    pri = np.array([np.prod([p_cause if x else 1 - p_cause for x in s]) for s in states])
    lik = np.array([noisy_or(s) for s in states])
    un = pri * lik
    return states, un / un.sum(), un.sum()

states, post, ev = joint_posterior(2)
print(f'P(o=1) = {ev:.6f}   surprise = {-np.log(ev):.4f} nats\\n')
for s, p in zip(states, post):
    print(f'  gust={s[0]}  baboon={s[1]}   posterior={p:.4f}')

m1 = sum(p for s, p in zip(states, post) if s[0] == 1)
m2 = sum(p for s, p in zip(states, post) if s[1] == 1)
joint_11 = post[states.index((1, 1))]
print(f'\\nmarginals: P(gust=1|o)={m1:.4f}  P(baboon=1|o)={m2:.4f}')
print(f'product of marginals at (1,1) = {m1*m2:.4f}')
print(f'true joint            at (1,1) = {joint_11:.4f}')
print(f'\\nP(baboon=1 | gust=1, o) = {joint_11/m1:.4f}   <- the gust explains it away')
print(f'P(baboon=1 | gust=0, o) = {post[states.index((0,1))]/(1-m1):.4f}')"""),

("md", """### How far off is the best factorised approximation?

The mean-field family cannot represent a correlated posterior at all. Measure the
gap in nats."""),

("code", """q = np.array([(1-m1)*(1-m2), m1*(1-m2), (1-m1)*m2, m1*m2])   # order: 00,10,01,11
p = np.array([post[states.index(s)] for s in [(0,0),(1,0),(0,1),(1,1)]])
print(f'D_KL[true posterior || product of marginals] = {np.sum(p*np.log(p/q)):.4f} nats')

print('\\nExplaining away strengthens as causes are added:')
for n in (2, 3, 4, 5):
    st, po, _ = joint_posterior(n)
    marg = sum(pp for s, pp in zip(st, po) if s[0] == 1)
    print(f'  {n} causes: P(cause 1 = 1 | o) = {marg:.4f}')"""),

("md", """## 4. Homeostasis: a body that acts, and one that does not

A two-dimensional random walk, absorbed at the unit circle. With `kappa = 0` it
is an unbiased walk and it always dies. With `kappa > 0` it does not diffuse at
all. Watch the entropy of the visited states, not the trajectory."""),

("code", """NB = 26                                   # histogram bins per axis
H_MAX = np.log(NB * NB)

def simulate(sigma=0.035, kappa=0.0, steps=4000, seed=None):
    r = np.random.default_rng(seed)
    x = np.zeros(2)
    bins = np.zeros((NB, NB))
    traj, ent = [], []
    for t in range(steps):
        x = x + sigma * r.normal(size=2) - kappa * x
        traj.append(x.copy())
        b = np.floor((x + 1) / 2 * NB).astype(int)
        if np.all((b >= 0) & (b < NB)):
            bins[b[1], b[0]] += 1
        q = bins / bins.sum()
        nz = q[q > 0]
        ent.append(-(nz * np.log(nz)).sum())
        if x @ x > 1:
            return np.array(traj), np.array(ent), t + 1     # absorbed
    return np.array(traj), np.array(ent), None              # survived

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for kappa, colour, label in [(0.0, '#9c4d2f', 'no action, $\\\\kappa=0$'),
                             (0.15, '#0f5f57', 'acting, $\\\\kappa=0.15$')]:
    traj, ent, died = simulate(kappa=kappa, seed=3)
    axes[0].plot(traj[:, 0], traj[:, 1], lw=0.7, color=colour, alpha=0.8, label=label)
    axes[1].plot(ent, color=colour, lw=1.8, label=label)
    print(f'{label:>22}: {"died at step " + str(died) if died else "survived 4000 steps"}, '
          f'final entropy {ent[-1]:.3f} nats')

theta = np.linspace(0, 2*np.pi, 200)
axes[0].plot(np.cos(theta), np.sin(theta), color='#3c3c42', lw=1.2)
axes[0].plot(0.35*np.cos(theta), 0.35*np.sin(theta), color='#0f5f57', lw=1, ls='--')
axes[0].set_aspect('equal'); axes[0].set_title('physiological state', loc='left')
axes[0].legend(fontsize=8, frameon=False)
axes[1].axhline(H_MAX, color='#90909a', ls=':', lw=1)
axes[1].text(0, H_MAX*0.96, 'uniform occupancy', fontsize=8, color='#90909a')
axes[1].set_xlabel('steps'); axes[1].set_ylabel('entropy of visited states, nats')
axes[1].set_title('staying alive is a low number', loc='left')
axes[1].legend(fontsize=8, frameon=False)
for a in axes: a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()"""),

("md", """### The survival curve

Exercise 6 on the problems page asks you to predict, from dimensional analysis
alone, that lifetime scales as $\\sigma^{-2}$. Here is the measurement. Predict
the slope before you run it."""),

("code", """sigmas = np.array([0.02, 0.03, 0.045, 0.06, 0.09])
means = []
for s in sigmas:
    lives = [simulate(sigma=s, kappa=0.0, steps=20000, seed=i)[2] or 20000
             for i in range(60)]
    means.append(np.mean(lives))
means = np.array(means)

slope, intercept = np.polyfit(np.log(sigmas), np.log(means), 1)
print(f'fitted exponent = {slope:.3f}   (dimensional analysis predicts -2)')

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.loglog(sigmas, means, 'o', color='#0f5f57')
ax.loglog(sigmas, np.exp(intercept) * sigmas**slope, color='#9c4d2f', lw=1.5,
          label=f'slope {slope:.2f}')
ax.set_xlabel('noise $\\\\sigma$'); ax.set_ylabel('mean steps before absorption')
ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()"""),

("md", """## Over to you

1. Change `prior` to the nervous hiker's `[0.40, 0.30, 0.30]` and recompute the
   surprise of *quiet*. Why is the nervous hiker more surprised by silence?
2. Build a two-state, three-observation model in which the rarest observation
   carries **zero** information. (Hint: make one row of `A` constant.)
3. In the noisy-OR model, set `strength = 0` so the observation is uninformative.
   Predict what happens to the mean-field gap before running it.
4. Add a third dimension to the homeostasis simulation. Does the lifetime scaling
   change? Should it?"""),
]

build(numpy_cells, os.path.join(OUT, "week01_numpy.ipynb"))


# ── JAX ──────────────────────────────────────────────────────────────────
jax_cells = [
("md", """# Week 1 — The hypothesis-testing agent (JAX, advanced stream)

The same Week 1 material, vectorised. This is the advanced stream: it exists
where doing many of something at once buys a genuinely different picture, not
merely a faster version of the same one.

Two places that applies this week:

- **Posteriors over a whole grid of priors at once**, so you can see the
  decision boundary of an inference problem rather than one posterior.
- **Ten thousand homeostasis runs in parallel**, so you get a survival
  *distribution* rather than a trajectory.

On Colab: `!pip install -q jax jaxlib` is usually unnecessary, JAX ships preinstalled."""),

("code", """import jax
import jax.numpy as jnp
from jax import vmap, jit, random
import numpy as np
import matplotlib.pyplot as plt

print('jax', jax.__version__, '| devices:', jax.devices())"""),

("md", """## 1. Every posterior at once

`vmap` turns a function written for one prior into one that handles a grid of
them, with no loop and no reshaping by hand."""),

("code", """A = jnp.array([[0.70, 0.15, 0.02],
               [0.25, 0.75, 0.08],
               [0.05, 0.10, 0.90]])

def posterior(prior, o):
    un = A[o] * prior
    ev = un.sum()
    return un / ev, ev

# a grid over the (leopard, baboon) simplex; 'nothing' takes up the slack
g = jnp.linspace(0.01, 0.60, 220)
L, B = jnp.meshgrid(g, g, indexing='ij')
valid = (L + B) < 0.98
priors = jnp.stack([L, B, 1 - L - B], axis=-1).reshape(-1, 3)

post, ev = jit(vmap(posterior, in_axes=(0, None)))(priors, 0)   # observe tawny flash
p_leopard = post[:, 0].reshape(L.shape)
print('grid of', priors.shape[0], 'posteriors computed in one call')"""),

("code", """fig, ax = plt.subplots(figsize=(6.2, 5))
masked = jnp.where(valid, p_leopard, jnp.nan)
im = ax.pcolormesh(np.array(B), np.array(L), np.array(masked),
                   cmap='BrBG', vmin=0, vmax=1, shading='auto')
cs = ax.contour(np.array(B), np.array(L), np.array(masked),
                levels=[0.5], colors='#1b1b1e', linewidths=2)
ax.clabel(cs, fmt={0.5: 'undecided'}, fontsize=9)
ax.set_xlabel('prior on baboon'); ax.set_ylabel('prior on leopard')
ax.set_title('P(leopard | tawny flash) across the prior simplex', loc='left')
fig.colorbar(im, ax=ax, label='posterior on leopard')
plt.tight_layout(); plt.show()

print('The black contour is the tipping point from the problems page.')
print('At a baboon prior of 0.22 it crosses the leopard prior at about 0.068.')"""),

("md", """## 2. Ten thousand lives at once

One trajectory tells you an anecdote. `vmap` over the random key gives you the
survival distribution, which is the thing the theory actually makes claims about."""),

("code", """@jit
def survive(key, sigma, kappa, steps=20000):
    \"\"\"Return the step at which the walk leaves the unit disc, or `steps`.\"\"\"
    noise = random.normal(key, (steps, 2)) * sigma

    def step(carry, inp):
        x, done, t = carry
        eps, i = inp
        x_new = jnp.where(done, x, x * (1 - kappa) + eps)
        out = (x_new @ x_new) > 1.0
        newly = out & (~done)
        t_new = jnp.where(newly, i, t)
        return (x_new, done | out, t_new), None

    idx = jnp.arange(steps)
    (_, died, t_death), _ = jax.lax.scan(
        step, (jnp.zeros(2), False, jnp.int32(steps)), (noise, idx))
    return t_death

keys = random.split(random.PRNGKey(0), 10000)
lifetimes = jit(vmap(survive, in_axes=(0, None, None)))(keys, 0.035, 0.0)
lifetimes = np.array(lifetimes)
print(f'10,000 unregulated lives: median {np.median(lifetimes):.0f} steps, '
      f'mean {lifetimes.mean():.0f}, max {lifetimes.max()}')"""),

("code", """fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

axes[0].hist(lifetimes, bins=60, color='#9c4d2f', alpha=0.85)
axes[0].set_xlabel('steps before absorption'); axes[0].set_ylabel('count')
axes[0].set_title('lifetime distribution, $\\\\kappa=0$', loc='left')

# the scaling law, measured rather than argued
sigmas = np.array([0.02, 0.03, 0.045, 0.06, 0.09])
med = []
for s in sigmas:
    lt = np.array(jit(vmap(survive, in_axes=(0, None, None)))(keys[:2000], float(s), 0.0))
    med.append(np.median(lt))
med = np.array(med)
slope = np.polyfit(np.log(sigmas), np.log(med), 1)[0]
axes[1].loglog(sigmas, med, 'o-', color='#0f5f57')
axes[1].set_xlabel('noise $\\\\sigma$'); axes[1].set_ylabel('median lifetime')
axes[1].set_title(f'measured exponent {slope:.2f}, theory $-2$', loc='left')

for a in axes: a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.show()"""),

("code", """# and with action. The interesting range is far smaller than you would guess.
print('kappa   survived 20,000 steps   median lifetime')
for kappa in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02):
    lt = np.array(jit(vmap(survive, in_axes=(0, None, None)))(keys[:3000], 0.035, float(kappa)))
    frac = (lt >= 20000).mean()
    med = np.median(lt)
    print(f'{kappa:<7.4f} {frac*100:>16.1f}%   {med:>14.0f}')"""),

("md", """That last table is the whole of Lesson 1 in four numbers. The physics did not
change and the noise did not change. What changed is that the system began to
move in a way that depends on where it is.

## Over to you

1. Sweep `kappa` finely between 0 and 0.05 and find where survival probability
   crosses one half. Is the transition sharp?
2. `survive` currently keeps integrating after death and masks the result. Rewrite
   it with `jax.lax.while_loop` to stop early. Is it faster? Why might it not be,
   under `vmap`?
3. Vectorise the *ambiguity* sweep from the NumPy notebook over both `lambda` and
   the prior at once, and plot information gained as a surface."""),
]

build(jax_cells, os.path.join(OUT, "week01_jax.ipynb"))
