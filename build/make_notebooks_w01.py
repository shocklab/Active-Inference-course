#!/usr/bin/env python3
"""Generate Week 1 notebooks from the reviewed cell sources below."""
from pathlib import Path
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]

NUMPY_CELLS = [('markdown',
  '# Week 1 : The hypothesis-testing agent (NumPy)\n'
  '\n'
  'Selected Week 1 examples implemented with NumPy, with intermediate arrays available to '
  'inspect.\n'
  '\n'
  'Run the cells in order. The exercises at the bottom are the ones from the\n'
  'lesson pages; the solutions are on the website, so try them here first.\n'
  '\n'
  '**Contents**\n'
  '1. Bayes with states and observations\n'
  '2. The evidence sum, and what it costs\n'
  '3. Explaining away, and what mean-field loses\n'
  '4. Homeostasis: a body that acts, and one that does not'),
 ('code',
  'import numpy as np\n'
  'import matplotlib.pyplot as plt\n'
  '\n'
  'np.set_printoptions(precision=4, suppress=True)\n'
  'rng = np.random.default_rng(0)'),
 ('markdown',
  '## 1. Bayes with states and observations\n'
  '\n'
  'The likelihood array `A` has observations down the **rows** and states across the\n'
  '**columns**, so `A[i, j] = P(o = i | s = j)`. Each column is a distribution over\n'
  'observations and therefore sums to one. Each row is a likelihood read as a\n'
  'function of the state and has no reason to sum to anything.'),
 ('code',
  "STATES = ['leopard', 'baboon', 'nothing']\n"
  "OBS    = ['tawny flash', 'branch shakes', 'quiet']\n"
  '\n'
  'A = np.array([\n'
  '    [0.70, 0.15, 0.02],   # tawny flash\n'
  '    [0.25, 0.75, 0.08],   # branch shakes\n'
  '    [0.05, 0.10, 0.90],   # quiet\n'
  '])\n'
  '\n'
  'prior = np.array([0.08, 0.22, 0.70])\n'
  '\n'
  "print('column sums (must all be 1):', A.sum(axis=0))\n"
  "print('row sums    (no reason to be 1):', A.sum(axis=1))"),
 ('code',
  'def posterior(A, prior, o):\n'
  '    """Bayes for a discrete model. Returns the posterior and the evidence P(o)."""\n'
  '    unnormalised = A[o] * prior          # elementwise: P(o|s) P(s), one per state\n'
  '    evidence = unnormalised.sum()        # the sum over every state in the model\n'
  '    return unnormalised / evidence, evidence\n'
  '\n'
  '\n'
  'for o, name in enumerate(OBS):\n'
  '    post, ev = posterior(A, prior, o)\n'
  '    surprise = -np.log(ev)\n'
  '    info = np.sum(post * np.log(post / prior))     # D_KL[posterior || prior]\n'
  "    print(f'{name:>14} | P(o)={ev:.4f}  surprise={surprise:.3f} nats'\n"
  "          f'  info gained={info:.3f} nats')\n"
  "    shown = '  '.join(f'{n}={v:.4f}' for n, v in zip(STATES, post))\n"
  '    print(f\'{"":>14} | posterior  {shown}\')'),
 ('markdown',
  'Those are the numbers quoted in Lesson 3. A tawny flash takes the leopard\n'
  'hypothesis from 8% to 54%: not because leopards became likely, but because\n'
  'nothing else explains tawny flashes.\n'
  '\n'
  '### The ambiguity slider, in code\n'
  '\n'
  'Blend the likelihood towards uniform and watch the observation stop telling you\n'
  'anything.'),
 ('code',
  'def blur(A, lam):\n'
  '    """lam=0 leaves A alone; lam=1 makes every state predict every observation equally."""\n'
  '    B = (1 - lam) * A + lam / A.shape[0]\n'
  '    return B / B.sum(axis=0, keepdims=True)      # keep the columns normalised\n'
  '\n'
  'lams = np.linspace(0, 0.98, 40)\n'
  'info = [np.sum(posterior(blur(A, l), prior, 0)[0]\n'
  '               * np.log(posterior(blur(A, l), prior, 0)[0] / prior)) for l in lams]\n'
  '\n'
  'fig, ax = plt.subplots(figsize=(7, 3.2))\n'
  "ax.plot(lams, info, color='#0f5f57', lw=2)\n"
  "ax.set_xlabel('ambiguity $\\\\lambda$'); ax.set_ylabel('information gained, nats')\n"
  "ax.set_title('An ambiguous channel carries no information', loc='left')\n"
  "ax.spines[['top', 'right']].set_visible(False)\n"
  'plt.tight_layout(); plt.show()'),
 ('markdown',
  '## 2. What the evidence sum costs\n'
  '\n'
  '`P(o)` requires visiting every state the model admits. With `n` factors of `k`\n'
  'values each, that is `k**n` terms.'),
 ('code',
  'for n in (5, 10, 20, 40, 89):\n'
  '    terms = 2.0 ** n\n'
  '    secs = terms / 1e9                      # one addition per nanosecond, generous\n'
  "    print(f'{n:>3} binary factors -> {terms:.3e} terms, {secs:.3e} s at 1e9 add/s')\n"
  '\n'
  'print()\n'
  "print('age of the universe in seconds ~ 4.35e17')\n"
  "print('so 89 binary factors already exceeds it.')"),
 ('markdown',
  '## 2b. Computing an observation probability in a chain\n'
  '\n'
  'Use the normalised transition matrix and final observation likelihood from\n'
  'Lesson 5. Enumeration and backward messages compute the same marginal\n'
  'likelihood. Compare them within floating-point tolerance; the order of addition\n'
  'can affect the last few bits.\n'),
 ('code',
  'import itertools\n'
  '\n'
  'p0 = np.array([0.4, 0.6])\n'
  'T = np.array([[0.7, 0.2], [0.3, 0.8]])  # row = next, column = current\n'
  'lik = np.array([0.1, 0.9])              # P(observed report | final state)\n'
  'assert np.allclose(T.sum(axis=0), 1)\n'
  '\n'
  'def brute_evidence(n):\n'
  '    total = 0.0\n'
  '    for path in itertools.product(range(2), repeat=n):\n'
  '        term = p0[path[0]] * lik[path[-1]]\n'
  '        for i in range(n - 1):\n'
  '            term *= T[path[i + 1], path[i]]\n'
  '        total += term\n'
  '    return total\n'
  '\n'
  'def message_evidence(n):\n'
  '    message = lik.copy()\n'
  '    for _ in range(n - 1):\n'
  '        message = T.T @ message\n'
  '    return p0 @ message\n'
  '\n'
  'for n in (3, 6, 9, 12):\n'
  '    enumerated, messages = brute_evidence(n), message_evidence(n)\n'
  '    assert np.isclose(enumerated, messages, rtol=1e-12, atol=1e-14)\n'
  "    print(f'n={n}: {2**n} joint states, evidence={messages:.8f}, '\n"
  "          f'difference={abs(enumerated-messages):.2e}')\n"),
 ('markdown',
  'Each backward step is a matrix-vector product. With `k` states, the work\n'
  'scales as `k**2` per step, while enumeration visits `k**n` joint states.\n'
  'This is a scaling statement, not a hardware-independent elapsed-time forecast.\n'),
 ('code',
  'for n in (20, 30, 40):\n'
  "    print(f'n={n}: {2**n} joint states; {n-1} binary message updates; '\n"
  "          f'evidence={message_evidence(n):.8f}')\n"),
 ('markdown',
  '## Intermediate factors\n'
  '\n'
  'The chain is cheap because the intermediate message has one state index.\n'
  'In a general tabular graph, elimination can create factors over many remaining\n'
  'variables. A factor with `w+1` indices of size `k` stores `k**(w+1)` entries.\n'
  'The largest such scope depends on the graph and the elimination order.\n'),
 ('code',
  'k = 2\n'
  'for width in (1, 4, 8, 12):\n'
  "    print(f'width={width}: largest tabular factor has {k**(width+1)} entries')\n"),
 ('markdown',
  'Posterior dependence does not by itself make exact inference expensive.\n'
  'An observed chain can contain dependent hidden states and still admit these\n'
  'small messages. We need to inspect the factor structure, not merely ask\n'
  'whether explaining away occurs.\n'),
 ('markdown',
  '## 3. What mean-field loses to explaining away\n'
  '\n'
  'Two independent causes, either of which can shake a branch. Independent priors,\n'
  'a shared observation, and a posterior that is anything but independent.'),
 ('code',
  'import itertools\n'
  '\n'
  'p_cause, strength, leak = 0.1, 0.9, 0.01\n'
  '\n'
  'def noisy_or(s):\n'
  '    """P(o=1 | s) for a vector of binary causes."""\n'
  '    return 1 - np.prod([1 - strength * x for x in s]) * (1 - leak)\n'
  '\n'
  'def joint_posterior(n_causes):\n'
  '    states = list(itertools.product([0, 1], repeat=n_causes))\n'
  '    pri = np.array([np.prod([p_cause if x else 1 - p_cause for x in s]) for s in states])\n'
  '    lik = np.array([noisy_or(s) for s in states])\n'
  '    un = pri * lik\n'
  '    return states, un / un.sum(), un.sum()\n'
  '\n'
  'states, post, ev = joint_posterior(2)\n'
  "print(f'P(o=1) = {ev:.6f}   surprise = {-np.log(ev):.4f} nats\\n')\n"
  'for s, p in zip(states, post):\n'
  "    print(f'  gust={s[0]}  baboon={s[1]}   posterior={p:.4f}')\n"
  '\n'
  'm1 = sum(p for s, p in zip(states, post) if s[0] == 1)\n'
  'm2 = sum(p for s, p in zip(states, post) if s[1] == 1)\n'
  'joint_11 = post[states.index((1, 1))]\n'
  "print(f'\\nmarginals: P(gust=1|o)={m1:.4f}  P(baboon=1|o)={m2:.4f}')\n"
  "print(f'product of marginals at (1,1) = {m1*m2:.4f}')\n"
  "print(f'true joint            at (1,1) = {joint_11:.4f}')\n"
  "print(f'\\nP(baboon=1 | gust=1, o) = {joint_11/m1:.4f}   <- the gust explains it away')\n"
  "print(f'P(baboon=1 | gust=0, o) = {post[states.index((0,1))]/(1-m1):.4f}')"),
 ('markdown',
  '### How far off is the best factorised approximation?\n'
  '\n'
  'The mean-field family cannot represent a correlated posterior at all. Measure the\n'
  'gap in nats.'),
 ('code',
  'q = np.array([(1-m1)*(1-m2), m1*(1-m2), (1-m1)*m2, m1*m2])   # order: 00,10,01,11\n'
  'p = np.array([post[states.index(s)] for s in [(0,0),(1,0),(0,1),(1,1)]])\n'
  "print(f'D_KL[true posterior || product of marginals] = {np.sum(p*np.log(p/q)):.4f} nats')\n"
  '\n'
  "print('\\nExplaining away strengthens as causes are added:')\n"
  'for n in (2, 3, 4, 5):\n'
  '    st, po, _ = joint_posterior(n)\n'
  '    marg = sum(pp for s, pp in zip(st, po) if s[0] == 1)\n'
  "    print(f'  {n} causes: P(cause 1 = 1 | o) = {marg:.4f}')"),
 ('markdown',
  '## 4. Homeostasis: a body that acts, and one that does not\n'
  '\n'
  'A two-dimensional random walk, absorbed at the unit circle. With `kappa = 0` it\n'
  'is an unbiased walk and it always dies. With `kappa > 0` it does not diffuse at\n'
  'all. Watch the entropy of the visited states, not the trajectory.'),
 ('code',
  'NB = 26                                   # histogram bins per axis\n'
  'H_MAX = np.log(NB * NB)\n'
  '\n'
  'def simulate(sigma=0.035, kappa=0.0, steps=4000, seed=None):\n'
  '    r = np.random.default_rng(seed)\n'
  '    x = np.zeros(2)\n'
  '    bins = np.zeros((NB, NB))\n'
  '    traj, ent = [], []\n'
  '    for t in range(steps):\n'
  '        x = x + sigma * r.normal(size=2) - kappa * x\n'
  '        traj.append(x.copy())\n'
  '        b = np.floor((x + 1) / 2 * NB).astype(int)\n'
  '        if np.all((b >= 0) & (b < NB)):\n'
  '            bins[b[1], b[0]] += 1\n'
  '        q = bins / bins.sum()\n'
  '        nz = q[q > 0]\n'
  '        ent.append(-(nz * np.log(nz)).sum())\n'
  '        if x @ x > 1:\n'
  '            return np.array(traj), np.array(ent), t + 1     # absorbed\n'
  '    return np.array(traj), np.array(ent), None              # survived\n'
  '\n'
  'fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n'
  "for kappa, colour, label in [(0.0, '#9c4d2f', 'no action, $\\\\kappa=0$'),\n"
  "                             (0.15, '#0f5f57', 'acting, $\\\\kappa=0.15$')]:\n"
  '    traj, ent, died = simulate(kappa=kappa, seed=3)\n'
  '    axes[0].plot(traj[:, 0], traj[:, 1], lw=0.7, color=colour, alpha=0.8, label=label)\n'
  '    axes[1].plot(ent, color=colour, lw=1.8, label=label)\n'
  '    print(f\'{label:>22}: {"died at step " + str(died) if died else "survived 4000 steps"}, \'\n'
  "          f'final entropy {ent[-1]:.3f} nats')\n"
  '\n'
  'theta = np.linspace(0, 2*np.pi, 200)\n'
  "axes[0].plot(np.cos(theta), np.sin(theta), color='#3c3c42', lw=1.2)\n"
  "axes[0].plot(0.35*np.cos(theta), 0.35*np.sin(theta), color='#0f5f57', lw=1, ls='--')\n"
  "axes[0].set_aspect('equal'); axes[0].set_title('physiological state', loc='left')\n"
  'axes[0].legend(fontsize=8, frameon=False)\n'
  "axes[1].axhline(H_MAX, color='#90909a', ls=':', lw=1)\n"
  "axes[1].text(0, H_MAX*0.96, 'uniform occupancy', fontsize=8, color='#90909a')\n"
  "axes[1].set_xlabel('steps'); axes[1].set_ylabel('entropy of visited states, nats')\n"
  "axes[1].set_title('staying alive is a low number', loc='left')\n"
  'axes[1].legend(fontsize=8, frameon=False)\n'
  "for a in axes: a.spines[['top', 'right']].set_visible(False)\n"
  'plt.tight_layout(); plt.show()'),
 ('markdown',
  '### The survival curve\n'
  '\n'
  'Exercise 6 on the problems page asks you to predict, from a small-step diffusion approximation, '
  'that lifetime scales as $\\sigma^{-2}$. Here is the measurement. Predict\n'
  'the slope before you run it.'),
 ('code',
  'sigmas = np.array([0.02, 0.03, 0.045, 0.06, 0.09])\n'
  'means = []\n'
  'for s in sigmas:\n'
  '    lives = [simulate(sigma=s, kappa=0.0, steps=20000, seed=i)[2] or 20000\n'
  '             for i in range(60)]\n'
  '    means.append(np.mean(lives))\n'
  'means = np.array(means)\n'
  '\n'
  'slope, intercept = np.polyfit(np.log(sigmas), np.log(means), 1)\n'
  "print(f'fitted exponent = {slope:.3f}   (dimensional analysis predicts -2)')\n"
  '\n'
  'fig, ax = plt.subplots(figsize=(6, 3.6))\n'
  "ax.loglog(sigmas, means, 'o', color='#0f5f57')\n"
  "ax.loglog(sigmas, np.exp(intercept) * sigmas**slope, color='#9c4d2f', lw=1.5,\n"
  "          label=f'slope {slope:.2f}')\n"
  "ax.set_xlabel('noise $\\\\sigma$'); ax.set_ylabel('mean steps before absorption')\n"
  "ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)\n"
  'plt.tight_layout(); plt.show()'),
 ('markdown',
  '## Over to you\n'
  '\n'
  "1. Change `prior` to the nervous hiker's `[0.40, 0.30, 0.30]` and recompute the\n"
  '   surprise of *quiet*. Why is the nervous hiker more surprised by silence?\n'
  '2. Build a two-state, three-observation model in which the rarest observation\n'
  '   carries **zero** information. (Hint: make one row of `A` constant.)\n'
  '3. In the noisy-OR model, set `strength = 0` so the observation is uninformative.\n'
  '   Predict what happens to the mean-field gap before running it.\n'
  '4. Add a third dimension to the homeostasis simulation. Does the lifetime scaling\n'
  '   change? Should it?')]

JAX_CELLS = [('markdown',
  '# Week 1 : The hypothesis-testing agent (JAX, advanced stream)\n'
  '\n'
  'The same Week 1 material, vectorised. This is the advanced stream: it exists\n'
  'where doing many of something at once buys a different picture, not\n'
  'merely a faster version of the same one.\n'
  '\n'
  'Two places that applies this week:\n'
  '\n'
  '- **Posteriors over a whole grid of priors at once**, so you can see the\n'
  '  decision boundary of an inference problem rather than one posterior.\n'
  '- **Ten thousand homeostasis runs in parallel**, so you get a survival\n'
  '  *distribution* rather than a trajectory.\n'
  '\n'
  'On Colab: `!pip install -q jax jaxlib` is usually unnecessary, JAX ships preinstalled.'),
 ('code',
  'import jax\n'
  'import jax.numpy as jnp\n'
  'from jax import vmap, jit, random\n'
  'import numpy as np\n'
  'import matplotlib.pyplot as plt\n'
  '\n'
  "print('jax', jax.__version__, '| devices:', jax.devices())"),
 ('markdown',
  '## 1. Every posterior at once\n'
  '\n'
  '`vmap` turns a function written for one prior into one that handles a grid of\n'
  'them, with no loop and no reshaping by hand.'),
 ('code',
  'A = jnp.array([[0.70, 0.15, 0.02],\n'
  '               [0.25, 0.75, 0.08],\n'
  '               [0.05, 0.10, 0.90]])\n'
  '\n'
  'def posterior(prior, o):\n'
  '    un = A[o] * prior\n'
  '    ev = un.sum()\n'
  '    return un / ev, ev\n'
  '\n'
  "# a grid over the (leopard, baboon) simplex; 'nothing' takes up the slack\n"
  'g = jnp.linspace(0.01, 0.60, 220)\n'
  "L, B = jnp.meshgrid(g, g, indexing='ij')\n"
  'valid = (L + B) < 0.98\n'
  'priors = jnp.stack([L, B, 1 - L - B], axis=-1).reshape(-1, 3)\n'
  '\n'
  'post, ev = jit(vmap(posterior, in_axes=(0, None)))(priors, 0)   # observe tawny flash\n'
  'p_leopard = post[:, 0].reshape(L.shape)\n'
  "print('grid of', priors.shape[0], 'posteriors computed in one call')"),
 ('code',
  'fig, ax = plt.subplots(figsize=(6.2, 5))\n'
  'masked = jnp.where(valid, p_leopard, jnp.nan)\n'
  'im = ax.pcolormesh(np.array(B), np.array(L), np.array(masked),\n'
  "                   cmap='BrBG', vmin=0, vmax=1, shading='auto')\n"
  'cs = ax.contour(np.array(B), np.array(L), np.array(masked),\n'
  "                levels=[0.5], colors='#1b1b1e', linewidths=2)\n"
  "ax.clabel(cs, fmt={0.5: 'undecided'}, fontsize=9)\n"
  "ax.set_xlabel('prior on baboon'); ax.set_ylabel('prior on leopard')\n"
  "ax.set_title('P(leopard | tawny flash) across the prior simplex', loc='left')\n"
  "fig.colorbar(im, ax=ax, label='posterior on leopard')\n"
  'plt.tight_layout(); plt.show()\n'
  '\n'
  "print('The black contour is the tipping point from the problems page.')\n"
  "print('At a baboon prior of 0.22 it crosses the leopard prior at about 0.068.')"),
 ('markdown',
  '## 2. Ten thousand lives at once\n'
  '\n'
  'One trajectory tells you an anecdote. `vmap` over the random key gives you the\n'
  'survival distribution, which is the thing the theory actually makes claims about.'),
 ('code',
  '@jit\n'
  'def survive(key, sigma, kappa, steps=20000):\n'
  '    """Return the step at which the walk leaves the unit disc, or `steps`."""\n'
  '    noise = random.normal(key, (steps, 2)) * sigma\n'
  '\n'
  '    def step(carry, inp):\n'
  '        x, done, t = carry\n'
  '        eps, i = inp\n'
  '        x_new = jnp.where(done, x, x * (1 - kappa) + eps)\n'
  '        out = (x_new @ x_new) > 1.0\n'
  '        newly = out & (~done)\n'
  '        t_new = jnp.where(newly, i, t)\n'
  '        return (x_new, done | out, t_new), None\n'
  '\n'
  '    idx = jnp.arange(steps)\n'
  '    (_, died, t_death), _ = jax.lax.scan(\n'
  '        step, (jnp.zeros(2), False, jnp.int32(steps)), (noise, idx))\n'
  '    return t_death\n'
  '\n'
  'keys = random.split(random.PRNGKey(0), 10000)\n'
  'lifetimes = jit(vmap(survive, in_axes=(0, None, None)))(keys, 0.035, 0.0)\n'
  'lifetimes = np.array(lifetimes)\n'
  "print(f'10,000 unregulated lives: median {np.median(lifetimes):.0f} steps, '\n"
  "      f'mean {lifetimes.mean():.0f}, max {lifetimes.max()}')"),
 ('code',
  'fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))\n'
  '\n'
  "axes[0].hist(lifetimes, bins=60, color='#9c4d2f', alpha=0.85)\n"
  "axes[0].set_xlabel('steps before absorption'); axes[0].set_ylabel('count')\n"
  "axes[0].set_title('lifetime distribution, $\\\\kappa=0$', loc='left')\n"
  '\n'
  '# the scaling law, measured rather than argued\n'
  'sigmas = np.array([0.02, 0.03, 0.045, 0.06, 0.09])\n'
  'med = []\n'
  'for s in sigmas:\n'
  '    lt = np.array(jit(vmap(survive, in_axes=(0, None, None)))(keys[:2000], float(s), 0.0))\n'
  '    med.append(np.median(lt))\n'
  'med = np.array(med)\n'
  'slope = np.polyfit(np.log(sigmas), np.log(med), 1)[0]\n'
  "axes[1].loglog(sigmas, med, 'o-', color='#0f5f57')\n"
  "axes[1].set_xlabel('noise $\\\\sigma$'); axes[1].set_ylabel('median lifetime')\n"
  "axes[1].set_title(f'measured exponent {slope:.2f}, theory $-2$', loc='left')\n"
  '\n'
  "for a in axes: a.spines[['top', 'right']].set_visible(False)\n"
  'plt.tight_layout(); plt.show()'),
 ('code',
  '# and with action. The interesting range is far smaller than you would guess.\n'
  "print('kappa   survived 20,000 steps   median lifetime')\n"
  'for kappa in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.02):\n'
  '    lt = np.array(jit(vmap(survive, in_axes=(0, None, None)))(keys[:3000], 0.035, '
  'float(kappa)))\n'
  '    frac = (lt >= 20000).mean()\n'
  '    med = np.median(lt)\n'
  "    print(f'{kappa:<7.4f} {frac*100:>16.1f}%   {med:>14.0f}')"),
 ('markdown',
  'That last table is the whole of Lesson 1 in four numbers. The physics did not\n'
  'change and the noise did not change. What changed is that the system began to\n'
  'move in a way that depends on where it is.\n'
  '\n'
  '## Over to you\n'
  '\n'
  '1. Sweep `kappa` finely between 0 and 0.05 and find where survival probability\n'
  '   crosses one half. Is the transition sharp?\n'
  '2. `survive` currently keeps integrating after death and masks the result. Rewrite\n'
  '   it with `jax.lax.while_loop` to stop early. Is it faster? Why might it not be,\n'
  '   under `vmap`?\n'
  '3. Vectorise the *ambiguity* sweep from the NumPy notebook over both `alpha` and\n'
  '   the prior at once, and plot information gained as a surface.')]

AI_ATTRIBUTION = "\n\n### AI use\n\nThis course notebook was developed with substantial assistance from Claude\n(Anthropic) and revised with Codex (OpenAI), under Jonathan Shock's direction.\nAI contributed to explanations and code. See [AI use and attribution](https://shocklab.github.io/Active-Inference-course/ai-use.html)\nfor the review process and its limits.\n"
for cells in (NUMPY_CELLS, JAX_CELLS):
    kind, text = cells[0]
    cells[0] = (kind, text.rstrip() + AI_ATTRIBUTION)

if __name__ == '__main__':
    for name, cells in [('numpy', NUMPY_CELLS), ('jax', JAX_CELLS)]:
        nb = nbf.v4.new_notebook(cells=[
            nbf.v4.new_markdown_cell(s) if kind == 'markdown' else nbf.v4.new_code_cell(s)
            for kind, s in cells])
        nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
        path = ROOT / 'notebooks' / 'week-01' / f'week01_{name}.ipynb'
        nbf.write(nb, path)
        print(path)
