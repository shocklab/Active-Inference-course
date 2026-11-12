"""The two-cause product-observation model of Lesson 4.5, computed.

Model (from the last cell of Jonathan Shock's Mathematica notebook):
    s_i ~ Exp(rate lambda_i), i = 1, 2, independent
    u_1 = s_2 s_1^2 + z_1,  u_2 = s_1 s_2^2 + z_2,  z_i ~ N(0, 1) independent.

What is here, all NumPy:
  * the seeded sample of N = 500 trials (mulberry32, shared with assets/js/w04.js);
  * the exact posterior of each trial by numerical integration in the cube roots of
    the noiseless sensor outputs, with separate refinement of grid spacing, limits
    and the grading near zero;
  * three mean-field families: free-form q1 q2 (coordinate ascent with 1-D
    quadrature), a product of Gammas and a product of Exponentials, each solved to a
    stated gradient or fixed-point tolerance from several starts;
  * variational EM for each family with E-steps warm-started from the previous Q,
    a check that every E-step and every M-step leaves total free energy no higher,
    and the M-step lambda_i = 1 / mean E_q[s_i];
  * the M-map Jacobians that give the feedback gain of Lesson 4.5.

`compute()` returns a dict of results. values.py caches it on disk, keyed by the hash
of this file, because the full computation takes tens of seconds.
"""
import numpy as np
from math import log, pi, sqrt, cos, lgamma

LN2PI = log(2.0 * pi)
LAM_TRUE = (0.9, 0.7)
LAM_INIT = (2.0, 0.5)
N_TRIALS = 500
SEED = 42
REP_U = (2.5, 3.0)          # the representative trial, also the widget default


# ── sample ───────────────────────────────────────────────────────────────
def mulberry32(seed):
    a = [seed & 0xFFFFFFFF]

    def imul(x, y):
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF

    def rnd():
        a[0] = (a[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a[0] ^ (a[0] >> 15), 1 | a[0])
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def sample(n=N_TRIALS, seed=SEED, lam=LAM_TRUE):
    """Exponential draws by inversion, Gaussian noise by Box-Muller, in the order
    s1, s2, z1, z2 per trial, exactly as w04.js and both notebooks draw them."""
    rf = mulberry32(seed)

    def normal():
        a, b = rf(), rf()
        return sqrt(-2.0 * log(1.0 - a)) * cos(2.0 * pi * b)
    s1 = np.empty(n); s2 = np.empty(n); u1 = np.empty(n); u2 = np.empty(n)
    for i in range(n):
        a = -log(1.0 - rf()) / lam[0]
        b = -log(1.0 - rf()) / lam[1]
        s1[i], s2[i] = a, b
        u1[i] = b * a * a + normal()
        u2[i] = b * b * a + normal()
    return s1, s2, u1, u2


# ── reference integration ────────────────────────────────────────────────
# With g1 = s2 s1^2, g2 = s1 s2^2 and y_i = g_i^(1/3): s1 = y1^2/y2, s2 = y2^2/y1, and
# the Jacobian d(s1, s2)/d(y1, y2) has determinant 3. So
#   P(u | lam) = (3 lam1 lam2 / 2 pi) * integral over y > 0 of
#                exp(-lam1 y1^2/y2 - lam2 y2^2/y1 - (u1 - y1^3)^2/2 - (u2 - y2^3)^2/2).
# The likelihood factorises in y, so each axis is laid out from its own observation.
DEFAULT_QUAD = dict(L=10.0, hy=1.5, ngrade=14, ratio=0.25, order=8)
# each refinement changes one setting of DEFAULT_QUAD
QUAD_VARIANTS = {"hy": dict(hy=0.75), "L": dict(L=14.0), "grade": dict(ngrade=24, ratio=0.2),
                 "order": dict(order=12)}


def gl_panels(breaks, order=8):
    gx, gw = np.polynomial.legendre.leggauss(order)
    a = breaks[:-1][:, None]
    b = breaks[1:][:, None]
    return ((0.5 * (a + b) + 0.5 * (b - a) * gx[None, :]).ravel(),
            (0.5 * (b - a) * gw[None, :]).ravel())


def y_nodes(u, L=10.0, hy=1.5, ngrade=14, ratio=0.25, order=8):
    """Nodes in y = g^(1/3) for one sensor. Panels of width hy in g over
    [max(0, u - L), max(u, 0) + L] (L noise standard deviations either side), mapped
    to y; when that range reaches g = 0 the first panel is replaced by panels graded
    geometrically towards y = 0."""
    glo = max(0.0, u - L)
    ghi = max(u, 0.0) + L
    npan = int(np.ceil((ghi - glo) / hy))
    yb = np.cbrt(np.linspace(glo, ghi, npan + 1))
    if glo == 0.0:
        grade = yb[1] * ratio ** np.arange(ngrade, 0, -1)
        yb = np.concatenate([[0.0], grade, yb[1:]])
    return gl_panels(yb, order)


class TrialGrid:
    """Tensor-product nodes for one trial; the parts that do not depend on lambda."""

    def __init__(self, u1, u2, **quad):
        q = dict(DEFAULT_QUAD, **quad)
        x1, v1 = y_nodes(u1, **q)
        x2, v2 = y_nodes(u2, **q)
        Y1, Y2 = np.meshgrid(x1, x2, indexing="ij")
        self.s1 = (Y1 * Y1 / Y2).ravel()
        self.s2 = (Y2 * Y2 / Y1).ravel()
        self.lw = (np.log(np.outer(v1, v2)) - 0.5 * (u1 - Y1 ** 3) ** 2
                   - 0.5 * (u2 - Y2 ** 3) ** 2).ravel()

    def post(self, r1, r2, full=True):
        lf = self.lw - r1 * self.s1 - r2 * self.s2
        mx = lf.max()
        p = np.exp(lf - mx)
        Z = p.sum()
        nll = -(mx + log(Z) + log(3.0 * r1 * r2) - LN2PI)
        p /= Z
        m1 = (p * self.s1).sum()
        m2 = (p * self.s2).sum()
        out = dict(nll=nll, m1=m1, m2=m2)
        if full:
            d1 = self.s1 - m1
            d2 = self.s2 - m2
            v1 = (p * d1 * d1).sum()
            v2 = (p * d2 * d2).sum()
            c = (p * d1 * d2).sum()
            out.update(v1=v1, v2=v2, sd1=sqrt(v1), sd2=sqrt(v2), corr=c / sqrt(v1 * v2))
        return out


def exact_batch(grids, r1, r2, full=True):
    rows = [g.post(r1, r2, full) for g in grids]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ── vectorised modified Newton ───────────────────────────────────────────
def newton_min(f, gh, x0, tol, maxit=200):
    """Minimise f row by row. Modified Newton (Hessian eigenvalues floored to make it
    positive definite), steps capped at 2 in every coordinate, backtracking with an
    Armijo condition. A row stops when its largest gradient component is below tol,
    when its Newton decrement is at the rounding level of f, or when no step lowers f
    (it is then at the rounding floor). Returns x, f, max |gradient| per row."""
    x = x0.copy()
    fx = f(x)
    n, _ = x.shape
    frozen = np.zeros(n, bool)
    for _ in range(maxit):
        g, H = gh(x)
        gn = np.max(np.abs(g), axis=1)
        ev, V = np.linalg.eigh(H)
        pd = ev.min(axis=1) > 0
        evf = np.maximum(np.abs(ev), 1e-8 * np.maximum(1.0, np.abs(ev).max(axis=1, keepdims=True)))
        step = -np.einsum("nij,nj->ni", V, np.einsum("nji,nj->ni", V, g) / evf)
        dec = -np.einsum("ni,ni->n", g, step)
        done = (dec <= 1e-28 * (1.0 + np.abs(fx)) ** 2) | (gn <= tol) | frozen
        local = pd & (dec < 1e-10 * (1.0 + np.abs(fx))) & ~done
        active = ~done
        if not active.any():
            break
        sn = np.max(np.abs(step), axis=1, keepdims=True)
        step = step * np.minimum(1.0, 2.0 / np.maximum(sn, 1e-300))
        t = np.ones((n, 1))
        accepted = ~active | local
        xn = x.copy(); fn = fx.copy()
        if local.any():
            xn[local] = x[local] + step[local]
            fn[local] = f(xn)[local]
        for _ in range(80):
            if accepted.all():
                break
            cand = x + t * step
            fc = f(cand)
            ok = (fc <= fx - 1e-4 * t[:, 0] * dec) & ~accepted
            ok |= (fc < fx) & ~accepted & (t[:, 0] < 1e-8)
            xn[ok] = cand[ok]; fn[ok] = fc[ok]
            accepted |= ok
            t[~accepted] *= 0.5
        frozen |= ~accepted
        x = xn; fx = fn
    g, _ = gh(x)
    return x, fx, np.max(np.abs(g), axis=1)


# ── product of Exponentials, parameterised by ln means ───────────────────
# F = sum_i [ -ln m_i - ln lam_i + lam_i m_i - 1 ] + ln 2pi
#     + u1^2/2 - 2 u1 m1^2 m2 + 24 m1^4 m2^2 + u2^2/2 - 2 u2 m1 m2^2 + 24 m1^2 m2^4
def F_exp(x, u1, u2, r1, r2):
    a, b = x[:, 0], x[:, 1]
    return (-a - np.log(r1) + r1 * np.exp(a) - 1.0 - b - np.log(r2) + r2 * np.exp(b) - 1.0 + LN2PI
            + 0.5 * u1 * u1 - 2.0 * u1 * np.exp(2 * a + b) + 24.0 * np.exp(4 * a + 2 * b)
            + 0.5 * u2 * u2 - 2.0 * u2 * np.exp(a + 2 * b) + 24.0 * np.exp(2 * a + 4 * b))


def gh_exp(x, u1, u2, r1, r2):
    a, b = x[:, 0], x[:, 1]
    A = np.exp(a); B = np.exp(b)
    E1 = np.exp(2 * a + b); E2 = np.exp(4 * a + 2 * b)
    E3 = np.exp(a + 2 * b); E4 = np.exp(2 * a + 4 * b)
    g1 = -1.0 + r1 * A - 4 * u1 * E1 + 96 * E2 - 2 * u2 * E3 + 48 * E4
    g2 = -1.0 + r2 * B - 2 * u1 * E1 + 48 * E2 - 4 * u2 * E3 + 96 * E4
    h11 = r1 * A - 8 * u1 * E1 + 384 * E2 - 2 * u2 * E3 + 96 * E4
    h12 = -4 * u1 * E1 + 192 * E2 - 4 * u2 * E3 + 192 * E4
    h22 = r2 * B - 2 * u1 * E1 + 96 * E2 - 8 * u2 * E3 + 384 * E4
    g = np.stack([g1, g2], -1)
    H = np.stack([np.stack([h11, h12], -1), np.stack([h12, h22], -1)], -2)
    return g, H


EXP_TOL = 1e-10


def solve_exp(u1, u2, r1, r2, x0):
    return newton_min(lambda x: F_exp(x, u1, u2, r1, r2),
                      lambda x: gh_exp(x, u1, u2, r1, r2), x0, EXP_TOL)


# ── product of Gammas, parameterised by ln means and ln shapes ───────────
def digamma(x):
    x = np.array(x, dtype=float, copy=True)
    out = np.zeros_like(x)
    while np.any(x < 10.0):
        m = x < 10.0
        out = np.where(m, out - 1.0 / x, out)
        x = np.where(m, x + 1.0, x)
    i = 1.0 / x; i2 = i * i
    return out + np.log(x) - 0.5 * i - i2 * (1/12 - i2 * (1/120 - i2 * (1/252 - i2 * (1/240 - i2 / 132))))


def trigamma(x):
    x = np.array(x, dtype=float, copy=True)
    out = np.zeros_like(x)
    while np.any(x < 10.0):
        m = x < 10.0
        out = np.where(m, out + 1.0 / (x * x), out)
        x = np.where(m, x + 1.0, x)
    i = 1.0 / x; i2 = i * i
    return out + i + 0.5 * i2 + i * i2 * (1/6 - i2 * (1/30 - i2 * (1/42 - i2 * (1/30 - i2 * 5/66))))


_lgam = np.vectorize(lgamma)


def _gam_pieces(x):
    m1, m2, k1, k2 = np.exp(x[:, 0]), np.exp(x[:, 1]), np.exp(x[:, 2]), np.exp(x[:, 3])
    c21, c22 = 1 + 1 / k1, 1 + 1 / k2
    c41 = c21 * (1 + 2 / k1) * (1 + 3 / k1)
    c42 = c22 * (1 + 2 / k2) * (1 + 3 / k2)
    return m1, m2, k1, k2, c21, c22, c41, c42


def F_gam(x, u1, u2, r1, r2):
    """KL(q_i || Exp(lam_i)) = -ln Gamma(k) + (k-1) psi(k) + ln k - k - ln m - ln lam + lam m;
    E[s^2] = m^2 (1 + 1/k), E[s^4] = m^4 (1+1/k)(1+2/k)(1+3/k)."""
    m1, m2, k1, k2, c21, c22, c41, c42 = _gam_pieces(x)
    kl1 = -_lgam(k1) + (k1 - 1) * digamma(k1) + np.log(k1) - k1 - x[:, 0] - np.log(r1) + r1 * m1
    kl2 = -_lgam(k2) + (k2 - 1) * digamma(k2) + np.log(k2) - k2 - x[:, 1] - np.log(r2) + r2 * m2
    e1 = u1 * u1 - 2 * u1 * m2 * m1 ** 2 * c21 + m2 ** 2 * c22 * m1 ** 4 * c41
    e2 = u2 * u2 - 2 * u2 * m1 * m2 ** 2 * c22 + m1 ** 2 * c21 * m2 ** 4 * c42
    return kl1 + kl2 + LN2PI + 0.5 * e1 + 0.5 * e2


def grad_gam(x, u1, u2, r1, r2):
    m1, m2, k1, k2, c21, c22, c41, c42 = _gam_pieces(x)
    dc21, dc22 = -1 / k1, -1 / k2                       # d c2 / d ln k
    dc41 = -c41 * (1 / (k1 + 1) + 2 / (k1 + 2) + 3 / (k1 + 3))   # d c4 / d ln k
    dc42 = -c42 * (1 / (k2 + 1) + 2 / (k2 + 2) + 3 / (k2 + 3))
    T1 = u1 * m2 * m1 ** 2 * c21
    Q1 = m2 ** 2 * c22 * m1 ** 4 * c41
    T2 = u2 * m1 * m2 ** 2 * c22
    Q2 = m1 ** 2 * c21 * m2 ** 4 * c42
    g_a1 = -1 + r1 * m1 + (-2 * T1 + 2 * Q1) + (-T2 + Q2)
    g_a2 = -1 + r2 * m2 + (-T1 + Q1) + (-2 * T2 + 2 * Q2)
    dkl1 = k1 * ((k1 - 1) * trigamma(k1) + 1 / k1 - 1)
    dkl2 = k2 * ((k2 - 1) * trigamma(k2) + 1 / k2 - 1)
    g_k1 = dkl1 - T1 * dc21 / c21 + 0.5 * Q1 * dc41 / c41 + 0.5 * Q2 * dc21 / c21
    g_k2 = dkl2 + 0.5 * Q1 * dc22 / c22 - T2 * dc22 / c22 + 0.5 * Q2 * dc42 / c42
    return np.stack([g_a1, g_a2, g_k1, g_k2], -1)


def _gh_fd(grad, h=1e-6):
    def gh(x):
        g0 = grad(x)
        n, p = x.shape
        H = np.empty((n, p, p))
        for j in range(p):
            e = np.zeros(p); e[j] = h
            H[:, :, j] = (grad(x + e) - grad(x - e)) / (2 * h)
        return g0, 0.5 * (H + np.transpose(H, (0, 2, 1)))
    return gh


GAM_TOL = 1e-9


def solve_gam(u1, u2, r1, r2, x0):
    return newton_min(lambda x: F_gam(x, u1, u2, r1, r2),
                      _gh_fd(lambda x: grad_gam(x, u1, u2, r1, r2)), x0, GAM_TOL)


# ── free-form mean-field ─────────────────────────────────────────────────
# ln q1(s1) = -E[s2^2]/2 s1^4 + (u1 E[s2] - E[s2^4]/2) s1^2 + (u2 E[s2^2] - lam1) s1 + const,
# and the same with the indices exchanged. Each factor is a density on s > 0 of the form
# exp(a s^4 + b s^2 + c s) with a < 0, integrated by 1-D quadrature.
FF_QUAD = dict(npan=32, ncoarse=256, window=60.0)


def quartic_moments(a, b, c, npan=32, ncoarse=256, window=60.0):
    """Rows of exp(a s^4 + b s^2 + c s) on s > 0, a < 0. A coarse scan finds where the
    exponent is within `window` nats of its maximum; composite 8-point Gauss-Legendre
    on that interval. Returns ln Z, E[s], E[s^2], E[s^4], and the interval."""
    n = a.shape[0]
    shi = 2.0 * np.maximum.reduce([np.ones(n), np.sqrt(np.abs(b / a)), np.cbrt(np.abs(c / a))]) + 1.0
    P = lambda s: a[:, None] * s ** 4 + b[:, None] * s ** 2 + c[:, None] * s
    t = np.linspace(0.0, 1.0, ncoarse)[None, :]
    Pc = P(shi[:, None] * t)
    keep = Pc > Pc.max(axis=1, keepdims=True) - window
    idx = np.arange(ncoarse)[None, :]
    lo = shi * np.maximum(np.where(keep, idx, ncoarse).min(axis=1) - 1, 0) / (ncoarse - 1)
    hi = shi * np.minimum(np.where(keep, idx, -1).max(axis=1) + 1, ncoarse - 1) / (ncoarse - 1)
    xs, ws = gl_panels(np.linspace(0, 1, npan + 1))
    s = lo[:, None] + (hi - lo)[:, None] * xs[None, :]
    w = (hi - lo)[:, None] * ws[None, :]
    Ps = P(s)
    m = Ps.max(axis=1, keepdims=True)
    e = np.exp(Ps - m) * w
    Z = e.sum(axis=1)
    p = e / Z[:, None]
    s2 = s * s
    return m[:, 0] + np.log(Z), (p * s).sum(1), (p * s2).sum(1), (p * s2 * s2).sum(1)


def ff_coef1(u1, u2, r1, mom2):
    e1, e2, e4 = mom2
    return -0.5 * e2, u1 * e1 - 0.5 * e4, u2 * e2 - r1


def ff_coef2(u1, u2, r2, mom1):
    e1, e2, e4 = mom1
    return -0.5 * e2, u2 * e1 - 0.5 * e4, u1 * e2 - r2


def F_ff(u1, u2, r1, r2, mom1, mom2, co1, lnZ1, co2, lnZ2):
    a1, a2, a4 = mom1
    b1, b2, b4 = mom2
    elnq1 = co1[0] * a4 + co1[1] * a2 + co1[2] * a1 - lnZ1
    elnq2 = co2[0] * b4 + co2[1] * b2 + co2[2] * b1 - lnZ2
    elnp = (np.log(r1) + np.log(r2) - r1 * a1 - r2 * b1 - LN2PI
            - 0.5 * (u1 * u1 - 2 * u1 * b1 * a2 + b2 * a4)
            - 0.5 * (u2 * u2 - 2 * u2 * b2 * a1 + b4 * a2))
    return elnq1 + elnq2 - elnp


FF_TOL = 1e-12


def cavi(u1, u2, r1, r2, mom2, tol=FF_TOL, maxsweep=20000, quad=None):
    """Sequential coordinate ascent, q1 then q2, starting from q2's moments. A trial stops
    when one sweep changes E[s1] and E[s2] by a relative amount below tol. Free energy
    never rises from sweep to sweep (asserted). Returns the moments of q1 and q2, F,
    and sweeps per trial."""
    kw = dict(FF_QUAD, **(quad or {}))
    n = u1.shape[0]
    mom2 = tuple(np.array(m, float, copy=True) for m in mom2)
    mom1 = tuple(np.zeros(n) for _ in range(3))
    F = np.full(n, np.inf)
    act = np.arange(n)
    sweeps = np.zeros(n, int)
    for _ in range(maxsweep):
        ua, va = u1[act], u2[act]
        co1 = ff_coef1(ua, va, r1, tuple(m[act] for m in mom2))
        lnZ1, *m1n = quartic_moments(*co1, **kw)
        co2 = ff_coef2(ua, va, r2, m1n)
        lnZ2, *m2n = quartic_moments(*co2, **kw)
        Fn = F_ff(ua, va, r1, r2, m1n, m2n, co1, lnZ1, co2, lnZ2)
        Fo = F[act]
        dm = np.maximum(np.abs(m1n[0] - mom1[0][act]) / m1n[0],
                        np.abs(m2n[0] - mom2[0][act]) / m2n[0])
        assert np.all(Fn <= Fo + 1e-9 * (1 + np.abs(Fo))), np.max(Fn - Fo)
        for M, Mn in ((mom1, m1n), (mom2, m2n)):
            for j in range(3):
                M[j][act] = Mn[j]
        F[act] = Fn
        sweeps[act] += 1
        act = act[~(dm < tol)]
        if act.size == 0:
            break
    assert act.size == 0, "coordinate ascent did not meet its tolerance"
    return mom1, mom2, F, sweeps


# ── families for variational EM ──────────────────────────────────────────
class Exact:
    name = "exact"

    def __init__(self, u1, u2, quad=None):
        self.grids = [TrialGrid(a, b, **(quad or {})) for a, b in zip(u1, u2)]

    def init(self, lam):
        return None

    def estep(self, lam, st):
        ex = exact_batch(self.grids, lam[0], lam[1], full=False)
        return None, ex["nll"], ex["m1"], ex["m2"]


EXP_STARTS = [(a, b) for a in (-2.0, 0.0, 1.5) for b in (-2.0, 0.0, 1.5)]
GAM_STARTS = [(a, k) for a in (-1.0, 0.0, 1.0) for k in (0.0, 2.0, 5.0)]


class ExpMF:
    name = "exp"

    def __init__(self, u1, u2):
        self.u1, self.u2 = u1, u2

    def multistart(self, lam, starts=EXP_STARTS):
        n = len(self.u1)
        best = None
        for a0, b0 in starts:
            x, f, _ = solve_exp(self.u1, self.u2, lam[0], lam[1], np.tile([a0, b0], (n, 1)))
            if best is None:
                best = [x, f]
            else:
                m = f < best[1]
                best[0][m] = x[m]; best[1][m] = f[m]
        return best[0]

    init = multistart

    def F(self, lam, x):
        return F_exp(x, self.u1, self.u2, lam[0], lam[1])

    def estep(self, lam, x0):
        x, f, g = solve_exp(self.u1, self.u2, lam[0], lam[1], x0)
        f0 = self.F(lam, x0)
        bad = f > f0              # keep the previous Q where the new one is not lower
        x[bad] = x0[bad]; f[bad] = f0[bad]
        self.gmax = float(g.max())
        return x, f, np.exp(x[:, 0]), np.exp(x[:, 1])


class GamMF:
    name = "gam"

    def __init__(self, u1, u2):
        self.u1, self.u2 = u1, u2

    def multistart(self, lam, starts=GAM_STARTS):
        n = len(self.u1)
        best = None
        for a0, k0 in starts:
            x, f, _ = solve_gam(self.u1, self.u2, lam[0], lam[1], np.tile([a0, a0, k0, k0], (n, 1)))
            if best is None:
                best = [x, f]
            else:
                m = f < best[1]
                best[0][m] = x[m]; best[1][m] = f[m]
        return best[0]

    init = multistart

    def F(self, lam, x):
        return F_gam(x, self.u1, self.u2, lam[0], lam[1])

    def estep(self, lam, x0):
        x, f, g = solve_gam(self.u1, self.u2, lam[0], lam[1], x0)
        f0 = self.F(lam, x0)
        bad = f > f0
        x[bad] = x0[bad]; f[bad] = f0[bad]
        self.gmax = float(g.max())
        return x, f, np.exp(x[:, 0]), np.exp(x[:, 1])


class FreeMF:
    name = "free"

    def __init__(self, u1, u2, quad=None):
        self.u1, self.u2, self.quad = u1, u2, quad

    def prior_moments(self, lam):
        n = len(self.u1); r = lam[1]
        return (np.full(n, 1 / r), np.full(n, 2 / r ** 2), np.full(n, 24 / r ** 4))

    init = prior_moments

    def estep(self, lam, mom2):
        m1, m2, F, sw = cavi(self.u1, self.u2, lam[0], lam[1], mom2, quad=self.quad)
        self.mom1, self.sweeps = m1, sw
        return m2, F, m1[0], m2[0]


def prior_shift(F, lam_old, lam_new, m1, m2):
    """Per-trial F after the prior rates change with Q held fixed: only the term
    -ln lam_i + lam_i E_q[s_i] of each KL depends on lam."""
    return (F - np.log(lam_new[0]) + lam_new[0] * m1 + np.log(lam_old[0]) - lam_old[0] * m1
              - np.log(lam_new[1]) + lam_new[1] * m2 + np.log(lam_old[1]) - lam_old[1] * m2)


EM_TOL = 1e-11


def em(fam, lam0=LAM_INIT, rule="HM", tol=EM_TOL, maxit=3000, state=None):
    """Variational EM from lam0. rule 'HM' is the M-step lam_i = 1/mean E_q[s_i], which
    minimises total F over lam; 'AM' sets lam_i = mean of 1/E_q[s_i], which is not an
    M-step of F. Every E-step is warm-started from the previous Q and checked not to
    raise total F; every HM M-step is checked likewise. Stops when lam changes by a
    relative amount below tol."""
    lam = np.array(lam0, float)
    st = fam.init(lam) if state is None else state
    st, F, m1, m2 = fam.estep(lam, st)
    Ftot = [F.sum()]; lams = [lam.copy()]; Fm_list = []
    rises = 0
    for it in range(maxit):
        if rule == "HM":
            new = np.array([1 / m1.mean(), 1 / m2.mean()])
        else:
            new = np.array([(1 / m1).mean(), (1 / m2).mean()])
        Fm = prior_shift(F, lam, new, m1, m2).sum()
        if rule == "HM":
            assert Fm <= Ftot[-1] + 1e-10 * abs(Ftot[-1]), ("M-step raised F", Fm - Ftot[-1])
        Fm_list.append(Fm)
        lam_old, lam = lam, new
        st, F, m1, m2 = fam.estep(lam, st)
        assert F.sum() <= Fm + 1e-10 * abs(Fm), ("E-step raised F", F.sum() - Fm)
        rises += F.sum() > Ftot[-1] + 1e-10 * abs(Ftot[-1])
        Ftot.append(F.sum()); lams.append(lam.copy())
        if np.max(np.abs(lam - lam_old) / lam) < tol:
            break
    else:
        raise RuntimeError("EM did not converge")
    return dict(lam=lam, F=F, m1=m1, m2=m2, state=st, its=it + 1, rises=int(rises),
                Ftot=np.array(Ftot), Fm=np.array(Fm_list), lams=np.array(lams))


def m_map(fam, lam, st):
    st2, _, m1, m2 = fam.estep(np.asarray(lam, float), st)
    return np.array([1 / m1.mean(), 1 / m2.mean()]), st2


def hess_gam(x, u1, u2, r1, r2, h=1e-5):
    """Hessian of F_gam: central differences of the analytic gradient, symmetrised,
    with no eigenvalue repair."""
    p = x.shape[1]
    H = np.empty((x.shape[0], p, p))
    for j in range(p):
        e = np.zeros(p); e[j] = h
        H[:, :, j] = (grad_gam(x + e, u1, u2, r1, r2) - grad_gam(x - e, u1, u2, r1, r2)) / (2 * h)
    return 0.5 * (H + np.transpose(H, (0, 2, 1)))


def _fam_gh(fam, lam, x):
    if fam.name == "exp":
        return gh_exp(x, fam.u1, fam.u2, lam[0], lam[1])
    return (grad_gam(x, fam.u1, fam.u2, lam[0], lam[1]),
            hess_gam(x, fam.u1, fam.u2, lam[0], lam[1]))


def jacobian_implicit(fam, lam, x, polish=8):
    """The M-map Jacobian of a parametric family from [eq:tc-jac-param]:
    J_ij = sum_n h_ni^T H_n^{-1} h_nj / (N mbar_i^2), with H_n the Hessian of F_n at the
    E-step solution (no eigenvalue repair) and h_ni = grad_x E_q[s_i] = m_i e_i in the
    ln-mean coordinates. Each solution is first polished by plain Newton steps, a step
    kept only where it does not raise F. Returns J, the smallest Hessian eigenvalue over
    trials and the largest gradient component after polishing."""
    lam = np.asarray(lam, float)
    x = x.copy()
    for _ in range(polish):
        g, H = _fam_gh(fam, lam, x)
        cand = x - np.linalg.solve(H, g[..., None])[..., 0]
        ok = fam.F(lam, cand) <= fam.F(lam, x)
        x[ok] = cand[ok]
    g, H = _fam_gh(fam, lam, x)
    mineig = float(np.linalg.eigvalsh(H).min())
    assert mineig > 0, ("Hessian not positive definite at an E-step solution", mineig)
    m = np.exp(x[:, :2])
    hv = np.zeros((len(x), x.shape[1], 2))
    hv[:, 0, 0] = m[:, 0]; hv[:, 1, 1] = m[:, 1]
    J = np.einsum("nki,nkj->ij", hv, np.linalg.solve(H, hv)) / len(x)
    J /= m.mean(axis=0)[:, None] ** 2
    return J, mineig, float(np.abs(g).max())


JAC_FD_STEPS = (1e-3, 3e-3, 1e-2)


def jacobian(fam, lam, st, h=1e-5):
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2); e[j] = h * lam[j]
        Mp, _ = m_map(fam, lam + e, st)
        Mm, _ = m_map(fam, lam - e, st)
        J[:, j] = (Mp - Mm) / (2 * e[j])
    return J


def ema_online(u1, u2, mu0=LAM_INIT, keep=0.99):
    """The notebook's loop: one pass over the trials; on each, the Exponential q at the
    current prior rates (best of 25 starts) and mu <- keep mu + (1 - keep) (1/m1, 1/m2)."""
    mu = np.array(mu0, float)
    starts = np.array([[a, b] for a in (-3, -1.5, 0, 1.5, 3) for b in (-3, -1.5, 0, 1.5, 3)], float)
    traj = [mu.copy()]
    for n in range(len(u1)):
        k = len(starts)
        x, f, _ = solve_exp(np.full(k, u1[n]), np.full(k, u2[n]), mu[0], mu[1], starts.copy())
        j = int(np.argmin(f))
        mu = keep * mu + (1 - keep) * np.exp(-x[j])
        traj.append(mu.copy())
    return np.array(traj)


def grad_diag(fam, lam, x, tol):
    """Largest |gradient| over trials at x, the number of trials above the stopping
    tolerance, and the largest Newton decrement g' H^-1 g among those trials (the
    decrease a further Newton step predicts)."""
    u1, u2 = fam.u1, fam.u2
    if fam.name == "exp":
        g, H = gh_exp(x, u1, u2, lam[0], lam[1])
    else:
        g, H = _gh_fd(lambda z: grad_gam(z, u1, u2, lam[0], lam[1]))(x)
    gn = np.max(np.abs(g), axis=1)
    over = gn > tol
    dec = 0.0
    if over.any():
        dec = float(np.max(np.einsum("ni,ni->n", g[over], np.linalg.solve(H[over], g[over][..., None])[..., 0])))
    return dict(gmax=float(gn.max()), over=int(over.sum()), dec=dec)


# ── densities on a plane grid, for the figure and the widget check ───────
def gamma_logpdf(s, m, k):
    return k * np.log(k / m) - lgamma(k) + (k - 1) * np.log(s) - k * s / m


def hpd_levels(P, masses):
    """Density thresholds whose super-level sets hold the given probability masses,
    for a density tabulated on a uniform grid (P normalised to sum to one)."""
    flat = np.sort(P.ravel())[::-1]
    cum = np.cumsum(flat)
    return [float(flat[np.searchsorted(cum, m)]) for m in masses]


def marching_squares(X, Y, Z, level):
    """Contour segments of Z = level on a rectilinear grid, chained into polylines."""
    segs = []
    above = Z > level
    cross = ((above[:-1, :-1] != above[1:, :-1]) | (above[:-1, :-1] != above[:-1, 1:])
             | (above[:-1, :-1] != above[1:, 1:]))
    for i, j in zip(*np.nonzero(cross)):
            c = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            v = [Z[a, b] - level for a, b in c]
            pts = []
            for k in range(4):
                a, b = c[k], c[(k + 1) % 4]
                va, vb = v[k], v[(k + 1) % 4]
                if (va > 0) != (vb > 0):
                    t = va / (va - vb)
                    pts.append((X[a[0]] + t * (X[b[0]] - X[a[0]]), Y[a[1]] + t * (Y[b[1]] - Y[a[1]])))
            if len(pts) == 2:
                segs.append((pts[0], pts[1]))
            elif len(pts) == 4:
                segs.append((pts[0], pts[1])); segs.append((pts[2], pts[3]))
    # chain
    key = lambda p: (round(p[0], 9), round(p[1], 9))
    ends = {}
    for idx, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append((idx, 0))
        ends.setdefault(key(b), []).append((idx, 1))
    used = [False] * len(segs)
    lines = []
    for idx in range(len(segs)):
        if used[idx]:
            continue
        used[idx] = True
        line = [segs[idx][0], segs[idx][1]]
        for direction in (1, 0):
            while True:
                tip = line[-1] if direction else line[0]
                nxt = None
                for j, side in ends.get(key(tip), []):
                    if not used[j]:
                        nxt = (j, side)
                        break
                if nxt is None:
                    break
                j, side = nxt
                used[j] = True
                other = segs[j][1 - side]
                if direction:
                    line.append(other)
                else:
                    line.insert(0, other)
        lines.append(line)
    return lines


def plane_densities(u1, u2, r1, r2, ex_q, gam_x, exp_x, ff_coefs, s1max, s2max, n=241):
    """Exact posterior and the three mean-field products on a uniform (s1, s2) grid."""
    g1 = np.linspace(s1max / (2 * n), s1max, n)
    g2 = np.linspace(s2max / (2 * n), s2max, n)
    S1, S2 = np.meshgrid(g1, g2, indexing="ij")
    out = {}
    lp = -r1 * S1 - r2 * S2 - 0.5 * (u1 - S2 * S1 ** 2) ** 2 - 0.5 * (u2 - S1 * S2 ** 2) ** 2
    out["exact"] = lp
    m1, m2 = np.exp(exp_x[0]), np.exp(exp_x[1])
    out["exp"] = -S1 / m1 - S2 / m2
    gm1, gm2, gk1, gk2 = np.exp(gam_x)
    out["gam"] = gamma_logpdf(S1, gm1, gk1) + gamma_logpdf(S2, gm2, gk2)
    (a1, b1, c1), (a2, b2, c2) = ff_coefs
    out["free"] = (a1 * S1 ** 4 + b1 * S1 ** 2 + c1 * S1) + (a2 * S2 ** 4 + b2 * S2 ** 2 + c2 * S2)
    dens = {}
    for k, v in out.items():
        p = np.exp(v - v.max())
        dens[k] = p / p.sum()
    return g1, g2, dens


# ── everything Lesson 4.5 quotes ─────────────────────────────────────────
def compute():
    R = {}
    s1, s2, u1, u2 = sample()
    lt = np.array(LAM_TRUE)
    R["u1"], R["u2"] = u1, u2
    R["s_mean"] = (float(s1.mean()), float(s2.mean()))

    # exact posterior at the true rates, every trial
    exf = Exact(u1, u2)
    ex = exact_batch(exf.grids, *lt)
    R["ex_true"] = ex
    # the M-map Jacobian of exact EM at the true rates, from posterior covariances:
    # d Lambda_i / d lambda_j = sum_n Cov(s_i, s_j | u_n) / (N mbar_i^2)
    cov = np.array([[ex["v1"], ex["corr"] * ex["sd1"] * ex["sd2"]],
                    [ex["corr"] * ex["sd1"] * ex["sd2"], ex["v2"]]]).mean(axis=2)
    mbar = np.array([ex["m1"].mean(), ex["m2"].mean()])
    R["jac_exact_cov"] = cov / mbar[:, None] ** 2

    # convergence of the reference integration: spacing, limits, grading, order, each alone
    conv = {}
    for k, q in QUAD_VARIANTS.items():
        gq = [TrialGrid(a, b, **q) for a, b in zip(u1, u2)]
        e2 = exact_batch(gq, *lt)
        conv[k] = {f: float(np.max(np.abs(e2[f] - ex[f]) / (np.abs(ex[f]) if f != "nll" else 1.0)))
                   for f in ("nll", "m1", "m2", "sd1", "sd2", "corr")}
        conv[k]["rate1"] = float(abs(1 / e2["m1"].mean() - 1 / ex["m1"].mean()))
        conv[k]["rate2"] = float(abs(1 / e2["m2"].mean() - 1 / ex["m2"].mean()))
        conv[k]["nodes"] = int(np.mean([g.s1.size for g in gq]))
    conv["base_nodes"] = int(np.mean([g.s1.size for g in exf.grids]))
    R["quad_conv"] = conv

    # one M-step from the true rates for each family
    fams = {"exact": exf, "free": FreeMF(u1, u2), "gam": GamMF(u1, u2), "exp": ExpMF(u1, u2)}
    pin, pin_state, pin_F = {}, {}, {}
    for k, fam in fams.items():
        st0 = fam.init(lt)
        st, F, m1, m2 = fam.estep(lt, st0)
        pin[k] = np.array([1 / m1.mean(), 1 / m2.mean()])
        pin_state[k], pin_F[k] = st, F
        R[f"pin_m_{k}"] = (m1, m2)
    R["pin"] = pin
    R["pin_F"] = pin_F
    R["pin_am_exp"] = np.array([(1 / R["pin_m_exp"][0]).mean(), (1 / R["pin_m_exp"][1]).mean()])
    R["ff_sweeps_true"] = fams["free"].sweeps
    R["ff_mom1_true"] = fams["free"].mom1
    R["pin_state"] = pin_state

    # free-form checks at the true rates: second start, finer quadrature
    ffm1, ffm2 = R["pin_m_free"]
    alt = (np.full(len(u1), 5.0), np.full(len(u1), 25.0), np.full(len(u1), 625.0))
    a1, a2, aF, _ = cavi(u1, u2, *lt, alt)
    R["ff_start_dm"] = float(np.max(np.abs(a1[0] - ffm1) / ffm1))
    R["ff_start_dF"] = float(np.max(np.abs(aF - pin_F["free"])))
    q1, q2, qF, _ = cavi(u1, u2, *lt, fams["free"].prior_moments(lt), quad=dict(npan=64, ncoarse=512, window=80.0))
    R["ff_quad_dm"] = float(np.max(np.abs(q1[0] - ffm1) / ffm1))
    R["ff_quad_dF"] = float(np.max(np.abs(qF - pin_F["free"])))

    # gradient checks for the closed-form families at an arbitrary point
    rng = np.random.default_rng(1)
    xg = rng.normal(size=(20, 4)) * 0.5
    ug1, ug2 = np.abs(rng.normal(size=20)) * 3, np.abs(rng.normal(size=20)) * 3
    ga = grad_gam(xg, ug1, ug2, *lt)
    gf = np.zeros_like(ga)
    for j in range(4):
        e = np.zeros(4); e[j] = 1e-6
        gf[:, j] = (F_gam(xg + e, ug1, ug2, *lt) - F_gam(xg - e, ug1, ug2, *lt)) / 2e-6
    R["gam_grad_relerr"] = float(np.max(np.abs(ga - gf) / (1 + np.abs(gf))))
    ge, He = gh_exp(xg[:, :2], ug1, ug2, *lt)
    gfe = np.zeros_like(ge); Hfe = np.zeros_like(He)
    for j in range(2):
        e = np.zeros(2); e[j] = 1e-6
        gfe[:, j] = (F_exp(xg[:, :2] + e, ug1, ug2, *lt) - F_exp(xg[:, :2] - e, ug1, ug2, *lt)) / 2e-6
        Hfe[:, :, j] = (gh_exp(xg[:, :2] + e, ug1, ug2, *lt)[0] - gh_exp(xg[:, :2] - e, ug1, ug2, *lt)[0]) / 2e-6
    R["exp_grad_relerr"] = float(np.max(np.abs(ge - gfe) / (1 + np.abs(gfe))))
    R["exp_hess_relerr"] = float(np.max(np.abs(He - Hfe) / (1 + np.abs(Hfe))))
    x1 = np.concatenate([xg[:, :2], np.zeros((20, 2))], axis=1)
    R["gam_k1_equals_exp"] = float(np.max(np.abs(F_gam(x1, ug1, ug2, *lt) - F_exp(xg[:, :2], ug1, ug2, *lt))))

    # closed-loop variational EM for every family from the notebook's start
    runs = {}
    for k, fam in fams.items():
        runs[k] = em(fam)
    runs["exp_am"] = em(fams["exp"], rule="AM")
    R["runs"] = runs
    R["ema"] = ema_online(u1, u2)

    # warm-started optima against multi-start optima at the fitted rates
    lf = runs["exp"]["lam"]
    fw = fams["exp"].F(lf, runs["exp"]["state"])
    fb = fams["exp"].F(lf, fams["exp"].multistart(lf))
    R["exp_fit_multistart"] = float(np.max(fw - fb))
    lf = runs["gam"]["lam"]
    fw = fams["gam"].F(lf, runs["gam"]["state"])
    fb = fams["gam"].F(lf, fams["gam"].multistart(lf))
    R["gam_fit_multistart"] = float(np.max(fw - fb))
    lf = runs["free"]["lam"]
    b1, b2, bF, _ = cavi(u1, u2, *lf, alt)
    R["ff_fit_multistart"] = float(np.max(np.abs(bF - runs["free"]["F"])))

    # stationarity of the closed-form E-steps at the true rates and at the fitted rates
    R["grad_diag"] = {}
    for k, tol in (("exp", EXP_TOL), ("gam", GAM_TOL)):
        R["grad_diag"][k] = dict(pin=grad_diag(fams[k], lt, pin_state[k], tol),
                                 fit=grad_diag(fams[k], runs[k]["lam"], runs[k]["state"], tol))
    R["gam_kmax"] = float(np.exp(pin_state["gam"][:, 2:]).max())

    # score of the exact log likelihood at the exact-EM fixed point (zero at the ML estimate)
    lhat = runs["exact"]["lam"]
    eml = exact_batch(exf.grids, *lhat)
    R["ml_score"] = np.array([len(u1) / lhat[0] - eml["m1"].sum(), len(u1) / lhat[1] - eml["m2"].sum()])
    R["ex_ml"] = eml
    # the ML fixed point under refined integration
    gq = [TrialGrid(a, b, hy=0.75, L=14.0, ngrade=24, ratio=0.2, order=12) for a, b in zip(u1, u2)]
    e3 = exact_batch(gq, *lhat, full=False)
    R["ml_refined_residual"] = np.array([1 / e3["m1"].mean(), 1 / e3["m2"].mean()]) - lhat

    # feedback: Jacobians of the M-map at the truth and at the fitted rates
    # exact and free-form: central differences of the map with steps of 1e-5 of each rate;
    # Exponential and Gamma: the implicit-function formula with the true Hessian, checked
    # against central differences of the map with much larger steps
    jac = {}
    R["jac_mineig"] = {}; R["jac_gres"] = {}; R["jac_fd_dev"] = {}
    for k, fam in fams.items():
        if k in ("exp", "gam"):
            jac[k] = {}
            for w, lam_w, x_w in (("true", lt, pin_state[k]), ("fit", runs[k]["lam"], runs[k]["state"])):
                J, me, gr = jacobian_implicit(fam, lam_w, x_w)
                jac[k][w] = J
                R["jac_mineig"][f"{k}_{w}"] = me
                R["jac_gres"][f"{k}_{w}"] = gr
                R["jac_fd_dev"][f"{k}_{w}"] = {str(hh): float(np.abs(jacobian(fam, lam_w, x_w, h=hh) - J).max())
                                               for hh in JAC_FD_STEPS}
        else:
            jac[k] = dict(true=jacobian(fam, lt, pin_state[k]),
                          fit=jacobian(fam, runs[k]["lam"], runs[k]["state"]))
    R["jac"] = jac

    # fitted-Q summaries: mean coefficient of variation of q(s1) after the final E-step
    R["cv1_fit"] = {"exact": float(np.mean(eml["sd1"] / eml["m1"]))}
    st = runs["free"]["state"]
    m1f, _, _, _ = cavi(u1, u2, *runs["free"]["lam"], st)
    R["cv1_fit"]["free"] = float(np.mean(np.sqrt(m1f[1] - m1f[0] ** 2) / m1f[0]))
    R["cv1_fit"]["gam"] = float(np.mean(np.exp(-0.5 * runs["gam"]["state"][:, 2])))
    R["cv1_fit"]["exp"] = 1.0

    # pinned summaries: CV of q(s1) and free-energy gaps per trial at the true rates
    m1t = R["ff_mom1_true"]
    R["cv1_pin"] = {"exact": float(np.mean(ex["sd1"] / ex["m1"])),
                    "free": float(np.mean(np.sqrt(m1t[1] - m1t[0] ** 2) / m1t[0])),
                    "gam": float(np.mean(np.exp(-0.5 * pin_state["gam"][:, 2]))),
                    "exp": 1.0}
    R["gap_pin"] = {k: float(np.mean(pin_F[k] - ex["nll"])) for k in ("free", "gam", "exp")}
    R["gap_pin_min"] = {k: float(np.min(pin_F[k] - ex["nll"])) for k in ("free", "gam", "exp")}
    _ge = np.sort(pin_F["exp"] - ex["nll"])[::-1]
    R["gap_exp_dist"] = dict(median=float(np.median(_ge)), top1=float(_ge[0] / _ge.sum()),
                             top10=float(_ge[:10].sum() / _ge.sum()),
                             imax=int(np.argmax(pin_F["exp"] - ex["nll"])))
    R["mean_ratio"] = {k: (float(np.mean(R[f"pin_m_{k}"][0] / ex["m1"])),
                           float(np.mean(R[f"pin_m_{k}"][1] / ex["m2"])))
                       for k in ("free", "gam", "exp")}
    R["mean_lower_frac"] = {k: (float(np.mean(R[f"pin_m_{k}"][0] < ex["m1"])),
                                float(np.mean(R[f"pin_m_{k}"][1] < ex["m2"])))
                            for k in ("free", "gam", "exp")}
    R["sd_ratio_free"] = (float(np.mean(np.sqrt(m1t[1] - m1t[0] ** 2) / ex["sd1"])),)

    # the representative trial
    ru = np.array([REP_U[0]]), np.array([REP_U[1]])
    g = TrialGrid(*REP_U)
    R["rep_ex"] = g.post(*lt)
    fe = ExpMF(*ru); xe = fe.multistart(lt)
    R["rep_exp_x"] = xe[0]; R["rep_exp_F"] = float(fe.F(lt, xe)[0])
    fg = GamMF(*ru); xg_ = fg.multistart(lt)
    R["rep_gam_x"] = xg_[0]; R["rep_gam_F"] = float(fg.F(lt, xg_)[0])
    ff = FreeMF(*ru)
    mA, mB, Ff, _ = cavi(*ru, *lt, ff.prior_moments(lt))
    R["rep_ff_mom"] = (tuple(float(m[0]) for m in mA), tuple(float(m[0]) for m in mB))
    R["rep_ff_F"] = float(Ff[0])
    co1 = ff_coef1(*ru, lt[0], mB)
    co2 = ff_coef2(*ru, lt[1], mA)
    R["rep_ff_coef"] = (tuple(float(c[0]) for c in co1), tuple(float(c[0]) for c in co2))
    return R


# ── what values.py keeps ─────────────────────────────────────────────────
def _thin(a, k):
    a = np.asarray(a)
    idx = list(range(0, len(a), k))
    if idx[-1] != len(a) - 1:
        idx.append(len(a) - 1)
    return a[idx].round(6).tolist()


def summary():
    """compute(), reduced to the scalars, small arrays and contour polylines the
    lesson, its figures and the parity checks use. JSON-serialisable."""
    R = compute()
    S = {}
    lt = np.array(LAM_TRUE)
    ex = R["ex_true"]
    S["n"] = N_TRIALS
    S["lam_true"] = list(LAM_TRUE)
    S["lam_init"] = list(LAM_INIT)
    S["s_mean"] = list(R["s_mean"])
    S["u_max"] = [float(R["u1"].max()), float(R["u2"].max())]
    S["quad"] = R["quad_conv"]
    S["quad_default"] = dict(DEFAULT_QUAD)
    S["quad_variants"] = QUAD_VARIANTS
    S["exp_starts"] = [list(x) for x in EXP_STARTS]
    S["gam_starts"] = [list(x) for x in GAM_STARTS]
    S["ex_true"] = dict(cv1=float(np.mean(ex["sd1"] / ex["m1"])), corr=float(np.mean(ex["corr"])),
                        corr_gt05=float(np.mean(np.abs(ex["corr"]) > 0.5)),
                        corr_min=float(ex["corr"].min()), corr_max=float(ex["corr"].max()))
    S["pin"] = {k: v.tolist() for k, v in R["pin"].items()}
    S["pin_am_exp"] = R["pin_am_exp"].tolist()
    S["fit"] = {k: r["lam"].tolist() for k, r in R["runs"].items()}
    S["its"] = {k: r["its"] for k, r in R["runs"].items()}
    S["rises"] = {k: r["rises"] for k, r in R["runs"].items()}
    S["Ftot_first_last"] = {k: [float(r["Ftot"][0]), float(r["Ftot"][-1])] for k, r in R["runs"].items()}
    S["traj"] = {k: _thin(r["lams"], 1) for k, r in R["runs"].items()}
    S["ema_final"] = R["ema"][-1].tolist()
    S["ema_last100"] = [R["ema"][-100:].min(0).tolist(), R["ema"][-100:].max(0).tolist()]
    S["ema_traj"] = _thin(R["ema"], 5)
    S["jac"] = {k: {w: J.tolist() for w, J in v.items()} for k, v in R["jac"].items()}
    S["jac_mineig"] = R["jac_mineig"]
    S["jac_gres"] = R["jac_gres"]
    S["jac_fd_dev"] = R["jac_fd_dev"]
    S["jac_fd_steps"] = list(JAC_FD_STEPS)
    S["cv1_fit"] = R["cv1_fit"]
    S["cv1_pin"] = R["cv1_pin"]
    S["gap_pin"] = R["gap_pin"]
    S["gap_pin_min"] = R["gap_pin_min"]
    S["gap_exp_dist"] = R["gap_exp_dist"]
    S["mean_ratio"] = R["mean_ratio"]
    S["mean_lower_frac"] = R["mean_lower_frac"]
    S["ml_score"] = R["ml_score"].tolist()
    S["ml_refined_residual"] = R["ml_refined_residual"].tolist()
    S["ml_cv1"] = R["cv1_fit"]["exact"]
    for k in ("ff_start_dm", "ff_start_dF", "ff_quad_dm", "ff_quad_dF", "gam_grad_relerr",
              "exp_grad_relerr", "exp_hess_relerr", "gam_k1_equals_exp", "exp_fit_multistart",
              "gam_fit_multistart", "ff_fit_multistart"):
        S[k] = R[k]
    S["ff_sweeps_true"] = [int(np.median(R["ff_sweeps_true"])), int(R["ff_sweeps_true"].max())]
    S["tol"] = dict(exp=EXP_TOL, gam=GAM_TOL, ff=FF_TOL, em=EM_TOL)
    S["grad_diag"] = R["grad_diag"]
    S["gam_kmax"] = R["gam_kmax"]
    S["jac_exact_cov"] = R["jac_exact_cov"].tolist()
    S["nll_max"] = float(ex["nll"].max())
    # the trial with the largest observations
    i = int(np.argmax(R["u1"] + R["u2"]))
    S["big"] = dict(i=i, u1=float(R["u1"][i]), u2=float(R["u2"][i]), nll=float(ex["nll"][i]),
                    m1=float(ex["m1"][i]), m2=float(ex["m2"][i]), sd1=float(ex["sd1"][i]),
                    sd2=float(ex["sd2"][i]), corr=float(ex["corr"][i]))
    # representative trial
    rx = R["rep_ex"]
    S["rep_u"] = list(REP_U)
    S["rep_ex"] = {k: float(v) for k, v in rx.items()}
    S["rep_exp"] = dict(m1=float(np.exp(R["rep_exp_x"][0])), m2=float(np.exp(R["rep_exp_x"][1])), F=R["rep_exp_F"])
    gx = np.exp(R["rep_gam_x"])
    S["rep_gam"] = dict(m1=float(gx[0]), m2=float(gx[1]), k1=float(gx[2]), k2=float(gx[3]), F=R["rep_gam_F"])
    (a1, a2, a4), (b1, b2, b4) = R["rep_ff_mom"]
    S["rep_ff"] = dict(m1=a1, m2=b1, sd1=sqrt(a2 - a1 * a1), sd2=sqrt(b2 - b1 * b1), F=R["rep_ff_F"],
                       coef1=list(R["rep_ff_coef"][0]), coef2=list(R["rep_ff_coef"][1]))
    # contours holding 50 and 90 per cent of the probability, on one grid
    s1max, s2max = 3.2, 4.4
    g1, g2, dens = plane_densities(*REP_U, *LAM_TRUE, None, R["rep_gam_x"], R["rep_exp_x"],
                                   R["rep_ff_coef"], s1max, s2max)
    cont = {}
    for k, P in dens.items():
        levels = hpd_levels(P, (0.5, 0.9))
        cont[k] = [[[[round(float(x), 4), round(float(y), 4)] for x, y in line]
                    for line in marching_squares(g1, g2, P, lv)] for lv in levels]
    S["contours"] = cont
    S["plane"] = [s1max, s2max]
    return S
