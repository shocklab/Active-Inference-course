"""Where three learning rules come to rest on Week 2's model (Lesson 4.4).

The world is Week 2's model at its true parameters: d from the prior N(d_p, Sigma_p)
restricted to d > 0, and u | d ~ N(theta / d^2, Sigma_u). One parameter at a time is
learned with the others held at their true values. For each rule the drift is the
rule's per-trial step averaged over the world, and the table in Lesson 4.4 quotes the
parameter value where that drift is zero.

  point rule     the derivative of the log joint at the global mode d-hat (Week 3)
  Laplace rule   minus the derivative of the Laplace free energy
                 F_Lap = -ell(d-hat) + (1/2) ln(-ell''(d-hat)) - (1/2) ln 2 pi,
                 whose derivative keeps the curvature term through d-hat
  exact rule     the posterior average of the derivative of the log joint

World average: Gauss-Legendre in d over (0, 7.5] times Gauss-Hermite in the sensory
noise. Posterior averages: Gauss-Legendre in g = theta / d^2, where the likelihood has a
fixed width, with extra breakpoints evenly spaced in d so that the prior's tail is
resolved too. The exact rule's drift vanishes at the truth (the score has mean zero),
which is a consistency check. Each rule's zero is also recomputed with a finer world
quadrature (REFINED); the Laplace rule's zero needs many more noise nodes than the
exact rule's to settle at four decimals.
"""
import numpy as np
from math import erf, sqrt, pi

TRUTH = dict(dp=2.0, vp=1.0, th=1.0, vu=0.1)


def _zp(dp, vp):
    return 0.5 * (1.0 + erf(dp / sqrt(2.0 * vp)))


def _world(n_d=60, order=16, n_z=120, dmax=7.5, p=TRUTH):
    gx, gw = np.polynomial.legendre.leggauss(order)
    br = np.linspace(0.0, dmax, n_d + 1)
    a, b = br[:-1, None], br[1:, None]
    d = (0.5 * (a + b) + 0.5 * (b - a) * gx).ravel()
    w = (0.5 * (b - a) * gw).ravel()
    pw = np.exp(-(d - p["dp"]) ** 2 / (2 * p["vp"])) / sqrt(2 * pi * p["vp"]) / _zp(p["dp"], p["vp"]) * w
    hz, hw = np.polynomial.hermite_e.hermegauss(n_z)
    hw = hw / hw.sum()
    U = (p["th"] / d[:, None] ** 2 + sqrt(p["vu"]) * hz[None, :]).ravel()
    W = (pw[:, None] * hw[None, :]).ravel()
    return U, W


def ell(d, u, dp, vp, th, vu):
    return (-(d - dp) ** 2 / (2 * vp) - 0.5 * np.log(2 * pi * vp) - np.log(_zp(dp, vp))
            - (u - th / d ** 2) ** 2 / (2 * vu) - 0.5 * np.log(2 * pi * vu))


def derivs(d, u, dp, vp, th, vu):
    g = th / d ** 2; g1 = -2 * th / d ** 3; g2 = 6 * th / d ** 4; g3 = -24 * th / d ** 5
    r = u - g
    return (-(d - dp) / vp + r * g1 / vu,
            -1 / vp + (-g1 ** 2 + r * g2) / vu,
            (-3 * g1 * g2 + r * g3) / vu)


_DS = np.exp(np.linspace(np.log(0.05), np.log(12.0), 600))


def modes(u, p):
    """Global mode of ell on d > 0: the highest point of a log-spaced scan, refined by
    Newton's method in ln d."""
    L = ell(_DS[None, :], u[:, None], *p)
    d = _DS[np.argmax(L, axis=1)].copy()
    for _ in range(100):
        l1, l2, _ = derivs(d, u, *p)
        step = np.where(l2 < 0, -l1 / l2, 0.1 * d * np.sign(l1))
        d = d * np.exp(np.clip(step / d, -0.3, 0.3))
    l1, l2, l3 = derivs(d, u, *p)
    assert np.max(np.abs(l1 / l2 / d)) < 1e-12 and np.all(l2 < 0)
    return d, l2, l3


def param_derivs(d, u, which, p):
    """d ell / d omega, and its first and second derivatives in d."""
    dp, vp, th, vu = p
    g = th / d ** 2; g1 = -2 * th / d ** 3; g2 = 6 * th / d ** 4
    if which == "vp":
        al = dp / sqrt(vp)
        phi = np.exp(-0.5 * al * al) / sqrt(2 * pi)
        lw = (d - dp) ** 2 / (2 * vp ** 2) - 1 / (2 * vp) + dp * phi / (2 * vp ** 1.5 * _zp(dp, vp))
        return lw, (d - dp) / vp ** 2, np.full_like(d, 1 / vp ** 2)
    r = u - g
    return r ** 2 / (2 * vu ** 2) - 1 / (2 * vu), -r * g1 / vu ** 2, (g1 ** 2 - r * g2) / vu ** 2


def post_avg(u, which, p, npan=48, half=12.0, dmax=12.0, nd=80):
    dp, vp, th, vu = p
    sd = sqrt(vu); gmin = th / dmax ** 2
    lo = np.maximum(gmin, u - half * sd)
    hi = np.maximum(u + half * sd, gmin * 1.5)
    xs, ws = np.polynomial.legendre.leggauss(8)
    t = np.linspace(0, 1, npan + 1)
    out = np.empty(u.size)
    for j in range(0, u.size, 4000):
        L0, H0, uu = lo[j:j + 4000, None], hi[j:j + 4000, None], u[j:j + 4000, None]
        bg = L0 + (H0 - L0) * t[None, :]
        bd = th / np.linspace(0.05, dmax, nd)[None, ::-1] ** 2 + 0 * L0
        brk = np.sort(np.concatenate([bg, np.clip(bd, L0, H0)], axis=1), axis=1)
        A, B = brk[:, :-1, None], brk[:, 1:, None]
        g = (0.5 * (A + B) + 0.5 * (B - A) * xs).reshape(len(uu), -1)
        w = np.maximum((0.5 * (B - A) * ws).reshape(len(uu), -1), 1e-300)
        d = np.sqrt(th / g)
        L = ell(d, uu, *p) + np.log(0.5 * sqrt(th)) - 1.5 * np.log(g) + np.log(w)
        P = np.exp(L - L.max(axis=1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        out[j:j + 4000] = (P * param_derivs(d, uu, which, p)[0]).sum(1)
    return out


def drift(rule, which, val, world):
    U, W = world
    p = [TRUTH["dp"], TRUTH["vp"], TRUTH["th"], TRUTH["vu"]]
    p[{"vp": 1, "vu": 3}[which]] = val
    if rule == "exact":
        return float((W * post_avg(U, which, p)).sum())
    d, l2, l3 = modes(U, p)
    lw, l1w, l2w = param_derivs(d, U, which, p)
    if rule == "point":
        return float((W * lw).sum())
    return float((W * (lw - 0.5 * (l2w + l3 * (-l1w / l2)) / l2)).sum())


def root(rule, which, lo, hi, world, tol=1e-10):
    """Zero of the drift in ln omega by the Illinois variant of regula falsi."""
    a, b = np.log(lo), np.log(hi)
    fa, fb = drift(rule, which, lo, world), drift(rule, which, hi, world)
    assert fa * fb < 0, (rule, which)
    side = 0
    for _ in range(200):
        c = (a * fb - b * fa) / (fb - fa)
        fc = drift(rule, which, float(np.exp(c)), world)
        if fc * fb > 0:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
        if abs(b - a) < tol:
            break
    return float(np.exp(0.5 * (a + b)))


REFINED = dict(n_d=120, order=20, n_z=160)


def drift_slope(rule, which, val, world, rel=1e-4):
    """d(drift)/d(ln omega) at val by central differences; negative means the zero
    attracts small deterministic steps."""
    up = drift(rule, which, val * (1 + rel), world)
    dn = drift(rule, which, val * (1 - rel), world)
    return float((up - dn) / (np.log1p(rel) - np.log1p(-rel)))


def summary():
    world = _world()
    fine = _world(**REFINED)
    S = {"world_mass": float(world[1].sum()), "refined": REFINED}
    truth = {"vp": TRUTH["vp"], "vu": TRUTH["vu"]}
    for which, t in truth.items():
        S[which] = {r: root(r, which, t * 0.01, t * 3, world) for r in ("point", "laplace", "exact")}
        S[which + "_fine"] = {r: root(r, which, t * 0.01, t * 3, fine) for r in ("point", "laplace", "exact")}
        S[which + "_slope"] = {r: drift_slope(r, which, S[which][r], world) for r in ("point", "laplace", "exact")}
        S[which + "_drift_truth"] = {r: drift(r, which, t, world) for r in ("point", "laplace", "exact")}
    return S
