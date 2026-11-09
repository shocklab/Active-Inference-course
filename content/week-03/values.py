"""Every derived number in Week 3, computed rather than typed.

Week 3 keeps Week 2's model unchanged: a source at distance d, an intensity
u = theta / d^2 heard through Gaussian noise of variance Sigma_u, and a
Gaussian prior on d with location d_p and variance Sigma_p, truncated to d > 0.
Week 2 held the four numbers (d_p, Sigma_p, Sigma_u, theta) fixed at
(2, 1, 0.1, 1). Week 3 lets the animal change them across many trials.

The simulations here are the reference for the widgets in assets/js/w03.js.
Both use the same seeded generator (mulberry32), the same inner loop and the
same update rules, in the same order of floating-point operations, so a widget
readout at its default settings should agree with the number quoted in the
prose. Run standalone to see the values:  python3 content/week-03/values.py
"""
from math import cos, erf, exp, log, pi, sqrt

import numpy as np

# ── Week 2's model, the starting point for every experiment ─────────────
D_PRIOR, VAR_PRIOR = 2.0, 1.0
U_OBS, VAR_OBS = 0.50, 0.10
THETA = 1.0                      # Week 2's "source power taken as one"

V = {
    "d_prior": D_PRIOR, "var_prior": VAR_PRIOR,
    "u_obs": U_OBS, "var_obs": VAR_OBS, "theta0": THETA,
    "prec_prior": 1 / VAR_PRIOR, "prec_obs": 1 / VAR_OBS,
}


def npdf(x, m, v):
    return exp(-(x - m) ** 2 / (2 * v)) / sqrt(2 * pi * v)


def zp(dp, vp):
    """Positive-half-line mass of N(dp, vp): Week 2's Z_p."""
    return 0.5 * (1 + erf(dp / sqrt(2 * vp)))


# ── the shared simulation engine (mirrored line for line in w03.js) ──────
ETA, KMAX, TOL = 0.05, 2000, 1e-12


def mulberry32(seed):
    """A 32-bit seeded generator, ported from the JavaScript original so that
    the widget and this file draw identical random numbers."""
    a = [seed & 0xFFFFFFFF]

    def imul(x, y):
        return ((x & 0xFFFFFFFF) * (y & 0xFFFFFFFF)) & 0xFFFFFFFF

    def rnd():
        a[0] = (a[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(a[0] ^ (a[0] >> 15), 1 | a[0])
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296
    return rnd


def normal(r):
    u1 = r()
    u2 = r()
    return sqrt(-2.0 * log(1.0 - u1)) * cos(2.0 * pi * u2)


def erf_as(x):
    """Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7. JavaScript
    has no erf, so the simulations on both sides use this one."""
    s = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * exp(-x * x)
    return s * y


def trunc_ratio(dp, vp):
    """N(0; dp, vp) / Z_p, the truncation correction in the prior updates."""
    dens = exp(-dp * dp / (2.0 * vp)) / sqrt(2.0 * pi * vp)
    return dens / (0.5 * (1.0 + erf_as(dp / sqrt(2.0 * vp))))


def settle(u, dp, vp, th, vu, path=None):
    """Week 2's ascent on the log joint, run within one trial.

    Each step is capped at the reciprocal of the Gauss-Newton curvature, and
    never removes more than half the current distance. The caps keep a loud,
    near source from throwing the iteration into the instability of Week 2's
    Lesson 3. A finite step can change which stationary point the climb reaches,
    so the reported runs are checked against the highest mode separately.
    Returns the state, the number of steps, and whether the stopping rule was
    met before KMAX steps ran out.
    """
    d = dp
    k = 0
    ok = False
    while k < KMAX:
        gp = -2.0 * th / (d * d * d)
        grad = (dp - d) / vp + (u - th / (d * d)) / vu * gp
        cap = 1.0 / (1.0 / vp + gp * gp / vu)
        step = (ETA if ETA < cap else cap) * grad
        if step < -0.5 * d:
            step = -0.5 * d
        d = d + step
        k += 1
        if path is not None:
            path.append(d)
        if abs(step) < TOL:
            ok = True
            break
    return d, k, ok


def world_trial(r, loc, var, th, vu):
    """One trial of the generative process: a truncated-Gaussian distance and a
    noisy inverse-square intensity."""
    while True:
        d = loc + sqrt(var) * normal(r)
        if d > 0:
            break
    return d, th / (d * d) + sqrt(vu) * normal(r)


NY = 801


def post_r2(u, dp, vp, th, vu):
    """Posterior expectation of the squared sensory residual (u - g(d))^2.

    When the likelihood's window of ten standard deviations either side of u
    lies at intensities the prior can reach, the integral runs over that window
    in residual coordinates rho = u - y, where y = theta / d^2 and the
    likelihood has a fixed width; the Jacobian |dd/dy| of Week 2's Lesson 2
    enters as y^(-3/2). Otherwise (a quiet or negative u) it runs over log
    distance, from where the likelihood has fallen by e^-72 to past the furthest
    the mode can be pushed. Checked against a brute-force grid for Sigma_u from
    0.001 to 1 and u from -3 to 3: largest relative difference 5e-15.
    """
    sd = sqrt(vu)
    lw = []
    r2 = []
    if u - 10.0 * sd > th / ((dp + 12.0 * sqrt(vp)) * (dp + 12.0 * sqrt(vp))):
        h = 20.0 * sd / (NY - 1)
        for i in range(NY):
            rho = -10.0 * sd + i * h
            y = u - rho
            d = sqrt(th / y)
            lw.append(-rho * rho / (2.0 * vu) - (d - dp) * (d - dp) / (2.0 * vp) - 1.5 * log(y))
            r2.append(rho * rho)
    else:
        up = u if u > 0.0 else 0.0
        da = sqrt(th / (up + 12.0 * sd))
        c = 2.0 * th * vp * (th / (dp * dp) + abs(u)) / vu
        db = dp + 12.0 * sqrt(vp) + sqrt(sqrt(c))
        if da > 0.5 * db:
            da = 0.5 * db
        sa = log(da)
        h = (log(db) - sa) / (NY - 1)
        for i in range(NY):
            q = sa + i * h
            d = exp(q)
            res = u - th / (d * d)
            lw.append(-res * res / (2.0 * vu) - (d - dp) * (d - dp) / (2.0 * vp) + q)
            r2.append(res * res)
    mx = max(lw)
    s0 = 0.0
    s2 = 0.0
    for i in range(NY):
        w = exp(lw[i] - mx) * (0.5 if i == 0 or i == NY - 1 else 1.0)
        s0 += w
        s2 += w * r2[i]
    return s2 / s0


def run(seed, n, world, model, learn, beta, rule="point", keep_paths=()):
    """Trials one after another. `world` is (location, variance, theta, Sigma_u)
    of the process; `model` holds the animal's (dp, vp, th, vu). Only the
    parameter named by `learn` changes."""
    r = mulberry32(seed)
    dp, vp, th, vu = model["dp"], model["vp"], model["th"], model["vu"]
    hist, errs, paths, unsettled = [], [], {}, 0
    for i in range(n):
        dtrue, u = world_trial(r, *world)
        path = [dp] if i in keep_paths else None
        dh, _, ok = settle(u, dp, vp, th, vu, path)
        if not ok:
            unsettled += 1
        if path is not None:
            paths[i] = path
        ep = (dp - dh) / vp
        eu = (u - th / (dh * dh)) / vu
        errs.append((ep, eu, dh, u, {"dp": dp, "vp": vp, "th": th, "vu": vu}[learn]))
        if not ok:
            pass                     # inference unfinished: no parameter update
        elif learn == "dp":
            dp = dp + beta * (-ep - trunc_ratio(dp, vp))
        elif learn == "th":
            th = th + beta * eu / (dh * dh)
        elif learn == "vu":
            res = u - th / (dh * dh)
            r2 = res * res if rule == "point" else post_r2(u, dp, vp, th, vu)
            vu = vu * exp(beta * 0.5 * (r2 / vu - 1.0))
        elif learn == "vp":
            dev = dh - dp
            vp = vp * exp(beta * (0.5 * (dev * dev / vp - 1.0)
                                  + 0.5 * dp * trunc_ratio(dp, vp)))
        hist.append({"dp": dp, "vp": vp, "th": th, "vu": vu}[learn])
    return hist, errs, paths, unsettled


WEEK2 = {"dp": D_PRIOR, "vp": VAR_PRIOR, "th": THETA, "vu": VAR_OBS}
N_TRIALS = 2000


def mode_gap(errs, model, learn):
    """Largest distance, over a run, between the state the inner loop settled
    at and the highest mode of that trial's log joint, found on a grid in log
    distance and polished by Newton's method. Zero to rounding means every
    trial settled on the highest mode, not merely on some stationary point."""
    E = np.array(errs, dtype=float)
    u, dh, val = E[:, 3], E[:, 2], E[:, 4]
    par = {k: np.full(len(E), model[k], dtype=float) for k in ("dp", "vp", "th", "vu")}
    par[learn] = val
    q = np.linspace(log(1e-3), log(40.0), 8000)
    gap = 0.0
    for i in range(0, len(E), 250):
        sl = slice(i, i + 250)
        dp, vp, th, vu = (par[k][sl, None] for k in ("dp", "vp", "th", "vu"))
        dg = np.exp(q)[None, :]
        lw = -(dg - dp) ** 2 / (2 * vp) - (u[sl, None] - th / dg ** 2) ** 2 / (2 * vu)
        d = dg[0, lw.argmax(1)]
        dp, vp, th, vu = (par[k][sl] for k in ("dp", "vp", "th", "vu"))
        for _ in range(60):
            g1 = (dp - d) / vp + (u[sl] - th / d ** 2) / vu * (-2 * th / d ** 3)
            g2 = -1 / vp - 4 * th * th / d ** 6 / vu + (u[sl] - th / d ** 2) / vu * 6 * th / d ** 4
            d = d - g1 / g2
        gap = max(gap, float(np.abs(d - dh[sl]).max()))
    return gap


# ════════════════════════════════════════════════════════════════════════
# Lesson 1 · two timescales
# ════════════════════════════════════════════════════════════════════════

def map_exact(u, dp=D_PRIOR, vp=VAR_PRIOR, th=THETA, vu=VAR_OBS):
    """The mode, by the inner loop and then Newton polishing on the exact
    curvature, so finite differences of the settled log joint are clean."""
    d, _, _ = settle(u, dp, vp, th, vu)
    for _ in range(50):
        g1 = (dp - d) / vp + (u - th / d ** 2) / vu * (-2 * th / d ** 3)
        g2 = (-1 / vp - (4 * th * th / d ** 6) / vu
              + (u - th / d ** 2) / vu * (6 * th / d ** 4))
        d -= g1 / g2
    return d


def ell(d, u, dp=D_PRIOR, vp=VAR_PRIOR, th=THETA, vu=VAR_OBS):
    """Week 2's log joint with every constant restored, including ln Z_p."""
    return (-(d - dp) ** 2 / (2 * vp) - 0.5 * log(2 * pi * vp) - log(zp(dp, vp))
            - (u - th / d ** 2) ** 2 / (2 * vu) - 0.5 * log(2 * pi * vu))


def ell_hat(u, **kw):
    return ell(map_exact(u, **kw), u, **kw)


_dhat = map_exact(U_OBS)
V["mode_w2"] = _dhat
V["eps_p_w2"] = (D_PRIOR - _dhat) / VAR_PRIOR
V["eps_u_w2"] = (U_OBS - THETA / _dhat ** 2) / VAR_OBS
V["ell_hat_w2"] = ell(_dhat, U_OBS)

# The envelope theorem, checked: the total derivative of the settled log joint
# with respect to Sigma_u equals the partial derivative at the settled state.
_h = 1e-6
V["env_total"] = (ell_hat(U_OBS, vu=VAR_OBS + _h) - ell_hat(U_OBS, vu=VAR_OBS - _h)) / (2 * _h)
V["env_partial"] = 0.5 * (V["eps_u_w2"] ** 2 - 1 / VAR_OBS)
assert abs(V["env_total"] - V["env_partial"]) < 1e-5, "envelope theorem check failed"
V["env_gap"] = abs(V["env_total"] - V["env_partial"])

# The same comparison when inference is stopped after three Week 2 steps,
# before it has settled: the state now depends on Sigma_u through the
# iteration, and the partial derivative no longer equals the total.
K_SHORT = 3


def _d_after(k, vu):
    d = D_PRIOR
    for _ in range(k):
        d += 0.05 * ((D_PRIOR - d) / VAR_PRIOR
                     + (U_OBS - THETA / d ** 2) / vu * (-2 * THETA / d ** 3))
    return d


def _ell_after(k, vu):
    return ell(_d_after(k, vu), U_OBS, vu=vu)


V["k_short"] = K_SHORT
V["d_short"] = _d_after(K_SHORT, VAR_OBS)
V["short_total"] = (_ell_after(K_SHORT, VAR_OBS + _h) - _ell_after(K_SHORT, VAR_OBS - _h)) / (2 * _h)
_eu_s = (U_OBS - THETA / V["d_short"] ** 2) / VAR_OBS
V["short_partial"] = 0.5 * (_eu_s ** 2 - 1 / VAR_OBS)
V["short_dl_dd"] = ((D_PRIOR - V["d_short"]) / VAR_PRIOR
                    + _eu_s * (-2 * THETA / V["d_short"] ** 3))

# ── the linear case: the slow update is a running average ────────────────
# With g(d) = d on the whole line, d_hat - d_p = Sigma_p/(Sigma_p+Sigma_u) (u - d_p),
# so Delta d_p = beta (u - d_p) / (Sigma_p + Sigma_u): an exponential moving
# average of the raw observations with gain kappa = beta/(Sigma_p + Sigma_u).
BETA_LIN = 0.02
V["beta_lin"] = BETA_LIN
V["kappa_lin"] = BETA_LIN / (VAR_PRIOR + VAR_OBS)
V["halflife_lin"] = log(2) / -log(1 - V["kappa_lin"])
V["sd_ratio_lin"] = sqrt(V["kappa_lin"] / (2 - V["kappa_lin"]))
# the spread of the observations in that linear world, and of d_p about its mean
V["var_u_lin"] = VAR_PRIOR + VAR_OBS
V["sd_dp_lin"] = sqrt(V["var_u_lin"]) * V["sd_ratio_lin"]


def _lin_ema_check(n=200000, seed=5):
    r = mulberry32(seed)
    dp, xs = D_PRIOR, []
    for i in range(n):
        d = D_PRIOR + sqrt(VAR_PRIOR) * normal(r)
        u = d + sqrt(VAR_OBS) * normal(r)
        dh = (VAR_OBS * dp + VAR_PRIOR * u) / (VAR_PRIOR + VAR_OBS)
        dp += BETA_LIN * (dh - dp) / VAR_PRIOR
        xs.append(dp)
    return float(np.std(xs[1000:]))


# the same spread at a quarter and a sixteenth of the rate: dividing the rate by
# four halves the standard deviation, which is of order sqrt(beta), not beta
for _k, _b in (("q", BETA_LIN / 4), ("s", BETA_LIN / 16)):
    _kap = _b / (VAR_PRIOR + VAR_OBS)
    V[f"beta_lin_{_k}"] = _b
    V[f"sd_dp_lin_{_k}"] = sqrt(V["var_u_lin"] * _kap / (2 - _kap))
assert abs(V["sd_dp_lin"] / V["sd_dp_lin_q"] - 2) < 0.02

V["sd_dp_lin_sim"] = _lin_ema_check()
assert abs(V["sd_dp_lin_sim"] / V["sd_dp_lin"] - 1) < 0.05, "EMA spread formula failed"

# ── widget 1 · two-timescales, at its default settings ───────────────────
W1 = dict(seed=7, loc=3.0, vu=0.01, beta=0.01)
_m = dict(WEEK2, vu=W1["vu"])
_KEEP = (0, 1, 2, 1999)
_h1, _e1, _p1, _bad1 = run(W1["seed"], N_TRIALS, (W1["loc"], VAR_PRIOR, THETA, W1["vu"]),
                    _m, "dp", W1["beta"], keep_paths=_KEEP)
V["w1_seed"], V["w1_loc"], V["w1_vu"], V["w1_beta"] = W1["seed"], W1["loc"], W1["vu"], W1["beta"]
V["n_trials"] = N_TRIALS
V["w1_final"] = _h1[-1]
V["w1_after_200"] = _h1[199]
V["w1_mean_late"] = float(np.mean(_h1[N_TRIALS // 2:]))
V["w1_sd_late"] = float(np.std(_h1[N_TRIALS // 2:]))
V["w1_first_steps"] = len(_p1[0]) - 1
V["w1_first_end"] = _p1[0][-1]
WIDGET_REF = {"w1": {"final": _h1[-1], "mean_late": V["w1_mean_late"],
                     "sd_late": V["w1_sd_late"], "steps0": V["w1_first_steps"]}}


# ════════════════════════════════════════════════════════════════════════
# Lesson 2 · the derivatives with respect to d_p, Sigma_p, Sigma_u
# ════════════════════════════════════════════════════════════════════════
V["trunc_corr"] = npdf(0, D_PRIOR, VAR_PRIOR) / zp(D_PRIOR, VAR_PRIOR)
V["zp_w2"] = zp(D_PRIOR, VAR_PRIOR)
V["n0_w2"] = npdf(0, D_PRIOR, VAR_PRIOR)

# d Z_p / d d_p = N(0; d_p, Sigma_p) and d Z_p / d Sigma_p = -(d_p / 2 Sigma_p) N(0; ...)
_e = 1e-6
_dz_ddp = (zp(D_PRIOR + _e, VAR_PRIOR) - zp(D_PRIOR - _e, VAR_PRIOR)) / (2 * _e)
_dz_dvp = (zp(D_PRIOR, VAR_PRIOR + _e) - zp(D_PRIOR, VAR_PRIOR - _e)) / (2 * _e)
assert abs(_dz_ddp - npdf(0, D_PRIOR, VAR_PRIOR)) < 1e-8
assert abs(_dz_dvp + D_PRIOR / (2 * VAR_PRIOR) * npdf(0, D_PRIOR, VAR_PRIOR)) < 1e-8
V["dz_dvp"] = _dz_dvp

# the three gradients at Week 2's single observation, at its mode
V["grad_dp_w2"] = -V["eps_p_w2"] - V["trunc_corr"]
V["grad_vp_w2"] = (0.5 * (V["eps_p_w2"] ** 2 - 1 / VAR_PRIOR)
                   + D_PRIOR / (2 * VAR_PRIOR) * V["trunc_corr"])
V["grad_vu_w2"] = 0.5 * (V["eps_u_w2"] ** 2 - 1 / VAR_OBS)
V["sq_err_u_w2"] = (U_OBS - THETA / _dhat ** 2) ** 2
V["sq_err_p_w2"] = (_dhat - D_PRIOR) ** 2
for _key, _kw in (("grad_dp_w2", "dp"), ("grad_vp_w2", "vp"), ("grad_vu_w2", "vu")):
    _base = {"dp": D_PRIOR, "vp": VAR_PRIOR, "vu": VAR_OBS}[_kw]
    _num = (ell(_dhat, U_OBS, **{_kw: _base + _e}) - ell(_dhat, U_OBS, **{_kw: _base - _e})) / (2 * _e)
    assert abs(_num - V[_key]) < 1e-6, f"hand derivative {_key} disagrees"

# the truncated prior's own mean and second moment about d_p, which the
# stationary conditions of the prior updates match
V["trunc_mean"] = D_PRIOR + VAR_PRIOR * V["trunc_corr"]
V["trunc_m2"] = VAR_PRIOR * (1 - D_PRIOR * V["trunc_corr"])
_x = np.linspace(0.0, 14.0, 400001)
_pr = np.exp(-(_x - D_PRIOR) ** 2 / (2 * VAR_PRIOR))
_pr /= np.trapezoid(_pr, _x)
assert abs(np.trapezoid(_x * _pr, _x) - V["trunc_mean"]) < 1e-8
assert abs(np.trapezoid((_x - D_PRIOR) ** 2 * _pr, _x) - V["trunc_m2"]) < 1e-8

# a plain Euler step on a variance with zero residual crosses zero below sqrt(beta/2)
BETA_VAR = 0.002
V["beta_var"] = BETA_VAR
V["var_cross"] = sqrt(BETA_VAR / 2)
# and with a residual, a small variance is thrown upwards: residual 0.1, variance 0.01
V["jump_r"], V["jump_var"] = 0.2, 0.01
V["jump_step"] = BETA_VAR * 0.5 * (V["jump_r"] ** 2 / V["jump_var"] ** 2 - 1 / V["jump_var"])


# ════════════════════════════════════════════════════════════════════════
# Lesson 3 · learning the link
# ════════════════════════════════════════════════════════════════════════
V["h_at_mode"] = 1 / _dhat ** 2
V["grad_th_w2"] = V["eps_u_w2"] * V["h_at_mode"]
_num = (ell(_dhat, U_OBS, th=THETA + _e) - ell(_dhat, U_OBS, th=THETA - _e)) / (2 * _e)
assert abs(_num - V["grad_th_w2"]) < 1e-6

# The scale symmetry: (theta, d_p, Sigma_p) and (c^2 theta, c d_p, c^2 Sigma_p) give
# the same evidence for every u. Checked by quadrature at c = 2.
def evidence(u, dp=D_PRIOR, vp=VAR_PRIOR, th=THETA, vu=VAR_OBS):
    d = np.linspace(1e-4, dp + 12 * sqrt(vp), 400001)
    f = (np.exp(-(d - dp) ** 2 / (2 * vp)) / sqrt(2 * pi * vp) / zp(dp, vp)
         * np.exp(-(u - th / d ** 2) ** 2 / (2 * vu)) / sqrt(2 * pi * vu))
    return float(np.trapezoid(f, d))


SCALE_C = 2.0
V["scale_c"] = SCALE_C
V["ev_w2"] = evidence(U_OBS)
V["ev_scaled"] = evidence(U_OBS, dp=SCALE_C * D_PRIOR, vp=SCALE_C ** 2 * VAR_PRIOR,
                          th=SCALE_C ** 2 * THETA)
assert abs(V["ev_w2"] / V["ev_scaled"] - 1) < 1e-6
V["scaled_dp"], V["scaled_vp"], V["scaled_th"] = SCALE_C * D_PRIOR, SCALE_C ** 2 * VAR_PRIOR, SCALE_C ** 2 * THETA

# ── widget 2 · wrong-link, at its default settings ───────────────────────
W2 = dict(seed=7, th_world=2.0, beta=0.01, window=200)
V["w2_th_world"], V["w2_beta"], V["w2_window"] = W2["th_world"], W2["beta"], W2["window"]
_h2, _e2, _, _bad2 = run(W2["seed"], N_TRIALS, (D_PRIOR, VAR_PRIOR, W2["th_world"], VAR_OBS),
                  dict(WEEK2), "th", W2["beta"])
_h2off, _e2off, _, _bad2off = run(W2["seed"], N_TRIALS, (D_PRIOR, VAR_PRIOR, W2["th_world"], VAR_OBS),
                        dict(WEEK2), "th", 0.0)
V["w2_final"] = _h2[-1]
V["w2_mean_late"] = float(np.mean(_h2[N_TRIALS // 2:]))
_e2 = np.array(_e2)
_e2off = np.array(_e2off)
_w = W2["window"]
V["w2_epsp_early"] = float(_e2[:_w, 0].mean())
V["w2_epsu_early"] = float(_e2[:_w, 1].mean())
V["w2_epsp_late"] = float(_e2[-_w:, 0].mean())
V["w2_epsu_late"] = float(_e2[-_w:, 1].mean())
V["w2_off_epsp"] = float(_e2off[-_w:, 0].mean())
V["w2_off_epsu"] = float(_e2off[-_w:, 1].mean())
V["w2_sig_early"] = float((_e2[:_w, 1] / _e2[:_w, 2] ** 2).mean())
V["w2_sig_late"] = float((_e2[-_w:, 1] / _e2[-_w:, 2] ** 2).mean())
V["w2_off_sig"] = float((_e2off[-_w:, 1] / _e2off[-_w:, 2] ** 2).mean())
_, _e2true, _, _bad2true = run(W2["seed"], N_TRIALS, (D_PRIOR, VAR_PRIOR, W2["th_world"], VAR_OBS),
                    dict(WEEK2, th=W2["th_world"]), "th", 0.0)
_e2true = np.array(_e2true)
V["w2_true_epsp"] = float(_e2true[:, 0].mean())
V["w2_true_epsu"] = float(_e2true[:, 1].mean())
# a world quieter than the model, theta* = 1/2, for the exercises
_hq, _eq, _, _badq = run(W2["seed"], N_TRIALS, (D_PRIOR, VAR_PRIOR, 0.5, VAR_OBS), dict(WEEK2), "th", W2["beta"])
_hqoff, _eqoff, _, _badqoff = run(W2["seed"], N_TRIALS, (D_PRIOR, VAR_PRIOR, 0.5, VAR_OBS), dict(WEEK2), "th", 0.0)
_eqoff = np.array(_eqoff)
V["w2q_th_world"] = 0.5
V["w2q_final"] = _hq[-1]
V["w2q_off_epsp"] = float(_eqoff[-_w:, 0].mean())
V["w2q_off_epsu"] = float(_eqoff[-_w:, 1].mean())
V["w2q_off_sig"] = float((_eqoff[-_w:, 1] / _eqoff[-_w:, 2] ** 2).mean())
WIDGET_REF["w2"] = {"final": V["w2_final"], "mean_late": V["w2_mean_late"],
                    "epsp_late": V["w2_epsp_late"], "epsu_late": V["w2_epsu_late"]}


# ════════════════════════════════════════════════════════════════════════
# Lesson 4 · what the point estimate leaves out
# ════════════════════════════════════════════════════════════════════════

# ── the Fisher identity at Week 2's single observation ───────────────────
# d ln p(u) / d omega equals the posterior average of d ell / d omega.
_G = np.linspace(1e-3, 14.0, 400001)


def _post_grid(u=U_OBS, dp=D_PRIOR, vp=VAR_PRIOR, th=THETA, vu=VAR_OBS):
    lw = -(_G - dp) ** 2 / (2 * vp) - (u - th / _G ** 2) ** 2 / (2 * vu)
    p = np.exp(lw - lw.max())
    return p / np.trapezoid(p, _G)


_p = _post_grid()
V["post_mean_w2"] = float(np.trapezoid(_G * _p, _G))
V["post_var_w2"] = float(np.trapezoid((_G - V["post_mean_w2"]) ** 2 * _p, _G))
_eu_grid = (U_OBS - THETA / _G ** 2) / VAR_OBS
V["fisher_avg_vu"] = float(np.trapezoid(0.5 * (_eu_grid ** 2 - 1 / VAR_OBS) * _p, _G))
V["fisher_num_vu"] = (log(evidence(U_OBS, vu=VAR_OBS + 1e-5))
                      - log(evidence(U_OBS, vu=VAR_OBS - 1e-5))) / 2e-5
assert abs(V["fisher_avg_vu"] - V["fisher_num_vu"]) < 1e-3, "Fisher identity check failed"
V["fisher_avg_dp"] = float(np.trapezoid(((_G - D_PRIOR) / VAR_PRIOR) * _p, _G)) - V["trunc_corr"]
V["fisher_num_dp"] = (log(evidence(U_OBS, dp=D_PRIOR + 1e-5))
                      - log(evidence(U_OBS, dp=D_PRIOR - 1e-5))) / 2e-5
assert abs(V["fisher_avg_dp"] - V["fisher_num_dp"]) < 1e-3
V["fisher_avg_vp"] = float(np.trapezoid(
    (0.5 * ((_G - D_PRIOR) ** 2 / VAR_PRIOR ** 2 - 1 / VAR_PRIOR)) * _p, _G)) \
    + D_PRIOR / (2 * VAR_PRIOR) * V["trunc_corr"]
V["fisher_num_vp"] = (log(evidence(U_OBS, vp=VAR_PRIOR + 1e-5))
                      - log(evidence(U_OBS, vp=VAR_PRIOR - 1e-5))) / 2e-5
assert abs(V["fisher_avg_vp"] - V["fisher_num_vp"]) < 1e-3

# ── the linear Gaussian case, where everything is exact ──────────────────
# World: d ~ N(d_p, S) on the whole line, u = d + noise of variance Sigma_u. The
# animal knows d_p and Sigma_u and learns Sigma_p =: x from MAP estimates.
LIN_S, LIN_VU = 1.0, 0.1                   # Week 2's linear comparison
V["lin_s"], V["lin_vu"] = LIN_S, LIN_VU


def lin_roots(S, vu):
    disc = (S - vu) ** 2 - 4 * vu ** 2
    if S - vu <= 0 or disc < 0:
        return None
    return ((S - vu) - sqrt(disc)) / 2, ((S - vu) + sqrt(disc)) / 2


V["lin_root_lo"], V["lin_root_hi"] = lin_roots(LIN_S, LIN_VU)
V["lin_root_hi_pct"] = 100 * (V["lin_root_hi"] / LIN_S - 1)
V["lin_ratio"] = 3
# check both roots satisfy (x + vu)^2 = x (S + vu)
for _x in (V["lin_root_lo"], V["lin_root_hi"]):
    assert abs((_x + LIN_VU) ** 2 - _x * (LIN_S + LIN_VU)) < 1e-12
LIN_VU_BAD = 0.5
V["lin_vu_bad"] = LIN_VU_BAD
assert lin_roots(LIN_S, LIN_VU_BAD) is None
V["lin_bad_threshold"] = 3 * LIN_VU_BAD


def _lin_learn(rule, n=60000, seed=13, beta=0.01, x0=VAR_PRIOR, vu=LIN_VU):
    """Learn Sigma_p in the linear world with the log-variance step."""
    r = mulberry32(seed)
    x, xs = x0, []
    for i in range(n):
        d = D_PRIOR + sqrt(LIN_S) * normal(r)
        u = d + sqrt(vu) * normal(r)
        dh = (vu * D_PRIOR + x * u) / (x + vu)
        sq = (dh - D_PRIOR) ** 2
        if rule == "post":
            sq += x * vu / (x + vu)        # the posterior variance
        x = x * exp(beta * 0.5 * (sq / x - 1.0))
        xs.append(x)
    return xs


_xs_pt = _lin_learn("point")
_xs_em = _lin_learn("post")
V["lin_sim_point"] = float(np.mean(_xs_pt[20000:]))
V["lin_sim_post"] = float(np.mean(_xs_em[20000:]))
assert abs(V["lin_sim_point"] / V["lin_root_hi"] - 1) < 0.03
assert abs(V["lin_sim_post"] / LIN_S - 1) < 0.03
_xs_bad = _lin_learn("point", n=4000, vu=LIN_VU_BAD)
V["lin_bad_after"] = _xs_bad[-1]
V["lin_bad_n"] = 4000

# the missing term: ln p(u) - ell_hat = (1/2) ln(2 pi v_post), exact here
V["lin_vpost"] = LIN_S * LIN_VU / (LIN_S + LIN_VU)

# figure: expected settled log joint and expected log evidence against Sigma_p
_FX = dict(x0=70.0, x1=640.0, ytop=30.0, ybase=250.0, lo=log(0.002), hi=log(10.0),
           ymin=-5.4, ymax=-0.8)
_xs_fig = np.exp(np.linspace(_FX["lo"], _FX["hi"], 240))
_Sbar = LIN_S + LIN_VU


def _f_hat(x):
    return -_Sbar / (2 * (x + LIN_VU)) - 0.5 * np.log(x) - log(2 * pi) - 0.5 * log(LIN_VU)


def _f_ev(x):
    return -_Sbar / (2 * (x + LIN_VU)) - 0.5 * np.log(x + LIN_VU) - 0.5 * log(2 * pi)


def _fx(x):
    return _FX["x0"] + (np.log(x) - _FX["lo"]) / (_FX["hi"] - _FX["lo"]) * (_FX["x1"] - _FX["x0"])


def _fy(y):
    return _FX["ybase"] - (y - _FX["ymin"]) / (_FX["ymax"] - _FX["ymin"]) * (_FX["ybase"] - _FX["ytop"])


def _path(f):
    pts = [f"{a:.1f} {b:.1f}" for a, b in zip(_fx(_xs_fig), _fy(f(_xs_fig)))]
    return "M" + " L".join(pts)


V["fig_hat_path"] = _path(_f_hat)
V["fig_ev_path"] = _path(_f_ev)
V["fig_hi_x"] = f'{float(_fx(V["lin_root_hi"])):.1f}'
V["fig_hi_y"] = f'{float(_fy(_f_hat(V["lin_root_hi"]))):.1f}'
V["fig_lo_x"] = f'{float(_fx(V["lin_root_lo"])):.1f}'
V["fig_lo_y"] = f'{float(_fy(_f_hat(V["lin_root_lo"]))):.1f}'
V["fig_ev_x"] = f'{float(_fx(LIN_S)):.1f}'
V["fig_ev_y"] = f'{float(_fy(_f_ev(LIN_S))):.1f}'
for _t in (0.01, 0.1, 1.0, 10.0):
    V[f"fig_tick_{str(_t).replace('.', '_')}"] = f"{float(_fx(_t)):.1f}"
for _t in (-5, -4, -3, -2, -1):
    V[f"fig_ytick_{-_t}"] = f"{float(_fy(_t)):.1f}"
# the evidence peak is where the derivative vanishes: x + vu = S + vu
_xe = _xs_fig[np.argmax(_f_ev(_xs_fig))]
assert abs(log(_xe) - log(LIN_S)) < 0.05

# ── the nonlinear world: stationary points of the trial-averaged updates ──
# Expectations over the world by quadrature: Gauss-Legendre in sqrt(d) over the
# whole of (0, loc + 9 sd], and Gauss-Hermite in the sensory noise. The settled
# state jumps between branches at some noise values, so the noise rule needs
# far more nodes than a smooth integrand would: 20 moved the fifth digit of
# some stationary points. `python3 values.py --refine` repeats them at
# (300, 160) nodes.
QUAD = dict(nd=200, nz=120)


def world_nodes(loc, var, th, vu, nd=None, nz=None):
    nd = nd or QUAD["nd"]
    nz = nz or QUAD["nz"]
    zn, zw = np.polynomial.hermite_e.hermegauss(nz)
    zw = zw / zw.sum()
    hi = loc + 9 * sqrt(var)
    x, w = np.polynomial.legendre.leggauss(nd)
    s = (x + 1) / 2 * sqrt(hi)
    d = s * s
    wd = w / 2 * sqrt(hi) * 2 * s
    pd = np.exp(-(d - loc) ** 2 / (2 * var)) / sqrt(2 * pi * var) / zp(loc, var)
    W = (wd * pd)[:, None] * zw[None, :]
    U = th / d[:, None] ** 2 + sqrt(vu) * zn[None, :]
    return U.ravel(), W.ravel() / W.sum()


def map_vec(u, dp, vp, th, vu, kmax=20000):
    """The settled state for many observations at once: the same capped ascent
    as `settle`, from the same start, until every step is below 1e-5 of the
    state, then Newton's method on the exact curvature until every correction
    is below 1e-14 of it. The tolerance is relative because a source a
    micrometre away gives u near 1e12, where an absolute one is never met."""
    d = np.full_like(u, dp)
    act = np.arange(u.size)                      # observations still climbing
    for _ in range(kmax):
        x, ua = d[act], u[act]
        gp = -2 * th / x ** 3
        grad = (dp - x) / vp + (ua - th / x ** 2) / vu * gp
        cap = 1 / (1 / vp + gp * gp / vu)
        step = np.maximum(np.minimum(ETA, cap) * grad, -0.5 * x)
        d[act] = x + step
        act = act[np.abs(step) > 1e-5 * d[act]]
        if act.size == 0:
            break
    else:
        raise RuntimeError("map_vec did not settle")
    for _ in range(30):
        res = u - th / d ** 2
        g1 = (dp - d) / vp + res / vu * (-2 * th / d ** 3)
        g2 = -1 / vp - 4 * th * th / d ** 6 / vu + res / vu * 6 * th / d ** 4
        assert np.all(g2 < 0), "settled at a point that is not a maximum"
        delta = g1 / g2
        d = d - delta
        if np.all(np.abs(delta) <= 1e-14 * d):
            return d
    raise RuntimeError("Newton polish did not converge")


def post_moments_vec(u, dp, vp, th, vu, ny=NY):
    """Posterior averages of d, d^2, the squared residual and the residual
    times h(d) = 1/d^2, for many u at once, by the two routes of post_r2."""
    if u.size > 3000:
        return np.concatenate([post_moments_vec(u[i:i + 3000], dp, vp, th, vu, ny)
                               for i in range(0, u.size, 3000)], axis=1)
    sd = sqrt(vu)
    out = np.zeros((5, u.size))
    near = u - 10 * sd > th / (dp + 12 * sqrt(vp)) ** 2
    t = np.linspace(0, 1, ny)
    tw = np.ones(ny)
    tw[0] = tw[-1] = 0.5
    if near.any():
        ua = u[near]
        rho = -10 * sd + 20 * sd * t
        y = ua[:, None] - rho[None, :]
        d = np.sqrt(th / y)
        lw = -rho[None, :] ** 2 / (2 * vu) - (d - dp) ** 2 / (2 * vp) - 1.5 * np.log(y)
        res = np.broadcast_to(rho[None, :], y.shape)
        out[:, near] = _moments(lw, tw, d, res, th)
    if (~near).any():
        ub = u[~near]
        up = np.maximum(ub, 0.0)
        da = np.sqrt(th / (up + 12 * sd))
        c = 2 * th * vp * (th / dp ** 2 + np.abs(ub)) / vu
        db = dp + 12 * sqrt(vp) + c ** 0.25
        da = np.minimum(da, 0.5 * db)
        q = np.log(da)[:, None] + (np.log(db) - np.log(da))[:, None] * t[None, :]
        d = np.exp(q)
        res = ub[:, None] - th / d ** 2
        lw = -res ** 2 / (2 * vu) - (d - dp) ** 2 / (2 * vp) + q
        out[:, ~near] = _moments(lw, tw, d, res, th)
    return out


def _moments(lw, tw, d, res, th):
    p = np.exp(lw - lw.max(1, keepdims=True)) * tw[None, :]
    p /= p.sum(1, keepdims=True)
    return np.array([(p * d).sum(1), (p * d * d).sum(1), (p * res ** 2).sum(1),
                     (p / d ** 2).sum(1), (p * res / d ** 2).sum(1)])


from scipy.optimize import brentq  # noqa: E402


def point_grad(which, val, world, model, nodes=None):
    """The expected update of the point rule, averaged over the world. At an
    exact settled state the stationary condition gives the sensory residual as
    (u - theta/d^2)/Sigma_u = (d_p - d) d^3 / (2 theta Sigma_p), which is used
    in place of the subtraction: for a source very near, u and theta/d^2 agree
    to every digit a double carries."""
    m = dict(model)
    m[which] = val
    U, W = nodes if nodes is not None else world_nodes(*world)
    dh = map_vec(U, m["dp"], m["vp"], m["th"], m["vu"])
    eu = (m["dp"] - dh) * dh ** 3 / (2 * m["th"] * m["vp"])     # epsilon_u at the mode
    if which == "dp":
        return float(np.sum(W * (dh - m["dp"])) / m["vp"] - trunc_ratio_exact(m["dp"], m["vp"]))
    if which == "th":
        return float(np.sum(W * eu / dh ** 2))
    if which == "vu":
        return float(np.sum(W * 0.5 * (eu ** 2 * m["vu"] - 1)))
    if which == "vp":
        return float(np.sum(W * 0.5 * ((dh - m["dp"]) ** 2 / m["vp"] - 1))
                     + 0.5 * m["dp"] * trunc_ratio_exact(m["dp"], m["vp"]))


def post_grad(which, val, world, model, nodes=None):
    """The same expected update with the posterior average in place of the mode."""
    m = dict(model)
    m[which] = val
    U, W = nodes if nodes is not None else world_nodes(*world)
    Ed, Ed2, Er2, Eh, Erh = post_moments_vec(U, m["dp"], m["vp"], m["th"], m["vu"])
    if which == "dp":
        return float(np.sum(W * (Ed - m["dp"])) / m["vp"] - trunc_ratio_exact(m["dp"], m["vp"]))
    if which == "th":
        return float(np.sum(W * Erh) / m["vu"])
    if which == "vu":
        return float(np.sum(W * 0.5 * (Er2 / m["vu"] - 1)))
    if which == "vp":
        dev2 = Ed2 - 2 * m["dp"] * Ed + m["dp"] ** 2
        return float(np.sum(W * 0.5 * (dev2 / m["vp"] - 1))
                     + 0.5 * m["dp"] * trunc_ratio_exact(m["dp"], m["vp"]))


def trunc_ratio_exact(dp, vp):
    return npdf(0, dp, vp) / zp(dp, vp)


_WW2 = (D_PRIOR, VAR_PRIOR, THETA, VAR_OBS)       # the world equal to Week 2's model

# prior location, world = Week 2's model: point rule stationary point, and the
# posterior-averaged rule checked to vanish at the truth
V["stat_dp_point"] = brentq(lambda x: point_grad("dp", x, _WW2, WEEK2), 1.0, 3.0, xtol=1e-9)
V["stat_dp_pct"] = 100 * (V["stat_dp_point"] / D_PRIOR - 1)
V["post_grad_dp_truth"] = post_grad("dp", D_PRIOR, _WW2, WEEK2)
assert abs(V["post_grad_dp_truth"]) < 2e-3, V["post_grad_dp_truth"]
V["point_grad_dp_truth"] = point_grad("dp", D_PRIOR, _WW2, WEEK2)

# widget 1's setting: world location 3, sharp ear
_WW1 = (W1["loc"], VAR_PRIOR, THETA, W1["vu"])
_M1 = dict(WEEK2, vu=W1["vu"])
V["w1_stat_point"] = brentq(lambda x: point_grad("dp", x, _WW1, _M1), 1.5, 4.0, xtol=1e-9)
V["w1_post_grad_truth"] = post_grad("dp", W1["loc"], _WW1, _M1)
assert abs(V["w1_post_grad_truth"]) < 2e-3, V["w1_post_grad_truth"]
# and with Week 2's ordinary ear
_M1b = dict(WEEK2)
V["w1_stat_point_vu1"] = brentq(lambda x: point_grad("dp", x, (W1["loc"], VAR_PRIOR, THETA, VAR_OBS), _M1b),
                                1.0, 4.0, xtol=1e-9)

# link parameter: widget 2's world, theta* = 2
_WW2b = (D_PRIOR, VAR_PRIOR, W2["th_world"], VAR_OBS)
V["stat_th_point"] = brentq(lambda x: point_grad("th", x, _WW2b, WEEK2), 1.0, 3.0, xtol=1e-9)
V["stat_th_pct"] = 100 * (V["stat_th_point"] / W2["th_world"] - 1)
V["post_grad_th_truth"] = post_grad("th", W2["th_world"], _WW2b, WEEK2)
assert abs(V["post_grad_th_truth"]) < 2e-3, V["post_grad_th_truth"]
# and in the world equal to Week 2's model, theta* = 1, for the table in Lesson 4
V["stat_th_point_w2"] = brentq(lambda x: point_grad("th", x, _WW2, WEEK2), 0.5, 1.5, xtol=1e-9)
V["stat_th_pct_w2"] = 100 * (V["stat_th_point_w2"] / THETA - 1)
assert abs(post_grad("th", THETA, _WW2, WEEK2)) < 2e-3

# sensory variance, world = Week 2's model and widget 3's noisier channel
V["stat_vu_point"] = brentq(lambda x: point_grad("vu", x, _WW2, WEEK2), 1e-3, 0.1, xtol=1e-10)
V["stat_vu_pct"] = 100 * (V["stat_vu_point"] / VAR_OBS - 1)
assert abs(post_grad("vu", VAR_OBS, _WW2, WEEK2)) < 2e-3
W3 = dict(seed=7, vu_world=0.3, beta=0.01)
_WW3 = (D_PRIOR, VAR_PRIOR, THETA, W3["vu_world"])
V["w3_vu_world"], V["w3_beta"] = W3["vu_world"], W3["beta"]
V["w3_stat_point"] = brentq(lambda x: point_grad("vu", x, _WW3, WEEK2), 1e-3, 0.3, xtol=1e-10)
V["w3_post_grad_truth"] = post_grad("vu", W3["vu_world"], _WW3, WEEK2)
assert abs(V["w3_post_grad_truth"]) < 2e-3, V["w3_post_grad_truth"]

# prior variance, world = Week 2's model: scan the sign of the expected point update
_vps = np.geomspace(1e-4, 100.0, 25)
_sg = np.array([point_grad("vp", x, _WW2, WEEK2) for x in _vps])
_roots = []
for i in range(len(_vps) - 1):
    if _sg[i] * _sg[i + 1] < 0:
        _roots.append(brentq(lambda x: point_grad("vp", x, _WW2, WEEK2), _vps[i], _vps[i + 1], xtol=1e-10))
V["stat_vp_n_roots"] = len(_roots)
V["n_vp_scan"] = len(_vps)
# the lesson says the sampled drift changes sign once, from positive to negative
assert len(_roots) == 1 and _sg[0] > 0 > _sg[-1], (_roots, _sg[0], _sg[-1])
V["stat_vp_point"] = _roots[-1] if _roots else float("nan")
V["stat_vp_pct"] = 100 * (V["stat_vp_point"] / VAR_PRIOR - 1)
if len(_roots) > 1:
    V["stat_vp_point_lo"] = _roots[0]
assert abs(post_grad("vp", VAR_PRIOR, _WW2, WEEK2)) < 2e-3

# ── widget 3 · precision-learning, at its default settings ───────────────
_h3p, _e3p, _, _bad3p = run(W3["seed"], N_TRIALS, _WW3, dict(WEEK2), "vu", W3["beta"], "point")
_h3e, _e3e, _, _bad3e = run(W3["seed"], N_TRIALS, _WW3, dict(WEEK2), "vu", W3["beta"], "post")
V["w3_point_final"] = _h3p[-1]
V["w3_point_mean_late"] = float(np.mean(_h3p[N_TRIALS // 2:]))
V["w3_post_final"] = _h3e[-1]
V["w3_post_mean_late"] = float(np.mean(_h3e[N_TRIALS // 2:]))
WIDGET_REF["w3"] = {"point_final": _h3p[-1], "point_mean_late": V["w3_point_mean_late"],
                    "post_final": _h3e[-1], "post_mean_late": V["w3_post_mean_late"]}


def ear_share(vu, d=D_PRIOR, vp=VAR_PRIOR, th=THETA):
    """The ear's share of the Gauss-Newton curvature at distance d."""
    s = (2 * th / d ** 3) ** 2 / vu
    return s / (1 / vp + s)


V["share_w2"] = ear_share(VAR_OBS)
V["share_world3"] = ear_share(W3["vu_world"])
V["share_point3"] = ear_share(V["w3_point_mean_late"])
V["share_post3"] = ear_share(V["w3_post_mean_late"])

# ── the prior variance collapsing, simulated ─────────────────────────────
_h4, _e4, _, _bad4 = run(W3["seed"], N_TRIALS, _WW2, dict(WEEK2), "vp", W3["beta"])
V["vp_sim_mean_late"] = float(np.mean(_h4[N_TRIALS // 2:]))
V["vp_sim_final"] = _h4[-1]

# ── the noisier ear, by expected drift rather than one run ───────────────
V["w3_share_stat"] = ear_share(V["w3_stat_point"])
V["w3_grad_start"] = point_grad("vu", VAR_OBS, _WW3, WEEK2)   # in log Sigma_u
assert V["w3_stat_point"] > VAR_OBS and V["w3_grad_start"] > 0

# ── the sharper ear of widget 1, and why it still falls short ────────────
V["w1_stat_pct"] = 100 * (V["w1_stat_point"] / W1["loc"] - 1)
V["w1_stat_pct_vu1"] = 100 * (V["w1_stat_point_vu1"] / W1["loc"] - 1)
V["w1_ear_prec"] = (2 * THETA / W1["loc"] ** 3) ** 2 / W1["vu"]      # g'(3)^2 / Sigma_u
V["w2_ear_prec"] = (2 * THETA / D_PRIOR ** 3) ** 2 / VAR_OBS

# ── the scale direction: the settled log joint falls by ln c ─────────────
# 2 theta d/dtheta + d_p d/dd_p + 2 Sigma_p d/dSigma_p of ell_hat equals -1
_e5 = 1e-5
_dl = {k: (ell_hat(U_OBS, **{k: b * (1 + _e5)}) - ell_hat(U_OBS, **{k: b * (1 - _e5)})) / (2 * _e5)
       for k, b in (("th", THETA), ("dp", D_PRIOR), ("vp", VAR_PRIOR))}
V["scale_identity"] = 2 * _dl["th"] + _dl["dp"] + 2 * _dl["vp"]     # each term is b * dl/db
assert abs(V["scale_identity"] + 1) < 1e-6, V["scale_identity"]
V["ell_hat_scaled"] = ell_hat(U_OBS, dp=SCALE_C * D_PRIOR, vp=SCALE_C ** 2 * VAR_PRIOR, th=SCALE_C ** 2 * THETA)
assert abs(V["ell_hat_scaled"] - (V["ell_hat_w2"] - log(SCALE_C))) < 1e-9

# ── a world slightly quieter than the model, theta* = 0.99 ───────────────
V["q99_th_world"] = 0.99
V["q99_grad"] = point_grad("th", THETA, (D_PRIOR, VAR_PRIOR, 0.99, VAR_OBS), WEEK2)
assert V["q99_grad"] > 0
V["q50_grad"] = point_grad("th", THETA, (D_PRIOR, VAR_PRIOR, 0.5, VAR_OBS), WEEK2)
assert V["q50_grad"] < 0

# ── every reported run settled, and settled on the highest mode ──────────
_runs = [(_bad1, _e1, _m, "dp"), (_bad2, _e2, WEEK2, "th"), (_bad2off, _e2off, WEEK2, "th"),
         (_bad2true, _e2true, dict(WEEK2, th=W2["th_world"]), "th"), (_badq, _eq, WEEK2, "th"),
         (_badqoff, _eqoff, WEEK2, "th"), (_bad3p, _e3p, WEEK2, "vu"), (_bad3e, _e3e, WEEK2, "vu"),
         (_bad4, _e4, WEEK2, "vp")]
assert all(b == 0 for b, *_ in _runs), "a reported run has a trial that did not settle"
V["n_runs_checked"] = len(_runs)
V["mode_gap_max"] = max(mode_gap(list(e), m, k) for _, e, m, k in _runs)
assert V["mode_gap_max"] < 1e-8, V["mode_gap_max"]

# ── the second-order expansion of the power's gap, against the exact gap ─
# Delta(u) = E_{d|u}[psi] - psi(d_hat), psi = eps_u h(d), h = 1/d^2, averaged over
# a world equal to the model, at theta* = 1 and theta* = 2. The expansion
# psi'(d_hat)(m - d_hat) + psi''(d_hat)[V + (m - d_hat)^2]/2 is split into its
# offset and variance parts. The residual at the mode comes from the stationary
# identity, as in point_grad.
def dev_moments_vec(u, dh, rh, dp, vp, th, vu, ny=NY):
    """Posterior averages of d - d_hat and (d - d_hat)^2 on the grids of
    post_moments_vec, with rh the residual at the mode. On the residual route
    d - d_hat = sqrt(theta)(rho - rh) / (sqrt(y) sqrt(y_hat) (sqrt(y) + sqrt(y_hat))),
    with y = u - rho and y_hat = u - rh, so that no two nearly equal distances
    are subtracted when the source is very near."""
    if u.size > 3000:
        return np.concatenate([dev_moments_vec(u[i:i + 3000], dh[i:i + 3000], rh[i:i + 3000],
                                               dp, vp, th, vu, ny)
                               for i in range(0, u.size, 3000)], axis=1)
    sd = sqrt(vu)
    out = np.zeros((2, u.size))
    near = u - 10 * sd > th / (dp + 12 * sqrt(vp)) ** 2
    t = np.linspace(0, 1, ny)
    tw = np.ones(ny)
    tw[0] = tw[-1] = 0.5
    if near.any():
        ua, ra = u[near], rh[near]
        rho = -10 * sd + 20 * sd * t
        y = ua[:, None] - rho[None, :]
        yh = (ua - ra)[:, None]
        d = np.sqrt(th / y)
        lw = -rho[None, :] ** 2 / (2 * vu) - (d - dp) ** 2 / (2 * vp) - 1.5 * np.log(y)
        dev = sqrt(th) * (rho[None, :] - ra[:, None]) / (np.sqrt(y * yh) * (np.sqrt(y) + np.sqrt(yh)))
        p = np.exp(lw - lw.max(1, keepdims=True)) * tw[None, :]
        p /= p.sum(1, keepdims=True)
        out[:, near] = [(p * dev).sum(1), (p * dev * dev).sum(1)]
    if (~near).any():
        ub, db_ = u[~near], dh[~near]
        up = np.maximum(ub, 0.0)
        da = np.sqrt(th / (up + 12 * sd))
        c = 2 * th * vp * (th / dp ** 2 + np.abs(ub)) / vu
        db = dp + 12 * sqrt(vp) + c ** 0.25
        da = np.minimum(da, 0.5 * db)
        q = np.log(da)[:, None] + (np.log(db) - np.log(da))[:, None] * t[None, :]
        d = np.exp(q)
        res = ub[:, None] - th / d ** 2
        lw = -res ** 2 / (2 * vu) - (d - dp) ** 2 / (2 * vp) + q
        dev = d - db_[:, None]
        p = np.exp(lw - lw.max(1, keepdims=True)) * tw[None, :]
        p /= p.sum(1, keepdims=True)
        out[:, ~near] = [(p * dev).sum(1), (p * dev * dev).sum(1)]
    return out


def power_gaps(th_world):
    world = (D_PRIOR, VAR_PRIOR, th_world, VAR_OBS)
    m = dict(WEEK2, th=th_world)
    U, W = world_nodes(*world)
    th, vu, dp, vp = m["th"], m["vu"], m["dp"], m["vp"]
    dh = map_vec(U, dp, vp, th, vu)
    eu = (dp - dh) * dh ** 3 / (2 * th * vp)          # residual at the mode / Sigma_u
    Erh = post_moments_vec(U, dp, vp, th, vu)[4]
    off, sq = dev_moments_vec(U, dh, eu * vu, dp, vp, th, vu)   # E[d - d_hat], E[(d - d_hat)^2]
    h, h1, h2 = dh ** -2, -2 * dh ** -3, 6 * dh ** -4
    p1 = h1 * eu - th * h * h1 / vu
    p2 = h2 * eu - th * (h * h2 + 2 * h1 * h1) / vu
    exact = float(np.sum(W * (Erh / vu - eu * h)))
    t_off = float(np.sum(W * p1 * off))
    t_var = float(np.sum(W * 0.5 * p2 * sq))
    return exact, t_off, t_var


for _k, _t in (("w2", THETA), ("loud", W2["th_world"])):
    _ex, _of, _va = power_gaps(_t)
    V[f"gap_exact_{_k}"], V[f"gap_off_{_k}"], V[f"gap_var_{_k}"] = _ex, _of, _va
    V[f"gap_two_{_k}"] = _of + _va
# the exact gaps have the signs of the table; the second-order sum does not in
# the louder world
assert V["gap_exact_w2"] < 0 < V["gap_exact_loud"] and V["gap_two_loud"] < 0
assert V["gap_off_loud"] > 0 > V["gap_var_loud"]

# ── one sound at which the mode lies further from d_p than the posterior does
V["sqdev_u"] = 1.0
V["sqdev_mode"] = (map_exact(V["sqdev_u"]) - D_PRIOR) ** 2
V["sqdev_post"] = float(np.trapezoid((_G - D_PRIOR) ** 2 * _post_grid(V["sqdev_u"]), _G))
assert V["sqdev_mode"] > V["sqdev_post"]

# ── hyperprior: inverse-gamma prior on a variance as n0 pseudo-trials ────
# maximising sum_n ell_n + ln p(Sigma) at fixed residuals gives
# Sigma = (sum r^2 + n0 Sigma0) / (N + n0 + 2)
V["hp_n0"], V["hp_s0"] = 4, 0.1

VALUES = V


def _refine():
    """The published stationary points again at (300, 160) quadrature nodes."""
    for name, which, world, model, a, b in (
            ("stat_dp_point", "dp", _WW2, WEEK2, 1.0, 3.0),
            ("stat_vp_point", "vp", _WW2, WEEK2, 0.2, 0.6),
            ("stat_vu_point", "vu", _WW2, WEEK2, 0.01, 0.05),
            ("stat_th_point_w2", "th", _WW2, WEEK2, 0.9, 1.3),
            ("stat_th_point", "th", _WW2b, WEEK2, 1.0, 3.0),
            ("w1_stat_point", "dp", _WW1, _M1, 1.5, 4.0),
            ("w3_stat_point", "vu", _WW3, WEEK2, 0.05, 0.2)):
        nodes = world_nodes(*world, nd=300, nz=160)
        x = brentq(lambda t: point_grad(which, t, world, model, nodes), a, b, xtol=1e-11)
        print(f"  {name:<20} {V[name]:.8f}  refined {x:.8f}  rel {abs(x / V[name] - 1):.1e}")
    QUAD.update(nd=300, nz=160)
    for k, t in (("w2", THETA), ("loud", W2["th_world"])):
        for name, x in zip(("exact", "off", "var"), power_gaps(t)):
            key = f"gap_{name}_{k}"
            print(f"  {key:<20} {V[key]:.8f}  refined {x:.8f}  rel {abs(x / V[key] - 1):.1e}")


if __name__ == "__main__":
    import sys
    if "--refine" in sys.argv:
        _refine()
        sys.exit()
    for k in sorted(VALUES):
        if not k.startswith("fig_"):
            print(f"  {k:<24} {VALUES[k]}")
