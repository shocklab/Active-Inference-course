"""Every derived number in Week 2, computed rather than typed.

The running example: an animal hears a sound of known power and infers how far
away the source is. Intensity falls as the inverse square, so the link from the
hidden variable to the observation is nonlinear, which is the whole point.

Run standalone to see the values:  python3 content/week-02/values.py
"""
from math import log, pi, sqrt

import numpy as np

# ── the model ────────────────────────────────────────────────────────────
SOURCE_POWER = 1.0          # chosen so intensity is 1 at unit distance
D_PRIOR, VAR_PRIOR = 2.0, 1.0       # the animal expects sources about 2 units off
U_OBS, VAR_OBS = 0.50, 0.10         # the intensity it actually heard, and its noise


def g(d):
    """Observation function: intensity at distance d."""
    return SOURCE_POWER / d ** 2


def g_prime(d):
    return -2.0 * SOURCE_POWER / d ** 3


V = {
    "d_prior": D_PRIOR, "var_prior": VAR_PRIOR,
    "u_obs": U_OBS, "var_obs": VAR_OBS,
    "prec_prior": 1 / VAR_PRIOR, "prec_obs": 1 / VAR_OBS,
    "g_at_prior": g(D_PRIOR),
    "d_from_data_alone": sqrt(SOURCE_POWER / U_OBS),
}

# ── the exact posterior, by quadrature ───────────────────────────────────
_grid = np.linspace(0.05, 8.0, 600000)
_post = (np.exp(-(_grid - D_PRIOR) ** 2 / (2 * VAR_PRIOR))
         * np.exp(-(U_OBS - g(_grid)) ** 2 / (2 * VAR_OBS)))
_post /= np.trapezoid(_post, _grid)

V["post_mode"] = float(_grid[np.argmax(_post)])
V["post_mean"] = float(np.trapezoid(_grid * _post, _grid))
V["post_sd"] = float(sqrt(np.trapezoid((_grid - V["post_mean"]) ** 2 * _post, _grid)))
V["post_skew"] = float(np.trapezoid(((_grid - V["post_mean"]) / V["post_sd"]) ** 3 * _post, _grid))
V["mean_mode_gap"] = V["post_mean"] - V["post_mode"]
V["mean_mode_pct"] = 100 * V["mean_mode_gap"] / V["post_mode"]

# ── the two prediction errors at the optimum ─────────────────────────────
_phi = V["post_mode"]
V["err_prior"] = (D_PRIOR - _phi) / VAR_PRIOR
V["err_obs"] = (U_OBS - g(_phi)) / VAR_OBS
V["err_obs_weighted"] = V["err_obs"] * g_prime(_phi)
V["g_at_mode"] = g(_phi)
V["gprime_at_mode"] = g_prime(_phi)

V["err_sum"] = V["err_prior"] + V["err_obs_weighted"]

# The same two errors evaluated away from the optimum, at the prior mean, where
# the sensory term is large and the prior term is exactly zero.
V["err_prior_at_start"] = (D_PRIOR - D_PRIOR) / VAR_PRIOR
V["err_obs_at_start"] = (U_OBS - g(D_PRIOR)) / VAR_OBS
V["err_obs_weighted_at_start"] = V["err_obs_at_start"] * g_prime(D_PRIOR)
V["gprime_at_prior"] = g_prime(D_PRIOR)

# ── gradient ascent from the prior mean ──────────────────────────────────
def ascend(phi0=D_PRIOR, rate=0.05, steps=400):
    phi, path = phi0, [phi0]
    for _ in range(steps):
        phi += rate * ((D_PRIOR - phi) / VAR_PRIOR
                       + (U_OBS - g(phi)) / VAR_OBS * g_prime(phi))
        path.append(phi)
    return path


_path = ascend()
V["ascent_rate"] = 0.05
V["ascent_final"] = _path[-1]
V["ascent_steps_to_1pct"] = next(
    i for i, p in enumerate(_path) if abs(p - V["post_mode"]) < 0.01 * V["post_mode"])

# ── how large a step the ascent will tolerate ────────────────────────────
# Linearising the update about the peak gives d_{k+1} - d* = (1 + eta F'')(d_k - d*),
# so the iteration converges only while |1 + eta F''| < 1, that is eta < 2/|F''|.
# Found by driving the widget's sensory-variance slider to its low end and
# watching the estimate settle into a two-cycle instead of onto the mode.
def _curvature(d, var_obs=VAR_OBS, h=1e-5):
    def F(x):
        return (-(x - D_PRIOR) ** 2 / (2 * VAR_PRIOR)
                - (U_OBS - g(x)) ** 2 / (2 * var_obs))
    return (F(d + h) - 2 * F(d) + F(d - h)) / h ** 2


V["curv_at_mode"] = _curvature(V["post_mode"])
V["eta_max"] = 2.0 / abs(V["curv_at_mode"])

# ── what happens as the sensory channel is trusted more or less ──────────
for _tag, _vo in (("sharp", 0.01), ("loose", 1.0)):
    _p = (np.exp(-(_grid - D_PRIOR) ** 2 / (2 * VAR_PRIOR))
          * np.exp(-(U_OBS - g(_grid)) ** 2 / (2 * _vo)))
    _p /= np.trapezoid(_p, _grid)
    V[f"mode_{_tag}"] = float(_grid[np.argmax(_p)])
    V[f"var_obs_{_tag}"] = _vo
    V[f"curv_{_tag}"] = _curvature(V[f"mode_{_tag}"], var_obs=_vo)
    V[f"eta_max_{_tag}"] = 2.0 / abs(V[f"curv_{_tag}"])

# The two-cycle the widget falls into when the rate exceeds that bound.
_osc = D_PRIOR
for _ in range(400):
    _osc += 0.05 * ((D_PRIOR - _osc) / VAR_PRIOR
                    + (U_OBS - g(_osc)) / 0.01 * g_prime(_osc))
V["osc_lo"] = min(_osc, _osc + 0.05 * ((D_PRIOR - _osc) / VAR_PRIOR
                                       + (U_OBS - g(_osc)) / 0.01 * g_prime(_osc)))
V["osc_hi"] = max(_osc, _osc + 0.05 * ((D_PRIOR - _osc) / VAR_PRIOR
                                       + (U_OBS - g(_osc)) / 0.01 * g_prime(_osc)))

# ── figure paths, generated from the curves above, never drawn by hand ───
# A hand-drawn curve in an SVG is a typed number wearing a disguise: it looks
# like the computation and is not bound to it. These are emitted from the same
# arrays the statistics come from, so the picture cannot drift from the prose.
_FIG = dict(x0=60.0, x1=640.0, ytop=34.0, ybase=250.0, dmax=6.0, n=145)


def _svg_path(y_of_d, *, box=_FIG):
    """Sample a curve across the plot box and emit an SVG polyline `d` string.

    Each curve is scaled to its own maximum, so the figure compares shapes and
    not heights; the caption says so.
    """
    ds = np.linspace(box["dmax"] / box["n"], box["dmax"], box["n"])
    ys = y_of_d(ds)
    ys = ys / ys.max()
    px = box["x0"] + (ds / box["dmax"]) * (box["x1"] - box["x0"])
    py = box["ybase"] - ys * (box["ybase"] - box["ytop"])
    pts = [f"{a:.1f} {b:.1f}" for a, b in zip(px, py)]
    return "M" + " L".join(pts)


def _svg_x(d, *, box=_FIG):
    return f'{box["x0"] + (d / box["dmax"]) * (box["x1"] - box["x0"]):.1f}'


def _svg_y(y_of_d, d, *, box=_FIG):
    ds = np.linspace(box["dmax"] / box["n"], box["dmax"], box["n"])
    ymax = y_of_d(ds).max()
    frac = float(y_of_d(np.array([d]))[0] / ymax)
    return f'{box["ybase"] - frac * (box["ybase"] - box["ytop"]):.1f}'


_prior_f = lambda d: np.exp(-(d - D_PRIOR) ** 2 / (2 * VAR_PRIOR))
_lik_f = lambda d: np.exp(-(U_OBS - g(d)) ** 2 / (2 * VAR_OBS))
_post_f = lambda d: _prior_f(d) * _lik_f(d)

V["fig_prior_path"] = _svg_path(_prior_f)
V["fig_lik_path"] = _svg_path(_lik_f)
V["fig_post_path"] = _svg_path(_post_f)
V["fig_mode_x"] = _svg_x(V["post_mode"])
V["fig_mode_y"] = _svg_y(_post_f, V["post_mode"])
V["fig_mean_x"] = _svg_x(V["post_mean"])
V["fig_mean_y"] = _svg_y(_post_f, V["post_mean"])

# ── the linear special case, where everything is exact ───────────────────
# Replacing 1/d^2 by the identity makes the posterior exactly Gaussian, so the
# mode and the mean coincide and the whole 35% gap of Lesson 2 is attributable
# to the nonlinearity and to nothing else.
_lin_num = D_PRIOR / VAR_PRIOR + U_OBS / VAR_OBS
_lin_den = 1 / VAR_PRIOR + 1 / VAR_OBS
V["lin_mode"] = _lin_num / _lin_den
V["lin_var"] = 1 / _lin_den

# Checked against quadrature rather than asserted: an algebraic identity that
# has never been evaluated is a claim, not a result.
#
# On a WIDE grid, deliberately. The first version of this check reused the
# d > 0.05 grid the nonlinear posterior lives on and reported a mean of 0.6550
# against an algebraic 0.6364, with a skew of 0.24 where the algebra says zero.
# Nothing was wrong with the algebra: the linear posterior is centred at 0.64
# with a standard deviation of 0.30, so cutting it off at 0.05 removes a real
# part of its left tail. The check had been measuring the truncation.
_wide = np.linspace(-6.0, 8.0, 600000)
_lp = (np.exp(-(_wide - D_PRIOR) ** 2 / (2 * VAR_PRIOR))
       * np.exp(-(U_OBS - _wide) ** 2 / (2 * VAR_OBS)))
_lp /= np.trapezoid(_lp, _wide)
V["lin_mean_quad"] = float(np.trapezoid(_wide * _lp, _wide))
V["lin_mode_quad"] = float(_wide[np.argmax(_lp)])
_lin_sd = sqrt(np.trapezoid((_wide - V["lin_mean_quad"]) ** 2 * _lp, _wide))
V["lin_sd_quad"] = float(_lin_sd)
V["lin_skew_quad"] = float(np.trapezoid(
    ((_wide - V["lin_mean_quad"]) / _lin_sd) ** 3 * _lp, _wide))

# How much of the linear posterior the physical domain d > 0 would cut off,
# and how much of the nonlinear one. The Gaussian prior over a distance is a
# convenience, and this is the size of the bill.
V["lin_mass_below_zero"] = float(np.trapezoid(_lp[_wide < 0], _wide[_wide < 0]))
# The nonlinear posterior needs no such apology. As d falls towards zero, g(d)
# runs to infinity, so the squared sensory error does too and the likelihood
# kills the density long before the prior's unphysical tail can matter.
_near = _grid < 0.5
V["nonlin_mass_below_half"] = float(np.trapezoid(_post[_near], _grid[_near]))

# ── how much of the curvature the g'' term accounts for ──────────────────
def g_second(d):
    return 6.0 * SOURCE_POWER / d ** 4


V["gsecond_at_mode"] = g_second(V["post_mode"])
V["curv_gaussnewton"] = -(1 / VAR_PRIOR + g_prime(V["post_mode"]) ** 2 / VAR_OBS)
V["curv_gpp_term"] = V["err_obs"] * g_second(V["post_mode"])
V["sensory_prec_on_state"] = g_prime(V["post_mode"]) ** 2 / VAR_OBS

# ── two things that go wrong with the mode, used in the problems ─────────
# 1. The MAP is not invariant under reparameterisation. Working in rho = ln d
#    instead of d changes the density by the Jacobian |dd/drho| = d, which adds
#    ln d to the log posterior and moves the peak. The posterior MEAN is
#    unaffected, being an expectation rather than a location on a density.
_logpost_d = (-(_grid - D_PRIOR) ** 2 / (2 * VAR_PRIOR)
              - (U_OBS - g(_grid)) ** 2 / (2 * VAR_OBS))
_logpost_rho = _logpost_d + np.log(_grid)
V["map_in_d"] = float(_grid[np.argmax(_logpost_d)])
V["map_via_logd"] = float(_grid[np.argmax(_logpost_rho)])
V["map_reparam_pct"] = 100 * (V["map_via_logd"] - V["map_in_d"]) / V["map_in_d"]

# 2. The log joint need not be unimodal. These settings give two peaks, and
#    ascent from the prior mean climbs the LOWER one and stops there.
BI = dict(dp=4.0, vp=0.5, u=1.5, vu=0.1)
_bg = np.linspace(0.05, 10.0, 400000)
_bq = (-(_bg - BI["dp"]) ** 2 / (2 * BI["vp"])
       - (BI["u"] - g(_bg)) ** 2 / (2 * BI["vu"]))
_bq = np.exp(_bq - _bq.max())
_pk = np.where((_bq[1:-1] > _bq[:-2]) & (_bq[1:-1] > _bq[2:]))[0] + 1
_pk = [i for i in _pk if _bq[i] > 1e-6]
V["bi_dp"], V["bi_vp"], V["bi_u"], V["bi_vu"] = BI["dp"], BI["vp"], BI["u"], BI["vu"]
V["bi_peak_lo"] = float(_bg[_pk[0]])
V["bi_peak_hi"] = float(_bg[_pk[1]])
V["bi_height_ratio"] = float(_bq[_pk[1]] / _bq[_pk[0]])


BI_ETA = 0.02          # one definition; the prose cites V["bi_eta"]
V["bi_eta"] = BI_ETA


def _bi_ascend(start, eta=BI_ETA, steps=6000):
    d = start
    for _ in range(steps):
        d += eta * ((BI["dp"] - d) / BI["vp"]
                    + (BI["u"] - g(d)) / BI["vu"] * g_prime(d))
        d = min(max(d, 0.05), 10.0)
    return d


V["bi_from_prior"] = _bi_ascend(BI["dp"])
V["bi_from_one"] = _bi_ascend(1.0)
# Started below the global peak, the step lands beyond BOTH: near d = 0.6 the
# predicted intensity is far above what was heard and the gradient is enormous,
# so a rate that is mild elsewhere throws the estimate clear across the range.
V["bi_from_low"] = _bi_ascend(0.6)
BI_START_LOW = 0.6
V["bi_start_low"] = BI_START_LOW
V["bi_g_at_low"] = g(BI_START_LOW)
V["bi_grad_at_low"] = ((BI["dp"] - BI_START_LOW) / BI["vp"]
                       + (BI["u"] - g(BI_START_LOW)) / BI["vu"] * g_prime(BI_START_LOW))
V["bi_step_at_low"] = BI_ETA * V["bi_grad_at_low"]

# ── the sensory channel's precision about the state, at three distances ──
for _d in (1.0, 4.0):
    _tag = str(int(_d))
    V[f"gprime_at_{_tag}"] = g_prime(_d)
    V[f"sensprec_at_{_tag}"] = g_prime(_d) ** 2 / VAR_OBS
V["sensprec_ratio_1_to_4"] = V["sensprec_at_1"] / V["sensprec_at_4"]

VALUES = V

if __name__ == "__main__":
    for k in sorted(VALUES):
        print(f"  {k:<24} {VALUES[k]}")
