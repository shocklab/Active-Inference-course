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

# ── what happens as the sensory channel is trusted more or less ──────────
for _tag, _vo in (("sharp", 0.01), ("loose", 1.0)):
    _p = (np.exp(-(_grid - D_PRIOR) ** 2 / (2 * VAR_PRIOR))
          * np.exp(-(U_OBS - g(_grid)) ** 2 / (2 * _vo)))
    _p /= np.trapezoid(_p, _grid)
    V[f"mode_{_tag}"] = float(_grid[np.argmax(_p)])
    V[f"var_obs_{_tag}"] = _vo

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

VALUES = V

if __name__ == "__main__":
    for k in sorted(VALUES):
        print(f"  {k:<24} {VALUES[k]}")
