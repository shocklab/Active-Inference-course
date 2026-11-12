"""Every derived number in Week 4, computed rather than typed.

Lessons:
  4.1 Bounding surprise: Jensen's inequality, the three decompositions of F[Q, o] on
      Week 1's three-state model, point masses, and differential entropy on Week 2's
      linear model.
  4.2 Mean-field and variational messages: the noisy-OR model of Week 1 Lesson 4,
      variational against sum-product messages, a second sensor, symmetry breaking,
      and Normal-Gamma conjugacy.
  4.3 The Laplace approximation on Week 2's sound model, its error expansion, and a
      log-normal recognition density.
  4.4 Learning with free energy: EM on Week 3's prior variance, and where the point,
      Laplace and exact rules come to rest on Week 2's model (laplace_learning.py).
  4.5 A mean-field learning failure: the two-cause model of twocause.py.
  4.6 Problems.

NumPy and the standard library only. The two heavy computations, twocause.py and
laplace_learning.py, are cached in JSON beside this file, keyed by the SHA-256 of the
module that produces them, so a changed module recomputes on the next build.

    python3 content/week-04/values.py      # print every value
"""
import hashlib
import importlib.util
import json
import os
from math import cos, erf, exp, lgamma, log, pi, sqrt

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E_CONST = exp(1.0)
V = {}


# ── cached heavy computations ────────────────────────────────────────────
def _cached(modname):
    src = os.path.join(HERE, modname + ".py")
    with open(src, "rb") as fh:
        key = hashlib.sha256(fh.read()).hexdigest()
    cache = os.path.join(HERE, modname + "-cache.json")
    try:
        with open(cache, encoding="utf-8") as fh:
            stored = json.load(fh)
        if stored.get("key") == key:
            return stored["data"]
    except (OSError, ValueError):
        pass
    spec = importlib.util.spec_from_file_location("w04_" + modname, src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = mod.summary()
    data = json.loads(json.dumps(data, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"key": key, "data": data}, fh)
    return data


# ── special functions ────────────────────────────────────────────────────
def digamma_np(x):
    """digamma by recurrence to x >= 10 and the asymptotic series."""
    x = np.asarray(x, dtype=float).copy()
    out = np.zeros_like(x)
    while np.any(x < 10.0):
        mask = x < 10.0
        out = np.where(mask, out - 1.0 / x, out)
        x = np.where(mask, x + 1.0, x)
    inv = 1.0 / x
    inv2 = inv * inv
    return out + (np.log(x) - 0.5 * inv
                  - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0))))


for _xt in (0.35, 1.0, 2.5, 12.0):
    _fd = (lgamma(_xt + 1e-5) - lgamma(_xt - 1e-5)) / 2e-5
    assert abs(float(digamma_np(_xt)) - _fd) < 1e-9


def mulberry32(seed):
    """32-bit seeded generator shared bit for bit with assets/js/w04.js."""
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


def gl(a, b, npan, order=16):
    """Composite Gauss-Legendre nodes and weights on [a, b]."""
    gx, gw = np.polynomial.legendre.leggauss(order)
    br = np.linspace(a, b, npan + 1)
    lo, hi = br[:-1, None], br[1:, None]
    return (0.5 * (lo + hi) + 0.5 * (hi - lo) * gx).ravel(), (0.5 * (hi - lo) * gw).ravel()


# ── figures: every coordinate below is computed ──────────────────────────
FONT = "font-family=\"'IBM Plex Sans',sans-serif\""
INK, MUTED, FRAME, PAPER = "#3c3c42", "#6d6d75", "#e4e1d9", "#fdfcf9"
GREEN, CLAY, BLUE, PLUM, OCHRE = "#0f5f57", "#9c4d2f", "#2c5f8a", "#6b4c7a", "#a8741a"


class Panel:
    """A rectangle of an SVG mapped linearly to data coordinates."""

    def __init__(self, x0, x1, y0, y1, X0, X1, Y0, Y1):
        self.px = (x0, x1)
        self.py = (y0, y1)          # y0 is the bottom edge in pixels
        self.dx = (X0, X1)
        self.dy = (Y0, Y1)

    def X(self, x):
        return self.px[0] + (x - self.dx[0]) / (self.dx[1] - self.dx[0]) * (self.px[1] - self.px[0])

    def Y(self, y):
        return self.py[0] - (y - self.dy[0]) / (self.dy[1] - self.dy[0]) * (self.py[0] - self.py[1])

    def path(self, xs, ys, close=False):
        pts = [f"{self.X(x):.1f} {self.Y(y):.1f}" for x, y in zip(xs, ys)]
        return "M" + " L".join(pts) + (" Z" if close else "")

    def axes(self, xticks, yticks, xlabel, ylabel, fmt="{:g}"):
        x0, x1 = self.px
        yb, yt = self.py
        out = [f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}"/>',
               f'<line x1="{x0}" y1="{yb}" x2="{x0}" y2="{yt}" stroke="{INK}"/>']
        for t in xticks:
            x = self.X(t)
            out.append(f'<line x1="{x:.1f}" y1="{yb}" x2="{x:.1f}" y2="{yb + 4}" stroke="{INK}"/>')
            out.append(text(x, yb + 16, fmt.format(t).replace("-", "&#8722;")))
        for t in yticks:
            y = self.Y(t)
            out.append(f'<line x1="{x0 - 4}" y1="{y:.1f}" x2="{x0}" y2="{y:.1f}" stroke="{INK}"/>')
            out.append(text(x0 - 7, y + 3.5, fmt.format(t).replace("-", "&#8722;"), anchor="end"))
        out.append(text(0.5 * (x0 + x1), yb + 31, xlabel, weight="600"))
        out.append(f'<text x="{x0 - 34}" y="{0.5 * (yb + yt):.1f}" text-anchor="middle" {FONT} '
                   f'font-size="10.5" font-weight="600" fill="{MUTED}" '
                   f'transform="rotate(-90 {x0 - 34} {0.5 * (yb + yt):.1f})">{ylabel}</text>')
        return out


def text(x, y, s, anchor="middle", size=10.5, fill=MUTED, weight=None, style=None):
    w = f' font-weight="{weight}"' if weight else ""
    st = f' font-style="{style}"' if style else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" {FONT} font-size="{size}"'
            f'{w}{st} fill="{fill}">{s}</text>')


def svg(w, h, label, body):
    head = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="{label}">')
    frame = f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" fill="{PAPER}" stroke="{FRAME}"/>'
    return "\n".join([head, frame] + body + ["</svg>"])


def legend(x, y, items):
    out = []
    for i, (col, dash, lab) in enumerate(items):
        yy = y + 18 * i
        d = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x}" y1="{yy}" x2="{x + 22}" y2="{yy}" stroke="{col}" stroke-width="2"{d}/>')
        out.append(text(x + 28, yy + 4, lab, anchor="start", fill=INK))
    return out


def dot(p, x, y, col, r=4, hollow=False):
    fill = PAPER if hollow else col
    return f'<circle cx="{p.X(x):.1f}" cy="{p.Y(y):.1f}" r="{r}" fill="{fill}" stroke="{col}" stroke-width="1.6"/>'


# ═════════════════════════════════════════════════════════════════════════
# 4.1  BOUNDING SURPRISE
# ═════════════════════════════════════════════════════════════════════════
V["jen_q"] = 0.5
V["jen_ey"] = 0.5 * 1.0 + 0.5 * 9.0
V["jen_ln_ey"] = log(V["jen_ey"])
V["jen_e_lny"] = 0.5 * log(1.0) + 0.5 * log(9.0)
V["jen_gap"] = V["jen_ln_ey"] - V["jen_e_lny"]

# Week 1's three-state model (leopard, baboon, nothing)
A_MAT = np.array([
    [0.70, 0.15, 0.02],   # o = tawny flash
    [0.25, 0.75, 0.08],   # o = branch shakes
    [0.05, 0.10, 0.90],   # o = quiet
])
PRIOR_3 = np.array([0.08, 0.22, 0.70])
V["p_leop"], V["p_bab"], V["p_noth"] = [float(x) for x in PRIOR_3]
V["a_tawny_leop"], V["a_tawny_bab"], V["a_tawny_noth"] = [float(x) for x in A_MAT[0]]
_joint_tawny = A_MAT[0] * PRIOR_3
_ev_tawny = float(_joint_tawny.sum())
_post_tawny = _joint_tawny / _ev_tawny
V["ev_tawny"] = _ev_tawny
V["surp_tawny"] = -log(_ev_tawny)
V["post_leop"], V["post_bab"], V["post_noth"] = [float(x) for x in _post_tawny]
V["joint_leop"], V["joint_bab"], V["joint_noth"] = [float(x) for x in _joint_tawny]


def _decomp_3(q_vec, o_idx=0):
    q = np.asarray(q_vec, dtype=float)
    lik = A_MAT[o_idx]
    joint = lik * PRIOR_3
    ev = float(joint.sum())
    post = joint / ev
    m = q > 0
    U = float(-np.sum(q[m] * np.log(joint[m])))
    H = float(-np.sum(q[m] * np.log(q[m]))) + 0.0
    comp = float(np.sum(q[m] * np.log(q[m] / PRIOR_3[m])))
    acc = float(np.sum(q[m] * np.log(lik[m])))
    kl_post = float(np.sum(q[m] * np.log(q[m] / post[m])))
    F1, F2, F3 = U - H, comp - acc, kl_post - log(ev)
    assert abs(F1 - F2) < 1e-14 and abs(F1 - F3) < 1e-14
    return dict(U=U, H=H, comp=comp, acc=acc, klpost=kl_post, F=F1)


_CANDIDATES = {
    "prior": PRIOR_3,
    "unif": np.array([1.0, 1.0, 1.0]) / 3.0,
    "sharp": np.array([0.96, 0.02, 0.02]),
    "point": np.array([1.0, 0.0, 0.0]),
    "post": _post_tawny,
}
V["unif_third"] = 1.0 / 3.0
V["q_sharp_leop"] = 0.96
V["q_sharp_other"] = 0.02
for _tag, _qv in _CANDIDATES.items():
    for _k, _val in _decomp_3(_qv, 0).items():
        V[f"d_{_tag}_{_k}"] = _val
# the point mass on the MAP state: zero entropy, F = -ln P(o, s_MAP)
assert int(np.argmax(_joint_tawny)) == 0
assert abs(V["d_point_F"] + log(_joint_tawny[0])) < 1e-14
assert abs(V["d_point_klpost"] + log(_post_tawny[0])) < 1e-14
assert V["d_point_H"] == 0.0
# complexity does not involve o; the posterior divergence does
assert V["d_prior_comp"] == 0.0 and abs(V["d_post_klpost"]) < 1e-15

V["w1_def_q1"], V["w1_def_q2"], V["w1_def_q3"] = 0.35, 0.40, 0.25
_w1 = _decomp_3([0.35, 0.40, 0.25], 0)
V["w1_def_F"] = _w1["F"]
V["w1_def_klpost"] = _w1["klpost"]

_P_OS = A_MAT * PRIOR_3[None, :]
V["joint_entropy_P"] = float(-np.sum(_P_OS * np.log(_P_OS)))

# Week 2's model and its linear relative
D_PRIOR, VAR_PRIOR = 2.0, 1.0
U_OBS, VAR_OBS = 0.50, 0.10
THETA = 1.0
Z_P = 0.5 * (1.0 + erf(D_PRIOR / sqrt(2.0 * VAR_PRIOR)))
V["d_prior"] = D_PRIOR
V["var_prior"] = VAR_PRIOR
V["u_obs"] = U_OBS
V["var_obs"] = VAR_OBS
V["theta0"] = THETA
V["zp_w2"] = Z_P
V["gauss_zero_H_var"] = 1.0 / (2.0 * pi * E_CONST)
V["gauss_zero_H_sd"] = sqrt(V["gauss_zero_H_var"])

# narrow Gaussians N(d_hat, Sigma_q) on the linear Gaussian model d ~ N(2, 1), u | d ~ N(d, 0.1)
_lin_spost = 1.0 / (1.0 / VAR_PRIOR + 1.0 / VAR_OBS)
_lin_dhat = _lin_spost * (D_PRIOR / VAR_PRIOR + U_OBS / VAR_OBS)
_lin_ell_hat = (-(_lin_dhat - D_PRIOR) ** 2 / (2.0 * VAR_PRIOR) - 0.5 * log(2.0 * pi * VAR_PRIOR)
                - (U_OBS - _lin_dhat) ** 2 / (2.0 * VAR_OBS) - 0.5 * log(2.0 * pi * VAR_OBS))
V["lin_spost"] = _lin_spost
V["lin_dhat"] = _lin_dhat
V["lin_neg_ell_hat"] = -_lin_ell_hat
V["lin_surp"] = -_lin_ell_hat - 0.5 * log(2.0 * pi * _lin_spost)
for _tag, _eps in (("opt", _lin_spost), ("e2", 0.01), ("e4", 1e-4), ("e8", 1e-8)):
    _U_eps = -_lin_ell_hat + _eps / (2.0 * _lin_spost)
    _H_eps = 0.5 * log(2.0 * pi * E_CONST * _eps)
    V[f"spike_{_tag}_var"] = _eps
    V[f"spike_{_tag}_U"] = _U_eps
    V[f"spike_{_tag}_H"] = _H_eps
    V[f"spike_{_tag}_F"] = _U_eps - _H_eps
assert abs(V["spike_opt_F"] - V["lin_surp"]) < 1e-14

# fixed-width means on the asymmetric potential of Problem 5, ell(x) = -(x^2/2 + e^x):
# F(m) = (m^2 + S)/2 + exp(m + S/2) - H, minimised where m + exp(m + S/2) = 0
_xh = -0.567
for _ in range(40):
    _xh -= (_xh + exp(_xh)) / (1.0 + exp(_xh))
for _tag, _S in (("a", 0.5), ("b", 0.05), ("c", 0.005)):
    _m = _xh
    for _ in range(60):
        _m -= (_m + exp(_m + 0.5 * _S)) / (1.0 + exp(_m + 0.5 * _S))
    V[f"fw_{_tag}_S"] = _S
    V[f"fw_{_tag}_m"] = _m
    V[f"fw_{_tag}_shift"] = _m - _xh
    # second-order prediction of the shift: -S ell'''/(2 ell'') = -S e^x/(2(1 + e^x))
    V[f"fw_{_tag}_pred"] = -_S * exp(_xh) / (2.0 * (1.0 + exp(_xh)))
V["fw_mode"] = _xh


# ═════════════════════════════════════════════════════════════════════════
# 4.2  MEAN-FIELD AND VARIATIONAL MESSAGES
# ═════════════════════════════════════════════════════════════════════════
P_CAUSE, W_NOR, LEAK_NOR = 0.10, 0.90, 0.01
V["nor_p"], V["nor_w"], V["nor_leak"] = P_CAUSE, W_NOR, LEAK_NOR


def _nor_lik_o1(s1, s2):
    return 1.0 - (1.0 - W_NOR * s1) * (1.0 - W_NOR * s2) * (1.0 - LEAK_NOR)


_nor_joint = np.zeros((2, 2))
for _i in (0, 1):
    for _j in (0, 1):
        _p1 = P_CAUSE if _i else 1.0 - P_CAUSE
        _p2 = P_CAUSE if _j else 1.0 - P_CAUSE
        _nor_joint[_i, _j] = _p1 * _p2 * _nor_lik_o1(_i, _j)
_nor_ev = float(_nor_joint.sum())
_nor_post = _nor_joint / _nor_ev
V["nor_ev"] = _nor_ev
V["nor_surp"] = -log(_nor_ev)
V["nor_lik_00"] = _nor_lik_o1(0, 0)
V["nor_lik_01"] = _nor_lik_o1(0, 1)
V["nor_lik_11"] = _nor_lik_o1(1, 1)
V["nor_prior_logodds"] = log(P_CAUSE / (1.0 - P_CAUSE))
V["nor_post_00"] = float(_nor_post[0, 0])
V["nor_post_01"] = float(_nor_post[0, 1])
V["nor_post_10"] = float(_nor_post[1, 0])
V["nor_post_11"] = float(_nor_post[1, 1])
V["nor_marg_1"] = float(_nor_post[1, :].sum())


def _nor_vfe(q1, q2, joint_mat):
    qs = np.outer([1.0 - q1, q1], [1.0 - q2, q2])
    m = qs > 0
    return float(np.sum(qs[m] * np.log(qs[m] / joint_mat[m])))


def _upd_q1(q2, joint_mat):
    w2 = np.array([1.0 - q2, q2])
    lg = np.sum(w2[None, :] * np.log(joint_mat), axis=1)
    lg -= lg.max()
    p = np.exp(lg)
    return float(p[1] / p.sum())


def _upd_q2(q1, joint_mat):
    w1 = np.array([1.0 - q1, q1])
    lg = np.sum(w1[:, None] * np.log(joint_mat), axis=0)
    lg -= lg.max()
    p = np.exp(lg)
    return float(p[1] / p.sum())


# sequential coordinate ascent from (0.10, 0.10); every half-step lowers F by a KL divergence
_q1_s, _q2_s = 0.10, 0.10
V["cavi_s0_q1"], V["cavi_s0_q2"] = _q1_s, _q2_s
V["cavi_s0_F"] = _nor_vfe(_q1_s, _q2_s, _nor_joint)
_seq_path = [(_q1_s, _q2_s)]
for _it in range(1, 5):
    _q1_old = _q1_s
    _q1_s = _upd_q1(_q2_s, _nor_joint)
    _seq_path.append((_q1_s, _q2_s))
    _F_mid = _nor_vfe(_q1_s, _q2_s, _nor_joint)
    _kl1 = (1 - _q1_old) * log((1 - _q1_old) / (1 - _q1_s)) + _q1_old * log(_q1_old / _q1_s)
    assert abs((V[f"cavi_s{_it - 1}_F"] - _F_mid) - _kl1) < 1e-14
    _q2_old = _q2_s
    _q2_s = _upd_q2(_q1_s, _nor_joint)
    _seq_path.append((_q1_s, _q2_s))
    _F_end = _nor_vfe(_q1_s, _q2_s, _nor_joint)
    _kl2 = (1 - _q2_old) * log((1 - _q2_old) / (1 - _q2_s)) + _q2_old * log(_q2_old / _q2_s)
    assert abs((_F_mid - _F_end) - _kl2) < 1e-14
    V[f"cavi_s{_it}_q1"], V[f"cavi_s{_it}_q2"] = _q1_s, _q2_s
    V[f"cavi_s{_it}_F_mid"], V[f"cavi_s{_it}_F"] = _F_mid, _F_end
    if _it == 1:
        V["cavi_s1_kl1"], V["cavi_s1_kl2"] = _kl1, _kl2
_n_seq = 4
while True:
    _a = _upd_q1(_q2_s, _nor_joint)
    _b = _upd_q2(_a, _nor_joint)
    _n_seq += 1
    _done = abs(_a - _q1_s) < 1e-15 and abs(_b - _q2_s) < 1e-15
    _q1_s, _q2_s = _a, _b
    if _done or _n_seq > 500:
        break
assert _n_seq < 500
V["cavi_lim_q1"], V["cavi_lim_q2"] = _q1_s, _q2_s
V["cavi_lim_F"] = _nor_vfe(_q1_s, _q2_s, _nor_joint)
V["cavi_lim_gap"] = V["cavi_lim_F"] - V["nor_surp"]
V["cavi_lim_sweeps"] = _n_seq

# the map R(q) = upd(q) and its diagonal fixed point
_lo, _hi = 0.01, 0.99
for _ in range(80):
    _mid = 0.5 * (_lo + _hi)
    if _upd_q1(_mid, _nor_joint) > _mid:
        _lo = _mid
    else:
        _hi = _mid
_q_sad = 0.5 * (_lo + _hi)
assert abs(_upd_q1(_q_sad, _nor_joint) - _q_sad) < 1e-13
V["cavi_sad_q"] = _q_sad
V["cavi_sad_F"] = _nor_vfe(_q_sad, _q_sad, _nor_joint)
V["cavi_sad_gap"] = V["cavi_sad_F"] - V["nor_surp"]
_b0 = V["nor_prior_logodds"] + log(V["nor_lik_01"] / V["nor_lik_00"])
_b1 = log(V["nor_lik_11"] * V["nor_lik_00"] / V["nor_lik_01"] ** 2)
V["cavi_logit_int"], V["cavi_logit_slope"] = _b0, _b1
V["cavi_sad_rprime"] = _q_sad * (1.0 - _q_sad) * _b1
_fdj = (_upd_q1(_q_sad + 1e-6, _nor_joint) - _upd_q1(_q_sad - 1e-6, _nor_joint)) / 2e-6
assert abs(_fdj - V["cavi_sad_rprime"]) < 1e-8 and V["cavi_sad_rprime"] < -1.0

# simultaneous updates from the diagonal stay on it and settle on a two-cycle of R
V["cyc_start"] = 0.10
_qj = V["cyc_start"]
for _ in range(400):
    _qj = _upd_q1(_qj, _nor_joint)
_qj2 = _upd_q1(_qj, _nor_joint)
assert abs(_upd_q1(_qj2, _nor_joint) - _qj) < 1e-14
V["cyc_lo"], V["cyc_hi"] = min(_qj, _qj2), max(_qj, _qj2)
V["cyc_F_lo"] = _nor_vfe(V["cyc_lo"], V["cyc_lo"], _nor_joint)
V["cyc_F_hi"] = _nor_vfe(V["cyc_hi"], V["cyc_hi"], _nor_joint)
# the two-cycle's points are the coordinates of the asymmetric minima
assert abs(V["cyc_hi"] - max(_q1_s, _q2_s)) < 1e-12 and abs(V["cyc_lo"] - min(_q1_s, _q2_s)) < 1e-12
_rp_lo = V["cyc_lo"] * (1 - V["cyc_lo"]) * _b1
_rp_hi = V["cyc_hi"] * (1 - V["cyc_hi"]) * _b1
V["cyc_rp_lo"], V["cyc_rp_hi"], V["cyc_rp_prod"] = _rp_lo, _rp_hi, _rp_lo * _rp_hi
assert abs(_rp_lo * _rp_hi) < 1.0

# global minimum over a grid, then refined: the asymmetric minima
_qg = np.linspace(0.001, 0.999, 500)
_Q1g, _Q2g = np.meshgrid(_qg, _qg, indexing="ij")
_Fg = (
    (1 - _Q1g) * (1 - _Q2g) * np.log((1 - _Q1g) * (1 - _Q2g) / _nor_joint[0, 0])
    + (1 - _Q1g) * _Q2g * np.log((1 - _Q1g) * _Q2g / _nor_joint[0, 1])
    + _Q1g * (1 - _Q2g) * np.log(_Q1g * (1 - _Q2g) / _nor_joint[1, 0])
    + _Q1g * _Q2g * np.log(_Q1g * _Q2g / _nor_joint[1, 1])
)
_im = np.unravel_index(np.argmin(_Fg), _Fg.shape)
assert abs(float(_Fg[_im]) - V["cavi_lim_F"]) < 1e-3
assert float(_Fg.min()) > V["cavi_lim_F"] - 1e-12

# Figure 4.2a: F over the square of mean-field beliefs, and the cobweb of the update map R
_ms_spec = importlib.util.spec_from_file_location("w04_twocause_ms", os.path.join(HERE, "twocause.py"))
_ms_mod = importlib.util.module_from_spec(_ms_spec)
_ms_spec.loader.exec_module(_ms_mod)
_ngr = np.linspace(0.002, 0.998, 300)
_A1, _A2 = np.meshgrid(_ngr, _ngr, indexing="ij")
_Fsq = np.zeros_like(_A1)
for _i in (0, 1):
    for _j in (0, 1):
        _qi = _A1 if _i else 1 - _A1
        _qj2 = _A2 if _j else 1 - _A2
        _Fsq += _qi * _qj2 * np.log(_qi * _qj2 / _nor_joint[_i, _j])
_lev = [V["cavi_lim_F"] + 0.004, V["cavi_sad_F"], V["cavi_lim_F"] + 0.05, V["cavi_lim_F"] + 0.15,
        V["cavi_lim_F"] + 0.4, V["cavi_lim_F"] + 0.8, V["cavi_lim_F"] + 1.4]
_pL = Panel(60, 320, 300, 40, 0.0, 1.0, 0.0, 1.0)
_pR = Panel(400, 660, 300, 40, 0.0, 1.0, 0.0, 1.0)
_tk = [0, 0.25, 0.5, 0.75, 1]
_body = _pL.axes(_tk, _tk, "q1  (gust)", "q2  (baboon)") + _pR.axes(_tk, _tk, "q", "R(q)")
for _lv in _lev:
    for _line in _ms_mod.marching_squares(_ngr, _ngr, _Fsq, _lv):
        _xs, _ys = [p[0] for p in _line], [p[1] for p in _line]
        _sw = "1.4" if _lv == V["cavi_sad_F"] else "0.9"
        _body.append(f'<path d="{_pL.path(_xs, _ys)}" fill="none" stroke="{MUTED}" stroke-width="{_sw}" opacity="0.7"/>')
_sp = np.array(_seq_path + [(V["cavi_lim_q1"], V["cavi_lim_q2"])])
_body.append(f'<path d="{_pL.path(_sp[:, 0], _sp[:, 1])}" fill="none" stroke="{BLUE}" stroke-width="1.8"/>')
for _x, _y in _seq_path:
    _body.append(dot(_pL, _x, _y, BLUE, r=2.4))
_body.append(f'<path d="{_pL.path([V["cyc_lo"], V["cyc_hi"]], [V["cyc_lo"], V["cyc_hi"]])}" fill="none" '
             f'stroke="{CLAY}" stroke-width="1.4" stroke-dasharray="4 3"/>')
for _c in (V["cyc_lo"], V["cyc_hi"]):
    _body.append(dot(_pL, _c, _c, CLAY, r=4.5, hollow=True))
_body.append(dot(_pL, V["cavi_lim_q1"], V["cavi_lim_q2"], GREEN, r=4.5))
_body.append(dot(_pL, V["cavi_lim_q2"], V["cavi_lim_q1"], GREEN, r=4.5))
_body.append(dot(_pL, V["cavi_sad_q"], V["cavi_sad_q"], OCHRE, r=4.5))
# right panel: R, the diagonal, and the simultaneous iteration from q = 0.10 as a cobweb
_qq = np.linspace(0.0, 1.0, 201)
_Rq = np.array([_upd_q1(x, _nor_joint) for x in _qq])
_body.append(f'<path d="{_pR.path([0, 1], [0, 1])}" fill="none" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>')
_body.append(f'<path d="{_pR.path(_qq, _Rq)}" fill="none" stroke="{GREEN}" stroke-width="2"/>')
_cx, _cy = [V["cyc_start"]], [0.0]
_qc = V["cyc_start"]
for _ in range(14):
    _qn = _upd_q1(_qc, _nor_joint)
    _cx += [_qc, _qn]
    _cy += [_qn, _qn]
    _qc = _qn
_body.append(f'<path d="{_pR.path(_cx, _cy)}" fill="none" stroke="{CLAY}" stroke-width="1.3"/>')
_body.append(dot(_pR, V["cavi_sad_q"], V["cavi_sad_q"], OCHRE, r=4.5))
for _c in (V["cyc_lo"], V["cyc_hi"]):
    _body.append(dot(_pR, _c, _upd_q1(_c, _nor_joint), CLAY, r=4.5, hollow=True))
_body.append(text(190, 28, "free energy over the square of beliefs", fill=INK))
_body.append(text(530, 28, "the update map R and the simultaneous iteration", fill=INK))
_lgy = 352
_items = [(BLUE, "sequential updates from (0.1, 0.1)"), (GREEN, "minima"), (OCHRE, "saddle"),
          (CLAY, "two-cycle of simultaneous updates")]
_lx0 = [40, 290, 370, 450]
for (_col, _lab), _x0 in zip(_items, _lx0):
    _body.append(f'<circle cx="{_x0}" cy="{_lgy}" r="4" fill="{_col if _col != CLAY else PAPER}" stroke="{_col}" stroke-width="1.6"/>')
    _body.append(text(_x0 + 9, _lgy + 4, _lab, anchor="start", fill=INK))
V["fig_nor_svg"] = svg(680, 372, "Left: contours of the mean-field free energy over the two Bernoulli "
                       "beliefs, with the two minima, the saddle on the diagonal, the path of sequential "
                       "updates, and the two diagonal points visited by simultaneous updates. Right: the "
                       "update map R against the diagonal, with the cobweb of the simultaneous iteration "
                       "settling on a two-cycle.", _body)


# messages: sum-product against variational, on the factor P(o1 = 1 | s1, s2), with the
# incoming belief about s2 equal to its prior in both
_psi = np.array([[_nor_lik_o1(i, j) for j in (0, 1)] for i in (0, 1)])
_m2 = np.array([1.0 - P_CAUSE, P_CAUSE])
_sp = _psi @ _m2
_vm = np.exp(np.log(_psi) @ _m2)
assert np.all(_vm <= _sp)
V["msg_sp0"], V["msg_sp1"] = float(_sp[0]), float(_sp[1])
V["msg_vm0"], V["msg_vm1"] = float(_vm[0]), float(_vm[1])
V["msg_sp_lo"] = log(_sp[1] / _sp[0])
V["msg_vm_lo"] = log(_vm[1] / _vm[0])
V["msg_sp_q1"] = 1.0 / (1.0 + exp(-(V["nor_prior_logodds"] + V["msg_sp_lo"])))
V["msg_vm_q1"] = 1.0 / (1.0 + exp(-(V["nor_prior_logodds"] + V["msg_vm_lo"])))
assert abs(V["msg_sp_q1"] - V["nor_marg_1"]) < 1e-14
assert abs(V["msg_vm_q1"] - V["cavi_s1_q1"]) < 1e-14

# chain of Week 1 Lesson 5: K variables of k values
V["chain_K"], V["chain_k"] = 40, 4
V["chain_edges"] = V["chain_K"] - 1      # one factor per neighbouring pair
V["chain_terms"] = float(V["chain_k"] ** V["chain_K"])
V["chain_sp_ops"] = V["chain_edges"] * V["chain_k"] ** 2
V["chain_naive_terms"] = float(V["chain_k"] ** V["chain_edges"])
V["chain_vmp_ops"] = 2 * V["chain_edges"] * V["chain_k"] ** 2   # each pairwise factor sends a message both ways
assert V["chain_sp_ops"] == 624 and V["chain_vmp_ops"] == 1248

# Figure 4.2b: the graphical Markov blanket of one node in a directed acyclic graph.
# A schematic with no data in it; the node layout is chosen, and the blanket is computed
# from the edge list rather than marked by hand.
_dag_nodes = {"c": (150, 52), "z": (600, 52), "a": (230, 122), "b": (360, 122), "y": (520, 122),
              "x": (300, 196), "e": (190, 268), "d": (440, 268), "f": (590, 268)}
_dag_edges = [("c", "a"), ("a", "x"), ("b", "x"), ("z", "y"), ("x", "e"), ("x", "d"), ("y", "d"), ("d", "f")]
_par = {n for n, m in _dag_edges if m == "x"}
_chi = {m for n, m in _dag_edges if n == "x"}
_cop = {n for n, m in _dag_edges if m in _chi and n != "x"}
_blanket = _par | _chi | _cop
assert _blanket == {"a", "b", "e", "d", "y"}
_role = {**{n: "parent" for n in _par}, **{n: "child" for n in _chi}, **{n: "co-parent" for n in _cop}}
_body = [f'<defs><marker id="w04arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
         f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="{INK}"/></marker></defs>']
_R = 19
for _u, _v in _dag_edges:
    (_x0, _y0), (_x1, _y1) = _dag_nodes[_u], _dag_nodes[_v]
    _L = sqrt((_x1 - _x0) ** 2 + (_y1 - _y0) ** 2)
    _ux, _uy = (_x1 - _x0) / _L, (_y1 - _y0) / _L
    _body.append(f'<line x1="{_x0 + _R * _ux:.1f}" y1="{_y0 + _R * _uy:.1f}" x2="{_x1 - (_R + 2) * _ux:.1f}" '
                 f'y2="{_y1 - (_R + 2) * _uy:.1f}" stroke="{INK}" stroke-width="1.4" marker-end="url(#w04arr)"/>')
for _n, (_x, _y) in _dag_nodes.items():
    if _n == "x":
        _fill, _st, _tc = GREEN, GREEN, PAPER
    elif _n in _blanket:
        _fill, _st, _tc = "#dcebe7", GREEN, INK
    else:
        _fill, _st, _tc = PAPER, MUTED, MUTED
    _body.append(f'<circle cx="{_x}" cy="{_y}" r="{_R}" fill="{_fill}" stroke="{_st}" stroke-width="1.6"/>')
    _lab = 's<tspan baseline-shift="sub" font-size="9">i</tspan>' if _n == "x" else ""
    if _lab:
        _body.append(text(_x, _y + 4, _lab, size=13, fill=_tc, style="italic"))
    if _n in _role:
        _off = {"a": (-27, 4, "end"), "b": (27, 4, "start"), "y": (27, 4, "start"),
                "e": (0, 36, "middle"), "d": (0, 36, "middle")}[_n]
        _body.append(text(_x + _off[0], _y + _off[1], _role[_n], anchor=_off[2], fill=INK))
V["fig_blanket_svg"] = svg(680, 320, "A directed acyclic graph of nine variables. One node is filled; its two "
                           "parents, its two children and the other parent of one child are shaded and form its "
                           "Markov blanket; the remaining three nodes are unshaded.", _body)


# a second sensor on s1: leaves that rustle in a gust, observed quiet
V["leaf_hit"], V["leaf_fp"] = 0.85, 0.05
_leaf0 = np.array([1.0 - V["leaf_fp"], 1.0 - V["leaf_hit"]])
V["leaf_o0_s0"], V["leaf_o0_s1"] = float(_leaf0[0]), float(_leaf0[1])
V["leaf_msg_logodds"] = log(_leaf0[1] / _leaf0[0])
_nor2_joint = _nor_joint * _leaf0[:, None]
_nor2_ev = float(_nor2_joint.sum())
_nor2_post = _nor2_joint / _nor2_ev
V["nor2_surp"] = -log(_nor2_ev)
V["nor2_exact_s1"] = float(_nor2_post[1, :].sum())
V["nor2_exact_s2"] = float(_nor2_post[:, 1].sum())
V["nor_marg_2"] = float(_nor_post[:, 1].sum())


def _m32(q1):
    """Message from the factor P(o1 = 1 | s1, s2) to s2: q1-average of its log."""
    return np.array([(1 - q1) * log(_nor_lik_o1(0, j)) + q1 * log(_nor_lik_o1(1, j)) for j in (0, 1)])


_qa, _qb = 0.10, 0.10
for _it in range(1, 200):
    _qa = _upd_q1(_qb, _nor2_joint)
    if _it == 1:
        V["nor2_s1_q1"] = _qa
        _mm = _m32(_qa)
        V["m32_two_0"], V["m32_two_1"] = float(_mm[0]), float(_mm[1])
        V["m32_two_lo"] = float(_mm[1] - _mm[0])
    _qb = _upd_q2(_qa, _nor2_joint)
    if _it == 1:
        V["nor2_s1_q2"] = _qb
        V["nor2_s1_F"] = _nor_vfe(_qa, _qb, _nor2_joint)
_mm = _m32(V["cavi_s1_q1"])
V["m32_one_0"], V["m32_one_1"] = float(_mm[0]), float(_mm[1])
V["m32_one_lo"] = float(_mm[1] - _mm[0])
V["nor2_vmp_q1"], V["nor2_vmp_q2"] = _qa, _qb
V["nor2_vmp_F"] = _nor_vfe(_qa, _qb, _nor2_joint)
# the fixed point is the same from every start on a grid
_ends = []
for _s1 in np.linspace(0.025, 0.975, 20):
    for _s2 in np.linspace(0.025, 0.975, 20):
        _x, _y = _s1, _s2
        for _ in range(300):
            _x = _upd_q1(_y, _nor2_joint)
            _y = _upd_q2(_x, _nor2_joint)
        _ends.append((_x, _y))
_ends = np.array(_ends)
V["nor2_grid_starts"] = len(_ends)
V["nor2_grid_spread"] = float(np.max(np.abs(_ends - _ends[0])))
assert V["nor2_grid_spread"] < 1e-12

# Normal-Gamma: d ~ N(2, 1), Pi_u ~ Gamma(shape k0 = 2, rate r0 = 0.2), u | d, Pi_u ~ N(d, 1/Pi_u)
NG_DP, NG_VP, NG_K0, NG_R0, NG_U = 2.0, 1.0, 2.0, 0.20, 0.50
V["ng_dp"], V["ng_vp"], V["ng_k0"], V["ng_r0"], V["ng_u"] = NG_DP, NG_VP, NG_K0, NG_R0, NG_U
V["ng_prior_mean_pi"] = NG_K0 / NG_R0
# exact evidence: P(u | Pi) = N(u; dp, vp + 1/Pi), integrated against the Gamma prior in ln Pi
_lx, _lw = gl(log(1e-6), log(1e4), 400)
_pis = np.exp(_lx)
_lp = (NG_K0 * log(NG_R0) - lgamma(NG_K0) + NG_K0 * _lx - NG_R0 * _pis
       - 0.5 * np.log(2 * pi * (NG_VP + 1 / _pis)) - 0.5 * (NG_U - NG_DP) ** 2 / (NG_VP + 1 / _pis))
_mx = float(_lp.max())
V["ng_surp"] = -(_mx + log(float(np.sum(_lw * np.exp(_lp - _mx)))))
_lx2, _lw2 = gl(log(1e-8), log(1e5), 800)
_lp2 = (NG_K0 * log(NG_R0) - lgamma(NG_K0) + NG_K0 * _lx2 - NG_R0 * np.exp(_lx2)
        - 0.5 * np.log(2 * pi * (NG_VP + np.exp(-_lx2))) - 0.5 * (NG_U - NG_DP) ** 2 / (NG_VP + np.exp(-_lx2)))
assert abs(V["ng_surp"] + (float(_lp2.max()) + log(float(np.sum(_lw2 * np.exp(_lp2 - _lp2.max())))))) < 1e-12


def _ng_vfe(md, vd, kq, rq):
    epi = kq / rq
    elpi = float(digamma_np(kq)) - log(rq)
    e_ln_pd = -0.5 * log(2.0 * pi * NG_VP) - (vd + (md - NG_DP) ** 2) / (2.0 * NG_VP)
    e_ln_ppi = NG_K0 * log(NG_R0) - lgamma(NG_K0) + (NG_K0 - 1.0) * elpi - NG_R0 * epi
    e_ln_lik = -0.5 * log(2.0 * pi) + 0.5 * elpi - 0.5 * epi * (vd + (NG_U - md) ** 2)
    h_qd = 0.5 * log(2.0 * pi * E_CONST * vd)
    h_qpi = kq - log(rq) + lgamma(kq) + (1.0 - kq) * float(digamma_np(kq))
    return -(e_ln_pd + e_ln_ppi + e_ln_lik) - (h_qd + h_qpi)


_kq, _rq = NG_K0, NG_R0
_Fprev = None
_ng_it = 0
while True:
    _ng_it += 1
    _epi = _kq / _rq
    _vd = 1.0 / (1.0 / NG_VP + _epi)
    _md = _vd * (NG_DP / NG_VP + _epi * NG_U)
    _kq = NG_K0 + 0.5
    _rq_new = NG_R0 + 0.5 * (_vd + (NG_U - _md) ** 2)
    _change = abs(_rq_new - _rq)
    _rq = _rq_new
    _Fn = _ng_vfe(_md, _vd, _kq, _rq)
    if _Fprev is not None:
        assert _Fn <= _Fprev + 1e-13
    _Fprev = _Fn
    if _ng_it in (1, 2, 5):
        V[f"ng_it{_ng_it}_md"] = _md
        V[f"ng_it{_ng_it}_vd"] = _vd
        V[f"ng_it{_ng_it}_epi"] = _kq / _rq
        V[f"ng_it{_ng_it}_F"] = _Fn
    if _change < 1e-15 or _ng_it > 200:
        break
assert _ng_it < 200
V["ng_sweeps"] = _ng_it
V["ng_md"], V["ng_vd"], V["ng_sd"] = _md, _vd, sqrt(_vd)
V["ng_kq"], V["ng_rq"], V["ng_epi"] = _kq, _rq, _kq / _rq
V["ng_elnpi"] = float(digamma_np(_kq)) - log(_rq)
V["ng_resid2"] = (NG_U - _md) ** 2 + _vd
V["ng_F"] = _ng_vfe(_md, _vd, _kq, _rq)
V["ng_gap"] = V["ng_F"] - V["ng_surp"]
assert V["ng_gap"] > 0


# ═════════════════════════════════════════════════════════════════════════
# 4.3  THE LAPLACE APPROXIMATION
# ═════════════════════════════════════════════════════════════════════════
def _ell_w2(d):
    return (-(d - D_PRIOR) ** 2 / (2.0 * VAR_PRIOR) - 0.5 * log(2.0 * pi * VAR_PRIOR) - log(Z_P)
            - (U_OBS - THETA / d ** 2) ** 2 / (2.0 * VAR_OBS) - 0.5 * log(2.0 * pi * VAR_OBS))


def _ell_w2_derivs(d):
    """First to fourth derivatives of ell in d on Week 2's model."""
    g1, g2, g3, g4 = -2 * THETA / d ** 3, 6 * THETA / d ** 4, -24 * THETA / d ** 5, 120 * THETA / d ** 6
    r = U_OBS - THETA / d ** 2
    return (-(d - D_PRIOR) / VAR_PRIOR + r * g1 / VAR_OBS,
            -1.0 / VAR_PRIOR + (r * g2 - g1 ** 2) / VAR_OBS,
            (-3 * g1 * g2 + r * g3) / VAR_OBS,
            (-3 * g2 ** 2 - 4 * g1 * g3 + r * g4) / VAR_OBS)


_d_hat = 1.55
for _ in range(40):
    _l1, _l2, _, _ = _ell_w2_derivs(_d_hat)
    _d_hat -= _l1 / _l2
assert abs(_ell_w2_derivs(_d_hat)[0]) < 1e-13
V["w2_mode"] = _d_hat
V["w2_ell_hat"] = _ell_w2(_d_hat)
V["w2_neg_ell_hat"] = -V["w2_ell_hat"]

# exact posterior by composite Gauss-Legendre on (0, 12]; doubling the panels agrees
_dq, _wq = gl(1e-9, 12.0, 300)
_lq = _ell_w2(_dq)
_mq = float(_lq.max())
_pq = _wq * np.exp(_lq - _mq)
_Zq = float(_pq.sum())
V["w2_surp"] = -(_mq + log(_Zq))
_pq /= _Zq
V["w2_mean"] = float(np.sum(_pq * _dq))
V["w2_sd"] = sqrt(float(np.sum(_pq * (_dq - V["w2_mean"]) ** 2)))
_dq2, _wq2 = gl(1e-9, 12.0, 600)
_l2q = _ell_w2(_dq2)
assert abs(-(float(_l2q.max()) + log(float(np.sum(_wq2 * np.exp(_l2q - _l2q.max()))))) - V["w2_surp"]) < 1e-12

_l1h, _l2h, _l3h, _l4h = _ell_w2_derivs(_d_hat)
_gp_hat = -2.0 * THETA / _d_hat ** 3
V["w2_abs_curv_hat"] = -_l2h
V["w2_abs_gn_curv_hat"] = 1.0 / VAR_PRIOR + _gp_hat ** 2 / VAR_OBS
V["w2_l3"], V["w2_l4"] = _l3h, _l4h

_lap_var = -1.0 / _l2h
V["lap_var"] = _lap_var
V["lap_sd"] = sqrt(_lap_var)
V["lap_width_term"] = 0.5 * log(2.0 * pi * _lap_var)
V["lap_F"] = -V["w2_ell_hat"] - V["lap_width_term"]
V["lap_err"] = V["lap_F"] - V["w2_surp"]
# leading correction to Laplace's method: ln P - ln P_Lap ~ l4/(8 l2^2) + 5 l3^2/(24 |l2|^3)
V["lap_c4"] = _l4h / (8.0 * _l2h ** 2)
V["lap_c3"] = 5.0 * _l3h ** 2 / (24.0 * abs(_l2h) ** 3)
V["lap_pred"] = V["lap_c4"] + V["lap_c3"]
assert V["lap_pred"] > 0 and V["lap_err"] > 0
# Stirling: Gamma(n + 1) = int x^n e^-x dx, correction 1/(12 n)
V["stir_n"] = 5
V["stir_err"] = lgamma(6.0) - (0.5 * log(2 * pi * 5) + 5 * log(5) - 5)
V["stir_pred"] = 1.0 / 60.0

V["lap_gn_sd"] = sqrt(1.0 / V["w2_abs_gn_curv_hat"])
V["lap_gn_F"] = -V["w2_ell_hat"] - 0.5 * log(2.0 * pi / V["w2_abs_gn_curv_hat"])
V["w2_gn_sens_term"] = _gp_hat ** 2 / VAR_OBS
V["w2_hess_res_term"] = (U_OBS - THETA / _d_hat ** 2) * (6.0 * THETA / _d_hat ** 4) / VAR_OBS


# minimising F_Lap(m) = -ell(m) + (1/2) ln(-ell''(m) / 2 pi) over m: -ell' + ell'''/(2 ell'') = 0
def _flap_grad(m):
    a1, a2, a3, _ = _ell_w2_derivs(m)
    return -a1 + 0.5 * a3 / a2


_lo, _hi = _d_hat, 2.6
assert _flap_grad(_lo) * _flap_grad(_hi) < 0
for _ in range(100):
    _mid = 0.5 * (_lo + _hi)
    if _flap_grad(_mid) * _flap_grad(_lo) > 0:
        _lo = _mid
    else:
        _hi = _mid
_ms = 0.5 * (_lo + _hi)
_a2 = _ell_w2_derivs(_ms)[1]
assert _a2 < 0
V["lap_shift_m"] = _ms
V["lap_shift_sd"] = sqrt(-1.0 / _a2)
V["lap_shift_F"] = -_ell_w2(_ms) + 0.5 * log(-_a2 / (2.0 * pi))
V["lap_shift_err"] = V["lap_shift_F"] - V["w2_surp"]
V["lap_shift_under"] = -V["lap_shift_err"]
assert V["lap_shift_F"] < V["w2_surp"]
# it is the global minimiser on the stretch of d where ell'' < 0
_mgrid = np.linspace(0.8, 3.0, 22001)
_fl = np.array([-_ell_w2(x) + 0.5 * log(-_ell_w2_derivs(x)[1] / (2 * pi)) if _ell_w2_derivs(x)[1] < 0 else np.inf
                for x in _mgrid])
assert abs(_mgrid[int(np.argmin(_fl))] - _ms) < 2e-4

# where the curvature is positive: a sharper ear and a louder sound
V["infl_vu"], V["infl_u"] = 0.02, 0.80
_gp2, _gpp2 = -2.0 * THETA / 8.0, 6.0 * THETA / 16.0
V["infl_curv_at_2"] = -1.0 / VAR_PRIOR + ((V["infl_u"] - THETA / 4.0) * _gpp2 - _gp2 ** 2) / V["infl_vu"]
V["infl_gn_at_2"] = -1.0 / VAR_PRIOR - _gp2 ** 2 / V["infl_vu"]
assert V["infl_curv_at_2"] > 0


# log-normal recognition density: rho = ln d ~ N(m, S)
def _lnmom(m, S):
    return (exp(m + 0.5 * S), exp(2 * m + 2 * S), exp(-2 * m + 2 * S), exp(-4 * m + 8 * S))


def _F_lognormal(m, S):
    e1, e2, em2, em4 = _lnmom(m, S)
    e_prior = -(e2 - 2.0 * D_PRIOR * e1 + D_PRIOR ** 2) / (2.0 * VAR_PRIOR) - 0.5 * log(2.0 * pi * VAR_PRIOR) - log(Z_P)
    e_lik = -(U_OBS ** 2 - 2.0 * U_OBS * THETA * em2 + THETA ** 2 * em4) / (2.0 * VAR_OBS) - 0.5 * log(2.0 * pi * VAR_OBS)
    return -(e_prior + e_lik) - (0.5 * log(2.0 * pi * E_CONST * S) + m)


def _gF_lognormal(m, S):
    e1, e2, em2, em4 = _lnmom(m, S)
    gm = (e2 - D_PRIOR * e1) / VAR_PRIOR + 2 * THETA * (U_OBS * em2 - THETA * em4) / VAR_OBS - 1.0
    gS = (2 * e2 - D_PRIOR * e1) / (2 * VAR_PRIOR) + 2 * THETA * (2 * THETA * em4 - U_OBS * em2) / VAR_OBS - 0.5 / S
    return np.array([gm, gS])


_x = np.array([log(_d_hat), 0.1])
for _ in range(60):
    _g = _gF_lognormal(*_x)
    _H = np.zeros((2, 2))
    for _j in range(2):
        _e = np.zeros(2)
        _e[_j] = 1e-6
        _H[:, _j] = (_gF_lognormal(*(_x + _e)) - _gF_lognormal(*(_x - _e))) / 2e-6
    _x = _x - np.linalg.solve(0.5 * (_H + _H.T), _g)
assert np.max(np.abs(_gF_lognormal(*_x))) < 1e-12
_fd = np.array([(_F_lognormal(_x[0] + 1e-6, _x[1]) - _F_lognormal(_x[0] - 1e-6, _x[1])) / 2e-6,
                (_F_lognormal(_x[0], _x[1] + 1e-6) - _F_lognormal(_x[0], _x[1] - 1e-6)) / 2e-6])
assert np.max(np.abs(_fd)) < 1e-8
_mr_opt, _sr_opt = float(_x[0]), float(_x[1])
V["ln_mr"], V["ln_sr"], V["ln_sd_rho"] = _mr_opt, _sr_opt, sqrt(_sr_opt)
V["ln_F"] = _F_lognormal(_mr_opt, _sr_opt)
V["ln_gap"] = V["ln_F"] - V["w2_surp"]
V["ln_mean_d"] = exp(_mr_opt + 0.5 * _sr_opt)
V["ln_mean_err"] = V["ln_mean_d"] - V["w2_mean"]
V["ln_sd_d"] = sqrt((exp(_sr_opt) - 1.0) * exp(2.0 * _mr_opt + _sr_opt))
V["ln_mode_d"] = exp(_mr_opt - _sr_opt)
V["lap_mean_err"] = _d_hat - V["w2_mean"]
assert V["ln_gap"] > 0 and V["ln_sd_d"] < V["w2_sd"]

# Figure 4.3: exact posterior, Laplace at the mode, log-normal
_p3 = Panel(58, 642, 232, 28, 0.0, 5.0, 0.0, 1.0)
_ds = np.linspace(0.08, 5.0, 180)
_pex = np.exp(_ell_w2(_ds) + V["w2_surp"])
_plap = np.exp(-(_ds - _d_hat) ** 2 / (2.0 * _lap_var)) / sqrt(2.0 * pi * _lap_var)
_pln = np.exp(-(np.log(_ds) - _mr_opt) ** 2 / (2.0 * _sr_opt)) / (_ds * sqrt(2.0 * pi * _sr_opt))
_p3.dy = (0.0, max(_pex.max(), _plap.max(), _pln.max()) * 1.05)
V["fig3_exact_path"] = _p3.path(_ds, _pex)
V["fig3_lap_path"] = _p3.path(_ds, _plap)
V["fig3_ln_path"] = _p3.path(_ds, _pln)
V["fig3_mode_x"] = f"{_p3.X(_d_hat):.1f}"
V["fig3_mean_x"] = f"{_p3.X(V['w2_mean']):.1f}"
for _t in (1, 2, 3, 4, 5):
    V[f"fig3_tick_{_t}_x"] = f"{_p3.X(float(_t)):.1f}"


# ═════════════════════════════════════════════════════════════════════════
# 4.4  LEARNING WITH FREE ENERGY
# ═════════════════════════════════════════════════════════════════════════
# EM on Week 3's linear prior-variance problem: d ~ N(2, Sigma_p* = 1), u | d ~ N(d, 1)
_r_em = mulberry32(19)
N_EM = 200
V["em_n"] = N_EM
V["em_vp_true"], V["em_vu"] = 1.0, 1.0
V["em_threshold"] = 3.0 * V["em_vu"]
_em_vp_init = 0.20
_u_em = np.array([D_PRIOR + normal(_r_em) + normal(_r_em) for _ in range(N_EM)])
_S_emp = float(np.mean((_u_em - D_PRIOR) ** 2))
V["em_S_emp"] = _S_emp
V["em_vp_mle"] = _S_emp - V["em_vu"]


def _em_bound(vp_q, vp_model, vu=1.0, S=_S_emp):
    """Per-trial -F[Q, u; vp_model] with Q_n = P(d | u_n; vp_q)."""
    s_post = vp_q * vu / (vp_q + vu)
    sh = vp_q / (vp_q + vu)
    e_dev2 = s_post + sh ** 2 * S
    e_res2 = s_post + (1.0 - sh) ** 2 * S
    return (-0.5 * log(2.0 * pi * vp_model) - 0.5 * e_dev2 / vp_model
            - 0.5 * log(2.0 * pi * vu) - 0.5 * e_res2 / vu + 0.5 * log(2.0 * pi * E_CONST * s_post))


def _em_logev(vp, vu=1.0, S=_S_emp):
    return -0.5 * log(2.0 * pi * (vp + vu)) - 0.5 * S / (vp + vu)


def _em_step(vp, vu=1.0, S=_S_emp):
    return vp * vu / (vp + vu) + (vp / (vp + vu)) ** 2 * S


_vp = _em_vp_init
V["em_0_vp"] = _vp
V["em_0_logev"] = _em_logev(_vp)
V["em_0_spost"] = _vp / (_vp + 1.0)
V["em_0_shrink"] = _vp / (_vp + 1.0)
_em_path = [_vp]
for _step in range(1, 7):
    assert abs(_em_bound(_vp, _vp) - _em_logev(_vp)) < 1e-14
    _nxt = _em_step(_vp)
    _bm = _em_bound(_vp, _nxt)
    assert _em_logev(_vp) <= _bm <= _em_logev(_nxt)
    V[f"em_{_step}_vp"], V[f"em_{_step}_elbo_m"], V[f"em_{_step}_logev"] = _nxt, _bm, _em_logev(_nxt)
    _vp = _nxt
    _em_path.append(_vp)
_k = 6
while abs(_em_step(_vp) - _vp) > 1e-15 and _k < 1000:
    _vp = _em_step(_vp)
    _k += 1
V["em_conv_vp"] = _vp
V["em_conv_its"] = _k
V["em_conv_logev"] = _em_logev(_vp)
assert abs(_vp - V["em_vp_mle"]) < 1e-12
assert abs(_em_bound(_vp, _vp) - V["em_conv_logev"]) < 1e-14
assert V["em_conv_logev"] >= _em_logev(V["em_vp_mle"] * 1.001) and V["em_conv_logev"] >= _em_logev(V["em_vp_mle"] * 0.999)
# the point rule: replace the posterior by its mode, so Sigma <- (Sigma/(Sigma + Sigma_u))^2 S
_vp_pt = _em_vp_init
for _step in range(1, 5):
    _vp_pt = (_vp_pt / (_vp_pt + 1.0)) ** 2 * _S_emp
    V[f"pt_{_step}_vp"] = _vp_pt
# its nonzero fixed points solve Sigma^2 + (2 Sigma_u - S) Sigma + Sigma_u^2 = 0
V["pt_disc"] = (2.0 - _S_emp) ** 2 - 4.0
assert V["pt_disc"] < 0
# the variance term the point rule drops, and its derivative in Sigma_p (linear model)
V["em_dlogspost"] = 1.0 / (2.0 * 1.0 * (1.0 + 1.0))

# Figure 4.4: log evidence and the bound at the first E-step's Q
_pe = Panel(66, 640, 236, 30, 0.0, 1.2, -1.95, -1.65)
_vs = np.linspace(0.04, 1.2, 240)
_le = np.array([_em_logev(v) for v in _vs])
_lb = np.array([_em_bound(_em_vp_init, v) for v in _vs])
_vis = _lb >= _pe.dy[0]
_fig = _pe.axes([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [-1.9, -1.8, -1.7], "prior variance Σp", "per trial (nats)",
                fmt="{:g}")
_fig += [f'<path d="{_pe.path(_vs, _le)}" fill="none" stroke="{GREEN}" stroke-width="2.4"/>',
         f'<path d="{_pe.path(_vs[_vis], _lb[_vis])}" fill="none" stroke="{CLAY}" stroke-width="2" stroke-dasharray="6 4"/>']
_v0, _v1 = _em_vp_init, V["em_1_vp"]
for _vv in _em_path[1:]:
    _fig.append(dot(_pe, _vv, _em_logev(_vv), GREEN, r=2.2))
_fig.append(dot(_pe, _v0, _em_logev(_v0), GREEN))
_fig.append(text(_pe.X(_v0) - 8, _pe.Y(_em_logev(_v0)) - 9, "Σp(0)", anchor="end", fill=GREEN))
_fig.append(dot(_pe, _v1, V["em_1_elbo_m"], CLAY, r=3.2))
_fig.append(f'<line x1="{_pe.X(_v1) + 3:.1f}" y1="{_pe.Y(V["em_1_elbo_m"]) + 3:.1f}" x2="{_pe.X(_v1) + 40:.1f}" '
            f'y2="{_pe.Y(V["em_1_elbo_m"]) + 34:.1f}" stroke="{CLAY}" stroke-width="1"/>')
_fig.append(text(_pe.X(_v1) + 44, _pe.Y(V["em_1_elbo_m"]) + 40, "bound after the first M-step", anchor="start", fill=CLAY))
_fig.append(text(_pe.X(_em_path[-1]) + 6, _pe.Y(_em_logev(_em_path[-1])) - 9, "Σp(6)", anchor="start", fill=GREEN))
_fig.append(dot(_pe, V["em_vp_mle"], V["em_conv_logev"], GREEN))
_fig.append(text(_pe.X(V["em_vp_mle"]), _pe.Y(V["em_conv_logev"]) - 10, "maximum", fill=GREEN))
_fig.append(f'<line x1="{_pe.X(_v1):.1f}" y1="{_pe.Y(V["em_1_elbo_m"]):.1f}" x2="{_pe.X(_v1):.1f}" '
            f'y2="{_pe.Y(V["em_1_logev"]):.1f}" stroke="{MUTED}" stroke-dasharray="2 2"/>')
_fig += legend(380, 188, [(GREEN, None, "log evidence per trial"), (CLAY, "6 4", "bound with Q from the E-step at Σp(0)")])
V["fig_em_svg"] = svg(680, 280, "The log evidence per trial as a function of the prior variance, with the "
                      "free-energy bound for the first E-step's recognition densities touching it at the starting value.", _fig)

# where the point, Laplace and exact rules come to rest on Week 2's model
_LL = _cached("laplace_learning")
for _w in ("vp", "vu"):
    for _r in ("point", "laplace", "exact"):
        V[f"ll_{_w}_{_r}"] = _LL[_w][_r]
        V[f"ll_{_w}_{_r}_pct"] = 100.0 * (_LL[_w][_r] / {"vp": VAR_PRIOR, "vu": VAR_OBS}[_w] - 1.0)
        V[f"ll_{_w}_{_r}_drift"] = _LL[_w + "_drift_truth"][_r]
    assert abs(_LL[_w]["exact"] / {"vp": VAR_PRIOR, "vu": VAR_OBS}[_w] - 1.0) < 1e-6
V["ll_world_mass"] = _LL["world_mass"]
V["ll_conv"] = max(abs(_LL[_w][_r] - _LL[_w + "_fine"][_r]) for _w in ("vp", "vu")
                  for _r in ("point", "laplace", "exact"))
V["ll_slope_max"] = max(_LL[_w + "_slope"][_r] for _w in ("vp", "vu") for _r in ("point", "laplace", "exact"))
assert V["ll_slope_max"] < 0
V["ll_fine_n_z"], V["ll_fine_order"] = _LL["refined"]["n_z"], _LL["refined"]["order"]
# the upper limit of the world average, read from the module's own default
_spec = importlib.util.spec_from_file_location("w04_ll_settings", os.path.join(HERE, "laplace_learning.py"))
_llm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_llm)
import inspect
V["ll_world_dmax"] = inspect.signature(_llm._world).parameters["dmax"].default
V["ll_n_z"] = inspect.signature(_llm._world).parameters["n_z"].default


# ═════════════════════════════════════════════════════════════════════════
# 4.5  A MEAN-FIELD LEARNING FAILURE
# ═════════════════════════════════════════════════════════════════════════
TC = _cached("twocause")
# probability inside the contour r standard deviations out, for a two-dimensional Gaussian
V["gauss_mass_1sd"] = 100.0 * (1.0 - exp(-0.5 * 1.0 ** 2))
V["gauss_mass_2sd"] = 100.0 * (1.0 - exp(-0.5 * 2.0 ** 2))
V["tc_gap_exp_median"] = TC["gap_exp_dist"]["median"]
V["tc_gap_exp_top1"] = 100.0 * TC["gap_exp_dist"]["top1"]
V["tc_gap_exp_top10"] = 100.0 * TC["gap_exp_dist"]["top10"]
V["tc_n"] = TC["n"]
V["r1_true"], V["r2_true"] = TC["lam_true"]
V["m1_prior_true"], V["m2_prior_true"] = 1.0 / TC["lam_true"][0], 1.0 / TC["lam_true"][1]
V["r1_init"], V["r2_init"] = TC["lam_init"]
V["ema_keep"], V["ema_rate"] = 0.99, 0.01
V["exp_pred_cv"] = sqrt(11.0)
V["tc_u_max1"], V["tc_u_max2"] = TC["u_max"]
# quadrature settings and the effect of refining each one alone, over all trials at the true rates
_qd = TC["quad_default"]
V["tc_quad_L"], V["tc_quad_hy"], V["tc_quad_order"] = _qd["L"], _qd["hy"], _qd["order"]
V["tc_quad_ngrade"], V["tc_quad_ratio"] = _qd["ngrade"], _qd["ratio"]
V["tc_quad_nodes"] = TC["quad"]["base_nodes"]
for _k in ("hy", "L", "grade", "order"):
    _c = TC["quad"][_k]
    V[f"tc_qc_{_k}_nll"] = _c["nll"]
    V[f"tc_qc_{_k}_m"] = max(_c["m1"], _c["m2"])
    V[f"tc_qc_{_k}_sd"] = max(_c["sd1"], _c["sd2"])
    V[f"tc_qc_{_k}_corr"] = _c["corr"]
    V[f"tc_qc_{_k}_rate"] = max(_c["rate1"], _c["rate2"])
    V[f"tc_qc_{_k}_nodes"] = _c["nodes"]
_qv = TC["quad_variants"]
V["tc_qv_hy"], V["tc_qv_L"], V["tc_qv_order"] = _qv["hy"]["hy"], _qv["L"]["L"], _qv["order"]["order"]
V["tc_qv_ngrade"], V["tc_qv_ratio"] = _qv["grade"]["ngrade"], _qv["grade"]["ratio"]
V["tc_qc_max_rate"] = max(max(TC["quad"][k]["rate1"], TC["quad"][k]["rate2"]) for k in ("hy", "L", "grade", "order"))
# the largest-observation trial
_b = TC["big"]
V["tc_big_i"] = _b["i"]
for _k in ("u1", "u2", "nll", "m1", "m2", "sd1", "sd2", "corr"):
    V[f"tc_big_{_k}"] = _b[_k]
# reference summaries
V["tc_ex_cv1"] = TC["ex_true"]["cv1"]
V["tc_ex_corr"] = TC["ex_true"]["corr"]
V["tc_ex_corr_pct"] = 100.0 * TC["ex_true"]["corr_gt05"]
V["tc_ex_corr_min"], V["tc_ex_corr_max"] = TC["ex_true"]["corr_min"], TC["ex_true"]["corr_max"]
# one M-step from the true rates, and fitted rates
for _k in ("exact", "free", "gam", "exp"):
    V[f"tc_pin_{_k}_1"], V[f"tc_pin_{_k}_2"] = TC["pin"][_k]
    V[f"tc_fit_{_k}_1"], V[f"tc_fit_{_k}_2"] = TC["fit"][_k]
    V[f"tc_its_{_k}"] = TC["its"][_k]
    V[f"tc_cv1_pin_{_k}"] = TC["cv1_pin"][_k]
    V[f"tc_cv1_fit_{_k}"] = TC["cv1_fit"][_k]
    _Jt = np.array(TC["jac"][_k]["true"])
    _Jf = np.array(TC["jac"][_k]["fit"])
    _b1v = np.array(TC["pin"][_k]) - np.array(TC["lam_true"])
    _pred = np.linalg.solve(np.eye(2) - _Jt, _b1v)
    V[f"tc_jt_{_k}_11"], V[f"tc_jt_{_k}_12"], V[f"tc_jt_{_k}_21"], V[f"tc_jt_{_k}_22"] = _Jt.ravel()
    V[f"tc_jf_{_k}_rho"] = float(max(abs(np.linalg.eigvals(_Jf))))
    V[f"tc_b_{_k}_1"], V[f"tc_b_{_k}_2"] = _b1v
    V[f"tc_pred_{_k}_1"], V[f"tc_pred_{_k}_2"] = _pred + np.array(TC["lam_true"])
    V[f"tc_disp_{_k}_1"], V[f"tc_disp_{_k}_2"] = np.array(TC["fit"][_k]) - np.array(TC["lam_true"])
    _G = np.linalg.inv(np.eye(2) - _Jt)
    V[f"tc_gain_{_k}_11"], V[f"tc_gain_{_k}_22"] = _G[0, 0], _G[1, 1]
    assert np.all(np.diag(_Jt) > 0) and max(abs(np.linalg.eigvals(_Jt))) < 1
for _k in ("free", "gam", "exp"):
    V[f"tc_gap_{_k}"] = TC["gap_pin"][_k]
    V[f"tc_gapmin_{_k}"] = TC["gap_pin_min"][_k]
    V[f"tc_mr_{_k}_1"], V[f"tc_mr_{_k}_2"] = TC["mean_ratio"][_k]
    V[f"tc_low_{_k}_1"], V[f"tc_low_{_k}_2"] = [100.0 * x for x in TC["mean_lower_frac"][_k]]
V["tc_fit_am_1"], V["tc_fit_am_2"] = TC["fit"]["exp_am"]
V["tc_its_am"] = TC["its"]["exp_am"]
V["tc_rises_am"] = TC["rises"]["exp_am"]
V["tc_pin_am_1"], V["tc_pin_am_2"] = TC["pin_am_exp"]
V["tc_am_over_hm_1"] = 100.0 * (TC["pin_am_exp"][0] / TC["pin"]["exp"][0] - 1.0)
V["tc_am_over_hm_2"] = 100.0 * (TC["pin_am_exp"][1] / TC["pin"]["exp"][1] - 1.0)
V["tc_ema_1"], V["tc_ema_2"] = TC["ema_final"]
V["tc_ema_lo_1"], V["tc_ema_lo_2"] = TC["ema_last100"][0]
V["tc_ema_hi_1"], V["tc_ema_hi_2"] = TC["ema_last100"][1]
V["tc_ema_ratio_1"] = TC["ema_final"][0] / TC["lam_true"][0]
V["tc_ema_ratio_2"] = TC["ema_final"][1] / TC["lam_true"][1]
for _k in ("exact", "free", "gam", "exp"):
    V[f"tc_fitpct_{_k}_1"] = 100.0 * (TC["fit"][_k][0] / TC["fit"]["exact"][0] - 1.0)
    V[f"tc_fitpct_{_k}_2"] = 100.0 * (TC["fit"][_k][1] / TC["fit"]["exact"][1] - 1.0)
V["tc_ml_score"] = max(abs(x) for x in TC["ml_score"])
V["tc_ml_refined"] = max(abs(x) for x in TC["ml_refined_residual"])
V["tc_ml_cv1"] = TC["ml_cv1"]
for _k in ("ff_start_dm", "ff_start_dF", "ff_quad_dm", "ff_quad_dF", "gam_grad_relerr",
           "exp_grad_relerr", "exp_hess_relerr", "gam_k1_equals_exp", "exp_fit_multistart",
           "gam_fit_multistart", "ff_fit_multistart"):
    V["tc_" + _k] = TC[_k]
V["tc_ff_sweeps_med"], V["tc_ff_sweeps_max"] = TC["ff_sweeps_true"]
V["tc_tol_exp"], V["tc_tol_gam"] = TC["tol"]["exp"], TC["tol"]["gam"]
V["tc_tol_ff"], V["tc_tol_em"] = TC["tol"]["ff"], TC["tol"]["em"]
V["tc_n_exp_starts"], V["tc_n_gam_starts"] = len(TC["exp_starts"]), len(TC["gam_starts"])
V["tc_exp_start_lo"] = min(a for a, _ in TC["exp_starts"])
V["tc_exp_start_hi"] = max(a for a, _ in TC["exp_starts"])
assert sorted({a for a, _ in TC["exp_starts"]}) == [V["tc_exp_start_lo"], 0.0, V["tc_exp_start_hi"]]
for _k in ("exp", "gam"):
    for _w in ("pin", "fit"):
        _gd = TC["grad_diag"][_k][_w]
        V[f"tc_gmax_{_k}_{_w}"] = _gd["gmax"]
        V[f"tc_gover_{_k}_{_w}"] = _gd["over"]
        V[f"tc_gdec_{_k}_{_w}"] = _gd["dec"]
V["tc_gam_kmax"] = TC["gam_kmax"]
for _k in ("exp", "gam"):
    for _w in ("pin", "fit"):
        V[f"tc_gpred_{_k}_{_w}"] = 0.5 * V[f"tc_gdec_{_k}_{_w}"]
# exact-EM Jacobian at the truth from posterior covariances, against the finite differences
_Jc = np.array(TC["jac_exact_cov"])
V["tc_jc_exact_11"], V["tc_jc_exact_12"], V["tc_jc_exact_21"], V["tc_jc_exact_22"] = _Jc.ravel()
V["tc_jc_vs_fd"] = float(np.max(np.abs(_Jc - np.array(TC["jac"]["exact"]["true"]))))
# parametric Jacobians from the implicit-function formula, against central differences
V["tc_jac_fd_dev"] = max(max(d.values()) for d in TC["jac_fd_dev"].values())
V["tc_jac_fd_step_lo"], V["tc_jac_fd_step_hi"] = min(TC["jac_fd_steps"]), max(TC["jac_fd_steps"])
V["tc_jac_mineig"] = min(TC["jac_mineig"].values())
assert V["tc_jac_mineig"] > 0
# truncation of the integration limits: what lies outside the box is at most exp(-L^2/2)/pi
V["tc_nll_max"] = TC["nll_max"]
V["tc_trunc_abs"] = exp(-0.5 * TC["quad_default"]["L"] ** 2) / pi
V["tc_trunc_rel"] = V["tc_trunc_abs"] * exp(TC["nll_max"])
# comparisons between rows of the separation table
for _k in ("free", "gam", "exp"):
    V[f"tc_pinpct_{_k}_1"] = 100.0 * (TC["pin"][_k][0] / TC["pin"]["exact"][0] - 1.0)
    V[f"tc_pinpct_{_k}_2"] = 100.0 * (TC["pin"][_k][1] / TC["pin"]["exact"][1] - 1.0)
V["tc_pinpct_am_1"] = 100.0 * (TC["pin_am_exp"][0] / TC["pin"]["exact"][0] - 1.0)
V["tc_pinpct_am_2"] = 100.0 * (TC["pin_am_exp"][1] / TC["pin"]["exact"][1] - 1.0)
V["tc_fitpct_am_1"] = 100.0 * (TC["fit"]["exp_am"][0] / TC["fit"]["exact"][0] - 1.0)
V["tc_fitpct_am_2"] = 100.0 * (TC["fit"]["exp_am"][1] / TC["fit"]["exact"][1] - 1.0)
for _w, _src in (("pin", TC["pin"]), ("fit", TC["fit"])):
    for _i in (0, 1):
        V[f"tc_gvf_{_w}_{_i + 1}"] = 100.0 * (_src["gam"][_i] / _src["free"][_i] - 1.0)
        V[f"tc_evg_{_w}_{_i + 1}"] = 100.0 * (_src["exp"][_i] / _src["gam"][_i] - 1.0)
V["tc_pred_err_max"] = max(abs(V[f"tc_pred_{_k}_{_i}"] - V[f"tc_fit_{_k}_{_i}"])
                           for _k in ("exact", "free", "gam") for _i in (1, 2))
V["tc_pred_err_exp"] = max(abs(V[f"tc_pred_exp_{_i}"] - V[f"tc_fit_exp_{_i}"]) for _i in (1, 2))
for _k in ("exact", "free", "gam", "exp"):
    for _i in (1, 2):
        assert abs(V[f"tc_disp_{_k}_{_i}"]) > abs(V[f"tc_b_{_k}_{_i}"]), (_k, _i)
    assert min(abs(V[f"tc_b_exp_{_i}"]) for _i in (1, 2)) > 1.0
V["tc_amfit_over_hm_1"] = 100.0 * (TC["fit"]["exp_am"][0] / TC["fit"]["exp"][0] - 1.0)
V["tc_amfit_over_hm_2"] = 100.0 * (TC["fit"]["exp_am"][1] / TC["fit"]["exp"][1] - 1.0)
for _k in ("exact", "free", "gam", "exp"):
    _f0, _f1 = TC["Ftot_first_last"][_k]
    V[f"tc_Fdrop_{_k}"] = _f0 - _f1

# representative trial
V["rep_u1"], V["rep_u2"] = TC["rep_u"]
_rx = TC["rep_ex"]
V["rep_nll"] = _rx["nll"]
V["rep_ex_m1"], V["rep_ex_m2"] = _rx["m1"], _rx["m2"]
V["rep_ex_sd1"], V["rep_ex_sd2"] = _rx["sd1"], _rx["sd2"]
V["rep_ex_corr"] = _rx["corr"]
V["rep_one_minus_c2"] = 1.0 - _rx["corr"] ** 2
V["rep_gauss_gap"] = -0.5 * log(1.0 - _rx["corr"] ** 2)
V["rep_sqrt_one_minus_c2"] = sqrt(1.0 - _rx["corr"] ** 2)
_re = TC["rep_exp"]
V["rep_exp_m1"], V["rep_exp_m2"], V["rep_exp_F"] = _re["m1"], _re["m2"], _re["F"]
V["rep_exp_gap"] = _re["F"] - _rx["nll"]
V["rep_exp_pred_m1"] = 2.0 * _re["m1"] ** 2 * _re["m2"]
V["rep_exp_pred_sd1"] = sqrt(44.0) * _re["m1"] ** 2 * _re["m2"]
_rg = TC["rep_gam"]
V["rep_gam_m1"], V["rep_gam_m2"] = _rg["m1"], _rg["m2"]
V["rep_gam_k1"], V["rep_gam_k2"] = _rg["k1"], _rg["k2"]
V["rep_gam_sd1"], V["rep_gam_sd2"] = _rg["m1"] / sqrt(_rg["k1"]), _rg["m2"] / sqrt(_rg["k2"])
V["rep_gam_F"], V["rep_gam_gap"] = _rg["F"], _rg["F"] - _rx["nll"]
_rf = TC["rep_ff"]
V["rep_ff_m1"], V["rep_ff_m2"] = _rf["m1"], _rf["m2"]
V["rep_ff_sd1"], V["rep_ff_sd2"] = _rf["sd1"], _rf["sd2"]
V["rep_ff_F"], V["rep_ff_gap"] = _rf["F"], _rf["F"] - _rx["nll"]
V["rep_ff_ratio1"], V["rep_ff_ratio2"] = _rf["sd1"] / _rx["sd1"], _rf["sd2"] / _rx["sd2"]
V["rep_gam_ratio1"], V["rep_gam_ratio2"] = V["rep_gam_sd1"] / _rx["sd1"], V["rep_gam_sd2"] / _rx["sd2"]

# Figure 4.5a: the posterior of the representative trial and the three products
_W5 = 680
_pa = Panel(60, 320, 300, 36, 0.0, TC["plane"][0], 0.0, TC["plane"][1])
_pb = Panel(400, 660, 300, 36, 0.0, TC["plane"][0], 0.0, TC["plane"][1])
_ticks = [0, 1, 2, 3]
_body = _pa.axes(_ticks, [0, 1, 2, 3, 4], "s1", "s2") + _pb.axes(_ticks, [0, 1, 2, 3, 4], "s1", "s2")
_styles = {"exact": (GREEN, None, 2.2), "free": (BLUE, None, 1.8), "gam": (PLUM, "5 3", 1.8),
           "exp": (CLAY, "2 3", 1.8)}


def _contours(panel, key):
    col, dash, wdt = _styles[key]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    out = []
    for li, lines in enumerate(TC["contours"][key]):
        op = "1" if li == 0 else "0.55"
        for line in lines:
            xs, ys = [p[0] for p in line], [p[1] for p in line]
            out.append(f'<path d="{panel.path(xs, ys)}" fill="none" stroke="{col}" stroke-width="{wdt}"'
                       f' opacity="{op}"{d}/>')
    return out


_body += _contours(_pa, "exact") + _contours(_pa, "free")
_body += _contours(_pb, "exact") + _contours(_pb, "gam") + _contours(_pb, "exp")
_body.append(text(190, 24, "exact posterior and free-form product", fill=INK))
_body.append(text(530, 24, "Gamma and Exponential products", fill=INK))
_body += legend(206, 62, [(GREEN, None, "exact"), (BLUE, None, "free-form")])
_body += legend(546, 62, [(GREEN, None, "exact"), (PLUM, "5 3", "Gamma"), (CLAY, "2 3", "Exponential")])
V["fig_ridge_svg"] = svg(_W5, 340, "Contours of the exact posterior over the two causes for the observation "
                         "(2.5, 3.0), a curved ridge, with the contours of the free-form, Gamma and Exponential "
                         "mean-field products.", _body)

# Figure 4.5b: the correlated Gaussian and its mean-field product
_c = _rx["corr"]
_pc = Panel(190, 490, 290, 30, -2.6, 2.6, -2.6, 2.6)
_t = np.linspace(0, 2 * pi, 241)
_Sig = np.array([[1.0, _c], [_c, 1.0]])
_ev, _evec = np.linalg.eigh(_Sig)
_circ = np.stack([np.cos(_t), np.sin(_t)])
_ell_pts = _evec @ (np.sqrt(_ev)[:, None] * _circ)
_mf_sd = sqrt(1.0 - _c ** 2)
_body = _pc.axes([-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], "x1 / σ1", "x2 / σ2")
for _sc, _op in ((1.0, "1"), (2.0, "0.55")):
    _body.append(f'<path d="{_pc.path(_sc * _ell_pts[0], _sc * _ell_pts[1], close=True)}" fill="none" '
                 f'stroke="{GREEN}" stroke-width="2.2" opacity="{_op}"/>')
    _body.append(f'<path d="{_pc.path(_sc * _mf_sd * _circ[0], _sc * _mf_sd * _circ[1], close=True)}" '
                 f'fill="none" stroke="{BLUE}" stroke-width="1.8" opacity="{_op}"/>')
_body += legend(512, 60, [(GREEN, None, "exact Gaussian"), (BLUE, None, "mean-field product")])
_body.append(text(512 + 2, 110, f"correlation {_c:.2f}".replace("-", "&#8722;"), anchor="start", fill=INK))
V["fig_ellipse_svg"] = svg(_W5, 330, "Contours of a correlated two-dimensional Gaussian and of its free-form "
                           "mean-field approximation, a circle that fits inside the tilted ellipse.", _body)

# Figure 4.5c: learned rates, every EM path from the notebook's start
_pr = Panel(70, 470, 300, 30, 0.6, 3.4, 0.4, 2.5)
_body = _pr.axes([1, 1.5, 2, 2.5, 3], [0.5, 1, 1.5, 2, 2.5], "λ1", "λ2")
_paths = [("exact", GREEN, None, "exact EM"), ("free", BLUE, None, "free-form"),
          ("gam", PLUM, "5 3", "Gamma"), ("exp", CLAY, None, "Exponential, M-step"),
          ("exp_am", CLAY, "2 3", "Exponential, average of rates")]
for _k, _col, _dash, _lab in _paths:
    _tr = np.array(TC["traj"][_k])
    _d = f' stroke-dasharray="{_dash}"' if _dash else ""
    _body.append(f'<path d="{_pr.path(_tr[:, 0], _tr[:, 1])}" fill="none" stroke="{_col}" stroke-width="1.6"{_d}/>')
    _body.append(dot(_pr, _tr[-1, 0], _tr[-1, 1], _col, r=3.5))
_tr = np.array(TC["ema_traj"])
_body.append(f'<path d="{_pr.path(_tr[:, 0], _tr[:, 1])}" fill="none" stroke="{OCHRE}" stroke-width="1.2"/>')
_body.append(dot(_pr, _tr[-1, 0], _tr[-1, 1], OCHRE, r=3.5))
_body.append(dot(_pr, TC["lam_init"][0], TC["lam_init"][1], INK, r=3.5, hollow=True))
_body.append(text(_pr.X(TC["lam_init"][0]) + 8, _pr.Y(TC["lam_init"][1]) + 4, "start", anchor="start", fill=INK))
_body.append(f'<path d="M{_pr.X(0.9) - 5:.1f} {_pr.Y(0.7):.1f} L{_pr.X(0.9) + 5:.1f} {_pr.Y(0.7):.1f} '
             f'M{_pr.X(0.9):.1f} {_pr.Y(0.7) - 5:.1f} L{_pr.X(0.9):.1f} {_pr.Y(0.7) + 5:.1f}" stroke="{INK}" stroke-width="1.6"/>')
_body.append(text(_pr.X(0.9) - 6, _pr.Y(0.7) + 15, "true", anchor="start", fill=INK))
_body += legend(490, 60, [(c, d, l) for _, c, d, l in _paths] + [(OCHRE, None, "online rule")])
V["fig_rates_svg"] = svg(_W5, 340, "Paths of the learned prior rates under exact EM and under variational EM "
                         "with free-form, Gamma and Exponential families, and the online rule, "
                         "all from the same starting rates.", _body)


# ═════════════════════════════════════════════════════════════════════════
# 4.6  PROBLEMS
# ═════════════════════════════════════════════════════════════════════════
V["ex_cg_c"] = 0.80
V["ex_cg_qvar"] = 1.0 - V["ex_cg_c"] ** 2
V["ex_cg_qsd"] = sqrt(V["ex_cg_qvar"])
V["ex_cg_kl"] = -0.5 * log(V["ex_cg_qvar"])
V["ex_cg_fwd_kl"] = V["ex_cg_kl"]   # KL(P || product of marginals) is the mutual information, the same value

# Laplace in the log coordinate rho = ln d on Week 2's model: ell_rho(rho) = ell(e^rho) + rho
_rh = log(V["w2_mode"])
for _ in range(60):
    _dd = exp(_rh)
    _l1, _l2, _, _ = _ell_w2_derivs(_dd)
    _g1 = _l1 * _dd + 1.0
    _g2 = _l2 * _dd ** 2 + _l1 * _dd
    _rh -= _g1 / _g2
_dd = exp(_rh)
_l1, _l2, _, _ = _ell_w2_derivs(_dd)
assert abs(_l1 * _dd + 1.0) < 1e-12 and _l2 * _dd ** 2 + _l1 * _dd < 0
V["lapr_rho"] = _rh
V["lapr_d"] = _dd
V["lapr_F"] = -(_ell_w2(_dd) + _rh) - 0.5 * log(2.0 * pi / -(_l2 * _dd ** 2 + _l1 * _dd))
V["lapr_err"] = V["lapr_F"] - V["w2_surp"]

# asymmetric potential V(x) = x^2/2 + e^x: density exp(-V)/Z on the real line
_xq, _xw = gl(-12.0, 6.0, 400)
_vx = 0.5 * _xq ** 2 + np.exp(_xq)
_Zas = float(np.sum(_xw * np.exp(-_vx)))
V["ex_as_surp"] = -log(_Zas)
_pxa = _xw * np.exp(-_vx) / _Zas
V["ex_as_mean"] = float(np.sum(_pxa * _xq))
V["ex_as_sd"] = sqrt(float(np.sum(_pxa * (_xq - V["ex_as_mean"]) ** 2)))
_v_xh = 0.5 * _xh ** 2 + exp(_xh)
_sig_lap_as = 1.0 / (1.0 + exp(_xh))
V["ex_as_xhat"] = _xh
V["ex_as_lap_var"] = _sig_lap_as
V["ex_as_lap_sd"] = sqrt(_sig_lap_as)
V["ex_as_lap_F"] = _v_xh - 0.5 * log(2.0 * pi * _sig_lap_as)
V["ex_as_lap_err"] = V["ex_as_lap_F"] - V["ex_as_surp"]
V["ex_as_lap_under"] = -V["ex_as_lap_err"]
# the Laplace-correction prediction: ell'' = -(1 + e^x), ell''' = ell'''' = -e^x at the mode
_ex = exp(_xh)
V["ex_as_c4"] = -_ex / (8.0 * (1.0 + _ex) ** 2)
V["ex_as_c3"] = 5.0 * _ex ** 2 / (24.0 * (1.0 + _ex) ** 3)
V["ex_as_pred"] = V["ex_as_c4"] + V["ex_as_c3"]
assert V["ex_as_pred"] < 0 and V["ex_as_lap_err"] < 0
_mg = -0.70
for _ in range(40):
    _s2g = 1.0 / (1.0 - _mg)
    _exg = exp(_mg + 0.5 * _s2g)
    _mg -= (_mg + _exg) / (1.0 + _exg * (1.0 + 0.5 / (1.0 - _mg) ** 2))
_s2g = 1.0 / (1.0 - _mg)
V["ex_as_gvi_m"] = _mg
V["ex_as_gvi_var"] = _s2g
V["ex_as_gvi_sd"] = sqrt(_s2g)
V["ex_as_gvi_F"] = 0.5 * (_mg ** 2 + _s2g) + exp(_mg + 0.5 * _s2g) - 0.5 * log(2.0 * pi * E_CONST * _s2g)
V["ex_as_gvi_gap"] = V["ex_as_gvi_F"] - V["ex_as_surp"]
assert V["ex_as_gvi_gap"] > 0

# one EM step by hand on a binary channel
_h, _f, _th0, _n1, _n0 = 0.80, 0.20, 0.30, 7, 3
_p1_0 = _h * _th0 + _f * (1.0 - _th0)
_p0_0 = 1.0 - _p1_0
_q_on = _h * _th0 / _p1_0
_q_off = (1.0 - _h) * _th0 / _p0_0
_th1 = (_n1 * _q_on + _n0 * _q_off) / (_n1 + _n0)


def _bin_elbo(q1, q0, th):
    def term(q, lk1, lk0):
        return q * log(lk1 * th / q) + (1.0 - q) * log(lk0 * (1.0 - th) / (1.0 - q))
    return _n1 * term(q1, _h, _f) + _n0 * term(q0, 1.0 - _h, 1.0 - _f)


_ll0 = _n1 * log(_p1_0) + _n0 * log(_p0_0)
_elbo_m1 = _bin_elbo(_q_on, _q_off, _th1)
_p1_1 = _h * _th1 + _f * (1.0 - _th1)
_ll1 = _n1 * log(_p1_1) + _n0 * log(1.0 - _p1_1)
assert abs(_bin_elbo(_q_on, _q_off, _th0) - _ll0) < 1e-13
assert _ll0 < _elbo_m1 < _ll1
_pi1 = _n1 / (_n1 + _n0)
V["ex_dem_hit"], V["ex_dem_fp"], V["ex_dem_emp"], V["ex_dem_slope"] = _h, _f, _pi1, _h - _f
V["ex_dem_th0"], V["ex_dem_p1_0"], V["ex_dem_p0_0"] = _th0, _p1_0, _p0_0
V["ex_dem_q_on"], V["ex_dem_q_off"], V["ex_dem_th1"] = _q_on, _q_off, _th1
V["ex_dem_ll0"], V["ex_dem_elbo_m1"], V["ex_dem_ll1"] = _ll0, _elbo_m1, _ll1
V["ex_dem_th_mle"] = (_pi1 - _f) / (_h - _f)
V["ex_dem_pi0"] = 1.0 - _pi1
V["ex_dem_sum"] = _pi1 + _h
V["ex_dem_prod"] = _pi1 * _h
# both roots of a^2 - (pi1 + h) a + pi1 h = 0 satisfy the fixed-point equation
for _a in (_pi1, _h):
    assert abs(_pi1 * _h / _a + (1 - _pi1) * (1 - _h) / (1 - _a) - 1.0) < 1e-14
_th = V["ex_dem_th_mle"]
_a = _f + (_h - _f) * _th
assert abs((_n1 * _h * _th / _a + _n0 * (1 - _h) * _th / (1 - _a)) / (_n1 + _n0) - _th) < 1e-14

VALUES = V

if __name__ == "__main__":
    for k in sorted(VALUES):
        v = VALUES[k]
        if isinstance(v, str) and len(v) > 60:
            print(f"{k:28s} = <svg, {len(v)} chars>")
        elif isinstance(v, float):
            print(f"{k:28s} = {v:.10g}")
        else:
            print(f"{k:28s} = {v}")
