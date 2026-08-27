"""Every derived number in Week 1, computed rather than typed.

The prose refers to these by key, as {{ev_tawny:.4f}}, and build_site.py
substitutes the computed value. A number that appears in a lesson but not here
is a number nobody has checked; build/check_numbers.py reports those.

Run standalone to see the values:  python3 content/week-01/numbers.py
"""
import itertools
from math import comb, exp, log, log2, sqrt

# ── the running discrete model (Lessons 3 and 5) ─────────────────────────
STATES = ["leopard", "baboon", "nothing"]
OBS = ["tawny flash", "branch shakes", "quiet"]

#            leopard baboon nothing
A = [[0.70, 0.15, 0.02],      # tawny flash
     [0.25, 0.75, 0.08],      # branch shakes
     [0.05, 0.10, 0.90]]      # quiet
PRIOR = [0.08, 0.22, 0.70]
PRIOR_NERVOUS = [0.40, 0.30, 0.30]


def posterior(A, prior, o):
    un = [A[o][s] * prior[s] for s in range(len(prior))]
    ev = sum(un)
    return [u / ev for u in un], ev


def kl(q, p):
    return sum(qi * log(qi / pi) for qi, pi in zip(q, p) if qi > 0)


V = {}

for oi, name in enumerate(OBS):
    tag = name.split()[0]                       # tawny / branch / quiet
    post, ev = posterior(A, PRIOR, oi)
    V[f"ev_{tag}"] = ev
    V[f"surprise_{tag}"] = -log(ev)
    V[f"info_{tag}"] = kl(post, PRIOR)
    for si, sname in enumerate(STATES):
        V[f"post_{tag}_{sname}"] = post[si]
    for si, sname in enumerate(STATES):
        V[f"num_{tag}_{sname}"] = A[oi][si] * PRIOR[si]

# nervous hiker observing quiet (Lesson 3 exercise)
post_n, ev_n = posterior(A, PRIOR_NERVOUS, 2)
V["ev_quiet_nervous"] = ev_n
V["surprise_quiet_nervous"] = -log(ev_n)
for si, sname in enumerate(STATES):
    V[f"post_quiet_nervous_{sname}"] = post_n[si]
    V[f"num_quiet_nervous_{sname}"] = A[2][si] * PRIOR_NERVOUS[si]

# rare-but-uninformative counterexample (Lesson 3 exercise)
A_CTR = [[0.60, 0.10], [0.35, 0.85], [0.05, 0.05]]
PRIOR_CTR = [0.5, 0.5]
for oi in range(3):
    post, ev = posterior(A_CTR, PRIOR_CTR, oi)
    V[f"ctr_ev_{oi+1}"] = ev
    V[f"ctr_surprise_{oi+1}"] = -log(ev)
    V[f"ctr_info_{oi+1}"] = kl(post, PRIOR_CTR)
    V[f"ctr_post_{oi+1}_a"] = post[0]
    V[f"ctr_post_{oi+1}_b"] = post[1]

# tipping-point prior (Lesson 5 exercise)
_coef = 0.15 * (22 / 92) + 0.02 * (70 / 92)
V["tip_coef"] = _coef
V["tip_denominator"] = 0.70 + _coef
V["tip_q"] = _coef / V["tip_denominator"]

# two observations in sequence (Lesson 5 exercise)
_p1, _ = posterior(A, PRIOR, 1)                 # branch shakes
_p2, _ev2 = posterior(A, _p1, 0)                # then tawny flash
V["seq_ev"] = _ev2
for si, sname in enumerate(STATES):
    V[f"seq_post_{sname}"] = _p2[si]
    V[f"seq_num_{sname}"] = A[0][si] * _p1[si]

# the reverse order of the same two observations (Lesson 5 exercise)
_a1, _ = posterior(A, PRIOR, 0)                 # tawny flash first
_a2, _aev = posterior(A, _a1, 1)                # then branch shakes
V["alt_ev"] = _aev
for si, sname in enumerate(STATES):
    V[f"alt_num_{sname}"] = A[1][si] * _a1[si]

# noisy-OR likelihood by number of causes present (Lesson 5 exercise)
for _k in range(4):
    V[f"noisy_lik_k{_k}"] = 1 - (0.1 ** _k) * 0.99

# lifetime scaling, stated as a multiple (Lesson 5 exercise)
# (sigma_ratio defined below, after the sigmas)

# mutual information, per model. The names carry the model deliberately: a bare
# "mutual_information" was quoted in the wrong example's solution and shipped a
# value that was right for a different matrix.
V["mi_leopard_model"] = sum(
    posterior(A, PRIOR, o)[1] * kl(posterior(A, PRIOR, o)[0], PRIOR) for o in range(3)
)
V["mi_counterexample"] = sum(
    posterior(A_CTR, PRIOR_CTR, o)[1] * kl(posterior(A_CTR, PRIOR_CTR, o)[0], PRIOR_CTR)
    for o in range(3)
)
V["ctr_prob_ratio"] = V["ctr_ev_1"] / V["ctr_ev_3"]

# ── explaining away (Lesson 4) ───────────────────────────────────────────
P_CAUSE, STRENGTH, LEAK = 0.1, 0.9, 0.01


def noisy_or(sv):
    prod = 1.0
    for x in sv:
        prod *= (1 - STRENGTH * x)
    return 1 - prod * (1 - LEAK)


def noisy_or_posterior(n):
    states = list(itertools.product([0, 1], repeat=n))
    pri = []
    for sv in states:
        p = 1.0
        for x in sv:
            p *= P_CAUSE if x else 1 - P_CAUSE
        pri.append(p)
    un = [p * noisy_or(sv) for p, sv in zip(pri, states)]
    ev = sum(un)
    return states, pri, [u / ev for u in un], ev


_st, _pri, _post, _ev = noisy_or_posterior(2)
V["noisy_ev"] = _ev
V["noisy_surprise"] = -log(_ev)
for sv, pr, po in zip(_st, _pri, _post):
    V[f"noisy_prior_{sv[0]}{sv[1]}"] = pr
    V[f"noisy_lik_{sv[0]}{sv[1]}"] = noisy_or(sv)
    V[f"noisy_prod_{sv[0]}{sv[1]}"] = pr * noisy_or(sv)
    V[f"noisy_post_{sv[0]}{sv[1]}"] = po

_m1 = V["noisy_post_10"] + V["noisy_post_11"]
_m2 = V["noisy_post_01"] + V["noisy_post_11"]
V["noisy_marg_gust"] = _m1
V["noisy_marg_bab"] = _m2
V["noisy_prod_marginals"] = _m1 * _m2
V["noisy_bab_given_gust1"] = V["noisy_post_11"] / _m1
V["noisy_bab_given_gust0"] = V["noisy_post_01"] / (1 - _m1)

# Two different "best factorised approximations", and they disagree. The forward
# direction uses the true marginals; variational inference minimises the REVERSE
# direction, which is what the course actually goes on to do.
_p = [V["noisy_post_00"], V["noisy_post_10"], V["noisy_post_01"], V["noisy_post_11"]]
_q_marg = [(1 - _m1) * (1 - _m2), _m1 * (1 - _m2), (1 - _m1) * _m2, _m1 * _m2]
V["noisy_gap_forward"] = kl(_p, _q_marg)


def _cavi(P, iters=400):
    """Coordinate ascent mean-field: the reverse-KL optimum over q(s1)q(s2).

    P is indexed [s1][s2]. This is the Week 4 algorithm, run here so the number
    quoted in Lesson 4 is the one that algorithm actually reaches. Started off
    the symmetric point, because the symmetric solution is a saddle here and a
    symmetric start would sit on it and report the wrong answer.
    """
    from math import exp, log
    q1, q2 = [0.55, 0.45], [0.45, 0.55]
    for _ in range(iters):
        for a in (0, 1):
            pass
        new1 = [exp(sum(q2[b] * log(max(P[a][b], 1e-300)) for b in (0, 1))) for a in (0, 1)]
        z = sum(new1); q1 = [x / z for x in new1]
        new2 = [exp(sum(q1[a] * log(max(P[a][b], 1e-300)) for a in (0, 1))) for b in (0, 1)]
        z = sum(new2); q2 = [x / z for x in new2]
    return q1, q2


_P = [[V["noisy_post_00"], V["noisy_post_01"]],
      [V["noisy_post_10"], V["noisy_post_11"]]]
_q1, _q2 = _cavi(_P)
V["noisy_mf_gust"] = _q1[1]
V["noisy_mf_bab"] = _q2[1]
_q_rev = [_q1[0] * _q2[0], _q1[1] * _q2[0], _q1[0] * _q2[1], _q1[1] * _q2[1]]
V["noisy_gap_reverse"] = kl(_q_rev, _p)
V["noisy_collapse_ratio"] = V["noisy_prod_marginals"] / V["noisy_post_11"]

# the reverse-KL optimum written out as a joint, and the symmetric product it beats
_qg, _qb = V["noisy_mf_gust"], V["noisy_mf_bab"]
for _a in (0, 1):
    for _b in (0, 1):
        V[f"noisy_revq_{_a}{_b}"] = (_qg if _a else 1 - _qg) * (_qb if _b else 1 - _qb)
_sym = V["noisy_marg_gust"]
_Qsym = {(a, b): (_sym if a else 1 - _sym) * (_sym if b else 1 - _sym)
         for a in (0, 1) for b in (0, 1)}
_Pj = {(0, 0): V["noisy_post_00"], (1, 0): V["noisy_post_10"],
       (0, 1): V["noisy_post_01"], (1, 1): V["noisy_post_11"]}
V["noisy_gap_reverse_symmetric"] = sum(_Qsym[k] * log(_Qsym[k] / _Pj[k]) for k in _Pj)

# three causes (Lesson 5 exercise)
_st3, _, _post3, _ev3 = noisy_or_posterior(3)
V["noisy3_ev"] = _ev3
V["noisy3_marg"] = sum(p for sv, p in zip(_st3, _post3) if sv[0] == 1)
V["noisy3_joint_mass"] = V["noisy3_marg"] * _ev3

# ── the intractability numbers (Lessons 3 and 4) ─────────────────────────
V["terms_15_binary"] = 2 ** 15
V["log10_universe"] = log2(4.4e26) * log(2) / log(10)      # log10 of the same figure
V["log2_universe"] = log2(4.4e26)
V["universe_ns"] = 4.4e26
V["factors_to_exceed_universe"] = int(log2(4.4e26)) + 1     # smallest n with 2^n >
V["factors_to_add"] = V["factors_to_exceed_universe"] - 15

# ── homeostasis scaling (Lesson 5 exercise) ──────────────────────────────
V["sigma_lo"], V["sigma_hi"] = 0.035, 0.09
V["lifetime_ratio"] = (V["sigma_lo"] / V["sigma_hi"]) ** 2
V["lifetime_factor"] = (V["sigma_hi"] / V["sigma_lo"]) ** 2
V["sigma_ratio"] = V["sigma_hi"] / V["sigma_lo"]

# percentages, for the prose that reads them as percentages
for _k in ("leopard", "baboon", "nothing"):
    V[f"prior_{_k}_pct"] = PRIOR[STATES.index(_k)] * 100
    V[f"post_tawny_{_k}_pct"] = V[f"post_tawny_{_k}"] * 100
    V[f"post_quiet_nervous_{_k}_pct"] = V[f"post_quiet_nervous_{_k}"] * 100
for _k in ("leopard", "baboon", "nothing"):
    V[f"prior_nervous_{_k}_pct"] = PRIOR_NERVOUS[STATES.index(_k)] * 100
V["noisy_marg_gust_pct"] = V["noisy_marg_gust"] * 100
V["noisy3_marg_pct"] = V["noisy3_marg"] * 100
V["tip_q_pct"] = V["tip_q"] * 100

# ── the homeostasis process, which the lesson now writes down ────────────
# x_{t+1} = (1 - kappa) x_t + sigma * xi_t,  xi ~ N(0, I_2), absorbed at |x| > 1
SIGMA, KAPPA = 0.035, 0.15
V["hs_sigma"], V["hs_kappa"] = SIGMA, KAPPA
V["hs_var_per_coord"] = SIGMA ** 2 / (1 - (1 - KAPPA) ** 2)   # stationary variance
V["hs_sd_per_coord"] = V["hs_var_per_coord"] ** 0.5
V["hs_rms_radius"] = (2 * V["hs_var_per_coord"]) ** 0.5
V["hs_boundary_in_sd"] = 1 / V["hs_sd_per_coord"]
V["hs_diffusive_steps"] = 1 / (2 * SIGMA ** 2)                # E|x|^2 = 2 t sigma^2 = 1
# the gain's full range: stability needs 0 < kappa < 2, and kappa(2-kappa) is
# maximised at kappa = 1, so the stationary variance is minimised there at sigma^2
# The widget bins position onto an NB x NB grid; keep this in step with NB in
# assets/js/w01.js, or the caption's maximum entropy silently stops matching the
# readout it describes.
HS_BINS = 26
V["hs_bins"] = HS_BINS
V["hs_bin_cells"] = HS_BINS ** 2
V["hs_max_entropy"] = log(HS_BINS ** 2)

V["hs_var_at_1"] = SIGMA ** 2
V["hs_sd_at_1"] = SIGMA
for _k in (0.5, 1.0, 1.5, 1.9):
    V[f"hs_var_k{str(_k).replace('.','')}"] = SIGMA ** 2 / (_k * (2 - _k))

# ── a worked entropy example (Lesson 1) ──────────────────────────────────
P_PEAKED = [0.94, 0.02, 0.02, 0.02]
V["ent_peaked"] = -sum(p * log(p) for p in P_PEAKED)
V["ent_uniform4"] = log(4)
V["surprise_common"] = -log(0.94)
V["surprise_rare"] = -log(0.02)
V["ent_peaked_check"] = sum(p * -log(p) for p in P_PEAKED)

# term-by-term KL contributions for the tawny flash, so the sum can be shown
for _si, _sn in enumerate(STATES):
    _q = V[f"post_tawny_{_sn}"]
    V[f"klterm_tawny_{_sn}"] = _q * log(_q / PRIOR[_si])
    V[f"ratio_tawny_{_sn}"] = _q / PRIOR[_si]

# ── the lamp behind a frosted screen (Lesson 2) ──────────────────────────
# Process: theta in {on, off}, P*(on) = 1/2. Sensor: o in {bright, dim}.
LAMP_PRIOR = 0.5
LAMP_TRUE = {"on": 0.8, "off": 0.1}          # P*(bright | theta)
LAMP_B = {"on": 0.99, "off": 0.01}           # agent B thinks its sensor is far better

def _lamp(em):
    br = LAMP_PRIOR * em["on"] + (1 - LAMP_PRIOR) * em["off"]
    return {"bright": br, "dim": 1 - br}

_star, _QB = _lamp(LAMP_TRUE), _lamp(LAMP_B)
V["lamp_prior"] = LAMP_PRIOR
for k, val in LAMP_TRUE.items():
    V[f"lamp_true_{k}"] = val
for k, val in LAMP_B.items():
    V[f"lamp_b_{k}"] = val
V["lamp_ev_bright_true"] = _star["bright"]
V["lamp_ev_dim_true"] = _star["dim"]
V["lamp_ev_bright_b"] = _QB["bright"]
V["lamp_surprise_bright_true"] = -log(_star["bright"])
V["lamp_surprise_bright_b"] = -log(_QB["bright"])
V["lamp_post_on_true"] = LAMP_PRIOR * LAMP_TRUE["on"] / _star["bright"]
V["lamp_post_on_b"] = LAMP_PRIOR * LAMP_B["on"] / _QB["bright"]
V["lamp_entropy_true"] = -sum(_star[o] * log(_star[o]) for o in _star)
V["lamp_avg_surprise_b"] = sum(_star[o] * -log(_QB[o]) for o in _star)
V["lamp_excess"] = V["lamp_avg_surprise_b"] - V["lamp_entropy_true"]
V["lamp_kl"] = sum(_star[o] * log(_star[o] / _QB[o]) for o in _star)

# ── the cost of the denominator, Lesson 5 ────────────────────────────────
# Everything here is an exact count except the four wall-clock figures, which
# are measurements and are labelled as such.
DEN_K, DEN_N = 4, 40
V["den_k"], V["den_n"] = DEN_K, DEN_N
V["den_terms"] = DEN_K ** DEN_N
V["den_chain_ops"] = (DEN_N - 1) * DEN_K ** 2

# Elimination costs k^(w+1) per bucket, with w the treewidth of the model's
# graph: 1 for a chain or any tree, about sqrt(n) for a square grid, n-1 when
# every variable touches every other.
V["den_grid_n"] = 36
V["den_grid_w"] = 6
V["den_grid_ops"] = V["den_grid_n"] * DEN_K ** (V["den_grid_w"] + 1)
V["den_grid_brute"] = DEN_K ** V["den_grid_n"]
V["den_dense_w"] = DEN_N - 1

# Measured on this machine (Apple M-series, CPython 3.13, single core,
# 2026-08-27) by the timing cell of the Week 1 notebook. Recorded rather than
# derived: they are observations of one computer, not properties of the problem,
# and the operation counts above are what the argument actually rests on.
# Two runs on the same machine gave 34969 ms and 34133 ms for brute force, and
# 0.31 ms and 0.07 ms for elimination. The brute-force figure is stable to a
# couple of per cent; the elimination figure is sub-millisecond and dominated by
# timer noise, so it is quoted to one significant figure and the ratio only to
# an order of magnitude. Quoting either to five figures, as a first draft did,
# would be reporting the noise.
V["den_brute_s_n12"] = 35
V["den_elim_ms_n12"] = 0.1
V["den_measured_n"] = 12
V["den_measured_terms"] = DEN_K ** 12
V["den_speedup_order"] = 5   # log10 of the ratio, which is all it supports

# Quadrature: with r bounded derivatives in n dimensions the best possible
# deterministic error from m points is of order m^(-r/n), so holding the error
# at eps needs m ~ eps^(-n/r) points. Bakhvalov's bound; r = 2 taken here.
V["quad_eps"] = 0.01
V["quad_r"] = 2
for _n in (1, 2, 5, 10, 20):
    V[f"quad_pts_n{_n}"] = V["quad_eps"] ** (-_n / V["quad_r"])

# Importance sampling escapes the dimension in its rate but not its constant.
# Effective sample size falls like exp(-KL), and over n independent coordinates
# the divergence adds, so the sample count needed grows exponentially again.
V["is_kl_per_coord"] = 0.5
V["is_target_ess"] = 1000
for _n in (1, 10, 20, 50):
    _kl = _n * V["is_kl_per_coord"]
    V[f"is_kl_n{_n}"] = _kl
    V[f"is_ess_frac_n{_n}"] = exp(-_kl)
    V[f"is_samples_n{_n}"] = V["is_target_ess"] * exp(_kl)

# ── constants quoted in prose ────────────────────────────────────────────
V["ln2"] = log(2)
V["nats_to_bits"] = 1 / log(2)

VALUES = V

if __name__ == "__main__":
    for k in sorted(VALUES):
        print(f"  {k:<34} {VALUES[k]}")
