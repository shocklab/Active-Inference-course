/* w04.js - interactive figures for Week 4: variational free energy.
 *
 * The numerical routines below follow content/week-04/values.py (the sound model of
 * Lesson 4.3) and content/week-04/twocause.py (the two-cause model of Lesson 4.5)
 * step for step, and are exported on AIF.w04 so that a parity check can run them
 * under node against the Python values. */
(function () {
  'use strict';
  var A = window.AIF, el = A.el, C = A.colours, M = A.math;

  /* ── Special functions ─────────────────────────────────────────────── */
  var LN2PI = Math.log(2.0 * Math.PI);
  var LANCZOS_C = [
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
  ];

  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1.0 - x);
    var z = x - 1.0;
    var acc = LANCZOS_C[0];
    for (var i = 1; i < LANCZOS_C.length; i++) acc += LANCZOS_C[i] / (z + i);
    var t = z + 7.5;
    return 0.5 * LN2PI + (z + 0.5) * Math.log(t) - t + Math.log(acc);
  }

  function digamma(x) {
    var out = 0.0;
    while (x < 10.0) { out -= 1.0 / x; x += 1.0; }
    var i = 1.0 / x, i2 = i * i;
    return out + Math.log(x) - 0.5 * i
      - i2 * (1 / 12 - i2 * (1 / 120 - i2 * (1 / 252 - i2 * (1 / 240 - i2 / 132))));
  }

  function trigamma(x) {
    var out = 0.0;
    while (x < 10.0) { out += 1.0 / (x * x); x += 1.0; }
    var i = 1.0 / x, i2 = i * i;
    return out + i + 0.5 * i2 + i * i2 * (1 / 6 - i2 * (1 / 30 - i2 * (1 / 42 - i2 * (1 / 30 - i2 * 5 / 66))));
  }

  /* Gauss-Legendre nodes and weights on [-1, 1], by Newton's method on P_n. */
  var GL_CACHE = {};
  function glRule(n) {
    if (GL_CACHE[n]) return GL_CACHE[n];
    var x = new Float64Array(n), w = new Float64Array(n), m = (n + 1) >> 1;
    for (var i = 0; i < m; i++) {
      var z = Math.cos(Math.PI * (i + 0.75) / (n + 0.5)), z1, pp = 1;
      for (var it = 0; it < 100; it++) {
        var p1 = 1.0, p2 = 0.0;
        for (var j = 0; j < n; j++) {
          var p3 = p2; p2 = p1;
          p1 = ((2 * j + 1) * z * p2 - j * p3) / (j + 1);
        }
        pp = n * (z * p1 - p2) / (z * z - 1.0);
        z1 = z; z = z1 - p1 / pp;
        if (Math.abs(z - z1) <= 1e-15) break;
      }
      x[i] = -z; x[n - 1 - i] = z;
      w[i] = w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
    }
    GL_CACHE[n] = { x: x, w: w };
    return GL_CACHE[n];
  }

  /* Composite Gauss-Legendre on the panels between consecutive breaks. */
  function glPanels(breaks, order) {
    var r = glRule(order), np = breaks.length - 1;
    var x = new Float64Array(np * order), w = new Float64Array(np * order);
    for (var p = 0; p < np; p++) {
      var a = breaks[p], b = breaks[p + 1];
      for (var k = 0; k < order; k++) {
        x[p * order + k] = 0.5 * (a + b) + 0.5 * (b - a) * r.x[k];
        w[p * order + k] = 0.5 * (b - a) * r.w[k];
      }
    }
    return { x: x, w: w };
  }

  function linspace(a, b, n) {
    var out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  }

  /* erf by 16-point Gauss-Legendre on eight panels of (0, |x|); beyond 6.5 the
   * difference from 1 is below double precision. */
  function erf(x) {
    var ax = Math.abs(x);
    if (ax > 6.5) return x > 0 ? 1.0 : -1.0;
    var r = glRule(16), npan = 8, h = ax / npan, s = 0.0;
    for (var p = 0; p < npan; p++) {
      for (var k = 0; k < 16; k++) {
        var t = (p + 0.5 * (1.0 + r.x[k])) * h;
        s += 0.5 * h * r.w[k] * Math.exp(-t * t);
      }
    }
    return (x < 0 ? -1.0 : 1.0) * 2.0 / Math.sqrt(Math.PI) * s;
  }

  function maxAbs(v) {
    var m = 0;
    for (var i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]));
    return m;
  }

  /* Eigen-decomposition of a small symmetric matrix by cyclic Jacobi rotations.
   * Returns eigenvalues ev and eigenvectors as the columns of V. */
  function symEig(H) {
    var n = H.length, a = H.map(function (r) { return r.slice(); }), V = [], i, k, p, q;
    for (i = 0; i < n; i++) { V.push(new Array(n).fill(0)); V[i][i] = 1; }
    for (var sweep = 0; sweep < 60; sweep++) {
      var off = 0, dg = 0;
      for (p = 0; p < n; p++) {
        dg += a[p][p] * a[p][p];
        for (q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      }
      if (off <= 1e-30 * dg || off === 0) break;
      for (p = 0; p < n; p++) {
        for (q = p + 1; q < n; q++) {
          var apq = a[p][q];
          if (apq === 0) continue;
          var theta = (a[q][q] - a[p][p]) / (2 * apq);
          var t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          if (!isFinite(t)) t = 0;
          var c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (k = 0; k < n; k++) {
            var akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
          }
          for (k = 0; k < n; k++) {
            var apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
          }
          for (k = 0; k < n; k++) {
            var vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    var ev = [];
    for (i = 0; i < n; i++) ev.push(a[i][i]);
    return { ev: ev, V: V };
  }

  /* Modified Newton, as newton_min in twocause.py: Hessian eigenvalues replaced by
   * their absolute values and floored at 1e-8 of the largest, steps capped at 2 in
   * every coordinate, backtracking with an Armijo condition. Stops when the largest
   * gradient component is below tol, when the Newton decrement is at the rounding
   * level of f, or when no step lowers f. */
  function newtonMin(f, gh, x0, tol, maxit) {
    var x = x0.slice(), fx = f(x), n = x.length, frozen = false, it, i, j;
    maxit = maxit || 200;
    for (it = 0; it < maxit; it++) {
      var r = gh(x), g = r.g;
      var gn = maxAbs(g);
      var e = symEig(r.H), ev = e.ev, V = e.V;
      var emin = Infinity, eabs = 0;
      for (j = 0; j < n; j++) { emin = Math.min(emin, ev[j]); eabs = Math.max(eabs, Math.abs(ev[j])); }
      var pd = emin > 0, floor = 1e-8 * Math.max(1.0, eabs);
      var step = new Array(n).fill(0);
      for (j = 0; j < n; j++) {
        var proj = 0;
        for (i = 0; i < n; i++) proj += V[i][j] * g[i];
        proj /= Math.max(Math.abs(ev[j]), floor);
        for (i = 0; i < n; i++) step[i] -= V[i][j] * proj;
      }
      var dec = 0;
      for (i = 0; i < n; i++) dec -= g[i] * step[i];
      var scale = 1 + Math.abs(fx);
      if (dec <= 1e-28 * scale * scale || gn <= tol || frozen) break;
      var local = pd && dec < 1e-10 * scale;
      var sn = maxAbs(step);
      if (sn > 2) for (i = 0; i < n; i++) step[i] *= 2 / sn;
      if (local) {
        for (i = 0; i < n; i++) x[i] += step[i];
        fx = f(x);
        continue;
      }
      var t = 1, ok = false;
      for (var bt = 0; bt < 80; bt++) {
        var cand = x.map(function (v, idx) { return v + t * step[idx]; });
        var fc = f(cand);
        if (fc <= fx - 1e-4 * t * dec || (fc < fx && t < 1e-8)) { x = cand; fx = fc; ok = true; break; }
        t *= 0.5;
      }
      if (!ok) frozen = true;
    }
    return { x: x, f: fx, gmax: maxAbs(gh(x).g), iters: it };
  }

  /* Hessian by central differences of an analytic gradient, symmetrised. */
  function ghFD(grad, h) {
    h = h || 1e-6;
    return function (x) {
      var g0 = grad(x), n = x.length, H = [], i, j;
      for (i = 0; i < n; i++) H.push(new Array(n));
      for (j = 0; j < n; j++) {
        var xp = x.slice(), xm = x.slice();
        xp[j] += h; xm[j] -= h;
        var gp = grad(xp), gm = grad(xm);
        for (i = 0; i < n; i++) H[i][j] = (gp[i] - gm[i]) / (2 * h);
      }
      for (i = 0; i < n; i++) for (j = 0; j < i; j++) H[i][j] = H[j][i] = 0.5 * (H[i][j] + H[j][i]);
      return { g: g0, H: H };
    };
  }

  function fmtSigned(v, dec) {
    var s = Math.abs(v).toFixed(dec == null ? 4 : dec);
    return (v < -1e-12 ? '−' : '') + s;
  }

  /* ── Shared simulation routines (exported on A.w04 for parity checks) ── */
  var A_MAT = [
    [0.70, 0.15, 0.02], /* tawny flash */
    [0.25, 0.75, 0.08], /* branch shakes */
    [0.05, 0.10, 0.90]  /* quiet */
  ];
  var PRIOR_3 = [0.08, 0.22, 0.70];
  var STATE_NAMES = ['leopard', 'baboon', 'nothing'];
  var OBS_NAMES = ['tawny flash', 'branch shakes', 'quiet'];

  function decomp3(qRaw, oIdx) {
    var q = M.normalise(qRaw);
    var lik = A_MAT[oIdx];
    var joint = [lik[0] * PRIOR_3[0], lik[1] * PRIOR_3[1], lik[2] * PRIOR_3[2]];
    var ev = joint[0] + joint[1] + joint[2];
    var post = [joint[0] / ev, joint[1] / ev, joint[2] / ev];
    var surp = -Math.log(ev);
    var U = 0, H = 0, comp = 0, acc = 0, klp = 0;
    for (var i = 0; i < 3; i++) {
      if (q[i] > 1e-300) {
        U -= q[i] * Math.log(joint[i]);
        H -= q[i] * Math.log(q[i]);
        comp += q[i] * Math.log(q[i] / PRIOR_3[i]);
        acc += q[i] * Math.log(lik[i]);
        klp += q[i] * Math.log(q[i] / post[i]);
      }
    }
    return {
      q: q, post: post, prior: PRIOR_3, ev: ev, surp: surp,
      U: U, H: H, comp: comp, acc: acc, klpost: klp, F: U - H
    };
  }


  /* ── Lesson 4.3: the sound model ────────────────────────────────────────
   * Prior N(d; dp, vp) truncated to d > 0, likelihood N(u; theta/d^2, vu); dp = 2 and
   * theta = 1 as in Week 2. Exact surprise and moments by composite 16-point
   * Gauss-Legendre on (0, dp + 12 sqrt(vp)); Laplace at the global mode; the
   * log-normal Q minimising F by modified Newton in (mean, ln variance) of ln d. */
  var SOUND_DP = 2.0, SOUND_TH = 1.0;

  function solveSound(u, vu, vp) {
    var dp = SOUND_DP, th = SOUND_TH;
    var zp = 0.5 * (1.0 + erf(dp / Math.sqrt(2.0 * vp)));
    var cPrior = -0.5 * Math.log(2.0 * Math.PI * vp) - Math.log(zp);
    var cLik = -0.5 * Math.log(2.0 * Math.PI * vu);
    function ell(d) {
      var r = u - th / (d * d);
      return -(d - dp) * (d - dp) / (2.0 * vp) + cPrior - r * r / (2.0 * vu) + cLik;
    }
    function ell1(d) {
      return -(d - dp) / vp + ((u - th / (d * d)) / vu) * (-2.0 * th / (d * d * d));
    }
    function ell2(d) {
      var gp = -2.0 * th / (d * d * d), gpp = 6.0 * th / (d * d * d * d);
      return -1.0 / vp + ((u - th / (d * d)) * gpp - gp * gp) / vu;
    }

    /* exact surprise, mean, variance and E[ln d], Var[ln d] */
    var dHi = dp + 12.0 * Math.sqrt(vp);
    var q = glPanels(linspace(1e-9, dHi, 301), 16), i, n = q.x.length;
    var ls = new Float64Array(n), mx = -Infinity;
    for (i = 0; i < n; i++) { ls[i] = ell(q.x[i]); if (ls[i] > mx) mx = ls[i]; }
    var Z = 0, s1 = 0, s2 = 0, l1 = 0, l2 = 0;
    for (i = 0; i < n; i++) {
      var w = q.w[i] * Math.exp(ls[i] - mx), d = q.x[i], ld = Math.log(d);
      Z += w; s1 += w * d; s2 += w * d * d; l1 += w * ld; l2 += w * ld * ld;
    }
    var surp = -(mx + Math.log(Z));
    var exMean = s1 / Z, exVar = s2 / Z - exMean * exMean;
    var lnMean = l1 / Z, lnVar = l2 / Z - lnMean * lnMean;

    /* modes: local maxima of ell on a scan, each polished by safeguarded Newton
     * on ell' inside its bracket; the Laplace Gaussian sits at the highest. */
    var NS = 3000, scan = linspace(1e-3, dHi, NS), lv = scan.map(ell), modes = [];
    for (i = 1; i < NS - 1; i++) {
      if (lv[i] >= lv[i - 1] && lv[i] > lv[i + 1]) {
        var lo = scan[i - 1], hi = scan[i + 1], x = scan[i];
        for (var it = 0; it < 200; it++) {
          var g1 = ell1(x), g2 = ell2(x);
          if (g1 > 0) lo = x; else hi = x;
          var xn = g2 < 0 ? x - g1 / g2 : 0.5 * (lo + hi);
          if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
          if (Math.abs(xn - x) <= 1e-15 * x) { x = xn; break; }
          x = xn;
        }
        modes.push(x);
      }
    }
    var dHat = modes[0];
    modes.forEach(function (m) { if (ell(m) > ell(dHat)) dHat = m; });
    var curv = ell2(dHat), lapOK = curv < 0;
    var lapVar = lapOK ? -1.0 / curv : NaN, lapSd = Math.sqrt(lapVar);
    var lapF = lapOK ? -ell(dHat) - 0.5 * Math.log(2.0 * Math.PI * lapVar) : NaN;
    var lapSpill = lapOK ? 0.5 * (1.0 - erf(dHat / (Math.SQRT2 * lapSd))) : NaN;

    /* log-normal: q(d) = LN(m, S), F(m, S) in closed form, as _F_lognormal in values.py */
    function lnmom(m, S) {
      return [Math.exp(m + 0.5 * S), Math.exp(2 * m + 2 * S), Math.exp(-2 * m + 2 * S), Math.exp(-4 * m + 8 * S)];
    }
    function fLN(x) {
      var m = x[0], S = Math.exp(x[1]), e = lnmom(m, S);
      var ePrior = -(e[1] - 2.0 * dp * e[0] + dp * dp) / (2.0 * vp) + cPrior;
      var eLik = -(u * u - 2.0 * u * th * e[2] + th * th * e[3]) / (2.0 * vu) + cLik;
      return -(ePrior + eLik) - (0.5 * Math.log(2.0 * Math.PI * Math.E * S) + m);
    }
    function gLN(x) {
      var m = x[0], S = Math.exp(x[1]), e = lnmom(m, S);
      var gm = (e[1] - dp * e[0]) / vp + 2 * th * (u * e[2] - th * e[3]) / vu - 1.0;
      var gS = (2 * e[1] - dp * e[0]) / (2 * vp) + 2 * th * (2 * th * e[3] - u * e[2]) / vu - 0.5 / S;
      return [gm, S * gS];
    }
    var starts = [[Math.log(dHat), Math.log(0.1)], [lnMean, Math.log(lnVar)]], best = null;
    starts.forEach(function (x0) {
      var r = newtonMin(fLN, ghFD(gLN, 1e-6), x0, 1e-10);
      if (!best || r.f < best.f) best = r;
    });
    var lnMr = best.x[0], lnSr = Math.exp(best.x[1]);
    return {
      surp: surp, exMean: exMean, exSd: Math.sqrt(exVar), zp: zp,
      dHat: dHat, nModes: modes.length, lapOK: lapOK, lapVar: lapVar, lapSd: lapSd, lapF: lapF,
      lapSpill: lapSpill,
      lnMr: lnMr, lnSr: lnSr, lnF: best.f, lnGmax: best.gmax,
      lnMean: Math.exp(lnMr + 0.5 * lnSr),
      lnSd: Math.sqrt((Math.exp(lnSr) - 1.0) * Math.exp(2.0 * lnMr + lnSr)),
      density: function (d) { return d > 0 ? Math.exp(ell(d) + surp) : 0; }
    };
  }

  /* ── Lesson 4.5: the two-cause model ────────────────────────────────────
   * s_i ~ Exp(lambda_i), u1 = s2 s1^2 + z1, u2 = s1 s2^2 + z2, unit noise.
   * Exact posterior by numerical integration in y_i = g_i^(1/3), where the
   * likelihood factorises (Jacobian determinant 3), as TrialGrid in twocause.py. */
  var TC_QUAD = { L: 10.0, hy: 1.5, ngrade: 14, ratio: 0.25, order: 8 };

  function yNodes(u, o) {
    var glo = Math.max(0.0, u - o.L), ghi = Math.max(u, 0.0) + o.L;
    var npan = Math.ceil((ghi - glo) / o.hy);
    var yb = linspace(glo, ghi, npan + 1).map(Math.cbrt);
    if (glo === 0.0) {
      var grade = [];
      for (var k = o.ngrade; k >= 1; k--) grade.push(yb[1] * Math.pow(o.ratio, k));
      yb = [0.0].concat(grade, yb.slice(1));
    }
    return glPanels(yb, o.order);
  }

  function TrialGrid(u1, u2, quad) {
    var o = Object.assign({}, TC_QUAD, quad || {});
    var a = yNodes(u1, o), b = yNodes(u2, o), n1 = a.x.length, n2 = b.x.length;
    this.s1 = new Float64Array(n1 * n2); this.s2 = new Float64Array(n1 * n2); this.lw = new Float64Array(n1 * n2);
    for (var i = 0; i < n1; i++) {
      var y1 = a.x[i], r1 = u1 - y1 * y1 * y1;
      for (var j = 0; j < n2; j++) {
        var y2 = b.x[j], r2 = u2 - y2 * y2 * y2, k = i * n2 + j;
        this.s1[k] = y1 * y1 / y2; this.s2[k] = y2 * y2 / y1;
        this.lw[k] = Math.log(a.w[i] * b.w[j]) - 0.5 * r1 * r1 - 0.5 * r2 * r2;
      }
    }
  }
  TrialGrid.prototype.post = function (r1, r2) {
    var n = this.lw.length, lf = new Float64Array(n), mx = -Infinity, k;
    for (k = 0; k < n; k++) { lf[k] = this.lw[k] - r1 * this.s1[k] - r2 * this.s2[k]; if (lf[k] > mx) mx = lf[k]; }
    var Z = 0, a1 = 0, a2 = 0, b11 = 0, b22 = 0, b12 = 0;
    for (k = 0; k < n; k++) {
      var p = Math.exp(lf[k] - mx), x = this.s1[k], y = this.s2[k];
      Z += p; a1 += p * x; a2 += p * y; b11 += p * x * x; b22 += p * y * y; b12 += p * x * y;
    }
    var m1 = a1 / Z, m2 = a2 / Z;
    var v1 = b11 / Z - m1 * m1, v2 = b22 / Z - m2 * m2, c = b12 / Z - m1 * m2;
    return {
      nll: -(mx + Math.log(Z) + Math.log(3.0 * r1 * r2) - LN2PI),
      m1: m1, m2: m2, sd1: Math.sqrt(v1), sd2: Math.sqrt(v2), corr: c / Math.sqrt(v1 * v2)
    };
  };

  /* Product of Exponentials, x = (ln m1, ln m2). */
  function Fexp(x, u1, u2, r1, r2) {
    var a = x[0], b = x[1];
    return (-a - Math.log(r1) + r1 * Math.exp(a) - 1.0 - b - Math.log(r2) + r2 * Math.exp(b) - 1.0 + LN2PI
      + 0.5 * u1 * u1 - 2.0 * u1 * Math.exp(2 * a + b) + 24.0 * Math.exp(4 * a + 2 * b)
      + 0.5 * u2 * u2 - 2.0 * u2 * Math.exp(a + 2 * b) + 24.0 * Math.exp(2 * a + 4 * b));
  }
  function ghExp(x, u1, u2, r1, r2) {
    var a = x[0], b = x[1], Aa = Math.exp(a), Bb = Math.exp(b);
    var E1 = Math.exp(2 * a + b), E2 = Math.exp(4 * a + 2 * b), E3 = Math.exp(a + 2 * b), E4 = Math.exp(2 * a + 4 * b);
    var g1 = -1.0 + r1 * Aa - 4 * u1 * E1 + 96 * E2 - 2 * u2 * E3 + 48 * E4;
    var g2 = -1.0 + r2 * Bb - 2 * u1 * E1 + 48 * E2 - 4 * u2 * E3 + 96 * E4;
    var h11 = r1 * Aa - 8 * u1 * E1 + 384 * E2 - 2 * u2 * E3 + 96 * E4;
    var h12 = -4 * u1 * E1 + 192 * E2 - 4 * u2 * E3 + 192 * E4;
    var h22 = r2 * Bb - 2 * u1 * E1 + 96 * E2 - 8 * u2 * E3 + 384 * E4;
    return { g: [g1, g2], H: [[h11, h12], [h12, h22]] };
  }
  var EXP_TOL = 1e-10, GAM_TOL = 1e-9, FF_TOL = 1e-12;
  var EXP_STARTS = [], GAM_STARTS = [];
  [-2.0, 0.0, 1.5].forEach(function (a) { [-2.0, 0.0, 1.5].forEach(function (b) { EXP_STARTS.push([a, b]); }); });
  [-1.0, 0.0, 1.0].forEach(function (a) { [0.0, 2.0, 5.0].forEach(function (k) { GAM_STARTS.push([a, a, k, k]); }); });

  function multistart(f, gh, starts, tol) {
    var best = null;
    starts.forEach(function (x0) {
      var r = newtonMin(f, gh, x0, tol);
      if (best === null || r.f < best.f) best = r;
    });
    return best;
  }

  function solveExp(u1, u2, r1, r2) {
    var r = multistart(function (x) { return Fexp(x, u1, u2, r1, r2); },
      function (x) { return ghExp(x, u1, u2, r1, r2); }, EXP_STARTS, EXP_TOL);
    return { x: r.x, F: r.f, gmax: r.gmax, m1: Math.exp(r.x[0]), m2: Math.exp(r.x[1]) };
  }

  /* Product of Gammas, x = (ln m1, ln m2, ln k1, ln k2). */
  function gamPieces(x) {
    var k1 = Math.exp(x[2]), k2 = Math.exp(x[3]);
    var c21 = 1 + 1 / k1, c22 = 1 + 1 / k2;
    return {
      m1: Math.exp(x[0]), m2: Math.exp(x[1]), k1: k1, k2: k2, c21: c21, c22: c22,
      c41: c21 * (1 + 2 / k1) * (1 + 3 / k1), c42: c22 * (1 + 2 / k2) * (1 + 3 / k2)
    };
  }
  function Fgam(x, u1, u2, r1, r2) {
    var P = gamPieces(x);
    var kl1 = -lgamma(P.k1) + (P.k1 - 1) * digamma(P.k1) + Math.log(P.k1) - P.k1 - x[0] - Math.log(r1) + r1 * P.m1;
    var kl2 = -lgamma(P.k2) + (P.k2 - 1) * digamma(P.k2) + Math.log(P.k2) - P.k2 - x[1] - Math.log(r2) + r2 * P.m2;
    var e1 = u1 * u1 - 2 * u1 * P.m2 * P.m1 * P.m1 * P.c21 + P.m2 * P.m2 * P.c22 * Math.pow(P.m1, 4) * P.c41;
    var e2 = u2 * u2 - 2 * u2 * P.m1 * P.m2 * P.m2 * P.c22 + P.m1 * P.m1 * P.c21 * Math.pow(P.m2, 4) * P.c42;
    return kl1 + kl2 + LN2PI + 0.5 * e1 + 0.5 * e2;
  }
  function gradGam(x, u1, u2, r1, r2) {
    var P = gamPieces(x), k1 = P.k1, k2 = P.k2;
    var dc21 = -1 / k1, dc22 = -1 / k2;
    var dc41 = -P.c41 * (1 / (k1 + 1) + 2 / (k1 + 2) + 3 / (k1 + 3));
    var dc42 = -P.c42 * (1 / (k2 + 1) + 2 / (k2 + 2) + 3 / (k2 + 3));
    var T1 = u1 * P.m2 * P.m1 * P.m1 * P.c21, Q1 = P.m2 * P.m2 * P.c22 * Math.pow(P.m1, 4) * P.c41;
    var T2 = u2 * P.m1 * P.m2 * P.m2 * P.c22, Q2 = P.m1 * P.m1 * P.c21 * Math.pow(P.m2, 4) * P.c42;
    var dkl1 = k1 * ((k1 - 1) * trigamma(k1) + 1 / k1 - 1);
    var dkl2 = k2 * ((k2 - 1) * trigamma(k2) + 1 / k2 - 1);
    return [
      -1 + r1 * P.m1 + (-2 * T1 + 2 * Q1) + (-T2 + Q2),
      -1 + r2 * P.m2 + (-T1 + Q1) + (-2 * T2 + 2 * Q2),
      dkl1 - T1 * dc21 / P.c21 + 0.5 * Q1 * dc41 / P.c41 + 0.5 * Q2 * dc21 / P.c21,
      dkl2 + 0.5 * Q1 * dc22 / P.c22 - T2 * dc22 / P.c22 + 0.5 * Q2 * dc42 / P.c42
    ];
  }
  function solveGam(u1, u2, r1, r2) {
    var r = multistart(function (x) { return Fgam(x, u1, u2, r1, r2); },
      ghFD(function (x) { return gradGam(x, u1, u2, r1, r2); }), GAM_STARTS, GAM_TOL);
    return {
      x: r.x, F: r.f, gmax: r.gmax, m1: Math.exp(r.x[0]), m2: Math.exp(r.x[1]),
      k1: Math.exp(r.x[2]), k2: Math.exp(r.x[3])
    };
  }

  /* Free-form mean-field: each factor is exp(a s^4 + b s^2 + c s) on s > 0 with a < 0.
   * A 256-point scan finds where the exponent is within 60 nats of its maximum, then
   * 32 panels of 8-point Gauss-Legendre on that interval, as quartic_moments. */
  var FF_UNIT = glPanels(linspace(0, 1, 33), 8);
  function quarticMoments(a, b, c) {
    var NC = 256, shi = 2.0 * Math.max(1.0, Math.sqrt(Math.abs(b / a)), Math.cbrt(Math.abs(c / a))) + 1.0;
    var P = function (s) { return a * s * s * s * s + b * s * s + c * s; };
    var pc = new Float64Array(NC), pmax = -Infinity, i;
    for (i = 0; i < NC; i++) { pc[i] = P(shi * i / (NC - 1)); if (pc[i] > pmax) pmax = pc[i]; }
    var first = NC, last = -1;
    for (i = 0; i < NC; i++) if (pc[i] > pmax - 60.0) { if (i < first) first = i; last = i; }
    var lo = shi * Math.max(first - 1, 0) / (NC - 1), hi = shi * Math.min(last + 1, NC - 1) / (NC - 1);
    var n = FF_UNIT.x.length, ps = new Float64Array(n), ss = new Float64Array(n), m = -Infinity;
    for (i = 0; i < n; i++) { ss[i] = lo + (hi - lo) * FF_UNIT.x[i]; ps[i] = P(ss[i]); if (ps[i] > m) m = ps[i]; }
    var Z = 0, e1 = 0, e2 = 0, e4 = 0;
    for (i = 0; i < n; i++) {
      var w = Math.exp(ps[i] - m) * (hi - lo) * FF_UNIT.w[i], s = ss[i], s2 = s * s;
      Z += w; e1 += w * s; e2 += w * s2; e4 += w * s2 * s2;
    }
    return { lnZ: m + Math.log(Z), e1: e1 / Z, e2: e2 / Z, e4: e4 / Z };
  }
  function Fff(u1, u2, r1, r2, q1, q2, co1, co2) {
    var elnq1 = co1[0] * q1.e4 + co1[1] * q1.e2 + co1[2] * q1.e1 - q1.lnZ;
    var elnq2 = co2[0] * q2.e4 + co2[1] * q2.e2 + co2[2] * q2.e1 - q2.lnZ;
    var elnp = Math.log(r1) + Math.log(r2) - r1 * q1.e1 - r2 * q2.e1 - LN2PI
      - 0.5 * (u1 * u1 - 2 * u1 * q2.e1 * q1.e2 + q2.e2 * q1.e4)
      - 0.5 * (u2 * u2 - 2 * u2 * q2.e2 * q1.e1 + q2.e4 * q1.e2);
    return elnq1 + elnq2 - elnp;
  }
  /* Sequential coordinate ascent, q1 then q2, from q2 equal to the prior Exp(lambda2).
   * Stops when one sweep changes both means by a relative amount below 1e-12. */
  function cavi(u1, u2, r1, r2, maxsweep) {
    var q2 = { e1: 1 / r2, e2: 2 / (r2 * r2), e4: 24 / Math.pow(r2, 4) }, q1 = { e1: 0 };
    var F = Infinity, rose = 0, sw, co1, co2, conv = false;
    maxsweep = maxsweep || 20000;
    for (sw = 1; sw <= maxsweep; sw++) {
      co1 = [-0.5 * q2.e2, u1 * q2.e1 - 0.5 * q2.e4, u2 * q2.e2 - r1];
      var n1 = quarticMoments(co1[0], co1[1], co1[2]);
      co2 = [-0.5 * n1.e2, u2 * n1.e1 - 0.5 * n1.e4, u1 * n1.e2 - r2];
      var n2 = quarticMoments(co2[0], co2[1], co2[2]);
      var Fn = Fff(u1, u2, r1, r2, n1, n2, co1, co2);
      var dm = Math.max(Math.abs(n1.e1 - q1.e1) / n1.e1, Math.abs(n2.e1 - q2.e1) / n2.e1);
      if (Fn > F + 1e-9 * (1 + Math.abs(F))) rose++;
      q1 = n1; q2 = n2; F = Fn;
      if (dm < FF_TOL) { conv = true; break; }
    }
    return {
      q1: q1, q2: q2, co1: co1, co2: co2, F: F, sweeps: sw, converged: conv, rose: rose,
      m1: q1.e1, m2: q2.e1, sd1: Math.sqrt(q1.e2 - q1.e1 * q1.e1), sd2: Math.sqrt(q2.e2 - q2.e1 * q2.e1)
    };
  }

  var GRID_CACHE = { key: null, grid: null };
  function solveTwoCause(u1, u2, r1, r2) {
    var key = u1 + ',' + u2;
    if (GRID_CACHE.key !== key) { GRID_CACHE.key = key; GRID_CACHE.grid = new TrialGrid(u1, u2); }
    return {
      ex: GRID_CACHE.grid.post(r1, r2),
      exp: solveExp(u1, u2, r1, r2),
      gam: solveGam(u1, u2, r1, r2),
      ff: cavi(u1, u2, r1, r2)
    };
  }

  A.w04 = {
    erf: erf, lgamma: lgamma, digamma: digamma, trigamma: trigamma, glRule: glRule, symEig: symEig,
    decomp3: decomp3, solveSound: solveSound, TrialGrid: TrialGrid, solveExp: solveExp,
    solveGam: solveGam, cavi: cavi, solveTwoCause: solveTwoCause, Fexp: Fexp, Fgam: Fgam, gradGam: gradGam
  };

  /* ── Shared canvas sub-panel helper ─────────────────────────────────── */
  function subPanel(p, x, y, w, h, xlim, ylim, title) {
    var c = p.ctx;
    c.strokeStyle = C.rule2;
    c.lineWidth = 1;
    c.strokeRect(x + 0.5, y + 0.5, w, h);
    if (title) {
      p.text(x, y - 9, title, { pixel: true, align: 'left', size: 10.5, colour: C.ink2, weight: '600' });
    }
    return {
      x: x, y: y, w: w, h: h, xlim: xlim, ylim: ylim,
      X: function (v) { return x + ((v - xlim[0]) / (xlim[1] - xlim[0])) * w; },
      Y: function (v) { return y + h - ((v - ylim[0]) / (ylim[1] - ylim[0])) * h; }
    };
  }

  /* Marching squares contour segments for a NY x NX grid */
  function drawContour(ctx, P, grid, nx, ny, xs, ys, level, col, width, dash) {
    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = width || 1.6;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath();
    for (var j = 0; j < ny - 1; j++) {
      var y0 = ys[j], y1 = ys[j + 1];
      for (var i = 0; i < nx - 1; i++) {
        var x0 = xs[i], x1 = xs[i + 1];
        var v00 = grid[j * nx + i] - level;
        var v10 = grid[j * nx + i + 1] - level;
        var v11 = grid[(j + 1) * nx + i + 1] - level;
        var v01 = grid[(j + 1) * nx + i] - level;
        var code = (v00 >= 0 ? 1 : 0) | (v10 >= 0 ? 2 : 0) | (v11 >= 0 ? 4 : 0) | (v01 >= 0 ? 8 : 0);
        if (code === 0 || code === 15) continue;
        var pts = [];
        /* bottom edge: (x0,y0)-(x1,y0) */
        if ((v00 >= 0) !== (v10 >= 0)) pts.push([x0 + (v00 / (v00 - v10)) * (x1 - x0), y0]);
        /* right edge: (x1,y0)-(x1,y1) */
        if ((v10 >= 0) !== (v11 >= 0)) pts.push([x1, y0 + (v10 / (v10 - v11)) * (y1 - y0)]);
        /* top edge: (x0,y1)-(x1,y1) */
        if ((v01 >= 0) !== (v11 >= 0)) pts.push([x0 + (v01 / (v01 - v11)) * (x1 - x0), y1]);
        /* left edge: (x0,y0)-(x0,y1) */
        if ((v00 >= 0) !== (v01 >= 0)) pts.push([x0, y0 + (v00 / (v00 - v01)) * (y1 - y0)]);
        if (pts.length === 2) {
          ctx.moveTo(P.X(pts[0][0]), P.Y(pts[0][1]));
          ctx.lineTo(P.X(pts[1][0]), P.Y(pts[1][1]));
        } else if (pts.length === 4) {
          ctx.moveTo(P.X(pts[0][0]), P.Y(pts[0][1]));
          ctx.lineTo(P.X(pts[1][0]), P.Y(pts[1][1]));
          ctx.moveTo(P.X(pts[2][0]), P.Y(pts[2][1]));
          ctx.lineTo(P.X(pts[3][0]), P.Y(pts[3][1]));
        }
      }
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ══════════════════════════════════════════════════════════════════════
   * 1. three-decompositions (Lesson 4.1)
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('three-decompositions', function (mount) {
    var oIdx = 0;
    var st = null;

    var s1 = A.slider({ label: 'weight on leopard, <i>w</i><sub>1</sub>', min: 0.001, max: 1.0, step: 0.0001, value: 0.35, decimals: 2 });
    var s2 = A.slider({ label: 'weight on baboon, <i>w</i><sub>2</sub>', min: 0.001, max: 1.0, step: 0.0001, value: 0.40, decimals: 2 });
    var s3 = A.slider({ label: 'weight on nothing, <i>w</i><sub>3</sub>', min: 0.001, max: 1.0, step: 0.0001, value: 0.25, decimals: 2 });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.44, pad: { l: 42, r: 16, t: 26, b: 36 } });

    var obsBtns = OBS_NAMES.map(function (nm, idx) {
      return A.button('o = ' + nm, function () {
        oIdx = idx;
        syncObsBtns();
        update();
      }, idx !== 0);
    });

    function syncObsBtns() {
      obsBtns.forEach(function (b, idx) {
        b.className = 'wbtn' + (idx === oIdx ? '' : ' ghost');
      });
    }

    var bPost = A.button('Set Q = P(s | o)', function () {
      var cur = decomp3([s1.get(), s2.get(), s3.get()], oIdx);
      s1.set(cur.post[0]); s2.set(cur.post[1]); s3.set(cur.post[2]);
      update();
    }, true);
    var bPrior = A.button('Set Q = P(s)', function () {
      s1.set(PRIOR_3[0]); s2.set(PRIOR_3[1]); s3.set(PRIOR_3[2]);
      update();
    }, true);
    var bSharp = A.button('Sharp spike', function () {
      s1.set(0.96); s2.set(0.02); s3.set(0.02);
      update();
    }, true);

    function update() {
      st = decomp3([s1.get(), s2.get(), s3.get()], oIdx);
      out.show([
        ['Q(s)', '(' + st.q.map(function (v) { return v.toFixed(2); }).join(', ') + ')'],
        ['F[Q, o]', st.F.toFixed(4) + ' nats'],
        ['−ln P(o)', st.surp.toFixed(4) + ' nats'],
        ['KL[Q || P(s|o)]', st.klpost.toFixed(4)],
        ['U[Q, o]', st.U.toFixed(4)],
        ['H[Q]', st.H.toFixed(4)],
        ['complexity', st.comp.toFixed(4)],
        ['accuracy', fmtSigned(st.acc, 4)]
      ]);
      plot.render();
    }

    plot.onDraw(function (p) {
      p.clear();
      if (!st) return;
      var c = p.ctx, gap = 50;
      var pw = (p.plotW() - gap) / 2, ph = p.plotH();
      var lx = p.o.pad.l, rx = lx + pw + gap, ty = p.o.pad.t;

      /* Left panel: distributions over the 3 hidden states */
      var L = subPanel(p, lx, ty, pw, ph, [0, 3], [0, 1.05], 'prior P(s), posterior P(s|o), belief Q(s)');
      [0, 0.25, 0.5, 0.75, 1.0].forEach(function (tv) {
        c.strokeStyle = C.rule; c.beginPath(); c.moveTo(lx, L.Y(tv)); c.lineTo(lx + pw, L.Y(tv)); c.stroke();
        p.text(lx - 6, L.Y(tv), tv.toFixed(2), { pixel: true, align: 'right', size: 9.5, colour: C.faint });
      });
      var series = [
        { vals: st.prior, col: C.rule2, lab: 'P(s)' },
        { vals: st.post, col: C.accent, lab: 'P(s|o)' },
        { vals: st.q, col: C.blue, lab: 'Q(s)' }
      ];
      var slotW = pw / 3, padX = slotW * 0.16, bw = (slotW - 2 * padX) / 3;
      for (var si = 0; si < 3; si++) {
        for (var gi = 0; gi < 3; gi++) {
          var v = series[gi].vals[si];
          var bx = lx + si * slotW + padX + gi * bw;
          var by = L.Y(v), b0 = L.Y(0);
          p.rect(bx, by, bw - 1.5, b0 - by, { fill: series[gi].col });
        }
        p.text(lx + (si + 0.5) * slotW, ty + ph + 14, STATE_NAMES[si], { pixel: true, size: 10, colour: C.ink2 });
      }
      /* Compact legend inside left panel */
      series.forEach(function (sr, k) {
        var kx = lx + 10 + k * 64, ky = ty + 13;
        p.rect(kx, ky - 4, 9, 8, { fill: sr.col });
        p.text(kx + 13, ky, sr.lab, { pixel: true, align: 'left', size: 9.5, colour: C.ink2 });
      });

      /* Right panel: three decompositions of F[Q, o] */
      var ymax = Math.max(st.U, st.F, st.surp, 3.2) * 1.14;
      var R = subPanel(p, rx, ty, pw, ph, [0, 3], [0, ymax], 'three decompositions of F[Q, o] (nats)');
      var stepT = ymax > 5 ? 2 : 1;
      for (var yv = 0; yv <= ymax; yv += stepT) {
        c.strokeStyle = C.rule; c.beginPath(); c.moveTo(rx, R.Y(yv)); c.lineTo(rx + pw, R.Y(yv)); c.stroke();
        p.text(rx - 6, R.Y(yv), yv.toFixed(0), { pixel: true, align: 'right', size: 9.5, colour: C.faint });
      }

      var rSlot = pw / 3, rBw = rSlot * 0.48;
      var yZero = R.Y(0), yF = R.Y(st.F), ySurp = R.Y(st.surp);

      /* Column 0: U - H = F (show full U outline + solid F + subtracted H cap) */
      var x0 = rx + 0.5 * rSlot - rBw / 2;
      var yU = R.Y(st.U);
      p.rect(x0, yF, rBw, yZero - yF, { fill: C.blue, alpha: 0.85 });
      if (st.H > 1e-4) {
        p.rect(x0, yU, rBw, yF - yU, { fill: C.goldTint, stroke: C.gold, width: 1 });
        p.text(x0 + rBw / 2, (yU + yF) / 2, '−H', { pixel: true, size: 9, colour: C.gold, weight: '600' });
      }
      p.text(x0 + rBw / 2, (yF + yZero) / 2, 'U−H', { pixel: true, size: 9, colour: '#ffffff', weight: '600' });

      /* Column 1: Complexity + (-Accuracy) = F */
      var x1 = rx + 1.5 * rSlot - rBw / 2;
      var negAcc = -st.acc;
      var yComp = R.Y(st.comp);
      p.rect(x1, yComp, rBw, yZero - yComp, { fill: C.violet, alpha: 0.82 });
      p.rect(x1, yF, rBw, yComp - yF, { fill: C.clay, alpha: 0.78 });
      if (yZero - yComp > 14) p.text(x1 + rBw / 2, (yComp + yZero) / 2, 'comp', { pixel: true, size: 8.5, colour: '#ffffff' });
      if (yComp - yF > 14 && negAcc > 0) p.text(x1 + rBw / 2, (yF + yComp) / 2, '−acc', { pixel: true, size: 8.5, colour: '#ffffff' });

      /* Column 2: Surprise + KL[Q || P(s|o)] = F */
      var x2 = rx + 2.5 * rSlot - rBw / 2;
      p.rect(x2, ySurp, rBw, yZero - ySurp, { fill: C.accent, alpha: 0.85 });
      if (st.klpost > 1e-4) {
        p.rect(x2, yF, rBw, ySurp - yF, { fill: C.clayTint, stroke: C.clay, width: 1 });
        if (ySurp - yF > 12) p.text(x2 + rBw / 2, (yF + ySurp) / 2, 'KL', { pixel: true, size: 8.5, colour: C.clay, weight: '600' });
      }
      if (yZero - ySurp > 14) p.text(x2 + rBw / 2, (ySurp + yZero) / 2, '−ln P(o)', { pixel: true, size: 8.5, colour: '#ffffff' });

      /* Horizontal level lines across all 3 columns: F[Q,o] and surprise floor */
      c.save();
      c.strokeStyle = C.ink; c.lineWidth = 1.3; c.setLineDash([2, 2]);
      c.beginPath(); c.moveTo(rx, yF); c.lineTo(rx + pw, yF); c.stroke();
      c.strokeStyle = C.accent; c.lineWidth = 1.4; c.setLineDash([5, 3]);
      c.beginPath(); c.moveTo(rx, ySurp); c.lineTo(rx + pw, ySurp); c.stroke();
      c.restore();

      p.text(rx + pw - 3, yF - 7, 'F = ' + st.F.toFixed(2), { pixel: true, align: 'right', size: 9.5, colour: C.ink, weight: '600' });

      ['U − H', 'comp − acc', 'surp + KL'].forEach(function (lb, k) {
        p.text(rx + (k + 0.5) * rSlot, ty + ph + 14, lb, { pixel: true, size: 9.8, colour: C.ink2 });
      });
    });

    s1.onchange(update); s2.onchange(update); s3.onchange(update);

    var btnRow = el('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;align-items:center;' },
      obsBtns.concat([el('span', { style: 'width:10px' }), bPost, bPrior, bSharp]));
    var body = el('div', {});
    mount.appendChild(A.panel([
      btnRow,
      A.row([s1.el, s2.el, s3.el]),
      body, out,
      A.note('The three sliders set unnormalised weights that are normalised to <i>Q</i>(<i>s</i>). '
        + 'In the right panel, every column lands on the same free energy <i>F</i>[<i>Q</i>, <i>o</i>] '
        + 'at or above the dashed green surprise floor −ln&nbsp;<i>P</i>(<i>o</i>). Clicking '
        + '<i>Set Q = P(s | o)</i> eliminates the top KL block in the third column without changing the '
        + 'observation floor.')
    ]));
    update();
    plot.attach(body);
    return null;
  });


  /* ══════════════════════════════════════════════════════════════════════
   * 2. laplace-vs-lognormal (Lesson 4.3)
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('laplace-vs-lognormal', function (mount) {
    var st = null;
    var sU = A.slider({ label: 'observation, <i>u</i>', min: -0.20, max: 1.50, step: 0.05, value: 0.50 });
    var sVu = A.slider({ label: 'sensory variance, &Sigma;<sub>u</sub>', min: 0.02, max: 0.40, step: 0.01, value: 0.10 });
    var sVp = A.slider({ label: 'prior variance, &Sigma;<sub>p</sub>', min: 0.20, max: 2.50, step: 0.05, value: 1.00 });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.44, pad: { l: 46, r: 16, t: 22, b: 36 }, xlim: [-0.8, 5.2], ylim: [0, 1] });

    function update() {
      st = solveSound(sU.get(), sVu.get(), sVp.get());
      var rows = [
        ['−ln P(u)', fmtSigned(st.surp, 4) + ' nats'],
        ['F, log-normal Q', fmtSigned(st.lnF, 4) + ' nats (above −ln P(u) by ' + (st.lnF - st.surp).toFixed(4) + ')'],
        ['F_Lap', st.lapOK ? fmtSigned(st.lapF, 4) + ' nats (minus −ln P(u): ' + fmtSigned(st.lapF - st.surp, 4) + ')'
          : 'undefined: zero curvature at the mode'],
        ['posterior mean', st.exMean.toFixed(4)],
        ['log-normal mean', st.lnMean.toFixed(4)],
        ['mode d̂', st.dHat.toFixed(4) + (st.nModes > 1 ? ' (highest of ' + st.nModes + ' modes)' : '')],
        ['Laplace mass at d ≤ 0', st.lapOK ? (100.0 * st.lapSpill).toFixed(2) + '%' : 'undefined']
      ];
      out.show(rows);
      plot.render();
    }

    plot.onDraw(function (p) {
      p.clear();
      if (!st) return;
      var xlo = -0.8, xhi = 5.2, N = 400;
      var exactPts = [], lnPts = [], lapPts = [], spillPts = [];
      var ymax = 0.2, i;
      for (i = 0; i <= N; i++) {
        var d = xlo + (i / N) * (xhi - xlo);
        if (d > 0) {
          var yEx = st.density(d);
          exactPts.push([d, yEx]);
          if (yEx > ymax) ymax = yEx;
          var z = (Math.log(d) - st.lnMr) / Math.sqrt(st.lnSr);
          var yLn = Math.exp(-0.5 * z * z) / (d * Math.sqrt(2.0 * Math.PI * st.lnSr));
          lnPts.push([d, yLn]);
          if (yLn > ymax) ymax = yLn;
        } else {
          exactPts.push([d, 0]);
          lnPts.push([d, 0]);
        }
        if (st.lapOK) {
          var yLap = Math.exp(-0.5 * (d - st.dHat) * (d - st.dHat) / st.lapVar) / Math.sqrt(2.0 * Math.PI * st.lapVar);
          lapPts.push([d, yLap]);
          if (yLap > ymax) ymax = yLap;
          if (d <= 0) spillPts.push([d, yLap]);
        }
      }
      if (spillPts.length) spillPts.push([0, spillPts[spillPts.length - 1][1]]);
      p.o.ylim = [0, ymax * 1.12];
      var yStep = ymax > 2.5 ? 1.0 : ymax > 1.1 ? 0.5 : 0.2;
      var yTicks = [];
      for (var yv = 0; yv <= p.o.ylim[1]; yv += yStep) yTicks.push(+yv.toFixed(2));
      p.grid(yTicks);
      p.frame();
      p.yaxis(yTicks, function (v) { return v.toFixed(1); }, { label: 'density' });
      p.xaxis([0, 1, 2, 3, 4, 5], function (v) { return String(v); }, { label: 'distance d' });

      var xZero = p.px(0), xLeft = p.px(xlo), yTop = p.o.pad.t, yBot = p.h - p.o.pad.b;
      p.rect(xLeft, yTop, xZero - xLeft, yBot - yTop, { fill: C.tint, alpha: 0.65 });
      if (spillPts.length > 1) p.area(spillPts, { colour: C.clay, alpha: 0.22, base: 0 });
      var c = p.ctx;
      c.save();
      c.strokeStyle = C.rule2; c.lineWidth = 1.2; c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(xZero, yTop); c.lineTo(xZero, yBot); c.stroke();
      c.restore();
      p.text(xZero - 5, yTop + 11, 'd ≤ 0', { pixel: true, align: 'right', size: 9.5, colour: C.muted });

      p.line(exactPts, { colour: C.accent, width: 2.4 });
      p.line(lnPts, { colour: C.blue, width: 2.0 });
      if (st.lapOK) p.line(lapPts, { colour: C.clay, width: 2.0, dash: [6, 4] });

      p.legend([
        { colour: C.accent, label: 'exact posterior' },
        { colour: C.blue, label: 'log-normal Q' },
        { colour: C.clay, label: 'Laplace at the mode', dash: true }
      ], p.o.pad.l + Math.max(70, xZero - xLeft + 14), yTop + 12, { gap: 22 });
    });

    sU.onchange(update); sVu.onchange(update); sVp.onchange(update);

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sU.el, sVu.el, sVp.el]),
      body, out,
      A.note('The prior mean <i>d</i><sub>p</sub> = 2 and the source power &theta; = 1 are '
        + 'Week&nbsp;2&rsquo;s values. The shaded band left of <i>d</i> = 0 holds the mass the '
        + 'Laplace Gaussian puts on negative distances; the log-normal density is zero there. The '
        + 'exact surprise and moments come from Gauss&ndash;Legendre quadrature, and the log-normal '
        + 'parameters from Newton&rsquo;s method on the closed-form free energy of this lesson.')
    ]));
    update();
    plot.attach(body);
    return null;
  });

  /* ══════════════════════════════════════════════════════════════════════
   * 3. mean-field-failure (Lesson 4.5)
   * ═════════════════════════════════════════════════════════════════════ */
  function gammaLogpdf(s, m, k) {
    return k * Math.log(k / m) - lgamma(k) + (k - 1) * Math.log(s) - k * s / m;
  }

  /* Density thresholds whose super-level sets hold the given masses (P sums to one). */
  function hpdLevels(P, masses) {
    var flat = Array.prototype.slice.call(P).sort(function (a, b) { return b - a; });
    return masses.map(function (m) {
      var cum = 0;
      for (var i = 0; i < flat.length; i++) { cum += flat[i]; if (cum >= m) return flat[i]; }
      return flat[flat.length - 1];
    });
  }

  A.register('mean-field-failure', function (mount) {
    var st = null;
    var sU1 = A.slider({ label: 'sensor 1, <i>u</i><sub>1</sub>', min: 0.50, max: 5.00, step: 0.10, value: 2.50 });
    var sU2 = A.slider({ label: 'sensor 2, <i>u</i><sub>2</sub>', min: 0.50, max: 5.00, step: 0.10, value: 3.00 });
    var sR1 = A.slider({ label: 'prior rate, &lambda;<sub>1</sub>', min: 0.30, max: 2.00, step: 0.05, value: 0.90 });
    var sR2 = A.slider({ label: 'prior rate, &lambda;<sub>2</sub>', min: 0.30, max: 2.00, step: 0.05, value: 0.70 });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.5, pad: { l: 44, r: 16, t: 26, b: 36 } });
    var NG = 161;
    var FAM = [
      { key: 'ex', lab: 'exact posterior', col: C.accent, dash: null },
      { key: 'ff', lab: 'free-form product', col: C.blue, dash: null },
      { key: 'gam', lab: 'Gamma product', col: C.violet, dash: [6, 3] },
      { key: 'exp', lab: 'Exponential product', col: C.clay, dash: [2, 3] }
    ];

    function densities(u1, u2, r1, r2, sol) {
      var mx1 = 0, mx2 = 0;
      [[sol.ex.m1, sol.ex.sd1, sol.ex.m2, sol.ex.sd2], [sol.ff.m1, sol.ff.sd1, sol.ff.m2, sol.ff.sd2],
        [sol.gam.m1, sol.gam.m1 / Math.sqrt(sol.gam.k1), sol.gam.m2, sol.gam.m2 / Math.sqrt(sol.gam.k2)]]
        .forEach(function (r) { mx1 = Math.max(mx1, r[0] + 4.5 * r[1]); mx2 = Math.max(mx2, r[2] + 4.5 * r[3]); });
      mx1 = Math.max(mx1, 4 * sol.exp.m1); mx2 = Math.max(mx2, 4 * sol.exp.m2);
      var g1 = [], g2 = [], i, j;
      for (i = 0; i < NG; i++) { g1.push(mx1 * (i + 0.5) / NG); g2.push(mx2 * (i + 0.5) / NG); }
      var co1 = sol.ff.co1, co2 = sol.ff.co2, gm = sol.gam, ep = sol.exp;
      var logd = {
        ex: function (a, b) { var d1 = u1 - b * a * a, d2 = u2 - a * b * b; return -r1 * a - r2 * b - 0.5 * (d1 * d1 + d2 * d2); },
        ff: function (a, b) { return co1[0] * a * a * a * a + co1[1] * a * a + co1[2] * a + co2[0] * b * b * b * b + co2[1] * b * b + co2[2] * b; },
        gam: function (a, b) { return gammaLogpdf(a, gm.m1, gm.k1) + gammaLogpdf(b, gm.m2, gm.k2); },
        exp: function (a, b) { return -a / ep.m1 - b / ep.m2; }
      };
      var grids = {}, levels = {}, marg = {};
      FAM.forEach(function (f) {
        var L = new Float64Array(NG * NG), mx = -Infinity, k;
        for (j = 0; j < NG; j++) for (i = 0; i < NG; i++) {
          k = j * NG + i; L[k] = logd[f.key](g1[i], g2[j]); if (L[k] > mx) mx = L[k];
        }
        var Z = 0;
        for (k = 0; k < L.length; k++) { L[k] = Math.exp(L[k] - mx); Z += L[k]; }
        for (k = 0; k < L.length; k++) L[k] /= Z;
        grids[f.key] = L;
        levels[f.key] = hpdLevels(L, [0.5, 0.9]);
        var p1 = new Array(NG).fill(0), p2 = new Array(NG).fill(0);
        for (j = 0; j < NG; j++) for (i = 0; i < NG; i++) { p1[i] += L[j * NG + i]; p2[j] += L[j * NG + i]; }
        marg[f.key] = [p1.map(function (v) { return v * NG / mx1; }), p2.map(function (v) { return v * NG / mx2; })];
      });
      return { g1: g1, g2: g2, mx1: mx1, mx2: mx2, grids: grids, levels: levels, marg: marg };
    }

    function update() {
      var u1 = sU1.get(), u2 = sU2.get(), r1 = sR1.get(), r2 = sR2.get();
      var sol = solveTwoCause(u1, u2, r1, r2);
      st = { sol: sol, dens: densities(u1, u2, r1, r2, sol) };
      var ex = sol.ex, ff = sol.ff, gm = sol.gam, ep = sol.exp;
      function pair(a, b, d) { return '(' + a.toFixed(d) + ', ' + b.toFixed(d) + ')'; }
      var rows = [
        ['−ln P(u)', ex.nll.toFixed(4) + ' nats'],
        ['exact', 'means ' + pair(ex.m1, ex.m2, 3) + ', sds ' + pair(ex.sd1, ex.sd2, 3) + ', correlation ' + fmtSigned(ex.corr, 3)],
        ['free-form', 'means ' + pair(ff.m1, ff.m2, 3) + ', sds ' + pair(ff.sd1, ff.sd2, 3) + ', KL ' + (ff.F - ex.nll).toFixed(4)],
        ['Gamma', 'means ' + pair(gm.m1, gm.m2, 3) + ', sds ' + pair(gm.m1 / Math.sqrt(gm.k1), gm.m2 / Math.sqrt(gm.k2), 3)
          + ', shapes ' + pair(gm.k1, gm.k2, 1) + ', KL ' + (gm.F - ex.nll).toFixed(4)],
        ['Exponential', 'means ' + pair(ep.m1, ep.m2, 3) + ', sds equal to the means, KL ' + (ep.F - ex.nll).toFixed(4)]
      ];
      var warn = [];
      if (ep.gmax > EXP_TOL) warn.push('Exponential gradient ' + ep.gmax.toExponential(1));
      if (gm.gmax > GAM_TOL) warn.push('Gamma gradient ' + gm.gmax.toExponential(1));
      if (!ff.converged) warn.push('free-form not converged after ' + ff.sweeps + ' sweeps');
      if (warn.length) rows.push(['solver stopped above tolerance', warn.join('; ')]);
      out.show(rows);
      plot.render();
    }

    plot.onDraw(function (p) {
      p.clear();
      if (!st) return;
      var c = p.ctx, gap = 56, D = st.dens, sol = st.sol;
      var ph = p.plotH(), pwL = Math.min((p.plotW() - gap) * 0.5, ph * 1.1);
      var pwR = p.plotW() - gap - pwL;
      var lx = p.o.pad.l, rx = lx + pwL + gap, ty = p.o.pad.t;

      var L = subPanel(p, lx, ty, pwL, ph, [0, D.mx1], [0, D.mx2], '50% and 90% regions in (s₁, s₂)');
      function ticks(mx) {
        var st0 = mx > 6 ? 2 : mx > 3 ? 1 : 0.5, t = [];
        for (var v = 0; v <= mx + 1e-9; v += st0) t.push(+v.toFixed(2));
        return t;
      }
      ticks(D.mx2).forEach(function (tv) {
        c.strokeStyle = C.rule; c.beginPath(); c.moveTo(lx, L.Y(tv)); c.lineTo(lx + pwL, L.Y(tv)); c.stroke();
        p.text(lx - 6, L.Y(tv), String(tv), { pixel: true, align: 'right', size: 9.5, colour: C.faint });
      });
      ticks(D.mx1).forEach(function (tv) {
        c.strokeStyle = C.rule; c.beginPath(); c.moveTo(L.X(tv), ty); c.lineTo(L.X(tv), ty + ph); c.stroke();
        p.text(L.X(tv), ty + ph + 13, String(tv), { pixel: true, size: 9.5, colour: C.faint });
      });
      p.text(lx + pwL / 2, p.h - 4, 's₁ (vertical axis: s₂)', { pixel: true, size: 10, colour: C.ink2, weight: '600' });

      FAM.slice().reverse().forEach(function (f) {
        var lv = D.levels[f.key];
        drawContour(c, L, D.grids[f.key], NG, NG, D.g1, D.g2, lv[1], f.col, 1.1, f.dash);
        drawContour(c, L, D.grids[f.key], NG, NG, D.g1, D.g2, lv[0], f.col, 2.0, f.dash);
      });
      [[sol.exp.m1, sol.exp.m2, C.clay], [sol.gam.m1, sol.gam.m2, C.violet],
        [sol.ff.m1, sol.ff.m2, C.blue], [sol.ex.m1, sol.ex.m2, C.accent]].forEach(function (d) {
        p.dot(L.X(d[0]), L.Y(d[1]), { pixel: true, r: 3.6, colour: d[2], ring: '#ffffff', ringWidth: 1.4 });
      });

      /* right: the two marginals, stacked */
      var hR = (ph - 30) / 2;
      [0, 1].forEach(function (ax) {
        var y0 = ty + ax * (hR + 30), mxS = ax === 0 ? D.mx1 : D.mx2, gs = ax === 0 ? D.g1 : D.g2, ymax = 0;
        FAM.forEach(function (f) { D.marg[f.key][ax].forEach(function (v) { if (v > ymax) ymax = v; }); });
        var R = subPanel(p, rx, y0, pwR, hR, [0, mxS], [0, ymax * 1.08], ax === 0 ? 'marginal density of s₁' : 'marginal density of s₂');
        ticks(mxS).forEach(function (tv) {
          p.text(R.X(tv), y0 + hR + 12, String(tv), { pixel: true, size: 9, colour: C.faint });
        });
        FAM.slice().reverse().forEach(function (f) {
          var pts = D.marg[f.key][ax];
          c.save();
          c.strokeStyle = f.col; c.lineWidth = f.key === 'ex' ? 2.2 : 1.7;
          if (f.dash) c.setLineDash(f.dash);
          c.beginPath();
          for (var i = 0; i < pts.length; i++) {
            if (i === 0) c.moveTo(R.X(gs[i]), R.Y(pts[i])); else c.lineTo(R.X(gs[i]), R.Y(pts[i]));
          }
          c.stroke();
          c.restore();
        });
      });

      FAM.forEach(function (f, idx) {
        var ky = ty + 12 + idx * 13, kx = lx + pwL - 118;
        c.save();
        c.strokeStyle = f.col; c.lineWidth = 2;
        if (f.dash) c.setLineDash(f.dash);
        c.beginPath(); c.moveTo(kx, ky); c.lineTo(kx + 16, ky); c.stroke();
        c.restore();
        p.text(kx + 20, ky, f.lab, { pixel: true, align: 'left', size: 9.2, colour: C.ink2 });
      });
    });

    sU1.onchange(update); sU2.onchange(update); sR1.onchange(update); sR2.onchange(update);

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sU1.el, sU2.el, sR1.el, sR2.el]),
      body, out,
      A.note('Each density is tabulated on a ' + NG + ' by ' + NG + ' grid to draw the contours and '
        + 'marginals; the readout comes from the lesson&rsquo;s solvers: numerical integration for the '
        + 'exact posterior, Newton&rsquo;s method from nine starts for the Exponential and Gamma products, '
        + 'and coordinate ascent for the free-form product. KL is each product&rsquo;s free energy minus '
        + '&minus;ln&nbsp;<i>P</i>(<i>u</i>). Dots mark the means.')
    ]));
    update();
    plot.attach(body);
    return null;
  });
})();
