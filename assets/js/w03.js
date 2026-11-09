/* w03.js — interactive figures for Week 3: learning the model. */
(function () {
  'use strict';
  var A = window.AIF, el = A.el, C = A.colours;

  /* ══════════════════════════════════════════════════════════════════════
   * The simulation engine, mirrored line for line from content/week-03/values.py.
   * Same generator, same inner loop, same update rules, in the same order of
   * floating-point operations, so that a readout at the default slider
   * positions agrees with the number the lesson prints beside it. Change one
   * side and the other must change with it; the parity check in the week's
   * report runs both.
   * ═════════════════════════════════════════════════════════════════════ */
  var ETA = 0.05, KMAX = 2000, TOL = 1e-12, NY = 801, NTRIALS = 2000;
  var WEEK2 = { dp: 2.0, vp: 1.0, th: 1.0, vu: 0.1 };

  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = ((t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normal(r) {
    var u1 = r();
    var u2 = r();
    return Math.sqrt(-2.0 * Math.log(1.0 - u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  /* Abramowitz and Stegun 7.1.26. JavaScript has no erf. */
  function erfAS(x) {
    var s = x >= 0 ? 1.0 : -1.0;
    x = Math.abs(x);
    var t = 1.0 / (1.0 + 0.3275911 * x);
    var y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  function truncRatio(dp, vp) {
    var dens = Math.exp(-dp * dp / (2.0 * vp)) / Math.sqrt(2.0 * Math.PI * vp);
    return dens / (0.5 * (1.0 + erfAS(dp / Math.sqrt(2.0 * vp))));
  }

  /* Week 2's ascent, capped at the reciprocal Gauss-Newton curvature and never
   * removing more than half the current distance. Returns the state, the
   * number of steps, and whether the stopping rule was met within KMAX. */
  function settle(u, dp, vp, th, vu, path) {
    var d = dp, k = 0, ok = false;
    while (k < KMAX) {
      var gp = -2.0 * th / (d * d * d);
      var grad = (dp - d) / vp + (u - th / (d * d)) / vu * gp;
      var cap = 1.0 / (1.0 / vp + gp * gp / vu);
      var step = (ETA < cap ? ETA : cap) * grad;
      if (step < -0.5 * d) step = -0.5 * d;
      d = d + step;
      k += 1;
      if (path) path.push(d);
      if (Math.abs(step) < TOL) { ok = true; break; }
    }
    return [d, k, ok];
  }

  function worldTrial(r, loc, v, th, vu) {
    var d;
    while (true) {
      d = loc + Math.sqrt(v) * normal(r);
      if (d > 0) break;
    }
    return [d, th / (d * d) + Math.sqrt(vu) * normal(r)];
  }

  /* Posterior expectation of the squared sensory residual. When the window of
   * ten likelihood widths either side of u lies at intensities the prior can
   * reach, by quadrature over it in residual coordinates rho = u - y, with
   * y = theta / d^2 and the Jacobian y^(-3/2); otherwise (a quiet or negative
   * u) over log distance. */
  function postR2(u, dp, vp, th, vu) {
    var sd = Math.sqrt(vu);
    var lw = new Array(NY), r2 = new Array(NY), mx = -Infinity, i, h, d, res;
    var far = dp + 12.0 * Math.sqrt(vp);
    if (u - 10.0 * sd > th / (far * far)) {
      h = 20.0 * sd / (NY - 1);
      for (i = 0; i < NY; i++) {
        var rho = -10.0 * sd + i * h;
        var y = u - rho;
        d = Math.sqrt(th / y);
        lw[i] = -rho * rho / (2.0 * vu) - (d - dp) * (d - dp) / (2.0 * vp) - 1.5 * Math.log(y);
        r2[i] = rho * rho;
        if (lw[i] > mx) mx = lw[i];
      }
    } else {
      var up = u > 0.0 ? u : 0.0;
      var da = Math.sqrt(th / (up + 12.0 * sd));
      var c = 2.0 * th * vp * (th / (dp * dp) + Math.abs(u)) / vu;
      var db = dp + 12.0 * Math.sqrt(vp) + Math.sqrt(Math.sqrt(c));
      if (da > 0.5 * db) da = 0.5 * db;
      var sa = Math.log(da);
      h = (Math.log(db) - sa) / (NY - 1);
      for (i = 0; i < NY; i++) {
        var q = sa + i * h;
        d = Math.exp(q);
        res = u - th / (d * d);
        lw[i] = -res * res / (2.0 * vu) - (d - dp) * (d - dp) / (2.0 * vp) + q;
        r2[i] = res * res;
        if (lw[i] > mx) mx = lw[i];
      }
    }
    var s0 = 0.0, s2 = 0.0;
    for (i = 0; i < NY; i++) {
      var w = Math.exp(lw[i] - mx) * ((i === 0 || i === NY - 1) ? 0.5 : 1.0);
      s0 += w;
      s2 += w * r2[i];
    }
    return s2 / s0;
  }

  /* Trials one after another. world = [location, variance, theta, Sigma_u] of
   * the process; model holds the animal's starting dp, vp, th, vu. Only the
   * parameter named by `learn` changes. Also keeps, per trial, what is needed
   * to replay its inner loop: the observation and the parameters it was
   * settled with. */
  function run(seed, n, world, model, learn, beta, rule) {
    var r = mulberry32(seed);
    var dp = model.dp, vp = model.vp, th = model.th, vu = model.vu;
    var hist = [], ep = [], eu = [], dh = [], us = [], before = [], truth = [], unsettled = 0;
    for (var i = 0; i < n; i++) {
      var wt = worldTrial(r, world[0], world[1], world[2], world[3]);
      var u = wt[1];
      before.push({ dp: dp, vp: vp, th: th, vu: vu });
      var s = settle(u, dp, vp, th, vu, null);
      var d = s[0];
      if (!s[2]) unsettled += 1;
      var e1 = (dp - d) / vp;
      var e2 = (u - th / (d * d)) / vu;
      ep.push(e1); eu.push(e2); dh.push(d); us.push(u); truth.push(wt[0]);
      if (!s[2]) {
        /* inference unfinished: the parameters are left as they were */
      } else if (learn === 'dp') {
        dp = dp + beta * (-e1 - truncRatio(dp, vp));
      } else if (learn === 'th') {
        th = th + beta * e2 / (d * d);
      } else if (learn === 'vu') {
        var res = u - th / (d * d);
        var r2 = rule === 'post' ? postR2(u, dp, vp, th, vu) : res * res;
        vu = vu * Math.exp(beta * 0.5 * (r2 / vu - 1.0));
      } else if (learn === 'vp') {
        var dev = d - dp;
        vp = vp * Math.exp(beta * (0.5 * (dev * dev / vp - 1.0)
                                   + 0.5 * dp * truncRatio(dp, vp)));
      }
      hist.push({ dp: dp, vp: vp, th: th, vu: vu }[learn]);
    }
    return { hist: hist, ep: ep, eu: eu, dh: dh, us: us, before: before, truth: truth,
             unsettled: unsettled };
  }

  /* A readout row that appears only when some trial ran out of steps before
   * meeting the stopping rule; the lesson's numbers assume none did. */
  function unsettledRows(rows, runs) {
    var n = 0;
    runs.forEach(function (r) { n += r.unsettled; });
    if (n > 0) rows.push(['trials not settled in ' + KMAX + ' steps (no update)', String(n)]);
    return rows;
  }

  function meanFrom(a, i0) {
    var s = 0;
    for (var i = i0; i < a.length; i++) s += a[i];
    return s / (a.length - i0);
  }

  /* The ear's share of the Gauss-Newton curvature at the prior location. */
  function earShare(vu) {
    var d = WEEK2.dp, s = Math.pow(2 * WEEK2.th / (d * d * d), 2) / vu;
    return s / (1 / WEEK2.vp + s);
  }

  /* exported for the parity check run under node */
  A.w03 = { run: run, settle: settle, postR2: postR2, truncRatio: truncRatio,
            earShare: earShare, meanFrom: meanFrom, WEEK2: WEEK2, NTRIALS: NTRIALS };

  /* A framed panel with a linear y axis. */
  function panel(p, x, y, w, h, ylo, yhi, xmax, title, ticks, fmt) {
    var c = p.ctx;
    c.strokeStyle = C.rule2; c.lineWidth = 1;
    c.strokeRect(x + 0.5, y + 0.5, w, h);
    p.text(x, y - 8, title, { pixel: true, align: 'left', size: 10.5, colour: C.ink2, weight: '600' });
    var P = {
      X: function (k) { return x + (k / xmax) * w; },
      Y: function (v) { return y + h - ((v - ylo) / (yhi - ylo)) * h; },
      x: x, y: y, w: w, h: h
    };
    (ticks || []).forEach(function (v) {
      if (v < ylo || v > yhi) return;
      c.strokeStyle = C.rule; c.lineWidth = 1;
      c.beginPath(); c.moveTo(x, P.Y(v)); c.lineTo(x + w, P.Y(v)); c.stroke();
      p.text(x - 6, P.Y(v), fmt ? fmt(v) : String(v), { pixel: true, align: 'right', size: 10, colour: C.faint });
    });
    return P;
  }

  function trace(p, P, arr, col, wid, dash, i0) {
    var c = p.ctx;
    c.strokeStyle = col; c.lineWidth = wid; c.setLineDash(dash || []);
    c.beginPath();
    var started = false;
    for (var k = i0 || 0; k < arr.length; k++) {
      var X = P.X(k), Y = P.Y(arr[k]);
      if (started) c.lineTo(X, Y); else { c.moveTo(X, Y); started = true; }
    }
    c.stroke(); c.setLineDash([]);
  }

  function hline(p, P, v, col, label, above) {
    var c = p.ctx;
    c.strokeStyle = col; c.lineWidth = 1.2; c.setLineDash([4, 3]);
    c.beginPath(); c.moveTo(P.x, P.Y(v)); c.lineTo(P.x + P.w, P.Y(v)); c.stroke();
    c.setLineDash([]);
    if (label) p.text(P.x + P.w - 4, P.Y(v) + (above ? -9 : 10), label,
      { pixel: true, align: 'right', size: 9.5, colour: col });
  }

  function key(p, x, y, items) {
    var c = p.ctx;
    items.forEach(function (it, i) {
      var ly = y + i * 14;
      c.strokeStyle = it[1]; c.lineWidth = 2; c.setLineDash(it[2] ? [4, 3] : []);
      c.beginPath(); c.moveTo(x, ly); c.lineTo(x + 16, ly); c.stroke();
      c.setLineDash([]);
      p.text(x + 21, ly, it[0], { pixel: true, align: 'left', size: 9.8, colour: C.ink2 });
    });
  }

  function niceTicks(lo, hi, n) {
    var span = hi - lo, raw = span / (n || 4);
    var mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var st = [1, 2, 2.5, 5, 10].map(function (m) { return m * mag; })
      .filter(function (s) { return s >= raw; })[0];
    var out = [];
    for (var v = Math.ceil(lo / st) * st; v <= hi + 1e-12; v += st) out.push(+v.toFixed(10));
    return out;
  }
  function tickFmt(ticks) {
    var dec = 0;
    ticks.forEach(function (v) {
      var s = String(v), i = s.indexOf('.');
      if (i >= 0) dec = Math.max(dec, Math.min(3, s.length - i - 1));
    });
    return function (v) { return v.toFixed(dec).replace('-', '−'); };
  }

  /* ══════════════════════════════════════════════════════════════════════
   * 1. two-timescales
   *    Left: one trial's inner loop, the fast timescale.
   *    Right: the prior location across all trials, the slow one.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('two-timescales', function (mount) {
    var SEED = 7, BETA = 0.01;
    var st = null;

    var sTrial = A.slider({ label: 'trial shown', min: 1, max: NTRIALS, step: 1, value: 1, decimals: 0 });
    var sLoc = A.slider({ label: 'world&rsquo;s source location, <i>d</i><sub>p</sub><sup>*</sup>', min: 1, max: 4, step: 0.1, value: 3, decimals: 1 });
    var sVu = A.slider({ label: 'sensory variance, &Sigma;<sub>u</sub>', min: 0.01, max: 0.3, step: 0.01, value: 0.01 });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.4, pad: { l: 44, r: 14, t: 24, b: 34 } });

    function simulate() {
      var vu = sVu.get();
      var res = run(SEED, NTRIALS, [sLoc.get(), WEEK2.vp, WEEK2.th, vu],
                    { dp: WEEK2.dp, vp: WEEK2.vp, th: WEEK2.th, vu: vu }, 'dp', BETA, 'point');
      st = { res: res, loc: sLoc.get(), vu: vu };
      pick();
    }

    function pick() {
      var i = Math.round(sTrial.get()) - 1, res = st.res, b = res.before[i];
      var path = [b.dp];
      var s = settle(res.us[i], b.dp, b.vp, b.th, b.vu, path);
      st.i = i; st.path = path; st.steps = s[1];
      out.show(unsettledRows([
        ['steps to settle', String(s[1])],
        ['settled at', s[0].toFixed(4)],
        ['true distance', res.truth[i].toFixed(4)],
        ['prior location after', res.hist[i].toFixed(4)],
        ['mean over trials 1001–2000', meanFrom(res.hist, NTRIALS / 2).toFixed(4)],
        ['final', res.hist[NTRIALS - 1].toFixed(4)]
      ], [res]));
      plot.render();
    }

    plot.onDraw(function (p) {
      p.clear();
      if (!st) return;
      var c = p.ctx, gap = 54;
      var pw = (p.plotW() - gap) / 2, ph = p.plotH();
      var lx = p.o.pad.l, rx = lx + pw + gap, ty = p.o.pad.t;
      var res = st.res, i = st.i;

      /* left: the inner loop of the chosen trial */
      var pts = st.path.concat([res.truth[i]]);
      var lo = Math.min.apply(null, pts), hi = Math.max.apply(null, pts);
      var padv = Math.max(0.1, 0.12 * (hi - lo));
      lo = Math.max(0, lo - padv); hi = hi + padv;
      var tl = niceTicks(lo, hi, 4);
      var L = panel(p, lx, ty, pw, ph, lo, hi, Math.max(st.path.length - 1, 1),
        'trial ' + (i + 1) + ': the state settling', tl, tickFmt(tl));
      hline(p, L, res.truth[i], C.clay, 'true distance', res.truth[i] > st.path[st.path.length - 1]);
      trace(p, L, st.path, C.accent, 2.2);
      c.fillStyle = C.gold;
      c.beginPath(); c.arc(L.X(0), L.Y(st.path[0]), 4, 0, 6.284); c.fill();
      p.text(L.X(0) + 7, L.Y(st.path[0]) - 9, 'prior location held', { pixel: true, align: 'left', size: 9.5, colour: C.gold });
      p.text(lx + pw / 2, p.h - 4, 'inference step within the trial', { pixel: true, size: 10.5, colour: C.ink2 });

      /* right: the prior location across trials */
      var hmin = Math.min(Math.min.apply(null, res.hist), WEEK2.dp, st.loc);
      var hmax = Math.max(Math.max.apply(null, res.hist), WEEK2.dp, st.loc);
      var pr = Math.max(0.1, 0.1 * (hmax - hmin));
      var tr = niceTicks(hmin - pr, hmax + pr, 4);
      var R = panel(p, rx, ty, pw, ph, hmin - pr, hmax + pr, NTRIALS,
        'prior location, across trials', tr, tickFmt(tr));
      hline(p, R, st.loc, C.clay, 'world’s location', true);
      trace(p, R, res.hist, C.accent, 1.6);
      c.strokeStyle = C.gold; c.lineWidth = 1.2; c.setLineDash([2, 3]);
      c.beginPath(); c.moveTo(R.X(i + 1), ty); c.lineTo(R.X(i + 1), ty + ph); c.stroke();
      c.setLineDash([]);
      c.fillStyle = C.gold;
      c.beginPath(); c.arc(R.X(i + 1), R.Y(res.hist[i]), 4, 0, 6.284); c.fill();
      p.text(rx + pw / 2, p.h - 4, 'trial', { pixel: true, size: 10.5, colour: C.ink2 });
    });

    sTrial.onchange(pick);
    sLoc.onchange(simulate);
    sVu.onchange(simulate);

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sTrial.el, sLoc.el, sVu.el]),
      body, out,
      A.note('Both panels come from one run of ' + NTRIALS + ' trials with a fixed seed, so moving '
        + 'the trial slider replays a trial rather than drawing a new one. On each trial the state '
        + 'starts at the prior location the animal held at that moment (gold) and climbs the log '
        + 'joint until its step falls below 10<sup>−12</sup> (a readout row appears if any trial needs '
        + 'more than ' + KMAX + ' steps, and such a trial changes no parameter); then the prior location takes one '
        + 'step of size &beta; = ' + BETA + ' along its derivative. The sensory variance slider sets the '
        + 'world and the model together. The prior variance and the source power are Week&nbsp;2&rsquo;s '
        + 'and are not learned here. The run settles below the world&rsquo;s location; Lesson&nbsp;4 '
        + 'explains why.')
    ]));
    simulate();
    plot.attach(body);
    return null;
  });

  /* ══════════════════════════════════════════════════════════════════════
   * 2. wrong-link
   *    A world louder than the model. Left: the learned power. Right: window
   *    averages of the two errors and of the learning signal.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('wrong-link', function (mount) {
    var SEED = 7, BETA = 0.01, WIN = 200;
    var st = null, learning = true;

    var sTh = A.slider({ label: 'world&rsquo;s source power, &theta;<sup>*</sup>', min: 0.5, max: 3, step: 0.1, value: 2, decimals: 1 });
    var btn = A.button('learning: on', function () {
      learning = !learning;
      btn.textContent = 'learning: ' + (learning ? 'on' : 'off');
      simulate();
    });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.4, pad: { l: 44, r: 14, t: 24, b: 34 } });

    function windowMeans(a) {
      var m = new Array(a.length), s = 0;
      for (var k = 0; k < a.length; k++) {
        s += a[k];
        if (k >= WIN) s -= a[k - WIN];
        m[k] = k >= WIN - 1 ? s / WIN : NaN;
      }
      return m;
    }

    function simulate() {
      var thw = sTh.get();
      var res = run(SEED, NTRIALS, [WEEK2.dp, WEEK2.vp, thw, WEEK2.vu],
                    { dp: WEEK2.dp, vp: WEEK2.vp, th: WEEK2.th, vu: WEEK2.vu },
                    'th', learning ? BETA : 0.0, 'point');
      var sig = res.eu.map(function (e, k) { return e / (res.dh[k] * res.dh[k]); });
      st = { res: res, thw: thw, sig: sig,
             mp: windowMeans(res.ep), mu: windowMeans(res.eu), ms: windowMeans(sig) };
      var last = NTRIALS - WIN;
      out.show(unsettledRows([
        ['final power', res.hist[NTRIALS - 1].toFixed(4)],
        ['mean over trials 1001–2000', meanFrom(res.hist, NTRIALS / 2).toFixed(4)],
        ['last ' + WIN + ' trials: prior error', meanFrom(res.ep, last).toFixed(4)],
        ['sensory error', meanFrom(res.eu, last).toFixed(4)],
        ['learning signal', meanFrom(sig, last).toFixed(4)]
      ], [res]));
      plot.render();
    }

    plot.onDraw(function (p) {
      p.clear();
      if (!st) return;
      var gap = 54;
      var pw = (p.plotW() - gap) / 2, ph = p.plotH();
      var lx = p.o.pad.l, rx = lx + pw + gap, ty = p.o.pad.t;
      var res = st.res;

      var hmin = Math.min(Math.min.apply(null, res.hist), WEEK2.th, st.thw);
      var hmax = Math.max(Math.max.apply(null, res.hist), WEEK2.th, st.thw);
      var pr = Math.max(0.1, 0.1 * (hmax - hmin));
      var tl = niceTicks(hmin - pr, hmax + pr, 4);
      var L = panel(p, lx, ty, pw, ph, hmin - pr, hmax + pr, NTRIALS,
        'source power the animal holds', tl, tickFmt(tl));
      hline(p, L, st.thw, C.clay, 'world’s power', st.thw >= WEEK2.th);
      trace(p, L, res.hist, C.accent, 1.8);
      p.text(lx + pw / 2, p.h - 4, 'trial', { pixel: true, size: 10.5, colour: C.ink2 });

      var all = [];
      [st.mp, st.mu, st.ms].forEach(function (a) {
        a.forEach(function (v) { if (isFinite(v)) all.push(v); });
      });
      var m = Math.max(0.05, Math.max.apply(null, all.map(Math.abs))) * 1.15;
      var tr = niceTicks(-m, m, 4);
      var R = panel(p, rx, ty, pw, ph, -m, m, NTRIALS,
        'averages over the last ' + WIN + ' trials', tr, tickFmt(tr));
      hline(p, R, 0, C.rule2, null);
      trace(p, R, st.mp, C.blue, 1.8, null, WIN - 1);
      trace(p, R, st.mu, C.gold, 1.8, null, WIN - 1);
      trace(p, R, st.ms, C.ink2, 2.2, [5, 3], WIN - 1);
      key(p, rx + 8, ty + 12, [['prior error εₚ', C.blue, false],
                              ['sensory error εᵤ', C.gold, false],
                              ['learning signal εᵤ h(d)', C.ink2, true]]);
      p.text(rx + pw / 2, p.h - 4, 'trial', { pixel: true, size: 10.5, colour: C.ink2 });
    });

    sTh.onchange(simulate);

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sTh.el, btn]),
      body, out,
      A.note('The animal starts from Week&nbsp;2&rsquo;s model, with source power 1, and learns the '
        + 'power alone with &beta; = ' + BETA + '; every other parameter of the world and the model is '
        + 'Week&nbsp;2&rsquo;s. Each average on the right is taken over the ' + WIN + ' trials ending at '
        + 'that point, so the curves start at trial ' + WIN + '. Errors are measured from the settled '
        + 'state. With learning on, the dashed learning signal is the quantity whose average vanishes '
        + 'where learning comes to rest; the two errors need not. Switch learning off to see what the mismatch does to inference '
        + 'alone. The same seed is used throughout, so the two settings see the same sounds.')
    ]));
    simulate();
    plot.attach(body);
    return null;
  });

  /* ══════════════════════════════════════════════════════════════════════
   * 3. precision-learning
   *    A noisier ear than the model believes. The sensory variance learned by
   *    the point rule and by the posterior-averaged rule, on the same trials.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('precision-learning', function (mount) {
    var SEED = 7, BETA = 0.01;
    var st = null;

    var sVw = A.slider({ label: 'world&rsquo;s sensory variance, &Sigma;<sub>u</sub><sup>*</sup>', min: 0.05, max: 0.5, step: 0.01, value: 0.3 });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.42, pad: { l: 50, r: 14, t: 24, b: 34 } });

    function simulate() {
      var vw = sVw.get();
      var world = [WEEK2.dp, WEEK2.vp, WEEK2.th, vw];
      var pt = run(SEED, NTRIALS, world, WEEK2, 'vu', BETA, 'point');
      var po = run(SEED, NTRIALS, world, WEEK2, 'vu', BETA, 'post');
      var mpt = meanFrom(pt.hist, NTRIALS / 2), mpo = meanFrom(po.hist, NTRIALS / 2);
      st = { pt: pt.hist, po: po.hist, vw: vw };
      out.show(unsettledRows([
        ['point rule: final', pt.hist[NTRIALS - 1].toFixed(4)],
        ['mean over trials 1001–2000', mpt.toFixed(4)],
        ['posterior-averaged: final', po.hist[NTRIALS - 1].toFixed(4)],
        ['mean over trials 1001–2000', mpo.toFixed(4)],
        ['ear’s share: at the start', earShare(WEEK2.vu).toFixed(3)],
        ['at the truth', earShare(vw).toFixed(3)],
        ['point rule', earShare(mpt).toFixed(3)],
        ['posterior-averaged', earShare(mpo).toFixed(3)]
      ], [pt, po]));
      plot.render();
    }

    plot.onDraw(function (p) {
      p.clear();
      if (!st) return;
      var pw = p.plotW(), ph = p.plotH(), lx = p.o.pad.l, ty = p.o.pad.t;
      var hi = Math.max(Math.max.apply(null, st.pt), Math.max.apply(null, st.po), st.vw) * 1.12;
      var tk = niceTicks(0, hi, 5);
      var P = panel(p, lx, ty, pw, ph, 0, hi, NTRIALS, 'sensory variance the animal holds', tk, tickFmt(tk));
      hline(p, P, st.vw, C.clay, 'world’s variance', true);
      hline(p, P, WEEK2.vu, C.faint, 'starting value', WEEK2.vu > st.vw * 0.7 ? false : true);
      trace(p, P, st.po, C.blue, 1.8);
      trace(p, P, st.pt, C.accent, 1.8);
      key(p, lx + 10, ty + 14, [['posterior-averaged rule', C.blue, false], ['point rule', C.accent, false]]);
      p.text(lx + pw / 2, p.h - 4, 'trial', { pixel: true, size: 10.5, colour: C.ink2 });
    });

    sVw.onchange(simulate);

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sVw.el]),
      body, out,
      A.note('Both rules see the same ' + NTRIALS + ' trials and take log-variance steps with &beta; = '
        + BETA + '. The point rule squares the residual left at the settled state; the posterior-averaged '
        + 'rule replaces it by its average over the posterior, computed on each trial by quadrature on '
        + NY + ' points. The ear&rsquo;s share is the ear&rsquo;s part of the curvature at the prior '
        + 'location, using the mean of each run over its second half. Those are averages of one '
        + 'run, and they move with the seed; Lesson&nbsp;4 computes where each rule&rsquo;s expected '
        + 'update vanishes. The other parameters are Week&nbsp;2&rsquo;s and are not learned.')
    ]));
    simulate();
    plot.attach(body);
    return null;
  });
})();
