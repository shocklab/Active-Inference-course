/* w02.js — interactive figures for Week 2: estimating a hidden state. */
(function () {
  'use strict';
  var A = window.AIF, el = A.el, C = A.colours;

  /* The model of Week 2, in one place. Changing it here changes every widget
   * on the week, which is the point: the reader should be able to see that the
   * same three lines drive all of them. */
  var DP = 2.0;                                    /* the prior mean, fixed */
  function g(d) { return 1 / (d * d); }
  function gp(d) { return -2 / (d * d * d); }

  /* The exact mode, by dense grid search rather than by a stored constant.
   * A hardcoded number would be right for the default slider positions and
   * quietly wrong for every other, which is worse than having no line at all. */
  function exactMode(u, vp, vu) {
    var best = 0.05, bestV = -Infinity;
    for (var i = 0; i < 8000; i++) {
      var d = 0.05 + i * (8 - 0.05) / 7999;
      var v = -(d - DP) * (d - DP) / (2 * vp) - (u - g(d)) * (u - g(d)) / (2 * vu);
      if (v > bestV) { bestV = v; best = d; }
    }
    return best;
  }

  /* ══════════════════════════════════════════════════════════════════════
   * 1. ascent-errors
   *    Left: the estimate climbing the log joint from the prior mean.
   *    Right: the two terms of the gradient, meeting where their sum is zero.
   *    The second panel is the one that matters. The trajectory shows THAT it
   *    converges; the error panel shows WHY it stops where it does.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('ascent-errors', function (mount) {
    var STEPS = 200;
    var st;

    var sU = A.slider({ label: 'intensity heard, <i>u</i>', min: 0.05, max: 1.2, step: 0.01, value: 0.5 });
    var sVp = A.slider({ label: 'prior variance, &Sigma;<sub>p</sub>', min: 0.05, max: 3, step: 0.05, value: 1 });
    var sVu = A.slider({ label: 'sensory variance, &Sigma;<sub>u</sub>', min: 0.005, max: 1, step: 0.005, value: 0.1, decimals: 3 });
    var sEta = A.slider({ label: 'rate, &eta;', min: 0.005, max: 0.3, step: 0.005, value: 0.05, decimals: 3 });

    function run() {
      var u = sU.get(), vp = sVp.get(), vu = sVu.get(), eta = sEta.get();
      var d = DP, path = [d], ep = [], eu = [];
      for (var k = 0; k < STEPS; k++) {
        var a = (DP - d) / vp;                  /* prior prediction error   */
        var b = ((u - g(d)) / vu) * gp(d);      /* sensory term, weighted   */
        ep.push(a); eu.push(b);
        d += eta * (a + b);
        /* The estimate is a distance. Left free, a large rate can drive it
         * negative, where 1/d^2 is finite but meaningless and the curve turns
         * into nonsense the reader has to be told to ignore. Clamped, an
         * over-large rate instead shows the honest failure: oscillation. */
        if (!isFinite(d) || d < 0.05) { d = 0.05; }
        if (d > 8) { d = 8; }
        path.push(d);
      }
      var mode = exactMode(u, vp, vu);
      var hit = path.findIndex(function (p) { return Math.abs(p - mode) < 0.01 * mode; });

      /* The landscape itself. The lesson is called "climbing the log joint" and
       * without this panel the hill is never drawn: the other two show the walk
       * against step number, which is a record of the climb, not the terrain. */
      var F = function (d) {
        return -(d - DP) * (d - DP) / (2 * vp)
               - (u - g(d)) * (u - g(d)) / (2 * vu);
      };
      st = { path: path, ep: ep, eu: eu, mode: mode, hit: hit, u: u, F: F };
      out.show([
        ['settles at', path[path.length - 1].toFixed(4)],
        ['exact mode', mode.toFixed(4)],
        ['steps to 1%', hit < 0 ? 'not within ' + STEPS : String(hit)],
        ['errors sum to', (st.ep[STEPS - 1] + st.eu[STEPS - 1]).toFixed(4)]
      ]);
      plot.render();
    }

    var plot = new A.Plot({ aspect: 0.34, pad: { l: 44, r: 14, t: 22, b: 38 } });
    var out = A.readout();

    plot.onDraw(function (p) {
      p.clear();
      var c = p.ctx;
      var gap = 40;
      var pw = (p.plotW() - 2 * gap) / 3, ph = p.plotH();
      var lx = p.o.pad.l, ax = lx + pw + gap, bx = ax + pw + gap, ty = p.o.pad.t;

      function box(x, ylo, yhi, title) {
        c.strokeStyle = C.rule2; c.lineWidth = 1;
        c.strokeRect(x + 0.5, ty + 0.5, pw, ph);
        p.text(x, ty - 8, title, { pixel: true, align: 'left', size: 10.5, colour: C.ink2, weight: '600' });
        return {
          X: function (k) { return x + (k / STEPS) * pw; },
          Y: function (v) { return ty + ph - ((v - ylo) / (yhi - ylo)) * ph; }
        };
      }

      /* ── left: the landscape, with the walk drawn on it ───────────────── */
      var DLO = 0.35, DHI = 5.0, NS = 260;
      var fs = [], fmin = Infinity, fmax = -Infinity;
      for (var i = 0; i <= NS; i++) {
        var dd = DLO + (DHI - DLO) * i / NS;
        var fv = st.F(dd);
        fs.push([dd, fv]);
        if (fv > fmax) fmax = fv;
        if (fv < fmin) fmin = fv;
      }
      /* The log joint dives to minus infinity as d -> 0, so a floor is needed or
       * the interesting part of the curve is a flat line at the top. Clip at a
       * fixed multiple of the peak-to-visible range rather than at the true
       * minimum, and say in the note that the bottom is cut. */
      var floor = fmax - Math.max(6, (fmax - fs[NS][1]) * 2.2);
      var Lx = function (d) { return lx + ((d - DLO) / (DHI - DLO)) * pw; };
      var Ly = function (v) {
        return ty + ph - ((Math.max(v, floor) - floor) / (fmax - floor)) * ph;
      };
      c.strokeStyle = C.rule2; c.lineWidth = 1;
      c.strokeRect(lx + 0.5, ty + 0.5, pw, ph);
      p.text(lx, ty - 8, 'the log joint F(d)',
        { pixel: true, align: 'left', size: 10.5, colour: C.ink2, weight: '600' });

      c.strokeStyle = C.ink2; c.lineWidth = 2;
      c.beginPath();
      fs.forEach(function (q, i) {
        i ? c.lineTo(Lx(q[0]), Ly(q[1])) : c.moveTo(Lx(q[0]), Ly(q[1]));
      });
      c.stroke();

      c.strokeStyle = C.clay; c.lineWidth = 1.2; c.setLineDash([3, 3]);
      c.beginPath(); c.moveTo(Lx(st.mode), ty); c.lineTo(Lx(st.mode), ty + ph); c.stroke();
      c.setLineDash([]);

      /* every fourth iterate, so the crowding near the peak stays readable */
      for (var k = 0; k < st.path.length; k += 4) {
        var dv = st.path[k];
        if (dv < DLO || dv > DHI) continue;
        c.globalAlpha = 0.25 + 0.75 * (1 - k / st.path.length);
        c.fillStyle = C.accent;
        c.beginPath(); c.arc(Lx(dv), Ly(st.F(dv)), 2.6, 0, 6.284); c.fill();
      }
      c.globalAlpha = 1;
      c.fillStyle = C.gold;
      c.beginPath(); c.arc(Lx(DP), Ly(st.F(DP)), 4.2, 0, 6.284); c.fill();
      p.text(Lx(DP), Ly(st.F(DP)) - 11, 'start',
        { pixel: true, size: 9.5, colour: C.gold });
      p.text(lx + pw - 4, ty + ph - 8, 'd \u2192',
        { pixel: true, align: 'right', size: 9.5, colour: C.faint });

      /* ── middle: the trajectory ───────────────────────────────────────── */
      var dlo = 0, dhi = Math.max(4, st.mode * 1.6, DP * 1.4);
      var Lp = box(ax, dlo, dhi, 'estimate, by step');

      [1, 2, 3, 4].forEach(function (v) {
        if (v > dhi) return;
        c.strokeStyle = C.rule; c.lineWidth = 1;
        c.beginPath(); c.moveTo(ax, Lp.Y(v)); c.lineTo(ax + pw, Lp.Y(v)); c.stroke();
        p.text(ax - 6, Lp.Y(v), String(v), { pixel: true, align: 'right', size: 10, colour: C.faint });
      });

      c.strokeStyle = C.clay; c.lineWidth = 1.2; c.setLineDash([4, 3]);
      c.beginPath(); c.moveTo(ax, Lp.Y(st.mode)); c.lineTo(ax + pw, Lp.Y(st.mode)); c.stroke();
      c.setLineDash([]);
      p.text(ax + pw - 4, Lp.Y(st.mode) - 9, 'exact mode', { pixel: true, align: 'right', size: 9.5, colour: C.clay });

      c.strokeStyle = C.accent; c.lineWidth = 2.2; c.beginPath();
      st.path.forEach(function (d, k) {
        var X = Lp.X(k), Y = Lp.Y(d);
        k ? c.lineTo(X, Y) : c.moveTo(X, Y);
      });
      c.stroke();
      p.text(Lp.X(0) + 3, Lp.Y(DP) - 9, 'starts at the prior mean',
        { pixel: true, align: 'left', size: 9.5, colour: C.muted });

      /* ── right: the two gradient terms ────────────────────────────────── */
      var m = 0;
      st.ep.forEach(function (v, i) {
        m = Math.max(m, Math.abs(v), Math.abs(st.eu[i]), Math.abs(v + st.eu[i]));
      });
      m = Math.max(m, 1e-6) * 1.15;
      var Rp = box(bx, -m, m, 'gradient terms');

      c.strokeStyle = C.rule2; c.lineWidth = 1;
      c.beginPath(); c.moveTo(bx, Rp.Y(0)); c.lineTo(bx + pw, Rp.Y(0)); c.stroke();
      p.text(bx - 6, Rp.Y(0), '0', { pixel: true, align: 'right', size: 10, colour: C.faint });
      p.text(bx - 6, Rp.Y(m * 0.82), (m * 0.82).toFixed(1), { pixel: true, align: 'right', size: 10, colour: C.faint });
      p.text(bx - 6, Rp.Y(-m * 0.82), (-m * 0.82).toFixed(1), { pixel: true, align: 'right', size: 10, colour: C.faint });

      function trace(arr, col, wid, dash) {
        c.strokeStyle = col; c.lineWidth = wid; c.setLineDash(dash || []);
        c.beginPath();
        arr.forEach(function (v, k) {
          var X = Rp.X(k), Y = Rp.Y(v);
          k ? c.lineTo(X, Y) : c.moveTo(X, Y);
        });
        c.stroke(); c.setLineDash([]);
      }
      trace(st.ep, C.blue, 1.8);
      trace(st.eu, C.gold, 1.8);
      trace(st.ep.map(function (v, i) { return v + st.eu[i]; }), C.ink2, 2.4, [5, 3]);

      /* A vertical key, drawn here rather than through Plot.legend: that one
       * lays its entries out in a row, which overflows a half-width panel. */
      [['\u03b5\u209a  prior', C.blue, false],
       ["g'(d)\u00b7\u03b5\u1d64  sensory", C.gold, false],
       ['their sum', C.ink2, true]].forEach(function (it, i) {
        var ly = ty + 12 + i * 14;
        c.strokeStyle = it[1]; c.lineWidth = 2;
        c.setLineDash(it[2] ? [4, 3] : []);
        c.beginPath(); c.moveTo(bx + 8, ly); c.lineTo(bx + 24, ly); c.stroke();
        c.setLineDash([]);
        p.text(bx + 29, ly, it[0], { pixel: true, align: 'left', size: 9.8, colour: C.ink2 });
      });

      p.text(ax + pw / 2, p.h - 4, 'gradient-ascent step',
        { pixel: true, size: 10.5, colour: C.ink2 });
      p.text(bx + pw / 2, p.h - 4, 'gradient-ascent step',
        { pixel: true, size: 10.5, colour: C.ink2 });
    });

    [sU, sVp, sVu, sEta].forEach(function (s) { s.onchange(run); });

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sU.el, sVp.el, sVu.el, sEta.el]),
      body, out,
      A.note('Left is the hill being climbed: <i>F</i> as a function of <i>d</i>, with the walk '
        + 'drawn on it and later steps fainter. Its bottom is cut off, because <i>F</i> runs to '
        + 'minus infinity as <i>d</i> approaches zero and plotting that flattens everything else. '
        + 'The other two panels show the same walk against step number. '
        + 'The prior mean is held at <i>d</i><sub>p</sub> = 2 throughout, and the estimate '
        + 'always starts there, so the prior error begins at exactly zero. Watch the right-hand '
        + 'panel: the two terms are not driven to zero, they are driven to cancel. Lower the '
        + 'sensory variance and the gold curve dominates, pulling the estimate towards what the '
        + 'data alone would say. Take it all the way to 0.010 with the rate left at 0.050 and '
        + 'the estimate stops converging: it cycles between two values either side of the peak, '
        + 'because a sharper peak admits a smaller step. The readout says so directly, in the '
        + 'residual gradient and in the step count that never arrives.')
    ]));
    run();
    plot.attach(body);
    return null;
  });
})();

(function () {
  'use strict';
  var A = window.AIF, el = A.el, C = A.colours;

  var DP = 2.0, U = 0.5;
  function g(d) { return 1 / (d * d); }

  /* ══════════════════════════════════════════════════════════════════════
   * 2. precision-posterior
   *    The posterior itself, as the two precisions move. Two things to see:
   *    the peak sliding between the prior mean and what the data alone say,
   *    and the mode parting company with the mean as the curve goes lopsided.
   *    The update of Lesson 4 finds the first of those and not the second.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('precision-posterior', function (mount) {
    var N = 1600, DMAX = 6;
    var st;

    var sVp = A.slider({ label: 'prior variance, &Sigma;<sub>p</sub>', min: 0.05, max: 4, step: 0.05, value: 1 });
    var sVu = A.slider({ label: 'sensory variance, &Sigma;<sub>u</sub>', min: 0.005, max: 1, step: 0.005, value: 0.1, decimals: 3 });
    var sLin = el('label', { class: 'wctl' }, []);
    var chk = el('input', { type: 'checkbox' });
    sLin.appendChild(el('div', { class: 'wlab' }, [el('b', { html: 'link function' })]));
    sLin.appendChild(chk);
    var linLab = el('span', { class: 'wval', text: 'g(d) = 1/d²' });
    sLin.querySelector('.wlab').appendChild(linLab);

    function build() {
      var vp = sVp.get(), vu = sVu.get(), lin = chk.checked;
      var link = lin ? function (d) { return d; } : g;
      linLab.textContent = lin ? 'g(d) = d' : 'g(d) = 1/d²';

      var xs = [], pr = [], lk = [], po = [];
      var h = DMAX / N;
      for (var i = 1; i <= N; i++) {
        var d = i * h;
        var a = Math.exp(-(d - DP) * (d - DP) / (2 * vp));
        var b = Math.exp(-(U - link(d)) * (U - link(d)) / (2 * vu));
        xs.push(d); pr.push(a); lk.push(b); po.push(a * b);
      }
      /* Normalise on the grid so mean and mode are the grid's own, not a
       * formula's. With the linear link the grid starts at 0 and the true
       * Gaussian has mass below it, so the mean shown is the mean of the
       * physically admissible part. The note says so. */
      var Z = 0; po.forEach(function (v) { Z += v * h; });
      var mean = 0, mode = xs[0], best = -1;
      po.forEach(function (v, i) {
        mean += xs[i] * (v / Z) * h;
        if (v > best) { best = v; mode = xs[i]; }
      });
      var sd = 0;
      po.forEach(function (v, i) { sd += (xs[i] - mean) * (xs[i] - mean) * (v / Z) * h; });
      sd = Math.sqrt(sd);
      var skew = 0;
      po.forEach(function (v, i) { skew += Math.pow((xs[i] - mean) / sd, 3) * (v / Z) * h; });

      st = { xs: xs, pr: pr, lk: lk, po: po, mean: mean, mode: mode, skew: skew, lin: lin };
      out.show([
        ['mode', mode.toFixed(4)],
        ['mean', mean.toFixed(4)],
        ['mean above mode by', (100 * (mean - mode) / mode).toFixed(1) + '%'],
        ['skew', skew.toFixed(3)]
      ]);
      plot.render();
    }

    var plot = new A.Plot({ aspect: 0.40, pad: { l: 30, r: 16, t: 18, b: 40 } });
    var out = A.readout();

    plot.onDraw(function (p) {
      p.clear();
      var c = p.ctx, pw = p.plotW(), ph = p.plotH();
      var ox = p.o.pad.l, oy = p.o.pad.t;
      var X = function (d) { return ox + (d / 6) * pw; };

      c.strokeStyle = C.rule2; c.lineWidth = 1;
      c.beginPath(); c.moveTo(ox, oy + ph); c.lineTo(ox + pw, oy + ph); c.stroke();
      [0, 1, 2, 3, 4, 5, 6].forEach(function (v) {
        p.text(X(v), oy + ph + 14, String(v), { pixel: true, size: 10, colour: C.faint });
      });
      p.text(ox + pw / 2, p.h - 5, 'distance d', { pixel: true, size: 10.5, colour: C.ink2 });

      function curve(arr, col, wid, dash) {
        var m = Math.max.apply(null, arr);
        c.strokeStyle = col; c.lineWidth = wid; c.setLineDash(dash || []);
        c.beginPath();
        arr.forEach(function (v, i) {
          var x = X(st.xs[i]), y = oy + ph - (v / m) * ph * 0.94;
          i ? c.lineTo(x, y) : c.moveTo(x, y);
        });
        c.stroke(); c.setLineDash([]);
      }
      curve(st.pr, C.faint, 1.6, [4, 3]);
      curve(st.lk, C.gold, 1.6, [2, 3]);
      curve(st.po, C.accent, 2.6);

      function marker(d, col, label, up) {
        c.strokeStyle = col; c.lineWidth = 1.2; c.setLineDash([2, 3]);
        c.beginPath(); c.moveTo(X(d), oy + 4); c.lineTo(X(d), oy + ph); c.stroke();
        c.setLineDash([]);
        p.text(X(d), oy + (up ? 10 : 26), label + ' ' + d.toFixed(3),
          { pixel: true, size: 9.8, colour: col });
      }
      marker(st.mode, C.accent, 'mode', true);
      marker(st.mean, C.clay, 'mean', false);

      [['prior', C.faint, true], ['likelihood', C.gold, true], ['posterior', C.accent, false]]
        .forEach(function (it, i) {
          var ly = oy + 12 + i * 14, lx = ox + pw - 108;
          c.strokeStyle = it[1]; c.lineWidth = 2;
          c.setLineDash(it[2] ? [4, 3] : []);
          c.beginPath(); c.moveTo(lx, ly); c.lineTo(lx + 16, ly); c.stroke();
          c.setLineDash([]);
          p.text(lx + 21, ly, it[0], { pixel: true, align: 'left', size: 9.8, colour: C.ink2 });
        });
    });

    [sVp, sVu].forEach(function (s) { s.onchange(build); });
    chk.addEventListener('change', build);

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sVp.el, sVu.el, sLin]),
      body, out,
      A.note('Each curve is scaled to its own peak. Switch the link to <i>g</i>(<i>d</i>) = <i>d</i> '
        + 'and the skew collapses by most of its value and the two markers nearly coincide: with a '
        + 'linear link the posterior is exactly Gaussian, whatever the precisions. What is left is '
        + 'not the link. The axis starts at zero because a negative distance is meaningless, and at '
        + 'these settings the linear posterior sits close enough to zero that cutting it there '
        + 'removes a slice of its left tail, which is what the residual skew is measuring. Widen '
        + 'the prior or sharpen the ear and watch that residual change without the link changing '
        + 'at all.')
    ]));
    build();
    plot.attach(body);
    return null;
  });
})();
