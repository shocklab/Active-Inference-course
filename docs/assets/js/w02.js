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
      st = { path: path, ep: ep, eu: eu, mode: mode, hit: hit, u: u };
      out.show([
        ['settles at', path[path.length - 1].toFixed(4)],
        ['exact mode', mode.toFixed(4)],
        ['steps to 1%', hit < 0 ? 'not within ' + STEPS : String(hit)],
        ['final Σε', (st.ep[STEPS - 1] + st.eu[STEPS - 1]).toFixed(4)]
      ]);
      plot.render();
    }

    var plot = new A.Plot({ aspect: 0.42, pad: { l: 44, r: 14, t: 22, b: 38 } });
    var out = A.readout();

    plot.onDraw(function (p) {
      p.clear();
      var c = p.ctx;
      var gap = 46;
      var pw = (p.plotW() - gap) / 2, ph = p.plotH();
      var ax = p.o.pad.l, bx = ax + pw + gap, ty = p.o.pad.t;

      function box(x, ylo, yhi, title) {
        c.strokeStyle = C.rule2; c.lineWidth = 1;
        c.strokeRect(x + 0.5, ty + 0.5, pw, ph);
        p.text(x, ty - 8, title, { pixel: true, align: 'left', size: 10.5, colour: C.ink2, weight: '600' });
        return {
          X: function (k) { return x + (k / STEPS) * pw; },
          Y: function (v) { return ty + ph - ((v - ylo) / (yhi - ylo)) * ph; }
        };
      }

      /* ── left: the trajectory ─────────────────────────────────────────── */
      var dlo = 0, dhi = Math.max(4, st.mode * 1.6, DP * 1.4);
      var Lp = box(ax, dlo, dhi, 'estimate d');

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

      p.text(p.o.pad.l + p.plotW() / 2, p.h - 4, 'gradient-ascent step',
        { pixel: true, size: 10.5, colour: C.ink2 });
    });

    [sU, sVp, sVu, sEta].forEach(function (s) { s.onchange(run); });

    var body = el('div', {});
    mount.appendChild(A.panel([
      A.row([sU.el, sVp.el, sVu.el, sEta.el]),
      body, out,
      A.note('The prior mean is held at <i>d</i><sub>p</sub> = 2 throughout, and the estimate '
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
