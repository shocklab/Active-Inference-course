/* w01.js — interactive figures for Week 1: the hypothesis-testing agent. */
(function () {
  'use strict';
  var A = window.AIF, el = A.el, C = A.colours, M = A.math;

  /* ══════════════════════════════════════════════════════════════════════
   * 1. homeostasis-drift
   *    A body in a two-dimensional physiological space. Noise pushes it out
   *    of the viable set. Action pulls it back. The readout that matters is
   *    the entropy of the visited states: staying alive IS staying low.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('homeostasis-drift', function (mount) {
    var NB = 26;                              /* histogram bins per axis */
    var HMAX = Math.log(NB * NB);             /* entropy of a uniform occupancy */
    var st, ghosts = [];

    function fresh() {
      return { x: 0, y: 0, t: 0, dead: false, trail: [],
               bins: new Float64Array(NB * NB), n: 0, hist: [], kappa: gain.get() };
    }

    var sigma = A.slider({ label: 'Noise &sigma;', min: 0.005, max: 0.09, step: 0.005, value: 0.035, decimals: 3 });
    var gain = A.slider({ label: 'Action strength &kappa;', min: 0, max: 0.30, step: 0.01, value: 0, decimals: 2 });
    var out = A.readout();

    /* Keep the finished curve so the next run can be compared against it.
     * Two runs at different kappa on the same axes is the whole argument. */
    function reset(keepGhost) {
      if (keepGhost && st && st.hist.length > 12) {
        ghosts.push({ hist: st.hist.slice(), kappa: st.kappa, dead: st.dead });
        if (ghosts.length > 2) ghosts.shift();
      }
      st = fresh();
      plot.render();
    }

    function entropy() {
      if (!st.n) return 0;
      var H = 0;
      for (var i = 0; i < st.bins.length; i++) {
        var q = st.bins[i] / st.n;
        if (q > 0) H -= q * Math.log(q);
      }
      return H;
    }

    function step() {
      if (st.dead) return;
      var s = sigma.get(), k = gain.get();
      st.kappa = k;
      st.x += s * M.randn() - k * st.x;
      st.y += s * M.randn() - k * st.y;
      st.t++;
      st.trail.push([st.x, st.y]);
      if (st.trail.length > 420) st.trail.shift();
      var bx = Math.floor((st.x + 1) / 2 * NB), by = Math.floor((st.y + 1) / 2 * NB);
      if (bx >= 0 && bx < NB && by >= 0 && by < NB) { st.bins[by * NB + bx]++; st.n++; }
      if (st.t % 5 === 0) st.hist.push([st.t, entropy()]);
      if (st.x * st.x + st.y * st.y > 1) st.dead = true;
    }

    var plot = new A.Plot({ aspect: 0.44, pad: { l: 8, r: 8, t: 8, b: 8 } });

    plot.onDraw(function (p) {
      p.clear();
      var c = p.ctx;
      var H = entropy();

      /* ── left panel: the body in its physiological space ─────────────── */
      var side = Math.min(p.h - 34, p.w * 0.46);
      var cx = 14 + side / 2, cy = p.h / 2 - 4, R = side / 2;
      function X(v) { return cx + v * R; }
      function Y(v) { return cy - v * R; }

      if (st.n > 0) {
        var mx = 0, i;
        for (i = 0; i < st.bins.length; i++) if (st.bins[i] > mx) mx = st.bins[i];
        var bw = (2 * R) / NB;
        for (var by = 0; by < NB; by++) {
          for (var bx = 0; bx < NB; bx++) {
            var v = st.bins[by * NB + bx];
            if (!v) continue;
            c.globalAlpha = 0.10 + 0.58 * Math.sqrt(v / mx);
            c.fillStyle = C.accent2;
            c.fillRect(X(-1) + bx * bw, Y(-1) - (by + 1) * bw, bw + 0.5, bw + 0.5);
          }
        }
        c.globalAlpha = 1;
      }

      c.save();
      c.strokeStyle = st.dead ? C.clay : C.ink2; c.lineWidth = 1.6;
      c.beginPath(); c.arc(cx, cy, R, 0, 2 * Math.PI); c.stroke();
      c.setLineDash([3, 4]); c.strokeStyle = C.accent; c.lineWidth = 1.2;
      c.beginPath(); c.arc(cx, cy, R * 0.35, 0, 2 * Math.PI); c.stroke();
      c.restore();

      if (st.trail.length > 1) {
        c.save();
        c.strokeStyle = st.dead ? C.clay : C.accent;
        c.lineWidth = 1.1; c.globalAlpha = 0.7; c.lineJoin = 'round';
        c.beginPath();
        st.trail.forEach(function (q, i) {
          if (i === 0) c.moveTo(X(q[0]), Y(q[1])); else c.lineTo(X(q[0]), Y(q[1]));
        });
        c.stroke(); c.restore();
      }
      c.save();
      c.fillStyle = st.dead ? C.clay : C.accent;
      c.beginPath(); c.arc(X(st.x), Y(st.y), 4.2, 0, 2 * Math.PI); c.fill();
      c.restore();

      p.text(cx, Y(1) - 13, st.dead ? 'boundary crossed' : 'viable set',
        { pixel: true, size: 10.5, weight: '600', colour: st.dead ? C.clay : C.muted });
      p.text(cx, cy - R * 0.35 - 8, 'preferred', { pixel: true, size: 9.5, colour: C.accent });
      p.text(cx, Y(-1) + 15, 'two physiological variables',
        { pixel: true, size: 9.5, colour: C.faint });

      /* ── right panel: entropy of the visited states against time ─────── */
      var gx = 14 + side + 58, gw = p.w - gx - 14, gy = 26, gh = p.h - 62;
      var tmax = Math.max(400, st.t * 1.05);
      ghosts.forEach(function (g) { if (g.hist.length) tmax = Math.max(tmax, g.hist[g.hist.length - 1][0] * 1.05); });

      function GX(t) { return gx + (t / tmax) * gw; }
      function GY(h) { return gy + gh - (h / HMAX) * gh; }

      c.save();
      c.strokeStyle = C.rule; c.lineWidth = 1;
      [0.25, 0.5, 0.75, 1].forEach(function (f) {
        c.beginPath(); c.setLineDash([2, 4]);
        c.moveTo(gx, GY(HMAX * f)); c.lineTo(gx + gw, GY(HMAX * f)); c.stroke();
      });
      c.setLineDash([]);
      c.beginPath(); c.moveTo(gx + 0.5, gy); c.lineTo(gx + 0.5, gy + gh); c.lineTo(gx + gw, gy + gh); c.stroke();
      c.restore();

      p.text(gx - 7, GY(HMAX), HMAX.toFixed(1), { pixel: true, align: 'right', size: 10, colour: C.faint });
      p.text(gx - 7, GY(0), '0', { pixel: true, align: 'right', size: 10, colour: C.faint });
      p.text(gx + gw / 2, gy - 13, 'entropy of visited states, nats',
        { pixel: true, size: 10.5, weight: '600', colour: C.ink2 });
      p.text(gx + gw / 2, gy + gh + 15, 'steps', { pixel: true, size: 9.5, colour: C.faint });
      p.text(gx + 4, GY(HMAX) - 11, 'uniform occupancy', { pixel: true, align: 'left', size: 9, colour: C.faint });

      ghosts.forEach(function (g, gi) {
        c.save();
        c.strokeStyle = g.dead ? C.clay : C.muted;
        c.globalAlpha = 0.42; c.lineWidth = 1.4; c.setLineDash([4, 3]);
        c.beginPath();
        g.hist.forEach(function (q, i) { i ? c.lineTo(GX(q[0]), GY(q[1])) : c.moveTo(GX(q[0]), GY(q[1])); });
        c.stroke(); c.restore();
        var last = g.hist[g.hist.length - 1];
        if (last) {
          p.text(GX(last[0]) - 4, GY(last[1]) - 9, 'κ = ' + g.kappa.toFixed(2),
            { pixel: true, align: 'right', size: 9.5, colour: g.dead ? C.clay : C.muted });
        }
      });

      if (st.hist.length > 1) {
        c.save();
        c.strokeStyle = st.dead ? C.clay : C.accent; c.lineWidth = 2;
        c.beginPath();
        st.hist.forEach(function (q, i) { i ? c.lineTo(GX(q[0]), GY(q[1])) : c.moveTo(GX(q[0]), GY(q[1])); });
        c.stroke(); c.restore();
        p.dot(GX(st.t), GY(H), { pixel: true, colour: st.dead ? C.clay : C.accent, r: 3.5 });
      }

      out.show([
        ['steps', String(st.t)],
        ['status', st.dead ? 'boundary crossed' : 'within bounds'],
        ['H[visited states]', H.toFixed(3) + ' / ' + HMAX.toFixed(3) + ' nats']
      ]);
    });

    var running = false, raf = null, runBtn;
    function halt() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      runBtn.textContent = 'Run';
    }
    function loop() {
      if (!running) return;
      for (var i = 0; i < 4; i++) step();
      plot.render();
      if (st.dead) { halt(); return; }   /* nothing left to simulate */
      raf = requestAnimationFrame(loop);
    }
    runBtn = A.button('Run', function () {
      if (running) { halt(); return; }
      if (st.dead) reset(true);          /* dead organism: start a fresh run */
      running = true;
      runBtn.textContent = 'Pause';
      loop();
    });

    var body = el('div', {});
    mount.appendChild(A.panel([
      body,
      A.row([sigma.el, gain.el]),
      el('div', { class: 'wbtns' }, [
        runBtn,
        A.button('New run, keep the old curve', function () { reset(true); }, true),
        A.button('Clear', function () { ghosts = []; reset(false); }, true)
      ]),
      out,
      A.note('Run it once with <b>&kappa; = 0</b> until the boundary is crossed. Then press '
        + '<b>New run</b>, raise <b>&kappa;</b>, and run again: the previous curve stays on the '
        + 'right panel as a dashed line so you can compare them directly. Same noise, same physics. '
        + 'The unregulated curve climbs towards uniform occupancy and stops only because the '
        + 'organism does. The regulated one flattens out well below it, and keeps flat. That '
        + 'plateau is what we are going to spend twelve weeks explaining how to achieve.')
    ]));
    reset(false);              /* state must exist before anything can draw */
    plot.attach(body);
    sigma.onchange(function () { plot.render(); });
    gain.onchange(function () { plot.render(); });
    return { stop: halt };
  });

  /* ══════════════════════════════════════════════════════════════════════
   * 2. bayes-discrete
   *    Prior, likelihood column, posterior, and the evidence that normalises
   *    it. Everything on one screen so the division is visible.
   * ═════════════════════════════════════════════════════════════════════ */
  var STATES = ['leopard', 'baboon', 'nothing'];
  var OBS = ['tawny flash', 'branch shakes', 'quiet'];

  /* Likelihood at full sharpness: rows = observations, cols = states. */
  var A_SHARP = [
    [0.70, 0.15, 0.02],
    [0.25, 0.75, 0.08],
    [0.05, 0.10, 0.90]
  ];

  function blurA(lambda) {
    /* lambda = 0 -> A_SHARP; lambda = 1 -> every state explains everything */
    return M.colNormalise(A_SHARP.map(function (r) {
      return r.map(function (v) { return (1 - lambda) * v + lambda * (1 / OBS.length); });
    }));
  }

  A.register('bayes-discrete', function (mount) {
    var pLeo = A.slider({ label: 'Prior on leopard', min: 0.01, max: 0.60, step: 0.01, value: 0.08 });
    var pBab = A.slider({ label: 'Prior on baboon', min: 0.01, max: 0.60, step: 0.01, value: 0.22 });
    var amb = A.slider({ label: 'Ambiguity &lambda;', min: 0, max: 0.95, step: 0.05, value: 0 });
    var obsIdx = 0;

    var tabs = el('div', { class: 'wtabs' });
    var tabEls = OBS.map(function (name, i) {
      var t = el('button', {
        class: 'wtab' + (i === 0 ? ' on' : ''), text: 'you observe: ' + name,
        onclick: function () {
          obsIdx = i;
          tabEls.forEach(function (x, j) { x.className = 'wtab' + (j === i ? ' on' : ''); });
          plot.render();
        }
      });
      tabs.appendChild(t);
      return t;
    });

    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.42, pad: { l: 44, r: 12, t: 22, b: 44 } });

    function current() {
      var a = M.clamp(pLeo.get(), 0.01, 0.9), b = M.clamp(pBab.get(), 0.01, 0.9);
      if (a + b > 0.97) { var s = 0.97 / (a + b); a *= s; b *= s; }
      var prior = [a, b, 1 - a - b];
      var Am = blurA(amb.get());
      var r = M.posterior(Am, prior, obsIdx);
      return { prior: prior, A: Am, post: r.post, ev: r.evidence };
    }

    plot.onDraw(function (p) {
      var d = current();
      p.clear();
      p.o.ylim = [0, 1];
      p.grid([0.25, 0.5, 0.75, 1.0], { dash: [2, 4] });
      p.frame();
      p.yaxis([0, 0.25, 0.5, 0.75, 1.0], function (v) { return v.toFixed(2); });
      p.bars([
        { values: d.prior, colour: C.faint },
        { values: OBS.map(function (_, i) { return d.A[obsIdx][i]; }).slice(0, 3), colour: C.gold },
        { values: d.post, colour: C.accent }
      ], STATES, { valueLabels: true });
      p.legend([
        { label: 'prior  P(s)', colour: C.faint },
        { label: 'likelihood  P(o|s)', colour: C.gold },
        { label: 'posterior  P(s|o)', colour: C.accent }
      ], p.o.pad.l + 2, 11);
      out.show([
        ['P(o)', d.ev.toFixed(4)],
        ['surprise  −ln P(o)', (-Math.log(d.ev)).toFixed(3) + ' nats'],
        ['D_KL[post || prior]', M.kl(d.post, d.prior).toFixed(3) + ' nats']
      ]);
    });

    var body = el('div', {});
    mount.appendChild(A.panel([
      tabs, body,
      A.row([pLeo.el, pBab.el, amb.el]),
      out,
      A.note('The likelihood bars are one <em>column</em> of the matrix: what each state predicts '
        + 'about the observation you actually got. The posterior is prior times likelihood, divided by '
        + 'the number in the readout. Push <b>&lambda;</b> up and the three likelihood bars level out: '
        + 'every state explains the data equally well, so the posterior collapses back onto the prior '
        + 'and the observation has told you nothing. That last readout is exactly how much it told you.')
    ]));
    plot.attach(body);
    [pLeo, pBab, amb].forEach(function (s) { s.onchange(function () { plot.render(); }); });
    return null;
  });

  /* ══════════════════════════════════════════════════════════════════════
   * 3. forward-inverse
   *    The same matrix read two ways. Down a column is free. Along a row you
   *    owe a sum over every state you did not observe.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('forward-inverse', function (mount) {
    var mode = 'forward', sel = 0;
    var prior = [0.08, 0.22, 0.70];
    var Am = A_SHARP;

    var tabs = el('div', { class: 'wtabs' });
    var modes = [['forward', 'Forward: pick a state'], ['inverse', 'Inverse: pick an observation']];
    var tEls = modes.map(function (m, i) {
      var t = el('button', {
        class: 'wtab' + (i === 0 ? ' on' : ''), text: m[1],
        onclick: function () {
          mode = m[0]; sel = 0;
          tEls.forEach(function (x, j) { x.className = 'wtab' + (j === i ? ' on' : ''); });
          plot.render();
        }
      });
      tabs.appendChild(t);
      return t;
    });

    var plot = new A.Plot({ aspect: 0.32, pad: { l: 96, r: 14, t: 34, b: 42 } });
    var out = A.readout();

    plot.onDraw(function (p) {
      p.clear();
      var c = p.ctx;
      var nS = STATES.length, nO = OBS.length;
      // Fill the canvas rather than sitting in the top-left of it: a fixed
      // 150px cap left most of a wide figure empty.
      var gw = Math.min(p.plotW() * 0.5, 300), gh = p.plotH();
      var cw = gw / nS, ch = gh / nO;
      var gx = p.o.pad.l, gy = p.o.pad.t + 6;

      /* the matrix */
      for (var i = 0; i < nO; i++) {
        for (var j = 0; j < nS; j++) {
          var v = Am[i][j];
          var lit = (mode === 'forward' && j === sel) || (mode === 'inverse' && i === sel);
          c.globalAlpha = lit ? 0.18 + 0.82 * v : 0.08 + 0.30 * v;
          c.fillStyle = lit ? (mode === 'forward' ? C.accent : C.clay) : C.muted;
          c.fillRect(gx + j * cw, gy + i * ch, cw - 1.5, ch - 1.5);
          c.globalAlpha = 1;
          p.text(gx + j * cw + cw / 2, gy + i * ch + ch / 2, v.toFixed(2),
            { pixel: true, size: 10.5, colour: v > 0.45 ? '#fff' : C.ink2, font: A.fonts.mono });
        }
        p.text(gx - 8, gy + i * ch + ch / 2, OBS[i],
          { pixel: true, align: 'right', size: 10, colour: mode === 'inverse' && i === sel ? C.clay : C.muted });
      }
      for (var j2 = 0; j2 < nS; j2++) {
        p.text(gx + j2 * cw + cw / 2, gy + gh + 13, STATES[j2],
          { pixel: true, size: 10, colour: mode === 'forward' && j2 === sel ? C.accent : C.muted });
      }
      p.text(gx + gw / 2, gy - 13, 'P(o | s)', { pixel: true, size: 10.5, weight: '600', colour: C.ink2 });

      /* the explanation panel */
      var bx = gx + gw + 34, bw = p.w - p.o.pad.r - bx;
      var lines;
      if (mode === 'forward') {
        var col = Am.map(function (r) { return r[sel]; });
        lines = [
          ['One column. Read it off.', C.ink2, '600'],
          ['P(o | s = ' + STATES[sel] + ') = [' + col.map(function (v) { return v.toFixed(2); }).join(', ') + ']', C.accent, '400'],
          ['It sums to ' + M.sum(col).toFixed(2) + ' by construction.', C.muted, '400'],
          ['Cost: zero arithmetic.', C.muted, '400']
        ];
        out.show([['direction', 'state → observation'], ['work', 'a table lookup']]);
      } else {
        var un = prior.map(function (pp, s) { return Am[sel][s] * pp; });
        var ev = M.sum(un);
        lines = [
          ['One row, and then a division.', C.ink2, '600'],
          ['numerators  P(o|s)P(s) = [' + un.map(function (v) { return v.toFixed(3); }).join(', ') + ']', C.clay, '400'],
          ['P(o) = Σ over all states = ' + ev.toFixed(3), C.clay, '400'],
          ['posterior = [' + M.normalise(un).map(function (v) { return v.toFixed(2); }).join(', ') + ']', C.ink2, '400'],
          ['Cost: one term per state. Always.', C.muted, '400']
        ];
        out.show([['direction', 'observation → state'],
                  ['work', 'a sum over the entire state space'],
                  ['P(o)', ev.toFixed(4)]]);
      }
      lines.forEach(function (L, i) {
        p.text(bx, gy + Math.max(6, (gh - lines.length * 21) / 2) + i * 21, L[0],
          { pixel: true, align: 'left', size: 10.8, colour: L[1], weight: L[2] });
      });

      /* clickable targets */
      p._hit = { gx: gx, gy: gy, cw: cw, ch: ch, gw: gw, gh: gh };
    });

    plot.canvas.style.cursor = 'pointer';
    plot.canvas.addEventListener('click', function (e) {
      var h = plot._hit; if (!h) return;
      var r = plot.canvas.getBoundingClientRect();
      var x = e.clientX - r.left, y = e.clientY - r.top;
      if (x < h.gx || x > h.gx + h.gw || y < h.gy || y > h.gy + h.gh) return;
      sel = mode === 'forward'
        ? Math.min(STATES.length - 1, Math.floor((x - h.gx) / h.cw))
        : Math.min(OBS.length - 1, Math.floor((y - h.gy) / h.ch));
      plot.render();
    });

    var body = el('div', {});
    mount.appendChild(A.panel([
      tabs, body, out,
      A.note('Click a cell to select a different state or observation. The matrix never changes. '
        + 'What changes is which way you read it, and only one of those two directions is free.')
    ]));
    plot.attach(body);
    return null;
  });

  /* ══════════════════════════════════════════════════════════════════════
   * 4. evidence-blowup
   *    Why "sum over all states" is not a small inconvenience.
   * ═════════════════════════════════════════════════════════════════════ */
  A.register('evidence-blowup', function (mount) {
    var nf = A.slider({ label: 'Number of state factors', min: 1, max: 40, step: 1, value: 12, decimals: 0 });
    var kf = A.slider({ label: 'Values per factor', min: 2, max: 12, step: 1, value: 4, decimals: 0 });
    var out = A.readout();
    var plot = new A.Plot({ aspect: 0.42, pad: { l: 58, r: 16, t: 16, b: 42 } });

    /* one addition per nanosecond is generous by a couple of orders of magnitude */
    var OPS_PER_SEC = 1e9;
    var AGE_UNIVERSE_S = 4.35e17;

    plot.onDraw(function (p) {
      var n = nf.get(), k = kf.get();
      var log10N = n * Math.log10(k);
      p.o.xlim = [1, 40];
      p.o.ylim = [0, Math.max(24, log10N * 1.15)];
      p.clear();
      var ticks = [];
      for (var t = 0; t <= p.o.ylim[1]; t += Math.ceil(p.o.ylim[1] / 6)) ticks.push(t);
      p.grid(ticks, { dash: [2, 4] });
      p.frame();
      p.yaxis(ticks, function (v) { return '10^' + v; }, { label: 'terms in the sum' });
      p.xaxis([1, 10, 20, 30, 40], null, { label: 'number of state factors' });

      [2, 4, 8].forEach(function (kk, i) {
        var pts = [];
        for (var x = 1; x <= 40; x++) pts.push([x, x * Math.log10(kk)]);
        p.line(pts, { colour: A.series[i], width: kk === k ? 2.4 : 1.2, alpha: kk === k ? 1 : 0.4 });
      });
      var pts2 = [];
      for (var x2 = 1; x2 <= 40; x2++) pts2.push([x2, x2 * Math.log10(k)]);
      p.line(pts2, { colour: C.clay, width: 2.4 });
      p.dot(n, log10N, { colour: C.clay, r: 4.5 });
      p.legend([
        { label: 'k = 2', colour: A.series[0] }, { label: 'k = 4', colour: A.series[1] },
        { label: 'k = 8', colour: A.series[2] }, { label: 'your setting', colour: C.clay }
      ], p.o.pad.l + 4, p.o.pad.t + 8);

      var secs = Math.pow(10, log10N) / OPS_PER_SEC;
      var human;
      if (secs < 60) human = secs.toPrecision(3) + ' s';
      else if (secs < 3.15e7) human = (secs / 86400).toPrecision(3) + ' days';
      else if (secs < AGE_UNIVERSE_S) human = (secs / 3.15e7).toPrecision(3) + ' years';
      else human = (secs / AGE_UNIVERSE_S).toPrecision(3) + ' × the age of the universe';

      out.show([
        ['state space size', k + '^' + n + ' ≈ 10^' + log10N.toFixed(1)],
        ['at 10⁹ additions/second', human]
      ]);
    });

    var body = el('div', {});
    mount.appendChild(A.panel([
      body, A.row([nf.el, kf.el]), out,
      A.note('A rat in a maze tracking position, hunger, the location of a predator, the smell in each '
        + 'arm and the time since the last reward already has a double-figure count of factors. '
        + 'The vertical axis is a logarithm: every step along the bottom multiplies the work. '
        + 'This is the wall that the whole of the next three weeks is built to get around.')
    ]));
    plot.attach(body);
    nf.onchange(function () { plot.render(); });
    kf.onchange(function () { plot.render(); });
    return null;
  });
})();
