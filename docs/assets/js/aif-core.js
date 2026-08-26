/* aif-core.js — the widget substrate for the Active Inference course.
 *
 * No dependencies, no build step. Widgets register themselves by name and are
 * mounted into any <div data-widget="name"> on the page.
 *
 *   AIF.register('bayes-discrete', function (mount, opts) { ... });
 *
 * Everything drawn goes through Plot, which handles device-pixel-ratio scaling
 * and resizing so canvases stay sharp on retina screens and reflow on phones.
 */
(function () {
  'use strict';

  var AIF = window.AIF = { widgets: {}, instances: [] };

  AIF.register = function (name, factory) { AIF.widgets[name] = factory; };

  /* ── palette (kept in sync with theme.py by eye, not by import) ──────── */
  var C = AIF.colours = {
    ink: '#1b1b1e', ink2: '#3c3c42', muted: '#6d6d75', faint: '#90909a',
    rule: '#e4e1d9', rule2: '#d3cfc4', paper: '#fdfcf9', card: '#ffffff',
    accent: '#0f5f57', accent2: '#14837a', accentTint: '#e9f2f0',
    clay: '#9c4d2f', clayTint: '#f9efe9', gold: '#8a6d1f', goldTint: '#faf5e6',
    blue: '#2c5f8a', violet: '#6b4a8a', tint: '#f4f2ec'
  };
  /* categorical series colours, chosen to stay distinguishable in greyscale */
  AIF.series = [C.accent, C.clay, C.blue, C.gold, C.violet, C.muted];

  var SANS = "'IBM Plex Sans',-apple-system,sans-serif";
  var SERIF = "'Source Serif 4',Georgia,serif";
  var MONO = "'IBM Plex Mono',Menlo,monospace";
  AIF.fonts = { sans: SANS, serif: SERIF, mono: MONO };

  /* ── DOM helpers ─────────────────────────────────────────────────────── */
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === 'class') n.className = attrs[k];
      else if (k === 'text') n.textContent = attrs[k];
      else if (k === 'html') n.innerHTML = attrs[k];
      else if (k.slice(0, 2) === 'on') n.addEventListener(k.slice(2), attrs[k]);
      else n.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) {
      n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return n;
  }
  AIF.el = el;

  /** A labelled range slider. Returns {el, get, set, onchange}. */
  function slider(o) {
    var fmt = o.format || function (v) { return v.toFixed(o.decimals != null ? o.decimals : 2); };
    var val = el('span', { class: 'wval' });
    var inp = el('input', {
      type: 'range', min: o.min, max: o.max,
      step: o.step != null ? o.step : (o.max - o.min) / 100, value: o.value
    });
    var lab = el('div', { class: 'wlab' }, [el('b', { html: o.label }), val]);
    var wrap = el('label', { class: 'wctl' }, [lab, inp]);
    var handlers = [];
    function sync() {
      val.textContent = fmt(parseFloat(inp.value));
      handlers.forEach(function (h) { h(parseFloat(inp.value)); });
    }
    inp.addEventListener('input', sync);
    val.textContent = fmt(parseFloat(inp.value));
    return {
      el: wrap, input: inp,
      get: function () { return parseFloat(inp.value); },
      set: function (v) { inp.value = v; sync(); },
      onchange: function (h) { handlers.push(h); return this; }
    };
  }
  AIF.slider = slider;

  function button(label, onclick, ghost) {
    return el('button', { class: 'wbtn' + (ghost ? ' ghost' : ''), text: label, onclick: onclick });
  }
  AIF.button = button;

  function readout() {
    var n = el('div', { class: 'wreadout' });
    n.show = function (pairs) {
      n.innerHTML = '';
      pairs.forEach(function (p) {
        n.appendChild(el('span', {}, [el('b', { text: p[0] + ' ' }), document.createTextNode(p[1])]));
      });
    };
    return n;
  }
  AIF.readout = readout;

  function note(html) { return el('div', { class: 'wnote', html: html }); }
  AIF.note = note;

  function row(kids) { return el('div', { class: 'wrow' }, kids); }
  AIF.row = row;

  function panel(kids) { return el('div', { class: 'wpanel' }, kids); }
  AIF.panel = panel;

  /* ── numeric helpers ─────────────────────────────────────────────────── */
  var M = AIF.math = {
    sum: function (a) { for (var s = 0, i = 0; i < a.length; i++) s += a[i]; return s; },
    normalise: function (a) {
      var s = M.sum(a);
      return s > 0 ? a.map(function (x) { return x / s; }) : a.map(function () { return 1 / a.length; });
    },
    /* numerically safe: 0 log 0 = 0 */
    xlogx: function (x) { return x > 1e-300 ? x * Math.log(x) : 0; },
    entropy: function (p) { return -M.sum(p.map(M.xlogx)); },
    kl: function (q, p) {
      var s = 0;
      for (var i = 0; i < q.length; i++) if (q[i] > 1e-300) s += q[i] * Math.log(q[i] / Math.max(p[i], 1e-300));
      return s;
    },
    softmax: function (v, beta) {
      beta = beta == null ? 1 : beta;
      var mx = Math.max.apply(null, v);
      var e = v.map(function (x) { return Math.exp(beta * (x - mx)); });
      return M.normalise(e);
    },
    /* matrix (rows x cols) times column vector */
    matvec: function (A, v) {
      return A.map(function (r) {
        for (var s = 0, j = 0; j < v.length; j++) s += r[j] * v[j];
        return s;
      });
    },
    matTvec: function (A, v) {
      var out = new Array(A[0].length).fill(0);
      for (var i = 0; i < A.length; i++) for (var j = 0; j < A[i].length; j++) out[j] += A[i][j] * v[i];
      return out;
    },
    /* Bayes: posterior over states given observation index o */
    posterior: function (A, prior, o) {
      var un = prior.map(function (p, s) { return A[o][s] * p; });
      return { post: M.normalise(un), evidence: M.sum(un) };
    },
    /* column-normalise a matrix so each column is a distribution over rows */
    colNormalise: function (A) {
      var nr = A.length, nc = A[0].length, out = A.map(function (r) { return r.slice(); });
      for (var j = 0; j < nc; j++) {
        var s = 0, i;
        for (i = 0; i < nr; i++) s += out[i][j];
        for (i = 0; i < nr; i++) out[i][j] = s > 0 ? out[i][j] / s : 1 / nr;
      }
      return out;
    },
    gaussian: function (x, mu, sd) {
      var z = (x - mu) / sd;
      return Math.exp(-0.5 * z * z) / (sd * Math.sqrt(2 * Math.PI));
    },
    randn: (function () {
      var spare = null;
      return function () {
        if (spare !== null) { var s = spare; spare = null; return s; }
        var u, v, r;
        do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; r = u * u + v * v; }
        while (r >= 1 || r === 0);
        var f = Math.sqrt(-2 * Math.log(r) / r);
        spare = v * f;
        return u * f;
      };
    })(),
    clamp: function (x, a, b) { return x < a ? a : x > b ? b : x; },
    lerp: function (a, b, t) { return a + (b - a) * t; }
  };

  /* ── Plot: a thin, sharp canvas wrapper ──────────────────────────────── */
  function Plot(opts) {
    this.o = Object.assign({
      aspect: 0.52,          /* height / width */
      pad: { l: 46, r: 14, t: 14, b: 34 },
      xlim: [0, 1], ylim: [0, 1]
    }, opts || {});
    this.canvas = el('canvas', { class: 'wcanvas' });
    this.ctx = this.canvas.getContext('2d');
    this.w = 0; this.h = 0;
    this._draw = opts && opts.draw;
    var self = this;
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(function () { self.resize(); });
    }
  }

  Plot.prototype.attach = function (parent) {
    parent.appendChild(this.canvas);
    if (this._ro) this._ro.observe(this.canvas.parentNode || parent);
    var self = this;
    /* Size it now if layout is already settled, and again on the next frame and
     * shortly after. requestAnimationFrame alone is not enough: a throttled or
     * hidden tab never fires it, and the figure would sit blank until the user
     * came back to it. */
    this.resize();
    requestAnimationFrame(function () { self.resize(); });
    setTimeout(function () { self.resize(); }, 60);
    return this;
  };

  Plot.prototype.resize = function () {
    var cssW = this.canvas.clientWidth || (this.canvas.parentNode && this.canvas.parentNode.clientWidth) || 0;
    if (!cssW) return;                       /* not laid out yet; a later call will catch it */
    var cssH = Math.round(cssW * this.o.aspect);
    var dpr = window.devicePixelRatio || 1;
    if (cssW === this.w && cssH === this.h && this._dpr === dpr) return;
    this.w = cssW; this.h = cssH; this._dpr = dpr;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.canvas.style.height = cssH + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  };

  Plot.prototype.render = function () { if (this._draw && this.w) this._draw(this); };
  Plot.prototype.onDraw = function (f) { this._draw = f; return this; };

  /* data-space -> pixel-space */
  Plot.prototype.px = function (x) {
    var p = this.o.pad, L = this.o.xlim;
    return p.l + (x - L[0]) / (L[1] - L[0]) * (this.w - p.l - p.r);
  };
  Plot.prototype.py = function (y) {
    var p = this.o.pad, L = this.o.ylim;
    return this.h - p.b - (y - L[0]) / (L[1] - L[0]) * (this.h - p.t - p.b);
  };
  Plot.prototype.plotW = function () { return this.w - this.o.pad.l - this.o.pad.r; };
  Plot.prototype.plotH = function () { return this.h - this.o.pad.t - this.o.pad.b; };

  Plot.prototype.clear = function (bg) {
    var c = this.ctx;
    c.clearRect(0, 0, this.w, this.h);
    if (bg !== false) { c.fillStyle = bg || C.card; c.fillRect(0, 0, this.w, this.h); }
    return this;
  };

  Plot.prototype.frame = function (o) {
    o = o || {};
    var c = this.ctx, p = this.o.pad;
    c.save();
    c.strokeStyle = C.rule; c.lineWidth = 1;
    c.beginPath();
    c.moveTo(p.l + 0.5, p.t); c.lineTo(p.l + 0.5, this.h - p.b + 0.5);
    c.lineTo(this.w - p.r, this.h - p.b + 0.5);
    c.stroke();
    c.restore();
    return this;
  };

  Plot.prototype.grid = function (yticks, o) {
    o = o || {};
    var c = this.ctx, self = this;
    c.save();
    c.strokeStyle = o.colour || C.rule; c.lineWidth = 1;
    if (o.dash) c.setLineDash(o.dash);
    yticks.forEach(function (y) {
      var Y = Math.round(self.py(y)) + 0.5;
      c.beginPath(); c.moveTo(self.o.pad.l, Y); c.lineTo(self.w - self.o.pad.r, Y); c.stroke();
    });
    c.restore();
    return this;
  };

  Plot.prototype.text = function (x, y, s, o) {
    o = o || {};
    var c = this.ctx;
    c.save();
    c.font = (o.weight ? o.weight + ' ' : '') + (o.size || 11) + 'px ' + (o.font || SANS);
    c.fillStyle = o.colour || C.muted;
    c.textAlign = o.align || 'center';
    c.textBaseline = o.baseline || 'middle';
    var X = o.pixel ? x : this.px(x), Y = o.pixel ? y : this.py(y);
    c.fillText(s, X, Y);
    c.restore();
    return this;
  };

  Plot.prototype.yaxis = function (ticks, fmt, o) {
    o = o || {}; fmt = fmt || function (v) { return String(v); };
    var self = this;
    ticks.forEach(function (t) {
      self.text(self.o.pad.l - 7, self.py(t), fmt(t),
        { pixel: true, align: 'right', size: o.size || 10.5, colour: o.colour || C.faint });
    });
    if (o.label) {
      var c = this.ctx;
      c.save();
      c.translate(11, this.o.pad.t + this.plotH() / 2);
      c.rotate(-Math.PI / 2);
      c.font = '600 10px ' + SANS; c.fillStyle = C.muted;
      c.textAlign = 'center'; c.textBaseline = 'middle';
      c.fillText(o.label, 0, 0);
      c.restore();
    }
    return this;
  };

  Plot.prototype.xaxis = function (ticks, fmt, o) {
    o = o || {}; fmt = fmt || function (v) { return String(v); };
    var self = this;
    ticks.forEach(function (t) {
      self.text(self.px(t), self.h - self.o.pad.b + 13, fmt(t),
        { pixel: true, size: o.size || 10.5, colour: o.colour || C.faint });
    });
    if (o.label) {
      this.text(this.o.pad.l + this.plotW() / 2, this.h - 5, o.label,
        { pixel: true, size: 10, weight: '600', colour: C.muted });
    }
    return this;
  };

  Plot.prototype.line = function (pts, o) {
    o = o || {};
    if (!pts.length) return this;
    var c = this.ctx, self = this;
    c.save();
    c.strokeStyle = o.colour || C.accent;
    c.lineWidth = o.width || 2;
    c.lineJoin = 'round'; c.lineCap = 'round';
    if (o.dash) c.setLineDash(o.dash);
    if (o.alpha != null) c.globalAlpha = o.alpha;
    c.beginPath();
    pts.forEach(function (p, i) {
      var X = self.px(p[0]), Y = self.py(p[1]);
      if (i === 0) c.moveTo(X, Y); else c.lineTo(X, Y);
    });
    c.stroke();
    c.restore();
    return this;
  };

  Plot.prototype.area = function (pts, o) {
    o = o || {};
    if (!pts.length) return this;
    var c = this.ctx, self = this, base = o.base != null ? o.base : this.o.ylim[0];
    c.save();
    c.fillStyle = o.colour || C.accentTint;
    if (o.alpha != null) c.globalAlpha = o.alpha;
    c.beginPath();
    c.moveTo(this.px(pts[0][0]), this.py(base));
    pts.forEach(function (p) { c.lineTo(self.px(p[0]), self.py(p[1])); });
    c.lineTo(this.px(pts[pts.length - 1][0]), this.py(base));
    c.closePath(); c.fill();
    c.restore();
    return this;
  };

  Plot.prototype.dot = function (x, y, o) {
    o = o || {};
    var c = this.ctx;
    c.save();
    c.fillStyle = o.colour || C.accent;
    c.beginPath();
    c.arc(o.pixel ? x : this.px(x), o.pixel ? y : this.py(y), o.r || 3.5, 0, 2 * Math.PI);
    c.fill();
    if (o.ring) { c.strokeStyle = o.ring; c.lineWidth = o.ringWidth || 2; c.stroke(); }
    c.restore();
    return this;
  };

  Plot.prototype.rect = function (x, y, w, h, o) {
    o = o || {};
    var c = this.ctx;
    c.save();
    if (o.fill) { c.fillStyle = o.fill; if (o.alpha != null) c.globalAlpha = o.alpha; c.fillRect(x, y, w, h); }
    if (o.stroke) { c.globalAlpha = 1; c.strokeStyle = o.stroke; c.lineWidth = o.width || 1; c.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); }
    c.restore();
    return this;
  };

  /**
   * A group of categorical bars. `groups` is an array of
   * {values:[...], colour, label}; bars are drawn side by side per category.
   */
  Plot.prototype.bars = function (groups, labels, o) {
    o = o || {};
    var n = labels.length, k = groups.length, self = this;
    var slotW = this.plotW() / n;
    var pad = slotW * (o.gap != null ? o.gap : 0.22);
    var barW = (slotW - 2 * pad) / k;
    var y0 = this.py(0);
    groups.forEach(function (g, gi) {
      g.values.forEach(function (v, i) {
        var x = self.o.pad.l + i * slotW + pad + gi * barW;
        var y = self.py(v);
        self.rect(x, Math.min(y, y0), barW - (k > 1 ? 1.5 : 0), Math.abs(y0 - y),
          { fill: g.colour || AIF.series[gi], alpha: g.alpha });
        if (o.valueLabels && Math.abs(v) > 1e-9) {
          self.text(x + barW / 2, y - 8, (o.valueFormat || function (t) { return t.toFixed(2); })(v),
            { pixel: true, size: 9.5, colour: g.colour || AIF.series[gi], weight: '600' });
        }
      });
    });
    labels.forEach(function (lb, i) {
      self.text(self.o.pad.l + (i + 0.5) * slotW, self.h - self.o.pad.b + 13, lb,
        { pixel: true, size: 10.5, colour: C.ink2 });
    });
    if (o.xlabel) {
      this.text(this.o.pad.l + this.plotW() / 2, this.h - 4, o.xlabel,
        { pixel: true, size: 10, weight: '600', colour: C.muted });
    }
    return this;
  };

  /** A small legend row drawn in pixel space. */
  Plot.prototype.legend = function (items, x, y, o) {
    o = o || {};
    var c = this.ctx, self = this, cx = x;
    items.forEach(function (it) {
      c.save();
      c.fillStyle = it.colour;
      if (it.dash) {
        c.strokeStyle = it.colour; c.lineWidth = 2; c.setLineDash([4, 3]);
        c.beginPath(); c.moveTo(cx, y); c.lineTo(cx + 14, y); c.stroke();
      } else {
        c.fillRect(cx, y - 5, 11, 10);
      }
      c.restore();
      self.text(cx + 18, y, it.label, { pixel: true, align: 'left', size: 10.5, colour: C.ink2 });
      cx += 18 + self.ctx.measureText(it.label).width * 0.62 + (o.gap || 26);
    });
    return this;
  };

  AIF.Plot = Plot;

  /** Convenience: build a Plot, attach it to a container div, return it. */
  AIF.plot = function (parent, opts) {
    var holder = el('div', {});
    parent.appendChild(holder);
    return new Plot(opts).attach(holder);
  };

  /* ── mounting ────────────────────────────────────────────────────────── */
  function mountAll() {
    document.querySelectorAll('[data-widget]').forEach(function (m) {
      if (m.dataset.mounted) return;
      var name = m.getAttribute('data-widget');
      var f = AIF.widgets[name];
      if (!f) {
        m.innerHTML = '<div class="wfallback">Widget <code>' + name + '</code> is not loaded on this page.</div>';
        return;
      }
      m.innerHTML = '';
      m.dataset.mounted = '1';
      try {
        AIF.instances.push(f(m, m.dataset) || null);
      } catch (e) {
        m.innerHTML = '<div class="wfallback">This figure failed to load.</div>';
        if (window.console) console.error('[AIF] widget ' + name + ' failed:', e);
      }
    });
  }
  AIF.mountAll = mountAll;

  /* Deferred scripts execute at readyState 'interactive', before
   * DOMContentLoaded. Mounting there would race the per-week widget files that
   * load after this one, so only mount immediately if the page is fully done. */
  if (document.readyState === 'complete') {
    mountAll();
  } else {
    document.addEventListener('DOMContentLoaded', mountAll);
  }
})();
