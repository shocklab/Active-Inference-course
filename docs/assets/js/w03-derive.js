/* w03-derive.js: the interactive problem "Deriving the missing term" on Week 3's
 * problems page.
 *
 * The first half of the file has no DOM in it: the simulation, an expression
 * parser and the checks of the two typed answers. Under Node it is exported as
 * a module, so the parser and the checks are tested outside the browser; in the
 * page it is used by the widget registered in the second half.
 *
 * The model is the linear case of Lesson 3.4. The world draws d ~ N(d_p, S),
 * S being Sigma_p^*, and the creature hears u = d + noise of variance Su. It
 * knows d_p and Su and learns its prior variance Sp by the log-variance step
 *
 *   Sp <- Sp * exp(beta * (target / Sp - 1) / 2),
 *
 * with target (dhat - d_p)^2 for the point rule, plus the student's own term in
 * the fourth step. runField mirrors _ex_run in content/week-03/values.py line for
 * line, in the same order of floating-point operations.
 */
(function (root) {
  'use strict';

  /* ── the generator of w03.js and values.py ─────────────────────────────── */
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

  /* ── the field world ──────────────────────────────────────────────────────
   * Lesson 3.4's collapse case: S = 1, Su = 0.5, so 3 Su exceeds S. The creature
   * starts at Sp = 1, the true value. A night is 20 sounds. Below a shrinkage
   * factor of DEAF the creature is counted deaf and the run ends with that night. */
  var FIELD = { dp: 2.0, S: 1.0, Su: 0.5, Sp0: 1.0, beta: 0.01, nights: 200, perNight: 20, seed: 2026 };
  var DEAF = 0.1;

  function shrink(Sp, Su) { return Sp / (Sp + Su); }

  function runField(extra) {
    var o = FIELD, r = mulberry32(o.seed);
    var Sp = o.Sp0, n = o.nights * o.perNight;
    var out = { d: [], u: [], dhat: [], nightSp: [], nightShrink: [], died: null };
    for (var i = 0; i < n; i++) {
      var d = o.dp + Math.sqrt(o.S) * normal(r);
      var u = d + Math.sqrt(o.Su) * normal(r);
      var dhat = (o.Su * o.dp + Sp * u) / (Sp + o.Su);
      var T = (dhat - o.dp) * (dhat - o.dp);
      if (extra) T = T + extra(Sp, o.Su);
      Sp = Sp * Math.exp(o.beta * 0.5 * (T / Sp - 1.0));
      out.d.push(d); out.u.push(u); out.dhat.push(dhat);
      if ((i + 1) % o.perNight === 0) {
        var k = shrink(Sp, o.Su);
        out.nightSp.push(Sp); out.nightShrink.push(k);
        if (k < DEAF) { out.died = out.nightSp.length; break; }
      }
    }
    return out;
  }

  /* ── the expression parser ────────────────────────────────────────────────
   * Recursive descent over numbers, names, + - * / ^ (or **), unary signs and
   * parentheses. Two factors written side by side multiply, as in 2Sp or
   * (Sp + Su)(Sp - Su), at the same precedence as * and /, so 1/2Sp is (1/2)Sp;
   * the rendering shows how it was read. ^ binds tighter than a unary minus and
   * associates to the right: -Sp^2 is -(Sp^2), and Sp^-1 is allowed.
   *
   *   expr    := term (('+' | '-') term)*
   *   term    := unary (('*' | '/') unary | factor-start unary)*
   *   unary   := ('+' | '-') unary | power
   *   power   := primary ('^' unary)?
   *   primary := number | name | '(' expr ')'
   */
  var NAMES = {
    Sp: { tex: '\\Sigma_p', what: 'the creature’s prior variance' },
    Su: { tex: '\\Sigma_u', what: 'the sensory variance' },
    Sps: { tex: '\\Sigma_p^{*}', what: 'the world’s prior variance' }
  };

  function ParseError(msg, pos) { this.message = msg; this.pos = pos; }

  function tokenise(src) {
    var toks = [], i = 0;
    src = src.replace(/−/g, '-').replace(/×|·/g, '*');
    while (i < src.length) {
      var c = src[i];
      if (/\s/.test(c)) { i++; continue; }
      var m = /^(\d+\.?\d*|\.\d+)/.exec(src.slice(i));
      if (m) { toks.push({ k: 'num', v: m[1], pos: i }); i += m[1].length; continue; }
      m = /^[A-Za-z_][A-Za-z0-9_]*/.exec(src.slice(i));
      if (m) { toks.push({ k: 'name', v: m[0], pos: i }); i += m[0].length; continue; }
      if (src.slice(i, i + 2) === '**') { toks.push({ k: 'op', v: '^', pos: i }); i += 2; continue; }
      if ('+-*/^()'.indexOf(c) >= 0) { toks.push({ k: 'op', v: c, pos: i }); i++; continue; }
      throw new ParseError('The character “' + c + '” is not part of an expression here.', i);
    }
    toks.push({ k: 'end', v: '', pos: src.length });
    return toks;
  }

  /* `allowed` lists the names this answer may use. A name outside it is an
   * error with its own message, so the student learns why, not only that. */
  function parse(src, allowed) {
    if (!/\S/.test(src)) return { ok: false, empty: true, error: 'Type an expression.' };
    var toks, i = 0;
    try { toks = tokenise(src); } catch (e) { return { ok: false, error: e.message, pos: e.pos }; }
    function peek() { return toks[i]; }
    function isOp(t, v) { return t.k === 'op' && t.v === v; }
    function describe(t) { return t.k === 'end' ? 'the end of the input' : '“' + t.v + '”'; }

    function expr() {
      var a = term();
      while (isOp(peek(), '+') || isOp(peek(), '-')) {
        var op = toks[i++].v;
        a = { t: 'bin', op: op, a: a, b: term() };
      }
      return a;
    }
    function startsFactor(t) { return t.k === 'num' || t.k === 'name' || isOp(t, '('); }
    function term() {
      var a = unary();
      for (;;) {
        var t = peek();
        if (isOp(t, '*') || isOp(t, '/')) { i++; a = { t: 'bin', op: t.v, a: a, b: unary() }; }
        else if (startsFactor(t)) a = { t: 'bin', op: '*', implicit: true, a: a, b: unary() };
        else return a;
      }
    }
    function unary() {
      if (isOp(peek(), '-')) { i++; return { t: 'neg', a: unary() }; }
      if (isOp(peek(), '+')) { i++; return unary(); }
      return power();
    }
    function power() {
      var a = primary();
      if (isOp(peek(), '^')) { i++; return { t: 'bin', op: '^', a: a, b: unary() }; }
      return a;
    }
    function primary() {
      var t = peek();
      if (t.k === 'num') { i++; return { t: 'num', v: parseFloat(t.v), s: t.v }; }
      if (t.k === 'name') {
        i++;
        if (allowed.indexOf(t.v) < 0) {
          var hint = NAMES[t.v]
            ? ' This answer may not use it.'
            : ' The names here are ' + allowed.join(', ') + '. Write a product with * or a space, as Sp*Su or Sp Su.';
          throw new ParseError('“' + t.v + '” is not a name this answer can use.' + hint, t.pos);
        }
        return { t: 'var', n: t.v };
      }
      if (isOp(t, '(')) {
        i++;
        var e = expr();
        if (!isOp(peek(), ')')) throw new ParseError('A “(” is not closed: expected “)” at ' + describe(peek()) + '.', peek().pos);
        i++;
        return e;
      }
      throw new ParseError('Expected a number, a name or “(” at ' + describe(t) + '.', t.pos);
    }

    try {
      var ast = expr();
      if (peek().k !== 'end') {
        var t = peek();
        throw new ParseError(isOp(t, ')') ? 'There is a “)” with no “(” before it.'
                                          : 'Unexpected ' + describe(t) + '.', t.pos);
      }
      return { ok: true, ast: ast, tex: toTex(ast), names: namesIn(ast) };
    } catch (e) {
      if (e instanceof ParseError) return { ok: false, error: e.message, pos: e.pos };
      throw e;
    }
  }

  function namesIn(n, acc) {
    acc = acc || {};
    if (n.t === 'var') acc[n.n] = true;
    if (n.a) namesIn(n.a, acc);
    if (n.b) namesIn(n.b, acc);
    return acc;
  }

  function evaluate(n, env) {
    switch (n.t) {
      case 'num': return n.v;
      case 'var': return env[n.n];
      case 'neg': return -evaluate(n.a, env);
      default:
        var a = evaluate(n.a, env), b = evaluate(n.b, env);
        if (n.op === '+') return a + b;
        if (n.op === '-') return a - b;
        if (n.op === '*') return a * b;
        if (n.op === '/') return a / b;
        return Math.pow(a, b);
    }
  }

  /* LaTeX for what was parsed, with only the parentheses the reading needs. */
  function prec(n) {
    if (n.t === 'bin') return n.op === '+' || n.op === '-' ? 1 : n.op === '^' ? 4 : 2;
    if (n.t === 'neg') return 3;
    return 5;
  }
  function paren(s) { return '\\left(' + s + '\\right)'; }
  function toTex(n) {
    if (n.t === 'num') return n.s;
    if (n.t === 'var') return NAMES[n.n].tex;
    if (n.t === 'neg') {
      var s = toTex(n.a);
      return '-' + (prec(n.a) <= 3 ? paren(s) : s);
    }
    var a = toTex(n.a), b = toTex(n.b);
    if (n.op === '/') return '\\frac{' + a + '}{' + b + '}';
    if (n.op === '^') return '{' + (prec(n.a) < 5 ? paren(a) : a) + '}^{' + b + '}';
    if (n.op === '+' || n.op === '-') {
      if (n.b.t === 'neg' || (n.op === '-' && prec(n.b) <= 1)) b = paren(b);
      return a + ' ' + n.op + ' ' + b;
    }
    if (prec(n.a) <= 1) a = paren(a);
    if (prec(n.b) <= 1 || n.b.t === 'neg') b = paren(b);
    var join = /^\{?[0-9.]/.test(b) ? ' \\cdot ' : '\\,';
    return a + join + b;
  }

  /* ── the checks ───────────────────────────────────────────────────────────
   * Each answer is evaluated at hidden parameter values drawn from a fixed
   * seed. Su is log-uniform on [0.05, 2]; the ratio S/Su is log-uniform on
   * [0.2, 2.8] for the first half of the points and on [3.2, 40] for the second,
   * so both sides of S = 3 Su are covered; the ratio Sp/S, used by (a) only, is
   * log-uniform on [0.02, 50].
   *
   * TOL is a relative tolerance. Twelve rearrangements of each right answer,
   * tested in game/week3-exercise/test_derive.js, differ from the reference by
   * at most 6e-14 at these points, the worst being an expanded form that loses
   * digits to cancellation. Every structurally wrong answer tested there is off
   * by more than 1e-3 at some point, and the right answer multiplied by
   * 1 + 1e-6 is off by 1e-6. A tolerance of 1e-9 passes any arrangement that
   * loses fewer than about six digits to cancellation and fails anything that
   * is not the same function to nine figures. */
  var TOL = 1e-9, NPTS = 24, SEED_A = 3401, SEED_B = 3402;
  var LIMIT_TOL = 1e-3, LIMIT_FAR = 1e6;

  function points(seed, n) {
    var r = mulberry32(seed), out = [];
    for (var i = 0; i < n; i++) {
      var Su = Math.exp(Math.log(0.05) + r() * (Math.log(2.0) - Math.log(0.05)));
      var lo = i < n / 2 ? 0.2 : 3.2, hi = i < n / 2 ? 2.8 : 40.0;
      var S = Su * Math.exp(Math.log(lo) + r() * (Math.log(hi) - Math.log(lo)));
      var Sp = S * Math.exp(Math.log(0.02) + r() * (Math.log(50.0) - Math.log(0.02)));
      out.push({ Sp: Sp, Su: Su, Sps: S, below: i < n / 2 });
    }
    return out;
  }
  var PTS_A = points(SEED_A, NPTS), PTS_B = points(SEED_B, NPTS);

  /* The reference answers. Used only to judge, never shown by the widget. */
  function refA(Sp, Sps, Su) { return Sp * Sp * (Sps + Su) / ((Sp + Su) * (Sp + Su)); }
  function refB(Sp, Su) { return Sp * Su / (Sp + Su); }

  function relErr(x, y) { return Math.abs(x - y) / Math.abs(y); }
  function agrees(x, y, tol) { return isFinite(x) && relErr(x, y) <= tol; }

  var SYM = { Sp: 'Σₚ', Su: 'Σᵤ', Sps: 'Σₚ*' };

  /* (a): the expected target as a function of Sp, Sps and Su. */
  function checkA(ast) {
    var f = function (Sp, Sps, Su) { return evaluate(ast, { Sp: Sp, Sps: Sps, Su: Su }); };
    var nAgree = 0, nBad = 0, maxErr = 0, ratios = [], neg = false;
    PTS_A.forEach(function (p) {
      var v = f(p.Sp, p.Sps, p.Su), g = refA(p.Sp, p.Sps, p.Su);
      if (!isFinite(v)) { nBad++; return; }
      if (v < 0) neg = true;
      var e = relErr(v, g);
      maxErr = Math.max(maxErr, e);
      if (e <= TOL) nAgree++;
      ratios.push(v / g);
    });
    var res = { pass: nAgree === NPTS, agree: nAgree, total: NPTS, maxErr: maxErr, notes: [] };
    if (res.pass) return res;
    if (nBad) res.notes.push('It is undefined, a division by zero or similar, at ' + nBad + ' of ' + NPTS + ' points.');
    if (neg) res.notes.push('It is negative at some points, but it is the average of a square.');
    if (!nBad && ratios.length) {
      var r0 = ratios[0];
      if (ratios.every(function (q) { return Math.abs(q / r0 - 1) <= TOL; }) && Math.abs(r0 - 1) > TOL)
        res.notes.push('It is ' + fmtG(r0) + ' times the right average at every point: the dependence on the variances is right and a constant factor is not.');
    }
    if (!res.notes.length || nAgree < NPTS) {
      /* behaviour in the two limits of Sp, and on the line Sp = Sps, at the
       * first six worlds of each regime */
      var ws = PTS_A.slice(0, 6).concat(PTS_A.slice(NPTS / 2, NPTS / 2 + 6));
      function all(pick) { return ws.every(function (p) { var q = pick(p); return agrees(f(q[0], q[1], q[2]), refA(q[0], q[1], q[2]), q[3]); }); }
      var small = all(function (p) { return [p.Su / LIMIT_FAR, p.Sps, p.Su, LIMIT_TOL]; });
      var large = all(function (p) { return [p.Su * LIMIT_FAR, p.Sps, p.Su, LIMIT_TOL]; });
      var diag = all(function (p) { return [p.Sps, p.Sps, p.Su, TOL]; });
      var lim = [];
      if (small && !large) lim.push('It agrees as ' + SYM.Sp + ' → 0 but not at large ' + SYM.Sp + '.');
      else if (large && !small) lim.push('It agrees at large ' + SYM.Sp + ' but not as ' + SYM.Sp + ' → 0.');
      else if (small && large) lim.push('It agrees in both limits, ' + SYM.Sp + ' → 0 and large ' + SYM.Sp + ', but not in between.');
      else lim.push('It agrees neither as ' + SYM.Sp + ' → 0 nor at large ' + SYM.Sp + '.');
      if (diag) lim.push('It agrees wherever ' + SYM.Sp + ' = ' + SYM.Sps + ', and nowhere else that was tested.');
      res.notes = res.notes.concat(lim);
      res.limits = { small: small, large: large, diag: diag };
    }
    return res;
  }

  /* (b): a term in Sp and Su that makes the drift vanish at Sp = Sps. The
   * dimensionless drift at the truth, E[target](Sps)/Sps - 1, is twice the
   * drift of ln Sp there. */
  function driftAtTruth(g, Sps, Su) { return (refA(Sps, Sps, Su) + g(Sps, Su)) / Sps - 1.0; }

  function checkB(ast) {
    var g = function (Sp, Su) { return evaluate(ast, { Sp: Sp, Su: Su }); };
    var nZero = 0, nBad = 0, down = 0, up = 0, zb = 0, za = 0, maxErr = 0;
    PTS_B.forEach(function (p) {
      var e = driftAtTruth(g, p.Sps, p.Su);
      if (!isFinite(e)) { nBad++; return; }
      maxErr = Math.max(maxErr, Math.abs(e));
      if (Math.abs(e) <= TOL) { nZero++; if (p.below) zb++; else za++; }
      else if (e < 0) down++;
      else up++;
    });
    var res = { pass: nZero === NPTS, agree: nZero, total: NPTS, maxErr: maxErr, notes: [] };
    if (res.pass) return res;
    var half = NPTS / 2, miss = NPTS - nZero;
    res.notes.push('The drift does not vanish at the truth in ' + miss + ' of ' + NPTS + ' worlds.');
    if (down || up) {
      var parts = [];
      if (down) parts.push('down in ' + down);
      if (up) parts.push('up in ' + up);
      res.notes.push('Started at the truth, the variance would move ' + parts.join(' and ') + ' of them.');
    }
    if (nBad) res.notes.push('The term is undefined, a division by zero or similar, in ' + nBad + ' worlds.');
    if (zb === half && za < half) res.notes.push('It vanishes in every world with ' + SYM.Sps + ' < 3' + SYM.Su + ' and not in all of the others.');
    if (za === half && zb < half) res.notes.push('It vanishes in every world with ' + SYM.Sps + ' > 3' + SYM.Su + ' and not in all of the others.');
    var ws = PTS_B.slice(0, 6).concat(PTS_B.slice(half, half + 6));
    function lim(scale) {
      return ws.every(function (p) {
        var Sps = p.Su * scale, e = driftAtTruth(g, Sps, p.Su);
        return isFinite(e) && Math.abs(e) <= LIMIT_TOL;
      });
    }
    var narrow = lim(1 / LIMIT_FAR), wide = lim(LIMIT_FAR);
    if (narrow && !wide) res.notes.push('It vanishes in the limit of worlds with ' + SYM.Sps + ' much smaller than ' + SYM.Su + ', but not when ' + SYM.Sps + ' is much larger.');
    else if (wide && !narrow) res.notes.push('It vanishes in the limit of worlds with ' + SYM.Sps + ' much larger than ' + SYM.Su + ', but not when ' + SYM.Sps + ' is much smaller.');
    else if (!wide && !narrow) res.notes.push('It vanishes neither when ' + SYM.Sps + ' is much smaller than ' + SYM.Su + ' nor when it is much larger.');
    res.limits = { narrow: narrow, wide: wide };
    return res;
  }

  /* enough figures to show how far from one a factor near one is */
  function fmtG(x) {
    var k = Math.abs(x - 1) > 0 ? Math.ceil(-Math.log(Math.abs(x - 1)) / Math.LN10) + 2 : 4;
    return Number(x.toPrecision(Math.min(15, Math.max(4, k)))).toString();
  }

  var CORE = {
    mulberry32: mulberry32, normal: normal, FIELD: FIELD, DEAF: DEAF, shrink: shrink, runField: runField,
    NAMES: NAMES, parse: parse, evaluate: evaluate, toTex: toTex,
    TOL: TOL, NPTS: NPTS, PTS_A: PTS_A, PTS_B: PTS_B, refA: refA, refB: refB,
    checkA: checkA, checkB: checkB, driftAtTruth: driftAtTruth
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = CORE; return; }
  if (!root.AIF) return;

  /* ══════════════════════════════════════════════════════════════════════
   * The widget: four steps, each opened by the one before.
   * ═════════════════════════════════════════════════════════════════════ */
  var A = root.AIF, el = A.el, C = A.colours, F = FIELD;
  var NSOUND = F.nights * F.perNight;
  function log10(x) { return Math.log(x) / Math.LN10; }
  function fmt(x, n) { return x.toFixed(n == null ? 3 : n); }
  function sci(x) { var s = x.toExponential(2).split('e'); return s[0] + ' × 10' + sup(s[1].replace('+', '')); }
  function sup(s) {
    var m = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
    return s.split('').map(function (c) { return m[c] || c; }).join('');
  }
  function spStr(x) { return x < 1e-3 ? sci(x) : fmt(x, 4); }

  var MATH_OPTS = {
    delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }],
    throwOnError: false
  };
  /* The page renders its maths at DOMContentLoaded, before widgets mount, so
   * the widget renders its own. */
  function html(tag, cls, s) {
    var n = el(tag, { class: cls || '', html: s });
    if (root.renderMathInElement) root.renderMathInElement(n, MATH_OPTS);
    return n;
  }
  function tex(node, s, display) {
    if (root.katex) root.katex.render(s, node, { throwOnError: false, displayMode: !!display });
    else node.textContent = s;
  }

  var PAD_TOP = { l: 50, r: 12, t: 28, b: 34 };

  /* A legend above the plot area, wrapping onto a second row when narrow. */
  function legend(p, items, x0) {
    var c = p.ctx, x = x0 == null ? p.o.pad.l : x0, y = 11;
    c.save();
    c.font = '10.5px ' + A.fonts.sans;
    items.forEach(function (it) {
      var w = 13 + c.measureText(it.label).width;
      if (x + w > p.w - p.o.pad.r && x > p.o.pad.l) { x = p.o.pad.l; y += 15; }
      c.fillStyle = it.colour; c.fillRect(x, y - 4, 9, 9);
      c.fillStyle = C.ink2; c.textAlign = 'left'; c.textBaseline = 'middle';
      c.fillText(it.label, x + 13, y);
      x += w + 14;
    });
    c.restore();
  }

  /* A taller plot, with room for two legend rows, when the column is narrow. */
  function responsive(plot, wide, narrow) {
    var resize = plot.resize;
    plot.resize = function () {
      var w = this.canvas.clientWidth || (this.canvas.parentNode && this.canvas.parentNode.clientWidth) || 0;
      var o = w && w < 480 ? narrow : wide;
      this.o.aspect = o.aspect; this.o.pad = o.pad;
      resize.call(this);
    };
  }

  A.register('derive-missing-term', function (mount) {
    /* An exercise box is narrower than the page's figure width. */
    var fig = mount.closest ? mount.closest('figure') : null;
    if (fig && fig.closest('.box')) { fig.style.width = 'auto'; fig.style.maxWidth = '100%'; }

    var run1 = runField(null), run2 = null, studentB = null;
    var st = { run: null, shown: 0, reveal: false, timer: null };

    var plotDev = new A.Plot({ aspect: 0.42, xlim: [0, NSOUND], ylim: [-4.2, 4.2], pad: PAD_TOP });
    var plotSp = new A.Plot({ aspect: 0.62, xlim: [0, F.nights], ylim: [-2.5, 0.5], pad: PAD_TOP });
    var plotSh = new A.Plot({ aspect: 0.62, xlim: [0, F.nights], ylim: [0, 1], pad: PAD_TOP });
    var PAD_TWO = { l: 50, r: 12, t: 42, b: 34 };
    responsive(plotDev, { aspect: 0.42, pad: PAD_TOP }, { aspect: 0.8, pad: PAD_TWO });
    responsive(plotSp, { aspect: 0.62, pad: PAD_TOP }, { aspect: 0.7, pad: PAD_TOP });
    responsive(plotSh, { aspect: 0.62, pad: PAD_TOP }, { aspect: 0.7, pad: PAD_TOP });
    var RUN2 = C.violet;

    function nightsShown(r) {
      if (!r || !r.started) return 0;
      return r === st.run ? st.shown : r.nightSp.length;
    }

    plotDev.onDraw(function (p) {
      p.clear().frame();
      p.grid([0], { colour: C.rule2 });
      p.yaxis([-4, -2, 0, 2, 4], function (v) { return String(v).replace('-', '−'); }, { label: 'deviation from dₚ' });
      p.xaxis([0, 1000, 2000, 3000, 4000], null, { label: 'sound' });
      var r = st.run;
      var items = [{ colour: C.faint, label: 'u − dₚ heard' }, { colour: r === run2 && r ? RUN2 : C.accent, label: 'd̂ − dₚ estimate' }];
      if (st.reveal) items.push({ colour: C.clay, label: 'd − dₚ true' });
      legend(p, items);
      if (!r) return;
      var n = st.shown * F.perNight, c = p.ctx;
      function dots(arr, colour, rad) {
        c.save(); c.fillStyle = colour;
        for (var i = 0; i < n; i++) {
          var y = Math.max(-4.2, Math.min(4.2, arr[i] - F.dp));
          c.fillRect(p.px(i) - rad / 2, p.py(y) - rad / 2, rad, rad);
        }
        c.restore();
      }
      dots(r.u, 'rgba(109,109,117,0.30)', 2);
      if (st.reveal) dots(r.d, 'rgba(156,77,47,0.50)', 2);
      dots(r.dhat, r === run2 ? RUN2 : C.accent, 2.2);
    });

    function lo() {
      var m = 1;
      [run1, run2].forEach(function (r) {
        var k = nightsShown(r);
        for (var i = 0; i < k; i++) m = Math.min(m, r.nightSp[i]);
      });
      return Math.min(-2.5, Math.floor(log10(m)) - 0.5);
    }

    plotSp.onDraw(function (p) {
      var bottom = lo();
      p.o.ylim = [bottom, 0.5];
      p.clear().frame();
      var ticks = [];
      for (var k = Math.ceil(bottom); k <= 0; k++) ticks.push(k);
      if (ticks.length > 6) ticks = ticks.filter(function (t) { return t % 2 === 0; });
      p.grid(ticks);
      p.yaxis(ticks, function (v) { return v === 0 ? '1' : '10' + sup(String(v)); }, { label: 'learned Σₚ, log scale' });
      p.xaxis([0, 50, 100, 150, 200], null, { label: 'night' });
      if (st.reveal) {
        p.line([[0, log10(F.S)], [F.nights, log10(F.S)]], { colour: C.clay, dash: [5, 4], width: 1.5 });
        p.text(F.nights - 2, log10(F.S) + 0.22, 'Σₚ* of the world', { align: 'right', colour: C.clay, size: 10.5 });
      }
      if (nightsShown(run2)) legend(p, [{ colour: C.accent, label: 'point rule' }, { colour: RUN2, label: 'with your c' }]);
      [[run1, C.accent], [run2, RUN2]].forEach(function (rc) {
        var r = rc[0], k = nightsShown(r);
        if (!r || !k) return;
        var pts = [[0, log10(F.Sp0)]];
        for (var i = 0; i < k; i++) pts.push([i + 1, log10(r.nightSp[i])]);
        p.line(pts, { colour: rc[1] });
      });
    });

    plotSh.onDraw(function (p) {
      p.clear().frame();
      p.grid([0.25, 0.5, 0.75, 1]);
      p.yaxis([0, 0.25, 0.5, 0.75, 1], function (v) { return v.toFixed(2); }, { label: 'shrinkage factor' });
      p.xaxis([0, 50, 100, 150, 200], null, { label: 'night' });
      p.rect(p.px(0), p.py(DEAF), p.px(F.nights) - p.px(0), p.py(0) - p.py(DEAF), { fill: C.clayTint });
      p.text(F.nights - 2, DEAF / 2, 'deaf', { align: 'right', colour: C.clay, size: 10.5 });
      if (st.reveal) {
        var k = shrink(F.S, F.Su);
        p.line([[0, k], [F.nights, k]], { colour: C.clay, dash: [5, 4], width: 1.5 });
        p.text(F.nights - 2, k + 0.07, 'the world’s value', { align: 'right', colour: C.clay, size: 10.5 });
      }
      [[run1, C.accent], [run2, RUN2]].forEach(function (rc) {
        var r = rc[0], n = nightsShown(r);
        if (!r || !n) return;
        var pts = [[0, shrink(F.Sp0, F.Su)]];
        for (var i = 0; i < n; i++) pts.push([i + 1, r.nightShrink[i]]);
        p.line(pts, { colour: rc[1] });
      });
    });

    function renderAll() { plotDev.render(); plotSp.render(); plotSh.render(); }

    function status(r, n) {
      var sp = n ? r.nightSp[n - 1] : F.Sp0, dead = r.died && n >= r.died;
      return [['night', n + ' of ' + F.nights], ['learned Σₚ', spStr(sp)],
              ['shrinkage factor', fmt(shrink(sp, F.Su), 3)],
              ['state', dead ? 'deaf' : n === F.nights ? 'still hearing at the end' : 'hearing']];
    }

    /* Reveal a run two nights per tick. setTimeout rather than
     * requestAnimationFrame, so a hidden tab still finishes. */
    function play(r, ro, done) {
      clearTimeout(st.timer);
      st.run = r; st.shown = 0; r.started = true;
      var total = r.nightSp.length;
      function tick() {
        st.shown = Math.min(total, st.shown + 2);
        renderAll(); ro.show(status(r, st.shown));
        if (st.shown < total) st.timer = setTimeout(tick, 25);
        else { st.tick = null; done(); }
      }
      st.tick = tick;
      tick();
    }
    /* Jump to the last night; the tick it calls runs the done handler. */
    function skip() {
      if (!st.tick) return;
      clearTimeout(st.timer);
      st.shown = st.run.nightSp.length - 1;
      st.tick();
    }

    /* ── layout ── */
    function step(num, title) {
      var box = el('div', { class: 'dstep locked' });
      box.appendChild(el('div', { class: 'dhd', html: '<span class="dno">' + num + '</span>' + title }));
      var body = el('div', { class: 'dbody' });
      box.appendChild(body);
      return { box: box, body: body, open: function () { box.classList.remove('locked'); } };
    }

    var style = el('style', { text:
      '.dstep{border-top:1px solid var(--rule);padding-top:.7rem}' +
      '.dstep.locked{opacity:.45}.dstep.locked .dbody{display:none}' +
      '.dhd{font-family:var(--sans);font-size:.78rem;font-weight:600;color:var(--ink2);letter-spacing:.03em;margin-bottom:.5rem}' +
      '.dno{display:inline-block;width:1.35rem;height:1.35rem;line-height:1.35rem;text-align:center;border-radius:50%;' +
      'background:var(--accent);color:#fff;font-size:.72rem;margin-right:.5rem}' +
      '.dstep.locked .dno{background:var(--faint)}' +
      '.dplots2{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}' +
      '@media (max-width:640px){.dplots2{grid-template-columns:1fr}}' +
      '.dtext{font-family:var(--sans);font-size:.82rem;line-height:1.55;color:var(--ink2);margin:.3rem 0 .6rem}' +
      '.dfacts{font-family:var(--sans);font-size:.82rem;color:var(--ink2);border-collapse:collapse;margin:.2rem 0 .4rem}' +
      '.dfacts td{padding:.18rem .9rem .18rem 0;vertical-align:baseline}' +
      '.dfacts td:last-child{font-family:var(--mono);font-variant-numeric:tabular-nums}' +
      '.dpart{margin:.6rem 0 1rem}' +
      '.dinput{width:100%;font-family:var(--mono);font-size:.9rem;padding:.4rem .5rem;border:1px solid var(--rule2);' +
      'border-radius:2px;background:var(--card);color:var(--ink)}' +
      '.dinput:focus{outline:2px solid var(--accent-tint);border-color:var(--accent)}' +
      '.dnames{font-family:var(--sans);font-size:.76rem;color:var(--muted);margin:.25rem 0}' +
      '.dpreview{min-height:2.4rem;overflow-x:auto;overflow-y:hidden;padding:.25rem 0;color:var(--ink)}' +
      '.dpreview .katex-display{margin:.2rem 0}' +
      '.dperr{font-family:var(--sans);font-size:.8rem;color:var(--clay)}' +
      '.dfeed{font-family:var(--sans);font-size:.82rem;line-height:1.5;padding:.5rem .7rem;border-radius:2px;margin-top:.5rem}' +
      '.dfeed.ok{background:var(--accent-tint);color:var(--accent-deep, #0a423d)}' +
      '.dfeed.bad{background:var(--clay-tint, #f9efe9);color:var(--ink2)}' +
      '.dfeed ul{margin:.3rem 0 0 1.1rem;padding:0}.dfeed li{font-size:.82rem;margin:.1rem 0}'
    });

    var s1 = step(1, 'Run'), s2 = step(2, 'The hidden world'), s3 = step(3, 'Derive'), s4 = step(4, 'Fix');
    s1.open();

    /* step 1 */
    var ro1 = A.readout();
    var bRun = A.button('Run the creature', function () { start1(); });
    var bSkip1 = A.button('Skip to the end', function () { skip(ro1); }, true);
    var plots = el('div', {});
    var devHolder = el('div', {}), spHolder = el('div', {}), shHolder = el('div', {});
    plots.appendChild(devHolder);
    plots.appendChild(el('div', { class: 'dplots2' }, [spHolder, shHolder]));
    var end1 = el('p', { class: 'dtext' });
    s1.body.appendChild(html('p', 'dtext',
      'The creature knows $d_p$ and $\\Sigma_u = ' + F.Su + '$, starts at $\\Sigma_p = ' + F.Sp0 +
      '$ and learns $\\Sigma_p$ by the point rule with $\\beta = ' + F.beta + '$, for up to ' + F.nights +
      ' nights of ' + F.perNight + ' sounds. You see what it sees: each sound, its estimate, its learned $\\Sigma_p$ and its shrinkage factor ' +
      '$\\Sigma_p/(\\Sigma_p + \\Sigma_u)$. Below ' + DEAF + ' it is counted deaf.'));
    s1.body.appendChild(el('div', { class: 'wbtns' }, [bRun, bSkip1]));
    s1.body.appendChild(ro1);
    s1.body.appendChild(end1);
    ro1.show(status(run1, 0));

    function start1() {
      bRun.disabled = true;
      run2 = null;
      play(run1, ro1, function () {
        bRun.disabled = false; bRun.textContent = 'Run again';
        end1.textContent = run1.died
          ? 'The creature went deaf on night ' + run1.died + '.'
          : 'The creature was still hearing after ' + F.nights + ' nights.';
        s2.open();
      });
    }

    /* step 2 */
    var facts = el('table', { class: 'dfacts' });
    var bReveal = A.button('Show the world', function () { reveal(); });
    s2.body.appendChild(html('p', 'dtext', 'The world the creature could not see: the true distances, now drawn in clay above, and the variance $\\Sigma_p^{*}$ they were drawn with.'));
    s2.body.appendChild(el('div', { class: 'wbtns' }, [bReveal]));
    s2.body.appendChild(facts);

    function factRow(label, value) {
      var tr = el('tr', {});
      tr.appendChild(html('td', '', label));
      tr.appendChild(el('td', { text: value }));
      facts.appendChild(tr);
    }
    function reveal() {
      if (st.reveal) return;
      st.reveal = true;
      bReveal.style.display = 'none';
      var last = run1.nightSp[run1.nightSp.length - 1];
      factRow('the world’s prior variance, $\\Sigma_p^{*}$', fmt(F.S, 3));
      factRow('the creature’s learned $\\Sigma_p$ at the end', spStr(last));
      factRow('shrinkage factor the world warrants, $\\Sigma_p^{*}/(\\Sigma_p^{*} + \\Sigma_u)$', fmt(shrink(F.S, F.Su), 3));
      factRow('the creature’s shrinkage factor at the end', fmt(shrink(last, F.Su), 3));
      renderAll();
      s3.open();
    }

    /* step 3 */
    function answerBox(label, names, check, onPass) {
      var wrap = el('div', { class: 'dpart' });
      var inp = el('input', { class: 'dinput', type: 'text', spellcheck: 'false', autocomplete: 'off',
                             autocapitalize: 'off', 'aria-label': label.replace(/<[^>]+>|\$/g, '') });
      var nm = html('div', 'dnames', 'Names: ' + names.map(function (n) {
        return '<code>' + n + '</code> for $' + NAMES[n].tex + '$';
      }).join(', ') + '. Use <code>+ - * / ^</code> and parentheses.');
      var prev = el('div', { class: 'dpreview' });
      var perr = el('div', { class: 'dperr' });
      var bCheck = A.button('Check', function () { doCheck(); });
      var feed = el('div', { class: 'dfeed', hidden: '' });
      var parsed = null;
      wrap.appendChild(html('p', 'dtext', label));
      wrap.appendChild(inp); wrap.appendChild(nm); wrap.appendChild(prev); wrap.appendChild(perr);
      wrap.appendChild(el('div', { class: 'wbtns' }, [bCheck]));
      wrap.appendChild(feed);
      function update() {
        parsed = parse(inp.value, names);
        feed.hidden = true;
        if (parsed.ok) { tex(prev, parsed.tex, true); perr.textContent = ''; }
        else {
          prev.textContent = '';
          perr.textContent = parsed.empty ? '' : parsed.error +
            (parsed.pos != null ? ' (at character ' + (parsed.pos + 1) + ')' : '');
        }
        bCheck.disabled = !parsed.ok;
      }
      function doCheck() {
        if (!parsed || !parsed.ok) return;
        var res = check(parsed.ast);
        feed.hidden = false;
        feed.innerHTML = '';
        if (res.pass) {
          feed.className = 'dfeed ok';
          feed.appendChild(el('div', { text: 'Correct: it agrees at all ' + res.total +
                                             ' hidden points to a relative tolerance of 10⁻⁹.' }));
          onPass(parsed.ast);
        } else {
          feed.className = 'dfeed bad';
          feed.appendChild(el('div', { text: 'Not yet. It agrees at ' + res.agree + ' of ' + res.total + ' hidden points.' }));
          var ul = el('ul', {});
          res.notes.forEach(function (t) { ul.appendChild(el('li', { text: t })); });
          if (res.notes.length) feed.appendChild(ul);
        }
        return res;
      }
      inp.addEventListener('input', update);
      inp.addEventListener('keydown', function (e) { if (e.key === 'Enter') doCheck(); });
      update();
      return { el: wrap, input: inp, check: doCheck, set: function (s) { inp.value = s; update(); } };
    }

    var partB = null;
    var partA = answerBox('(a) The average over the world of the squared deviation of the mode, $\\mathbb E_{P^*}\\!\\left[(\\hat d - d_p)^2\\right]$, ' +
      'as a function of $\\Sigma_p$, $\\Sigma_p^{*}$ and $\\Sigma_u$.', ['Sp', 'Sps', 'Su'], checkA, function () {
        partB.el.style.display = '';
      });
    partB = answerBox('(b) The term $c(\\Sigma_p, \\Sigma_u)$ which, added to $(\\hat d - d_p)^2$ in the step, ' +
      'makes the average of the sum equal $\\Sigma_p$ at $\\Sigma_p = \\Sigma_p^{*}$ in every world.', ['Sp', 'Su'], checkB, function (ast) {
        studentB = ast;
        s4.open();
      });
    partB.el.style.display = 'none';
    s3.body.appendChild(html('p', 'dtext', 'Each answer is evaluated at ' + NPTS + ' hidden values of the parameters, half with ' +
      '$\\Sigma_p^{*} < 3\\Sigma_u$ and half with $\\Sigma_p^{*} > 3\\Sigma_u$. Part (b) opens when (a) is right.'));
    s3.body.appendChild(partA.el);
    s3.body.appendChild(partB.el);

    /* step 4 */
    var ro2 = A.readout();
    var end2 = el('p', { class: 'dtext' });
    var bRun2 = A.button('Run again with your term', function () { start2(); });
    var bSkip2 = A.button('Skip to the end', function () { skip(ro2); }, true);
    s4.body.appendChild(html('p', 'dtext', 'The same world and the same sounds, with your $c$ from (b) added to $(\\hat d - d_p)^2$ on every sound, evaluated at the creature’s current $\\Sigma_p$. The new run is drawn in violet.'));
    s4.body.appendChild(el('div', { class: 'wbtns' }, [bRun2, bSkip2]));
    s4.body.appendChild(ro2);
    s4.body.appendChild(end2);

    function start2() {
      if (!studentB) return;
      var ast = studentB;
      run2 = runField(function (Sp, Su) { return evaluate(ast, { Sp: Sp, Su: Su }); });
      bRun2.disabled = true;
      play(run2, ro2, function () {
        bRun2.disabled = false;
        var last = run2.nightSp[run2.nightSp.length - 1];
        end2.innerHTML = '';
        end2.appendChild(document.createTextNode(run2.died
          ? 'The creature went deaf on night ' + run2.died + '.' :
          'After ' + F.nights + ' nights the learned Σₚ is ' + fmt(last, 3) + ' against the world’s ' + fmt(F.S, 3) +
          ', and the shrinkage factor ' + fmt(shrink(last, F.Su), 3) + ' against ' + fmt(shrink(F.S, F.Su), 3) + '.'));
      });
    }

    mount.appendChild(style);
    mount.appendChild(A.panel([s1.box, plots, s2.box, s3.box, s4.box]));
    plotDev.attach(devHolder); plotSp.attach(spHolder); plotSh.attach(shHolder);

    /* a hook for the headless check: the same handlers the buttons call */
    return { start1: start1, skip1: function () { skip(ro1); }, reveal: reveal, partA: partA, partB: partB,
             start2: start2, skip2: function () { skip(ro2); }, state: st,
             end1: end1, end2: end2, facts: facts, run1: run1, run2: function () { return run2; } };
  });
})(this);
