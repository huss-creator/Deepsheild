/*
 * DeepShield face viewer: draws the REAL uploaded photo on a canvas and layers on top of it
 *   - the detected face box (corner brackets) and the five YuNet landmarks when available
 *   - a "scanner" mask built from the face's own pixels: mosaic dissolve, halftone dots and a
 *     feature mesh whose nodes sit on the face's real edges (eyes, brows, nose, mouth, jaw)
 *   - on the result page: combined Grad-CAM heat map and the numbered suspicious regions
 * No library, no network. Nothing here decides a verdict; it only visualises data the server
 * already computed. Respects prefers-reduced-motion (draws one static frame).
 *
 * Usage:
 *   var fm = new DSFaceMask(containerEl, {src, width, height, faces, mode: 'processing' | 'result'});
 *   fm.setFaces(faces, {width, height});   fm.setStage('AI_ANALYSIS');   fm.setLayers({box:true, ...});
 */
(function () {
  'use strict';

  var ACCENT = [99, 102, 241];   // indigo  (--accent)
  var CYAN = [6, 182, 212];      // cyan    (--accent-2)
  var DANGER = [239, 68, 68];
  var TAU = Math.PI * 2;
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function ease(t) { t = clamp(t, 0, 1); return 1 - Math.pow(1 - t, 3); }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t].map(Math.round); }
  function rand(seed) { var x = Math.sin(seed * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }

  function DSFaceMask(root, opts) {
    this.root = root;
    this.mode = opts.mode || 'result';
    this.imgInfo = { width: opts.width || 0, height: opts.height || 0 };
    this.faces = [];
    this.stage = 'VALIDATING';
    this.progress = 0;
    this.active = 0;
    this.label = '';
    this.layers = { box: true, mask: false, regions: false, heat: false, landmarks: true };
    this.t0 = performance.now();
    this.maskFade = this.mode === 'result' ? 1 : 0;          // result page: mask dissolves away on load
    this.maskFadeStart = this.t0;
    this.img = new Image();
    this.ready = false;
    this.failed = false;
    this.cache = {};
    this.canvas = document.createElement('canvas');
    this.canvas.className = 'fm-canvas';
    this.canvas.setAttribute('role', 'img');
    this.canvas.setAttribute('aria-label', opts.label || 'Uploaded photo with the detected face highlighted');
    root.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.scratch = document.createElement('canvas');
    this.sctx = this.scratch.getContext('2d', { willReadFrequently: true });
    var self = this;
    this.img.onload = function () {
      self.ready = true;
      if (!self.imgInfo.width) { self.imgInfo.width = self.img.naturalWidth; }
      if (!self.imgInfo.height) { self.imgInfo.height = self.img.naturalHeight; }
      self.layout();
      self.start();
    };
    this.img.onerror = function () { self.failed = true; self.layout(); self.start(); };
    this.img.src = opts.src;
    if (opts.faces) { this.setFaces(opts.faces, this.imgInfo); }
    this._ro = null;
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(function () { self.layout(); self.draw(performance.now()); });
      this._ro.observe(root);
    }
  }

  var P = DSFaceMask.prototype;

  // ---------------------------------------------------------------- public API
  P.setFaces = function (faces, info) {
    var now = performance.now();
    if (info && info.width) { this.imgInfo = { width: info.width, height: info.height }; }
    var have = this.faces.length;
    this.faces = (faces || []).map(function (f, i) {
      return Object.assign({}, f, { lockAt: i < have && this.faces[i] ? this.faces[i].lockAt : now });
    }, this);
    this.cache = {};
    if (this.ready) { this.layout(); }
  };
  P.setStage = function (stage, progress) {
    this.stage = stage;
    if (typeof progress === 'number') { this.progress = progress; }
  };
  P.setLabel = function (text) { this.label = text || ''; };
  P.setLayers = function (l) { Object.assign(this.layers, l); this.draw(performance.now()); };
  P.setActive = function (i) { this.active = i; this.draw(performance.now()); };
  P.stop = function () { this.running = false; if (this._ro) { this._ro.disconnect(); } };
  P.replayUnmask = function () { this.maskFade = 1; this.maskFadeStart = performance.now(); this.start(); };

  /* Regions for the legend: [{n, label, x, y}] in crop-fraction terms for the active face. */
  P.getRegions = function (faceIndex) {
    var f = this.faces[faceIndex == null ? this.active : faceIndex];
    return f ? this._regions(f) : [];
  };

  // ---------------------------------------------------------------- layout
  P.layout = function () {
    var w = this.root.clientWidth || 400;
    var iw = this.imgInfo.width || 4, ih = this.imgInfo.height || 3;
    var h = Math.round(w * ih / iw);
    var maxH = Math.round(Math.max(360, window.innerHeight * 0.72));
    if (h > maxH) { h = maxH; w = Math.round(h * iw / ih); }
    this.cssW = w; this.cssH = h;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.dpr = dpr;
    this.s = w / iw;                                           // image px -> css px
    this.cache = {};
  };

  P.start = function () {
    if (this.running) { return; }
    this.running = true;
    var self = this;
    if (reduce) { this.maskFade = 0; this.draw(performance.now()); this.running = false; return; }
    var last = 0, FRAME = 1000 / 12;           // ~12 fps is plenty and keeps the page responsive
    (function loop(t) {
      if (!self.running) { return; }
      if (t - last >= FRAME) {
        last = t;
        self.draw(t);
        if (self.mode !== 'processing' && t - self.maskFadeStart > 1700) {
          self.running = false; self.draw(t); return;   // result page: stop after the fade
        }
      }
      requestAnimationFrame(loop);
    })(performance.now());
  };

  // ---------------------------------------------------------------- per-face precomputation
  P._faceData = function (f, idx) {
    var key = 'f' + idx;
    if (this.cache[key]) { return this.cache[key]; }
    var b = f.bbox, d = { pts: [], edges: [], lum: null };
    var pad = 0.06;
    var x = b[0] - b[2] * pad, y = b[1] - b[3] * pad, w = b[2] * (1 + 2 * pad), h = b[3] * (1 + 2 * pad);
    d.rect = [x, y, w, h];
    // sample the face on a small grid: luminance for the halftone + gradients for mesh nodes
    var gw = 48, gh = Math.max(8, Math.round(48 * h / w));
    d.gw = gw; d.gh = gh;
    var lum = new Float32Array(gw * gh), mag = new Float32Array(gw * gh);
    var ok = false;
    try {
      this.scratch.width = gw; this.scratch.height = gh;
      this.sctx.drawImage(this.img, x, y, w, h, 0, 0, gw, gh);
      var px = this.sctx.getImageData(0, 0, gw, gh).data;
      for (var i = 0; i < gw * gh; i++) { lum[i] = (0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2]) / 255; }
      ok = true;
    } catch (e) { /* tainted canvas (cross-origin storage): fall back to a regular pattern */ }
    if (ok) {
      for (var yy = 1; yy < gh - 1; yy++) {
        for (var xx = 1; xx < gw - 1; xx++) {
          var gx = lum[yy * gw + xx + 1] - lum[yy * gw + xx - 1], gy = lum[(yy + 1) * gw + xx] - lum[(yy - 1) * gw + xx];
          mag[yy * gw + xx] = Math.sqrt(gx * gx + gy * gy);
        }
      }
    }
    d.lum = lum; d.ok = ok;
    // feature nodes: strongest local edges, spaced apart (these land on eyes, brows, nose, lips, jaw)
    var cand = [];
    for (var j = 0; j < gw * gh; j++) {
      var cx = j % gw, cy = (j / gw) | 0, nx = (cx + 0.5) / gw - 0.5, ny = (cy + 0.5) / gh - 0.5;
      if (nx * nx / 0.2 + ny * ny / 0.25 > 1) { continue; }       // keep inside the face ellipse
      cand.push([ok ? mag[j] : rand(j), cx, cy]);
    }
    cand.sort(function (a, b) { return b[0] - a[0]; });
    var chosen = [], minD = ok ? 4.2 : 6;
    for (var c = 0; c < cand.length && chosen.length < 64; c++) {
      var near = false;
      for (var k = 0; k < chosen.length; k++) {
        var dx = chosen[k][0] - cand[c][1], dy = chosen[k][1] - cand[c][2];
        if (dx * dx + dy * dy < minD * minD) { near = true; break; }
      }
      if (!near) { chosen.push([cand[c][1], cand[c][2]]); }
    }
    // outline nodes on the face ellipse
    for (var o = 0; o < 18; o++) {
      var an = o / 18 * TAU;
      chosen.push([gw * (0.5 + 0.47 * Math.cos(an)), gh * (0.5 + 0.49 * Math.sin(an))]);
    }
    d.pts = chosen.map(function (p) { return [p[0] / gw, p[1] / gh]; });           // 0..1 in the face rect
    var pts = d.pts, seen = {};
    for (var a = 0; a < pts.length; a++) {
      var ds = [];
      for (var q = 0; q < pts.length; q++) {
        if (q !== a) { ds.push([(pts[a][0] - pts[q][0]) * (pts[a][0] - pts[q][0]) + (pts[a][1] - pts[q][1]) * (pts[a][1] - pts[q][1]) * (h / w) * (h / w), q]); }
      }
      ds.sort(function (m, n) { return m[0] - n[0]; });
      for (var e = 0; e < 3; e++) {
        var lo = Math.min(a, ds[e][1]), hi = Math.max(a, ds[e][1]);
        if (!seen[lo + '_' + hi]) { seen[lo + '_' + hi] = 1; d.edges.push([lo, hi]); }
      }
    }
    this.cache[key] = d;
    return d;
  };

  // ---------------------------------------------------------------- drawing
  P.draw = function (t) {
    var ctx = this.ctx;
    if (!this.cssW) { return; }
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    ctx.fillStyle = '#0b1120';
    ctx.fillRect(0, 0, this.cssW, this.cssH);
    if (this.failed || !this.ready) {
      ctx.fillStyle = '#94A3B8'; ctx.font = '14px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText(this.failed ? 'Image could not be loaded' : 'Loading image…', this.cssW / 2, this.cssH / 2);
      return;
    }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this.img, 0, 0, this.cssW, this.cssH);
    var sec = (t - this.t0) / 1000;
    var proc = this.mode === 'processing';
    if (proc) {
      ctx.fillStyle = 'rgba(5,8,20,0.32)';
      ctx.fillRect(0, 0, this.cssW, this.cssH);
      if (!this.faces.length) { this._searching(ctx, sec); }
    }
    for (var i = 0; i < this.faces.length; i++) {
      var f = this.faces[i], isActive = proc || i === this.active;
      var lock = ease((t - f.lockAt) / 700);
      var d = this._faceData(f, i);
      if (proc) {
        this._scanMask(ctx, f, d, sec, i);
      } else {
        if (this.layers.heat && isActive) { this._heat(ctx, f, i); }
        var fadeT = 1 - ease((t - this.maskFadeStart) / 1500);
        var maskAmt = this.layers.mask && isActive ? 1 : (isActive ? fadeT : 0);
        if (maskAmt > 0.01) { this._mosaic(ctx, f, d, this.layers.mask ? 7 : 4 + 16 * fadeT, maskAmt, 1); this._dots(ctx, f, d, maskAmt * 0.8, sec); this._mesh(ctx, f, d, sec, maskAmt * 0.8, false); }
        if (this.layers.regions && isActive) { this._regionsDraw(ctx, f, i, sec); }
      }
      if (this.layers.box || proc) { this._box(ctx, f, i, lock, isActive, sec); }
      if (this.layers.landmarks && f.landmarks && (proc || isActive)) { this._landmarks(ctx, f, lock, sec); }
    }
    if (proc) { this._hud(ctx, sec); }
  };

  P._searching = function (ctx, sec) {
    var W = this.cssW, H = this.cssH, step = 22;
    ctx.fillStyle = rgba(CYAN, 0.16);
    for (var y = step / 2; y < H; y += step) {
      for (var x = step / 2; x < W; x += step) {
        var tw = 0.5 + 0.5 * Math.sin(sec * 2 + x * 0.05 + y * 0.03);
        ctx.globalAlpha = 0.25 + 0.55 * tw; ctx.fillRect(x - 1, y - 1, 2, 2);
      }
    }
    ctx.globalAlpha = 1;
    var by = ((sec * 0.35) % 1) * H;
    var g = ctx.createLinearGradient(0, by - 60, 0, by);
    g.addColorStop(0, rgba(CYAN, 0)); g.addColorStop(1, rgba(CYAN, 0.35));
    ctx.fillStyle = g; ctx.fillRect(0, by - 60, W, 60);
    ctx.fillStyle = rgba(CYAN, 0.9); ctx.fillRect(0, by, W, 1.5);
  };

  // rectangle in css px for a face's padded region
  P._r = function (d) { var s = this.s; return [d.rect[0] * s, d.rect[1] * s, d.rect[2] * s, d.rect[3] * s]; };
  P._ellipsePath = function (ctx, r) {
    ctx.beginPath(); ctx.ellipse(r[0] + r[2] / 2, r[1] + r[3] / 2, r[2] * 0.48, r[3] * 0.5, 0, 0, TAU);
  };

  // pixelated copy of the face region, clipped to the face ellipse
  P._mosaic = function (ctx, f, d, block, alpha, coverage, beamFrac) {
    var r = this._r(d);
    var bs = Math.max(2, block), sw = Math.max(2, Math.round(r[2] / bs)), sh = Math.max(2, Math.round(r[3] / bs));
    this.scratch.width = sw; this.scratch.height = sh;
    this.sctx.imageSmoothingEnabled = true;
    this.sctx.drawImage(this.img, d.rect[0], d.rect[1], d.rect[2], d.rect[3], 0, 0, sw, sh);
    ctx.save();
    this._ellipsePath(ctx, r); ctx.clip();
    if (beamFrac != null) { ctx.beginPath(); ctx.rect(r[0] - 2, r[1] - 2, r[2] + 4, (r[3] + 4) * beamFrac); ctx.clip(); this._ellipsePath(ctx, r); ctx.clip(); }
    ctx.imageSmoothingEnabled = false;
    ctx.globalAlpha = alpha * coverage;
    ctx.drawImage(this.scratch, 0, 0, sw, sh, r[0], r[1], r[2], r[3]);
    // colour grade (indigo to cyan) so the mask reads as an overlay rather than a blur
    var g = ctx.createLinearGradient(r[0], r[1], r[0] + r[2], r[1] + r[3]);
    g.addColorStop(0, rgba(ACCENT, 0.42)); g.addColorStop(1, rgba(CYAN, 0.34));
    ctx.globalCompositeOperation = 'overlay'; ctx.globalAlpha = alpha * coverage;
    ctx.fillStyle = g; ctx.fillRect(r[0], r[1], r[2], r[3]);
    ctx.restore();
  };

  // halftone dots: radius follows the real luminance of the face
  P._dots = function (ctx, f, d, alpha, sec) {
    if (!d.lum) { return; }
    var r = this._r(d), cw = r[2] / d.gw, chh = r[3] / d.gh;
    ctx.save(); this._ellipsePath(ctx, r); ctx.clip();
    var st = d.gw * d.gh > 600 ? 2 : 1;   // thin out the grid on big faces
    for (var y = 0; y < d.gh; y += st) {
      for (var x = 0; x < d.gw; x += st) {
        var L = d.lum[y * d.gw + x], rad = (0.15 + 0.5 * (1 - L)) * Math.min(cw, chh) * 0.55;
        var tone = mix(CYAN, ACCENT, (x / d.gw * 0.6 + y / d.gh * 0.4));
        var tw = 0.8;
        ctx.fillStyle = rgba(tone, alpha * (0.35 + 0.5 * tw));
        ctx.beginPath(); ctx.arc(r[0] + (x + 0.5) * cw, r[1] + (y + 0.5) * chh, Math.max(0.4, rad), 0, TAU); ctx.fill();
      }
    }
    ctx.restore();
  };

  // wireframe through real edge points
  P._mesh = function (ctx, f, d, sec, alpha, pulses) {
    var r = this._r(d), p = d.pts;
    ctx.save();
    ctx.lineWidth = 1; ctx.strokeStyle = rgba(CYAN, 0.38 * alpha);
    ctx.beginPath();
    for (var i = 0; i < d.edges.length; i++) {
      var a = p[d.edges[i][0]], b = p[d.edges[i][1]];
      ctx.moveTo(r[0] + a[0] * r[2], r[1] + a[1] * r[3]); ctx.lineTo(r[0] + b[0] * r[2], r[1] + b[1] * r[3]);
    }
    ctx.stroke();
    ctx.fillStyle = rgba([180, 245, 255], 0.9 * alpha);
    for (var n = 0; n < p.length; n++) {
      ctx.beginPath(); ctx.arc(r[0] + p[n][0] * r[2], r[1] + p[n][1] * r[3], 1.7, 0, TAU); ctx.fill();
    }
    if (pulses) {                                                // light travelling along random edges
      ctx.lineWidth = 2; ctx.strokeStyle = rgba([255, 255, 255], 0.85 * alpha);
      for (var k = 0; k < 7; k++) {
        var e = d.edges[Math.floor(rand(k * 13.7 + Math.floor(sec * 1.5 + k)) * d.edges.length)];
        var A = p[e[0]], B = p[e[1]], tt = (sec * 1.5 + k * 0.37) % 1;
        var x1 = r[0] + A[0] * r[2], y1 = r[1] + A[1] * r[3], x2 = r[0] + B[0] * r[2], y2 = r[1] + B[1] * r[3];
        ctx.beginPath(); ctx.moveTo(x1 + (x2 - x1) * Math.max(0, tt - 0.25), y1 + (y2 - y1) * Math.max(0, tt - 0.25));
        ctx.lineTo(x1 + (x2 - x1) * tt, y1 + (y2 - y1) * tt); ctx.stroke();
      }
    }
    ctx.restore();
  };

  // the processing-page effect: a scan beam converts the face into mosaic + dots + mesh
  P._scanMask = function (ctx, f, d, sec, idx) {
    var r = this._r(d), st = this.stage;
    var running = st === 'AI_ANALYSIS', forensic = st === 'FORENSIC_ANALYSIS' || st === 'AGGREGATING' || st === 'REPORT_GENERATION' || st === 'COMPLETED';
    var period = running ? 2.2 : 3.2;
    var phase = reduce ? 0.6 : ((sec / period) % 1), tri = phase < 0.5 ? phase * 2 : 2 - phase * 2;   // 0..1..0
    var block = forensic ? 5 + 2 * Math.sin(sec * 2) : 9 + 7 * (1 - tri) + 2 * Math.sin(sec * 3);
    var lockedFor = performance.now() - f.lockAt;
    var cover = ease(lockedFor / 1200);
    this._mosaic(ctx, f, d, block, 0.95, cover, forensic ? 1 : 0.18 + 0.82 * tri);
    this._dots(ctx, f, d, 0.75 * cover, sec);
    this._mesh(ctx, f, d, sec, cover, true);
    // random glitch cells above the beam
    ctx.save(); this._ellipsePath(ctx, r); ctx.clip();
    for (var g = 0; g < 14; g++) {
      var seed = g * 7.3 + Math.floor(sec * 6);
      var gx = r[0] + rand(seed) * r[2], gy = r[1] + rand(seed + 1.1) * r[3] * (forensic ? 1 : tri);
      var sz = 5 + rand(seed + 2.2) * 12;
      ctx.fillStyle = rgba(rand(seed + 3.3) > 0.5 ? CYAN : ACCENT, 0.55);
      ctx.fillRect(gx, gy, sz, sz * (0.5 + rand(seed + 4.4)));
    }
    if (forensic) {                                               // forensic phase: spectrum rings + noise grain
      var cx = r[0] + r[2] / 2, cy = r[1] + r[3] / 2;
      ctx.lineWidth = 1;
      for (var q = 1; q <= 4; q++) {
        ctx.strokeStyle = rgba(CYAN, 0.35 - q * 0.05);
        ctx.beginPath(); ctx.ellipse(cx, cy, r[2] * 0.12 * q + 4 * Math.sin(sec * 2 + q), r[3] * 0.12 * q, 0, 0, TAU); ctx.stroke();
      }
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      for (var n = 0; n < 60; n++) { ctx.fillRect(r[0] + rand(n + Math.floor(sec * 12)) * r[2], r[1] + rand(n * 1.7 + Math.floor(sec * 12)) * r[3], 1.5, 1.5); }
    }
    ctx.restore();
    // the beam itself
    if (!forensic) {
      var by = r[1] + r[3] * tri;
      var gr = ctx.createLinearGradient(0, by - 26, 0, by + 2);
      gr.addColorStop(0, rgba(CYAN, 0)); gr.addColorStop(1, rgba(CYAN, 0.5));
      ctx.save(); ctx.fillStyle = gr; ctx.fillRect(r[0] - 10, by - 26, r[2] + 20, 28);
      ctx.fillStyle = 'rgba(255,255,255,0.95)'; ctx.fillRect(r[0] - 10, by, r[2] + 20, 1.6);
      ctx.restore();
    }
  };

  P._box = function (ctx, f, idx, lock, isActive, sec) {
    var s = this.s, b = f.bbox;
    var cx = (b[0] + b[2] / 2) * s, cy = (b[1] + b[3] / 2) * s;
    var k = 1 + (1 - lock) * 0.45;
    var w = b[2] * s * k, h = b[3] * s * k, x = cx - w / 2, y = cy - h / 2, L = Math.min(w, h) * 0.2;
    var col = this.mode === 'processing' ? CYAN : (isActive ? DANGER : [148, 163, 184]);
    if (this.mode === 'result' && isActive && f.label !== 'POTENTIALLY_MANIPULATED') { col = f.label === 'AUTHENTIC' ? [34, 197, 94] : [245, 158, 11]; }
    ctx.save();
    ctx.globalAlpha = 0.25 + 0.75 * lock;
    ctx.strokeStyle = rgba(col, 1); ctx.lineWidth = 3; ctx.lineCap = 'square';
    ctx.shadowColor = rgba(col, 0.8); ctx.shadowBlur = this.mode === 'processing' ? 10 : 6;
    ctx.beginPath();
    ctx.moveTo(x, y + L); ctx.lineTo(x, y); ctx.lineTo(x + L, y);
    ctx.moveTo(x + w - L, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + L);
    ctx.moveTo(x + w, y + h - L); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - L, y + h);
    ctx.moveTo(x + L, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - L);
    ctx.stroke();
    ctx.shadowBlur = 0; ctx.lineWidth = 1.2; ctx.globalAlpha *= 0.8;
    ctx.beginPath();                                              // centre tick marks
    ctx.moveTo(cx - 6, cy); ctx.lineTo(cx + 6, cy); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx, cy + 6);
    ctx.stroke();
    ctx.restore();
    // label chip
    var txt = 'FACE ' + String(idx + 1).padStart(2, '0') + (f.confidence ? '  ' + Math.round(f.confidence * 100) + '%' : '');
    ctx.save();
    ctx.font = '600 11px "JetBrains Mono", "Courier New", monospace';
    var tw = ctx.measureText(txt).width + 14, ty = Math.max(2, y - 22);
    ctx.fillStyle = rgba(col, 0.92); ctx.fillRect(x, ty, tw, 18);
    ctx.fillStyle = '#04111c'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(txt, x + 7, ty + 9.5);
    ctx.restore();
  };

  P._landmarks = function (ctx, f, lock, sec) {
    var s = this.s;
    ctx.save();
    for (var i = 0; i < f.landmarks.length; i++) {
      var x = f.landmarks[i][0] * s, y = f.landmarks[i][1] * s, pr = 0.5 + 0.5 * Math.sin(sec * 3 + i);
      ctx.fillStyle = rgba(CYAN, 0.25 * lock); ctx.beginPath(); ctx.arc(x, y, 5 + 3 * pr, 0, TAU); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,' + lock + ')'; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, TAU); ctx.fill();
    }
    ctx.restore();
  };

  P._hud = function (ctx, sec) {
    var W = this.cssW, H = this.cssH;
    ctx.save();
    ctx.font = '600 11px "JetBrains Mono", "Courier New", monospace';
    ctx.textBaseline = 'top'; ctx.fillStyle = rgba(CYAN, 0.95);
    var n = this.faces.length;
    ctx.fillText(n ? ('FACES  ' + n + '   ·   ' + (this.label || this.stage.replace(/_/g, ' '))) : 'SEARCHING FOR FACES…', 12, 10);
    ctx.fillStyle = rgba(CYAN, 0.6);
    ctx.fillText(new Date().toISOString().slice(11, 19), 12, H - 20);
    var ph = (sec * 10) % 100;                                    // tiny activity ticker
    ctx.textAlign = 'right'; ctx.fillText('SCAN ' + String(Math.floor(ph)).padStart(2, '0'), W - 12, H - 20);
    ctx.restore();
  };

  // ---------------------------------------------------------------- result-page layers
  P._xform = function (ctx, f) {                                  // crop space -> photo space
    var g = f.crop_geo, s = this.s;
    if (!g) { return false; }
    ctx.translate(g.cx * s, g.cy * s); ctx.rotate((g.angle || 0) * Math.PI / 180); ctx.translate(-g.cx * s, -g.cy * s);
    return true;
  };

  P._combined = function (f, idx) {
    var key = 'c' + idx;
    if (this.cache[key] !== undefined) { return this.cache[key]; }
    var grids = f.grids || [], N = 32, acc = null;
    for (var g = 0; g < grids.length; g++) {
      var G = grids[g].grid, gh = G.length, gw = G[0].length;
      var c = document.createElement('canvas'); c.width = gw; c.height = gh;
      var cx = c.getContext('2d'), id = cx.createImageData(gw, gh);
      for (var y = 0; y < gh; y++) { for (var x = 0; x < gw; x++) { var v = Math.round(clamp(G[y][x], 0, 1) * 255), o = (y * gw + x) * 4; id.data[o] = v; id.data[o + 1] = v; id.data[o + 2] = v; id.data[o + 3] = 255; } }
      cx.putImageData(id, 0, 0);
      var up = document.createElement('canvas'); up.width = N; up.height = N;
      var ux = up.getContext('2d'); ux.imageSmoothingEnabled = true; ux.imageSmoothingQuality = 'high';
      ux.drawImage(c, 0, 0, gw, gh, 0, 0, N, N);
      var data = ux.getImageData(0, 0, N, N).data;
      if (!acc) { acc = new Float32Array(N * N); }
      for (var i = 0; i < N * N; i++) { acc[i] += data[i * 4] / 255 / grids.length; }
    }
    if (!acc) { this.cache[key] = null; return null; }
    var max = 0; for (var m = 0; m < acc.length; m++) { max = Math.max(max, acc[m]); }
    if (max <= 0) { this.cache[key] = null; return null; }
    for (var n = 0; n < acc.length; n++) { acc[n] /= max; }
    var hc = document.createElement('canvas'); hc.width = N; hc.height = N;
    var hx = hc.getContext('2d'), hd = hx.createImageData(N, N);
    for (var p = 0; p < N * N; p++) {
      var val = acc[p], col = val < 0.5 ? mix([40, 50, 160], [168, 60, 190], val * 2) : mix([168, 60, 190], [255, 120, 40], (val - 0.5) * 2);
      hd.data[p * 4] = col[0]; hd.data[p * 4 + 1] = col[1]; hd.data[p * 4 + 2] = col[2];
      hd.data[p * 4 + 3] = Math.round(clamp((val - 0.18) / 0.82, 0, 1) * 215);
    }
    hx.putImageData(hd, 0, 0);
    var res = { canvas: hc, acc: acc, N: N };
    this.cache[key] = res;
    return res;
  };

  P._heat = function (ctx, f, idx) {
    var h = this._combined(f, idx), g = f.crop_geo;
    if (!h || !g) { return; }
    ctx.save();
    if (this._xform(ctx, f)) {
      var s = this.s;
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.globalAlpha = 0.78;
      ctx.drawImage(h.canvas, g.x1 * s, g.y1 * s, (g.x2 - g.x1) * s, (g.y2 - g.y1) * s);
    }
    ctx.restore();
  };

  P._regions = function (f) {
    var idx = this.faces.indexOf(f), h = this._combined(f, idx);
    if (!h) { return []; }
    var N = h.N, acc = Float32Array.from(h.acc), out = [], names = [];
    for (var r = 0; r < 3; r++) {
      var best = -1, bi = 0;
      for (var i = 0; i < N * N; i++) { if (acc[i] > best) { best = acc[i]; bi = i; } }
      if (best < 0.55) { break; }
      var px = (bi % N + 0.5) / N, py = ((bi / N) | 0) / N + 0.5 / N;
      out.push({ n: r + 1, fx: px, fy: py, value: best });
      for (var j = 0; j < N * N; j++) {                           // suppress the neighbourhood
        var dx = (j % N + 0.5) / N - px, dy = ((j / N) | 0) / N + 0.5 / N - py;
        if (dx * dx + dy * dy < 0.045) { acc[j] = 0; }
      }
    }
    out.forEach(function (o) {
      var v = o.fy < 0.34 ? 'forehead / eyes' : (o.fy < 0.58 ? 'nose / cheek' : (o.fy < 0.8 ? 'mouth area' : 'chin / jaw'));
      var side = o.fx < 0.38 ? 'left ' : (o.fx > 0.62 ? 'right ' : '');
      o.label = (v.indexOf('cheek') >= 0 && side ? side + 'cheek' : side + v);
    });
    return out;
  };

  P._regionsDraw = function (ctx, f, idx, sec) {
    var regs = this._regions(f), g = f.crop_geo;
    if (!regs.length || !g) { return; }
    var s = this.s, cw = (g.x2 - g.x1) * s, ch = (g.y2 - g.y1) * s;
    ctx.save();
    if (!this._xform(ctx, f)) { ctx.restore(); return; }
    var pulse = 0.5 + 0.5 * Math.sin(sec * 2.4);
    regs.forEach(function (o) {
      var w = cw * 0.34, h = ch * 0.26, x = g.x1 * s + o.fx * cw - w / 2, y = g.y1 * s + o.fy * ch - h / 2;
      ctx.fillStyle = rgba(DANGER, 0.12 + 0.08 * pulse); ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = rgba(DANGER, 0.95); ctx.lineWidth = 1.8; ctx.setLineDash([6, 4]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
      ctx.fillStyle = rgba(DANGER, 1); ctx.beginPath(); ctx.arc(x, y, 10, 0, TAU); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.font = '700 11px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(String(o.n), x, y + 0.5);
    });
    ctx.restore();
  };

  window.DSFaceMask = DSFaceMask;
})();
