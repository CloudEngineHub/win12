/* Win12 WebGL renderer.
 *
 * Modes (persisted in localStorage 'webgl-mode'):
 *   off     - pure browser DOM/CSS rendering (default).
 *   partial - legacy experimental WebGL overlay on top of the DOM.
 *   full    - WebGL is the primary *visual* layer, composed in the same
 *             stacking order as the DOM:
 *
 *                 ┌──────────────────────── viewport ───────────────────────┐
 *                 │  wallpaper canvas (z-index:-1, one WebGL context)       │
 *                 │                                                          │
 *                 │   .window (stacking ctx)  ── own panel canvas (child,    │
 *                 │   .window (stacking ctx)  ── own panel canvas  z:-1 in   │
 *                 │   #start-menu (z:91)      ── own panel canvas   element) │
 *                 │   .dock / #cm / widget    ── own panel canvas            │
 *                 │                                                          │
 *                 │   text / icons / inputs / iframes stay DOM, ON TOP of   │
 *                 │   the canvas inside each element.                       │
 *                 └──────────────────────────────────────────────────────────┘
 *
 *             Each panel's canvas lives INSIDE that element's stacking
 *             context, so when windows overlap the upper window's WebGL
 *             material paints ABOVE the lower window's DOM content - the
 *             browser z-ordering composites everything correctly. A single
 *             canvas behind the whole DOM cannot do this (the lower window's
 *             content would ghost through the upper window).
 *
 *   Panel contexts come from a small pool (browsers cap live WebGL contexts,
 *   commonly ~8-16). Elements without a pool slot keep the normal DOM
 *   acrylic (the .gpu-panel paint-stripping class is only added when a
 *   layer is attached), so degradation is per-panel, never a hard failure.
 *
 *   Blur is GPU-only: a downscaled (~384px) copy of the wallpaper is shared
 *   as an ImageBitmap and uploaded once per panel context; the panel shader
 *   samples it with cover mapping + ring taps. No DOM capture, no
 *   readPixels, no per-frame buffer/shader/texture creation. Static layers
 *   are not redrawn.
 *
 * Public API (window.win12WebGL):
 *   init() / apply(mode) / setMode(mode) / getMode()
 *   start(mode) / stop() / resize() / render() / destroy()
 *   supported() / getState()
 */
(function () {
  'use strict';

  const KEY = 'webgl-mode';
  const DEFAULT_MODE = 'off';
  const MODES = ['off', 'partial', 'full'];
  const MAX_DPR = 2;
  const POOL_SIZE = 12;         // max simultaneous panel WebGL contexts
  const BLUR_SRC_MAX = 384;    // px of the shared downscaled wallpaper

  // ---------------------------------------------------------------- utils

  function parseColor(str) {
    if (!str) return null;
    str = str.trim();
    if (str[0] === '#') {
      const h = str.slice(1);
      if (h.length === 3 || h.length === 4) {
        const r = parseInt(h[0] + h[0], 16), g = parseInt(h[1] + h[1], 16),
          b = parseInt(h[2] + h[2], 16), a = h.length === 4 ? parseInt(h[3] + h[3], 16) : 255;
        return [r / 255, g / 255, b / 255, a / 255];
      }
      if (h.length === 6 || h.length === 8) {
        const r = parseInt(h.slice(0, 2), 16), g = parseInt(h.slice(2, 4), 16),
          b = parseInt(h.slice(4, 6), 16), a = h.length === 8 ? parseInt(h.slice(6, 8), 16) : 255;
        return [r / 255, g / 255, b / 255, a / 255];
      }
      return null;
    }
    const m = str.match(/rgba?\(([^)]+)\)/);
    if (m) {
      const p = m[1].split(',').map(s => parseFloat(s));
      return [p[0] / 255, p[1] / 255, p[2] / 255, p.length > 3 ? p[3] : 1];
    }
    return null;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function dpr() {
    return Math.min(window.devicePixelRatio || 1, MAX_DPR);
  }

  function linkProgram(gl, vsSrc, fsSrc) {
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(s);
        gl.deleteShader(s);
        throw new Error('shader compile failed: ' + log);
      }
      return s;
    };
    const p = gl.createProgram();
    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('program link failed: ' + log);
    }
    return p;
  }

  function uniformMap(gl, program) {
    const map = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) || 0;
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(program, i);
      map[info.name] = gl.getUniformLocation(program, info.name);
    }
    return map;
  }

  // -------------------------------------------------------------- shaders
  // GLSL ES 1.00 (WebGL1 + WebGL2 compatible).

  const QUAD_VS = `
    attribute vec2 aPos; // 0..1 unit quad
    varying vec2 vUV;
    void main() {
      vUV = aPos;
      gl_Position = vec4(aPos * 2.0 - 1.0, 0.0, 1.0);
    }`;

  const WALLPAPER_FS = `
    precision mediump float;
    varying vec2 vUV;
    uniform sampler2D uTex;
    uniform float uHasTex;
    uniform vec2 uCoverScale;
    uniform vec2 uCoverOffset;
    uniform vec4 uColA;
    uniform vec4 uColB;
    void main() {
      vec4 c;
      if (uHasTex > 0.5) {
        c = texture2D(uTex, vUV * uCoverScale + uCoverOffset);
      } else {
        c = mix(uColB, uColA, vUV.y);
      }
      gl_FragColor = vec4(c.rgb * c.a, c.a);
    }`;

  // One panel: wallpaper-blur "mica" fill / solid tint, inner border and a
  // soft drop shadow via a rounded-rect SDF. All geometry in CSS px; the
  // backing store is CSS*DPR and gl.viewport handles the scaling.
  const PANEL_VS = `
    attribute vec2 aPos; // 0..1 over (panel + 2*pad)
    uniform vec2 uViewport;   // CSS px
    uniform vec4 uRect;       // panel x,y,w,h in CSS px (y down)
    uniform float uPad;
    varying vec2 vLocal;      // px, origin at panel top-left, may exceed it
    varying vec2 vScreenN;    // 0..1 of viewport, y down
    void main() {
      vec2 css = uRect.xy - uPad + aPos * (uRect.zw + 2.0 * uPad);
      vLocal = css - uRect.xy;
      vScreenN = css / uViewport;
      gl_Position = vec4(css.x / uViewport.x * 2.0 - 1.0,
                         1.0 - css.y / uViewport.y * 2.0, 0.0, 1.0);
    }`;

  const PANEL_FS = `
    precision mediump float;
    varying vec2 vLocal;
    varying vec2 vScreenN;
    uniform vec2 uSize;
    uniform float uRadius;
    uniform vec4 uTint;
    uniform float uBlurMix;
    uniform sampler2D uWall;
    uniform vec2 uCoverScale;
    uniform vec2 uCoverOffset;
    uniform vec2 uTexel;
    uniform vec4 uShadowColor;
    uniform vec2 uShadowOffset;
    uniform float uShadowBlur;
    uniform vec4 uBorderColor;
    uniform float uBorderWidth;
    uniform float uOpacity;

    float rbox(vec2 p, vec2 b, float r) {
      vec2 q = abs(p) - b + r;
      return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    vec3 wallpaperBlur(vec2 wuv) {
      vec3 c = texture2D(uWall, wuv).rgb * 0.30;
      float wsum = 0.30;
      // two 8-tap rings; source texture is tiny so the bilinear upscale is
      // already very soft - these widen it to roughly CSS blur(60px).
      for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.7853981; // pi/4
        vec2 d1 = vec2(cos(a), sin(a)) * uTexel * 1.8;
        vec2 d2 = vec2(cos(a), sin(a)) * uTexel * 4.2;
        c += texture2D(uWall, wuv + d1).rgb * 0.05;
        c += texture2D(uWall, wuv + d2).rgb * 0.0375;
        wsum += 0.0875;
      }
      return c / wsum;
    }

    void main() {
      vec2 halfSize = uSize * 0.5;
      vec2 p = vLocal - halfSize;
      float d = rbox(p, halfSize, uRadius);

      // --- shadow, outside the body only ---
      float ds = rbox(p - uShadowOffset, halfSize, uRadius);
      float fillMask = 1.0 - smoothstep(-0.75, 0.75, d);
      float shadowA = (1.0 - smoothstep(0.0, uShadowBlur, ds))
                    * uShadowColor.a * (1.0 - fillMask) * uOpacity;
      vec3 accRGB = uShadowColor.rgb * shadowA;
      float accA = shadowA;

      // --- fill: solid tint, or tint over blurred wallpaper ("mica") ---
      vec3 fillRGB = uTint.rgb;
      float fillA = uTint.a;
      if (uBlurMix > 0.001) {
        vec2 wuv = vScreenN * uCoverScale + uCoverOffset;
        vec3 blur = wallpaperBlur(wuv);
        // approx CSS saturate(1.6) contrast(0.85) used by the acrylic rules
        float luma = dot(blur, vec3(0.299, 0.587, 0.114));
        blur = mix(vec3(luma), blur, 1.6);
        blur = mix(vec3(0.5), blur, 0.85);
        vec3 mixRGB = mix(blur, uTint.rgb, uTint.a);
        float mixA = uTint.a + (1.0 - uTint.a);
        fillRGB = mix(uTint.rgb, mixRGB, uBlurMix);
        fillA = mix(uTint.a, mixA, uBlurMix);
      }

      // --- border stroke hugging the inside edge ---
      float bw = max(uBorderWidth, 0.0);
      float borderMask = 0.0;
      if (bw > 0.001 && uBorderColor.a > 0.001) {
        borderMask = (1.0 - smoothstep(-0.75, 0.75, abs(d + bw * 0.5) - bw * 0.5));
      }
      fillRGB = mix(fillRGB, uBorderColor.rgb, borderMask * uBorderColor.a);
      float fillAlpha = fillMask * fillA * uOpacity;

      vec3 outRGB = fillRGB * fillAlpha + accRGB * (1.0 - fillAlpha);
      float outA = fillAlpha + accA * (1.0 - fillAlpha);
      gl_FragColor = vec4(outRGB, outA);
    }`;

  // ---------------------------------------------------------- GL resources

  function createUnitQuad(gl) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER,
      new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    return buf;
  }

  function coverTransform(viewW, viewH, imgW, imgH) {
    if (!imgW || !imgH) return { scale: [1, 1], offset: [0, 0] };
    const s = Math.max(viewW / imgW, viewH / imgH);
    const drawnW = imgW * s, drawnH = imgH * s;
    const offX = (viewW - drawnW) / 2, offY = (viewH - drawnH) / 2;
    return {
      scale: [viewW / drawnW, viewH / drawnH],
      offset: [-offX / drawnW, -offY / drawnH],
    };
  }

  // ------------------------------------------------------ wallpaper canvas

  class WallpaperLayer {
    constructor() {
      this.canvas = null;
      this.gl = null;
      this.program = null;
      this.quad = null;
      this.u = null;
      this.tex = null;
      this.texSize = [0, 0];
      this.dirty = true;
      this.lost = false;
    }

    init(onLost, onRestored) {
      const canvas = document.createElement('canvas');
      canvas.id = 'win12-webgl-canvas';
      canvas.className = 'win12-gl-bg';
      canvas.setAttribute('aria-hidden', 'true');
      const gl = canvas.getContext('webgl2', {
        alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
      }) || canvas.getContext('webgl', {
        alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false,
      });
      if (!gl) return false;
      this.canvas = canvas;
      this.gl = gl;

      canvas.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        this.lost = true;
        onLost();
      });
      canvas.addEventListener('webglcontextrestored', () => {
        this.lost = false;
        this._build();
        this.dirty = true;
        onRestored();
      });

      this._build();
      return true;
    }

    _build() {
      const gl = this.gl;
      this.program = linkProgram(gl, QUAD_VS, WALLPAPER_FS);
      this.u = uniformMap(gl, this.program);
      this.aPos = gl.getAttribLocation(this.program, 'aPos');
      this.quad = createUnitQuad(gl);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      this._uploadCurrent();
    }

    setWallpaper(image) {
      const gl = this.gl;
      if (this.tex) { gl.deleteTexture(this.tex); this.tex = null; }
      this.texSize = [0, 0];
      if (!image) { this.dirty = true; return; }
      try {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        this.tex = tex;
        this.texSize = [image.naturalWidth || image.width, image.naturalHeight || image.height];
      } catch (err) {
        // e.g. SVG without intrinsic dimensions: gradient fallback
        console.warn('Win12 WebGL: wallpaper texture rejected, using gradient:', err && err.message);
        if (this.tex) { gl.deleteTexture(this.tex); this.tex = null; }
      }
      this.dirty = true;
    }

    _uploadCurrent() { /* placeholder for restore: setWallpaper is re-run by manager */ }

    resize() {
      const s = dpr();
      const w = Math.max(1, Math.round(window.innerWidth * s));
      const h = Math.max(1, Math.round(window.innerHeight * s));
      if (this.canvas.width !== w || this.canvas.height !== h) {
        this.canvas.width = w;
        this.canvas.height = h;
        this.dirty = true;
      }
    }

    render(colA, colB) {
      if (!this.gl || this.lost || !this.dirty) return;
      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.program);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(this.aPos);
      gl.vertexAttribPointer(this.aPos, 2, gl.FLOAT, false, 0, 0);
      const cov = coverTransform(window.innerWidth, window.innerHeight,
        this.texSize[0], this.texSize[1]);
      gl.uniform2f(this.u.uCoverScale, cov.scale[0], cov.scale[1]);
      gl.uniform2f(this.u.uCoverOffset, cov.offset[0], cov.offset[1]);
      if (this.tex) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.uniform1i(this.u.uTex, 0);
        gl.uniform1f(this.u.uHasTex, 1);
      } else {
        gl.uniform1f(this.u.uHasTex, 0);
      }
      gl.uniform4fv(this.u.uColA, colA);
      gl.uniform4fv(this.u.uColB, colB);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.dirty = false;
    }

    destroy() {
      this.canvas?.remove();
      this.canvas = null;
      this.gl = null;
    }
  }

  // -------------------------------------------------------- one panel layer
  // A canvas attached INSIDE one panel element, with its own GL context.

  let layerSeq = 0;

  class PanelLayer {
    constructor(el) {
      this.id = ++layerSeq;
      this.el = el;
      this.canvas = document.createElement('canvas');
      this.canvas.className = 'win12-gl-panel';
      this.canvas.setAttribute('aria-hidden', 'true');
      this.gl = null;
      this.program = null;
      this.quad = null;
      this.u = null;
      this.aPos = 0;
      this.tex = null;
      this.texKey = null;
      this.desc = null;
      this.dirty = true;
      this.lost = false;
      this._sig = null;
    }

    /** @returns {boolean} context acquired */
    attach() {
      const gl = this.canvas.getContext('webgl2', {
        alpha: true, premultipliedAlpha: true, antialias: false,
        depth: false, stencil: false, powerPreference: 'low-power',
      }) || this.canvas.getContext('webgl', {
        alpha: true, premultipliedAlpha: true, antialias: false,
        depth: false, stencil: false, powerPreference: 'low-power',
      });
      if (!gl) return false;
      this.gl = gl;
      this.el.insertBefore(this.canvas, this.el.firstChild);
      this.el.classList.add('gpu-panel');

      this.canvas.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        this.lost = true;
        this.onLost?.(this);
      });
      this.canvas.addEventListener('webglcontextrestored', () => {
        this.lost = false;
        this._buildGL();
        this.dirty = true;
      });

      this._buildGL();
      return true;
    }

    _buildGL() {
      const gl = this.gl;
      this.program = linkProgram(gl, PANEL_VS, PANEL_FS);
      this.u = uniformMap(gl, this.program);
      this.aPos = gl.getAttribLocation(this.program, 'aPos');
      this.quad = createUnitQuad(gl);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      this.tex = null;
      this.texKey = null;
    }

    /** Upload the shared (downscaled wallpaper) bitmap. key avoids reuploads. */
    setSource(bitmap, key) {
      if (!this.gl || key === this.texKey) return;
      const gl = this.gl;
      if (this.tex) gl.deleteTexture(this.tex);
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      this.tex = tex;
      this.texKey = key;
      this.dirty = true;
    }

    update(desc) {
      const s = dpr();
      const padCss = desc.shadowBlur * 2 + Math.abs(desc.shadowOffset[0]) +
        Math.abs(desc.shadowOffset[1]) + 4;
      const cw = Math.max(1, Math.round((desc.w + 2 * padCss) * s));
      const ch = Math.max(1, Math.round((desc.h + 2 * padCss) * s));
      // cheap signature: redraw only when something actually changed
      const sig = [cw, ch, desc.x, desc.y, desc.w, desc.h, desc.radius,
        desc.opacity, desc.blurMix, desc.shadowBlur,
        desc.tint.join(','), desc.shadowColor.join(','),
        desc.shadowOffset.join(','), desc.borderColor.join(','),
        desc.borderWidth, padCss].join('|');
      if (sig === this._sig) return;
      this._sig = sig;
      if (this.canvas.width !== cw || this.canvas.height !== ch) {
        this.canvas.width = cw;
        this.canvas.height = ch;
      }
      const cs = this.canvas.style;
      const left = desc.x - padCss, top = desc.y - padCss;
      const widthCss = (desc.w + 2 * padCss) + 'px';
      const heightCss = (desc.h + 2 * padCss) + 'px';
      if (cs.left !== left + 'px') cs.left = left + 'px';
      if (cs.top !== top + 'px') cs.top = top + 'px';
      if (cs.width !== widthCss) cs.width = widthCss;
      if (cs.height !== heightCss) cs.height = heightCss;
      this.desc = desc;
      this._pad = padCss;
      this.dirty = true;
    }

    render(viewportW, viewportH, sourceSize) {
      if (!this.gl || this.lost || !this.dirty || !this.desc) return;
      const gl = this.gl;
      const d = this.desc;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(this.program);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(this.aPos);
      gl.vertexAttribPointer(this.aPos, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(this.u.uViewport, viewportW, viewportH);
      gl.uniform4f(this.u.uRect, d.x, d.y, d.w, d.h);
      gl.uniform1f(this.u.uPad, this._pad);
      gl.uniform2f(this.u.uSize, d.w, d.h);
      gl.uniform1f(this.u.uRadius, d.radius);
      gl.uniform4fv(this.u.uTint, d.tint);
      gl.uniform1f(this.u.uBlurMix, d.blurMix);
      gl.uniform4fv(this.u.uShadowColor, d.shadowColor);
      gl.uniform2f(this.u.uShadowOffset, d.shadowOffset[0], d.shadowOffset[1]);
      gl.uniform1f(this.u.uShadowBlur, d.shadowBlur);
      gl.uniform4fv(this.u.uBorderColor, d.borderColor);
      gl.uniform1f(this.u.uBorderWidth, d.borderWidth);
      gl.uniform1f(this.u.uOpacity, d.opacity);

      if (this.tex) {
        const cov = coverTransform(viewportW, viewportH, sourceSize[0], sourceSize[1]);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.tex);
        gl.uniform1i(this.u.uWall, 0);
        gl.uniform2f(this.u.uCoverScale, cov.scale[0], cov.scale[1]);
        gl.uniform2f(this.u.uCoverOffset, cov.offset[0], cov.offset[1]);
        gl.uniform2f(this.u.uTexel, 1 / sourceSize[0], 1 / sourceSize[1]);
      }
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      this.dirty = false;
    }

    destroy() {
      this.el.classList.remove('gpu-panel');
      this.canvas.remove();
      this.gl = null;
      this.el = null;
      this.canvas = null;
    }
  }

  // ---------------------------------------------------- wallpaper source IO

  const Wallpaper = {
    url: null,
    image: null,
    failed: false,

    currentURL() {
      const bg = getComputedStyle(document.body).backgroundImage;
      const m = bg && bg.match(/url\(["']?([^"')]+)["']?\)/);
      return m ? m[1] : null;
    },

    /** @returns {{changed:boolean}} (image may be null -> gradient fallback) */
    sync(done) {
      const url = this.currentURL();
      if (url === this.url && (this.image || this.failed)) {
        done(this.image, false);
        return;
      }
      this.url = url;
      this.image = null;
      this.failed = !url;
      if (!url) { done(null, true); return; }
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => { this.image = img; done(img, true); };
      img.onerror = () => { this.failed = true; done(null, true); };
      img.src = url;
    },
  };

  /**
   * Shared tiny bitmap every panel context uploads. Rebuilt only when the
   * wallpaper image, theme gradient or viewport aspect changes materially.
   */
  const BlurSource = {
    bitmap: null,
    key: '',
    size: [0, 0],

    async rebuild(image, colA, colB) {
      let w, h, source, usedImage = false;
      if (image) {
        const iw = image.naturalWidth || image.width;
        const ih = image.naturalHeight || image.height;
        if (iw > 0 && ih > 0) {
          const k = Math.min(1, BLUR_SRC_MAX / Math.max(iw, ih));
          w = Math.max(1, Math.round(iw * k));
          h = Math.max(1, Math.round(ih * k));
          source = image;
          usedImage = true;
        }
      }
      if (!usedImage) {
        w = 64; h = 64;
        const c = document.createElement('canvas');
        c.width = w; c.height = h;
        const ctx = c.getContext('2d');
        const g = ctx.createLinearGradient(0, 0, w * 0.6, h);
        const rgba = (cc) => `rgba(${cc.map(v => Math.round(v * 255)).join(',')})`;
        g.addColorStop(0, rgba(colA));
        g.addColorStop(1, rgba(colB));
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, w, h);
        source = c;
      }
      let bitmap = null;
      if (usedImage) {
        try {
          bitmap = await createImageBitmap(source,
            { resizeWidth: w, resizeHeight: h, resizeQuality: 'low' });
        } catch (_) {
          try {
            // pre-scale via 2D canvas, then bitmap the canvas
            const c = document.createElement('canvas');
            c.width = w; c.height = h;
            c.getContext('2d').drawImage(source, 0, 0, w, h);
            bitmap = await createImageBitmap(c);
          } catch (__) {
            bitmap = null; // caller falls back to the gradient key next frame
          }
        }
      }
      if (!bitmap) {
        const c = document.createElement('canvas');
        c.width = 64; c.height = 64;
        const ctx = c.getContext('2d');
        const g = ctx.createLinearGradient(0, 0, 38, 64);
        const rgba = (cc) => `rgba(${cc.map(v => Math.round(v * 255)).join(',')})`;
        g.addColorStop(0, rgba(colA));
        g.addColorStop(1, rgba(colB));
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, 64, 64);
        bitmap = await createImageBitmap(c);
        w = 64; h = 64;
      }
      this.bitmap = bitmap;
      this.size = [w, h];
      this.key = (Wallpaper.url || 'grad') + ':' + w + 'x' + h +
        ':' + colA.join(',') + ':' + colB.join(',');
    },
  };

  // ------------------------------------------------------- DOM -> scene

  /**
   * Elements whose panel material is GPU-appropriate. webapp windows are
   * excluded: they are opaque iframes and need no blur.
   * priority: higher = first in line for a pool slot.
   */
  const PANEL_RULES = [
    { sel: '#cm.show-begin', kind: 'cm', base: 600 },
    { sel: '#start-menu.show-begin, #search-win.show-begin, #widgets.show-begin, #datebox.show-begin, #control.show-begin', kind: 'menu', base: 500 },
    { sel: '#dock-box>.dock', kind: 'dock', base: 400 },
    // windows must beat desktop widgets for pool slots
    { sel: '.window.show-begin:not(.webapp)', kind: 'window', base: 200 },
    { sel: '#desktop-widgets>*:not(.widgets-move)', kind: 'widget', base: 50 },
  ];

  function snapshotPanels() {
    const root = document.documentElement;
    const v = (n, fb) => parseColor(cssVar(n)) || fb;
    const shadow = v('--shadow', [0.13, 0.13, 0.13, 0.19]);
    const unfoc = v('--unfoc', [0.92, 0.92, 0.92, 1]);
    const bg70 = v('--bg70', [1, 1, 1, 0.75]);
    const bg50 = v('--bg50', [1, 1, 1, 0.63]);
    const ctxMenu = v('--contextmeu', [0.97, 0.97, 0.97, 0.73]);
    const winBorder = parseColor('#6f6f6f30');
    const menuBorder = parseColor('#99999950');
    const moreBlur = root.classList.contains('moreblur');

    const M = {
      window: { shadowBlur: 24, shadowOffset: [2, 6], borderColor: winBorder, borderWidth: 1.5, shadowColor: shadow },
      widget: { tint: bg50, blurMix: 1, shadowBlur: 20, shadowOffset: [3, 3], borderColor: [0, 0, 0, 0], borderWidth: 0, shadowColor: shadow },
      dock:   { tint: ctxMenu, blurMix: 0.9, shadowBlur: 18, shadowOffset: [0, 3], borderColor: menuBorder, borderWidth: 1, shadowColor: shadow },
      menu:   { tint: bg50, blurMix: 1, shadowBlur: 22, shadowOffset: [3, 4], borderColor: menuBorder, borderWidth: 1.5, shadowColor: shadow },
      cm:     { tint: ctxMenu, blurMix: 0.9, shadowBlur: 20, shadowOffset: [3, 3], borderColor: winBorder, borderWidth: 1.5, shadowColor: shadow },
    };

    const out = [];
    for (const rule of PANEL_RULES) {
      const els = document.querySelectorAll(rule.sel);
      for (let i = 0; i < els.length; i++) {
        const el = els[i];
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden') continue;
        const opacity = parseFloat(cs.opacity);
        if (!(opacity > 0.01)) continue;
        const r = el.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) continue;
        // generous cull so slide-in menus allocate before entering view
        if (r.right < -1200 || r.bottom < -1200 ||
            r.left > window.innerWidth + 1200 || r.top > window.innerHeight + 1200) continue;
        let radius = parseFloat(cs.borderTopLeftRadius);
        if (!isFinite(radius)) radius = 0;
        radius = Math.min(radius, r.width / 2, r.height / 2);

        const b = M[rule.kind];
        const desc = {
          el, kind: rule.kind,
          x: r.left, y: r.top, w: r.width, h: r.height,
          radius, opacity,
          tint: b.tint, blurMix: b.blurMix,
          shadowColor: b.shadowColor, shadowOffset: b.shadowOffset,
          shadowBlur: b.shadowBlur, borderColor: b.borderColor,
          borderWidth: b.borderWidth,
          priority: rule.base,
        };
        if (rule.kind === 'window') {
          const foc = el.classList.contains('foc');
          desc.tint = foc ? bg70 : unfoc;
          desc.blurMix = foc ? 1 : (moreBlur ? 0.5 : 0);
          if (!foc) { desc.shadowBlur = 10; desc.shadowOffset = [1, 2]; }
          desc.priority = rule.base + (parseInt(cs.zIndex, 10) || 0);
        }
        out.push(desc);
      }
    }
    return out;
  }

  function fallbackColors() {
    return {
      colA: parseColor(cssVar('--theme-1')) || [0.68, 0.43, 0.79, 1],
      colB: parseColor(cssVar('--theme-2')) || [0.23, 0.57, 0.85, 1],
    };
  }

  // -------------------------------------------------------- panel pool mgr

  class PanelManager {
    /** @param {() => void} onDirty request a frame (so a lost layer can be re-allocated) */
    constructor(onDirty) {
      this.onDirty = onDirty;
      /** @type {Map<Element, PanelLayer>} */
      this.layers = new Map();
      this.limit = POOL_SIZE;
    }

    /**
     * Reconcile layers against the current scene.
     * @param {Array} descs visible panel descriptors
     * @param {ImageBitmap} bitmap shared blur source
     * @param {string} key source identity
     * @param {number[]} sourceSize bitmap dimensions
     */
    sync(descs, bitmap, key, sourceSize) {
      const want = new Set(descs.map(d => d.el));

      // release layers that disappeared
      for (const [el, layer] of this.layers) {
        if (!want.has(el)) {
          layer.destroy();
          this.layers.delete(el);
        }
      }

      // allocate pool slots, highest priority first
      const sorted = descs.slice().sort((a, b) => b.priority - a.priority);
      for (const d of sorted) {
        if (this.layers.has(d.el)) continue;
        if (this.layers.size >= this.limit) {
          // evict the lowest-priority ACTIVE layer if this one outranks it
          let victim = null, victimPri = Infinity;
          for (const [el, layer] of this.layers) {
            const p = layer.desc ? layer.desc.priority : -1;
            if (p < victimPri) { victimPri = p; victim = el; }
          }
          if (victim && d.priority > victimPri) {
            const l = this.layers.get(victim);
            l.destroy();
            this.layers.delete(victim);
          } else {
            d.el.classList.remove('gpu-panel'); // DOM fallback for this panel
            continue;
          }
        }
        const layer = new PanelLayer(d.el);
        layer.onLost = (l) => {
          // context is gone: fully remove this canvas (give the panel back
          // to the DOM) and let the next frame allocate a fresh layer.
          const el = l.el;
          l.destroy();
          this.layers.delete(el);
          this.onDirty();
        };
        let attached = false;
        try {
          attached = layer.attach();
        } catch (err) {
          // shader/context failure for this one panel must not kill all
          console.warn('Win12 WebGL: panel layer failed, using DOM for it:', err && err.message);
          layer.destroy();
          this.limit = this.layers.size;
          continue;
        }
        if (!attached) {
          layer.destroy();
          this.limit = this.layers.size; // device refused more contexts
          continue;
        }
        this.layers.set(d.el, layer);
      }

      // push geometry + shared texture, mark dirty
      for (const d of descs) {
        const layer = this.layers.get(d.el);
        if (!layer) continue;
        if (bitmap) layer.setSource(bitmap, key);
        layer.update(d);
      }
    }

    renderAll(viewportW, viewportH, sourceSize) {
      for (const layer of this.layers.values()) {
        layer.render(viewportW, viewportH, sourceSize);
      }
    }

    destroy() {
      for (const layer of this.layers.values()) layer.destroy();
      this.layers.clear();
    }
  }

  // ----------------------------------------------------- change detection

  class SurfaceTracker {
    constructor(onDirty) {
      this.onDirty = onDirty;
      // While a CSS transition is running we must keep polling rects.
      // transitionstart/end are not reliably paired (interrupted transitions
      // can omit end), so use a sliding deadline instead of a counter:
      // every start extends "live" until TRANSITION_GRACE_MS after the last.
      this.liveUntil = 0;
      this._observer = null;
      this._resizeObserver = null;
      this._onStart = null;
      this._onResize = null;
    }

    start() {
      this._observer = new MutationObserver(() => this.onDirty());
      this._observer.observe(document.body, {
        attributes: true, attributeFilter: ['class', 'style'],
        childList: true, subtree: true,
      });
      this._observer.observe(document.documentElement, {
        attributes: true, attributeFilter: ['class', 'style'],
      });
      if (window.ResizeObserver) {
        this._resizeObserver = new ResizeObserver(() => this.onDirty());
        this._resizeObserver.observe(document.body);
      }
      document.addEventListener('transitionstart',
        this._onStart = () => {
          this.liveUntil = performance.now() + 1200; // Win12 transitions <= 700ms
          this.onDirty();
        }, true);
      window.addEventListener('resize', this._onResize = () => this.onDirty());
    }

    isLive() { return performance.now() < this.liveUntil; }

    stop() {
      this._observer?.disconnect();
      this._observer = null;
      this._resizeObserver?.disconnect();
      this._resizeObserver = null;
      if (this._onStart) document.removeEventListener('transitionstart', this._onStart, true);
      if (this._onResize) window.removeEventListener('resize', this._onResize);
      this.liveUntil = 0;
    }
  }

  // --------------------------------------------------------- full compositor

  const FULL_STYLE_ID = 'win12-webgl-full-style';

  const FULL_CSS = `
    html.webgl-full body { background: none !important; }

    /* Paint stripping applies ONLY to panels that own a live GL layer. */
    html.webgl-full .window.gpu-panel,
    html.webgl-full.mica .window.gpu-panel.foc {
      background: transparent !important;
      background-color: transparent !important;
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      box-shadow: none !important;
      border-color: transparent !important;
      isolation: isolate;
    }
    /* the panel canvas spills a shadow margin outside the box; re-clip the
       children to the rounded corners instead of relying on overflow:hidden */
    html.webgl-full .window.gpu-panel { overflow: visible !important; }
    html.webgl-full .window.gpu-panel > .titbar {
      border-top-left-radius: inherit;
      border-top-right-radius: inherit;
    }
    html.webgl-full .window.gpu-panel > .content {
      border-bottom-left-radius: inherit;
      border-bottom-right-radius: inherit;
    }

    html.webgl-full #dock-box .dock.gpu-panel {
      background: transparent !important;
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      box-shadow: none !important;
      outline-color: transparent !important;
      position: relative;
      isolation: isolate;
    }
    html.webgl-full #start-menu.gpu-panel,
    html.webgl-full #search-win.gpu-panel,
    html.webgl-full #widgets.gpu-panel,
    html.webgl-full #datebox.gpu-panel,
    html.webgl-full #control.gpu-panel,
    html.webgl-full #cm.gpu-panel {
      background: transparent !important;
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      box-shadow: none !important;
      border-color: transparent !important;
      isolation: isolate;
    }
    html.webgl-full #desktop-widgets > *.gpu-panel {
      background: transparent !important;
      backdrop-filter: none !important;
      -webkit-backdrop-filter: none !important;
      box-shadow: none !important;
      isolation: isolate;
    }

    .win12-gl-bg {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      z-index: -1;
      pointer-events: none;
    }
    .win12-gl-panel {
      position: absolute;
      z-index: -1;
      pointer-events: none;
    }
  `;

  class FullCompositor {
    constructor() {
      this.bg = new WallpaperLayer();
      this.panels = new PanelManager(() => this._markDirty());
      this.tracker = null;
      this.styleEl = null;
      this.active = false;
      this.raf = 0;
      this.needsFrame = false;
      this.sourceKey = '';
      this.frameToken = 0;
      this.lastError = null;
      this._errTimer = 0;
      this._onVis = null;
    }

    start() {
      if (this.active) return;
      if (!this.bg.init(
        () => this._bgLost(),
        () => {
          // wallpaper context restored: re-upload texture and repaint
          this.bg.setWallpaper(Wallpaper.image);
          this.styleEl && (this.styleEl.disabled = false);
          this._markDirty();
        },
      )) throw new Error('no-webgl');

      this._injectStyles();
      document.documentElement.classList.add('webgl-full');
      document.body.prepend(this.bg.canvas);
      this.bg.resize();

      this.tracker = new SurfaceTracker(() => this._markDirty());
      this.tracker.start();

      this._onVis = () => {
        if (document.hidden) this._cancel();
        else this._markDirty();
      };
      document.addEventListener('visibilitychange', this._onVis);

      this.active = true;
      this._markDirty();
    }

    _bgLost() {
      // wallpaper context gone: DOM cannot paint the wallpaper while body
      // background is stripped - release CSS until restore re-enables it.
      this.styleEl && (this.styleEl.disabled = true);
    }

    _injectStyles() {
      if (this.styleEl) return;
      this.styleEl = document.createElement('style');
      this.styleEl.id = FULL_STYLE_ID;
      this.styleEl.textContent = FULL_CSS;
      document.head.appendChild(this.styleEl);
    }

    resize() {
      if (!this.active) return;
      this.bg.resize();
      this._markDirty();
    }

    render() { this._markDirty(); }

    _markDirty() {
      if (!this.active) return;
      this.needsFrame = true;
      if (!this.raf && !document.hidden) {
        this.raf = requestAnimationFrame(() => this._frame());
      }
    }

    _cancel() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = 0;
      this.needsFrame = false;
    }

    async _frame() {
      this.raf = 0;
      if (!this.active) return;
      if (!this.needsFrame && !this.tracker.isLive()) return;
      this.needsFrame = false;
      const token = ++this.frameToken;
      try {
        await this._renderFrame(token);
      } catch (err) {
        // never let a frame error kill the whole compositor silently
        this.lastError = String(err && (err.stack || err.message || err));
        console.warn('Win12 WebGL: frame error:', err);
        this._scheduleAfterError();
      }
      if (this.active && token === this.frameToken && this.tracker.isLive()) {
        this._markDirty();
      }
    }

    _scheduleAfterError() {
      clearTimeout(this._errTimer);
      this._errTimer = setTimeout(() => this._markDirty(), 500);
    }

    async _renderFrame(token) {

      const { colA, colB } = fallbackColors();
      this.bg.resize();

      // 1. wallpaper image -> wallpaper canvas texture (when changed)
      Wallpaper.sync((img, changed) => {
        if (!changed) return;
        this.bg.setWallpaper(img);
        this._markDirty(); // covers async image decode finishing later
      });
      if (this.bg.dirty) this.bg.render(colA, colB);

      // 2. shared blur bitmap (rebuilt only on wallpaper/theme change)
      const gradKey = colA.join(',') + '|' + colB.join(',');
      const desiredKey = (Wallpaper.image ? 'img:' : 'grad:') +
        (Wallpaper.url || 'grad') + '#' + gradKey;
      if (desiredKey !== this.sourceKey || !BlurSource.bitmap) {
        this.sourceKey = desiredKey;
        await BlurSource.rebuild(Wallpaper.image, colA, colB);
      }
      // mode may have switched (or a newer frame started) during the await
      if (!this.active || token !== this.frameToken) return;

      // 3. reconcile panel layers with the DOM, draw dirty layers
      const descs = snapshotPanels();
      this.panels.sync(descs, BlurSource.bitmap, BlurSource.key, BlurSource.size);
      this.panels.renderAll(window.innerWidth, window.innerHeight, BlurSource.size);
      this.lastError = null;

      if (this.tracker.isLive()) this._markDirty();
    }

    stop() {
      if (!this.active) return;
      this.active = false;
      this._cancel();
      this.tracker?.stop();
      this.tracker = null;
      if (this._onVis) document.removeEventListener('visibilitychange', this._onVis);
      this.panels.destroy();
      this.styleEl?.remove();
      this.styleEl = null;
      document.documentElement.classList.remove('webgl-full');
      this.bg.destroy();
      Wallpaper.url = null;
      Wallpaper.image = null;
      Wallpaper.failed = false;
      BlurSource.bitmap = null;
      this.sourceKey = '';
    }
  }

  // ------------------------------------------------------- partial overlay

  class PartialOverlay {
    constructor() {
      this.canvas = null;
      this.raf = 0;
      this.styleEl = null;
      this._resize = null;
    }

    start() {
      const style = document.createElement('style');
      style.textContent = `
        #win12-webgl-layer {
          position: fixed; inset: 0; width: 100vw; height: 100vh;
          pointer-events: none; z-index: 0; opacity: .3; mix-blend-mode: multiply;
        }`;
      document.head.appendChild(style);
      this.styleEl = style;

      const canvas = document.createElement('canvas');
      canvas.id = 'win12-webgl-layer';
      canvas.setAttribute('aria-hidden', 'true');
      document.body.prepend(canvas);
      const gl = canvas.getContext('webgl2', { alpha: true }) ||
                 canvas.getContext('webgl', { alpha: true });
      if (!gl) throw new Error('no-webgl');
      this.canvas = canvas;

      const program = linkProgram(gl,
        'attribute vec2 p;void main(){gl_Position=vec4(p,0.,1.);}',
        `precision mediump float;uniform float t;
         void main(){
           vec2 p = gl_FragCoord.xy / vec2(1200.0, 800.0);
           float v = 0.5 + 0.5 * sin(t * 0.00025 + p.x * 3.0 + p.y * 2.0);
           gl_FragColor = vec4(0.05 + 0.08 * v, 0.28 + 0.12 * v, 0.52 + 0.18 * v, 0.12);
         }`);
      gl.useProgram(program);
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(program, 'p');
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      const time = gl.getUniformLocation(program, 't');

      const resize = () => {
        const s = dpr();
        canvas.width = innerWidth * s;
        canvas.height = innerHeight * s;
        gl.viewport(0, 0, canvas.width, canvas.height);
      };
      window.addEventListener('resize', this._resize = resize);
      resize();

      const frame = (now) => {
        if (!this.canvas) return;
        gl.uniform1f(time, now);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        this.raf = requestAnimationFrame(frame);
      };
      this.raf = requestAnimationFrame(frame);

      canvas.addEventListener('webglcontextlost', (e) => {
        e.preventDefault();
        this.stop();
        document.documentElement.classList.add('webgl-fallback');
      });
      document.documentElement.classList.add('webgl-partial');
    }

    stop() {
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = 0;
      if (this._resize) window.removeEventListener('resize', this._resize);
      this._resize = null;
      this.canvas?.remove();
      this.canvas = null;
      this.styleEl?.remove();
      this.styleEl = null;
      document.documentElement.classList.remove('webgl-partial');
    }
  }

  // ------------------------------------------------------------ controller

  function supported() {
    try {
      const probe = document.createElement('canvas');
      return !!(probe.getContext('webgl2', { alpha: true }) ||
                probe.getContext('webgl', { alpha: true }));
    } catch (_) {
      return false;
    }
  }

  const full = new FullCompositor();
  const partial = new PartialOverlay();
  let currentMode = DEFAULT_MODE;

  function stopAll() {
    full.stop();
    partial.stop();
    document.documentElement.classList.remove('webgl-full', 'webgl-partial', 'webgl-fallback');
    currentMode = 'off';
  }

  function start(mode) {
    stopAll();
    if (mode === 'off') return 'off';
    if (!supported()) {
      document.documentElement.classList.add('webgl-fallback');
      return 'off';
    }
    try {
      if (mode === 'full') full.start();
      else partial.start();
      currentMode = mode;
      return mode;
    } catch (err) {
      console.warn('Win12 WebGL: falling back to DOM rendering:', err && err.message);
      full.stop();
      partial.stop();
      document.documentElement.classList.add('webgl-fallback');
      currentMode = 'off';
      return 'off';
    }
  }

  window.win12WebGL = {
    getMode: () => localStorage.getItem(KEY) || DEFAULT_MODE,
    apply(mode) {
      mode = MODES.includes(mode) ? mode : DEFAULT_MODE;
      localStorage.setItem(KEY, mode);
      return start(mode);
    },
    setMode(mode) { return this.apply(mode); },
    start,
    stop: stopAll,
    resize() { if (currentMode === 'full') full.resize(); },
    render() { if (currentMode === 'full') full.render(); },
    destroy() { stopAll(); },
    supported,
    getState() {
      return {
        v: 'layered-3',
        mode: currentMode,
        supported: supported(),
        fallback: document.documentElement.classList.contains('webgl-fallback'),
        layers: full.active ? full.panels.layers.size : 0,
        lastError: full.lastError,
      };
    },
    init() { return start(this.getMode()); },
  };

  document.addEventListener('DOMContentLoaded', () => window.win12WebGL.init());
})();
