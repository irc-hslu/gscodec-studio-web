import { GSPLAT_VS_INSTANCED, GSPLAT_FS } from "./gsplat-shaders.js";
import { sortSplatsBackToFront } from "./splat-sort.js";

/** @typedef {{ count: number, positions: Float32Array, scales: Float32Array, rotations: Float32Array, colors: Float32Array }} RenderableSplats */

export class SplatRenderer {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
    if (!this.gl) throw new Error("WebGL2 required");

    const gl = this.gl;
    this.prog = createProgram(gl, GSPLAT_VS_INSTANCED, GSPLAT_FS);
    this.uView = gl.getUniformLocation(this.prog, "uView");
    this.uProj = gl.getUniformLocation(this.prog, "uProj");
    this.uViewport = gl.getUniformLocation(this.prog, "uViewport");

    const corners = new Float32Array([
      -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1,
    ]);
    this.cornerVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerVbo);
    gl.bufferData(gl.ARRAY_BUFFER, corners, gl.STATIC_DRAW);

    this.buffers = {
      center: gl.createBuffer(),
      scale: gl.createBuffer(),
      rot: gl.createBuffer(),
      color: gl.createBuffer(),
    };

    this.instanceCount = 0;
    this.fov = Math.PI / 4;
    this.near = 0.05;
    this.far = 500;
    /** @type {RenderableSplats | null} */
    this._full = null;
    this._center = null;
  }

  /**
   * @param {RenderableSplats} renderable
   * @param {{ maxSplats?: number }} [opts]
   */
  setSplats(renderable, opts = {}) {
    const max = opts.maxSplats ?? renderable.count;
    const n = Math.min(renderable.count, max);
    const stride = Math.max(1, Math.floor(renderable.count / n));
    this.instanceCount = Math.ceil(renderable.count / stride);

    this._full = {
      count: renderable.count,
      positions: renderable.positions,
      scales: renderable.scales,
      rotations: renderable.rotations,
      colors: renderable.colors,
    };
    this._stride = stride;
    this._center = renderable.positions;
  }

  sceneBounds() {
    if (!this._center) return { center: [0, 0, 0], radius: 1 };
    const n = this.instanceCount * this._stride;
    const c = this._center;
    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;
    const limit = Math.min(n, c.length / 3);
    for (let i = 0; i < limit; i++) {
      const x = c[i * 3];
      const y = c[i * 3 + 1];
      const z = c[i * 3 + 2];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
      minZ = Math.min(minZ, z);
      maxZ = Math.max(maxZ, z);
    }
    const center = [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5];
    const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5 || 1;
    return { center, radius };
  }

  /**
   * @param {Float32Array} view column-major 4x4
   * @param {Float32Array} proj column-major 4x4
   */
  draw(view, proj) {
    if (!this._full || this.instanceCount === 0) return;

    const gl = this.gl;
    const w = this.canvas.clientWidth || this.canvas.width;
    const h = this.canvas.clientHeight || this.canvas.height;
    if (this.canvas.width !== w) this.canvas.width = w;
    if (this.canvas.height !== h) this.canvas.height = h;

    gl.viewport(0, 0, w, h);
    gl.clearColor(0.04, 0.05, 0.07, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);

    const order = buildOrderedSubset(
      this._full,
      this._stride,
      this.instanceCount,
      view,
    );
    packInstanceBuffers(gl, this.buffers, this._full, order);

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.uView, false, view);
    gl.uniformMatrix4fv(this.uProj, false, proj);
    gl.uniform2f(this.uViewport, w, h);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.cornerVbo);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const bindInstance = (loc, buf, size) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      gl.vertexAttribDivisor(loc, 1);
    };
    bindInstance(1, this.buffers.center, 3);
    bindInstance(2, this.buffers.scale, 3);
    bindInstance(3, this.buffers.rot, 4);
    bindInstance(4, this.buffers.color, 4);

    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, order.length);

    for (let i = 1; i <= 4; i++) gl.vertexAttribDivisor(i, 0);
  }
}

function buildOrderedSubset(full, stride, cap, view) {
  const ids = [];
  for (let i = 0; i < full.count && ids.length < cap; i += stride) ids.push(i);
  return sortSplatsBackToFront(full.positions, view, ids);
}

function packInstanceBuffers(gl, buffers, full, ids) {
  const n = ids.length;
  const center = new Float32Array(n * 3);
  const scale = new Float32Array(n * 3);
  const rot = new Float32Array(n * 4);
  const color = new Float32Array(n * 4);
  for (let j = 0; j < n; j++) {
    const i = ids[j];
    center.set(full.positions.subarray(i * 3, i * 3 + 3), j * 3);
    scale.set(full.scales.subarray(i * 3, i * 3 + 3), j * 3);
    rot.set(full.rotations.subarray(i * 4, i * 4 + 4), j * 4);
    color.set(full.colors.subarray(i * 4, i * 4 + 4), j * 4);
  }
  for (const [name, data] of [
    ["center", center],
    ["scale", scale],
    ["rot", rot],
    ["color", color],
  ]) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers[name]);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
  }
}

function createProgram(gl, vs, fs) {
  const v = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(v, vs);
  gl.compileShader(v);
  if (!gl.getShaderParameter(v, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(v) || "VS compile failed");
  }
  const f = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(f, fs);
  gl.compileShader(f);
  if (!gl.getShaderParameter(f, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(f) || "FS compile failed");
  }
  const p = gl.createProgram();
  gl.attachShader(p, v);
  gl.attachShader(p, f);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p) || "Link failed");
  }
  return p;
}
