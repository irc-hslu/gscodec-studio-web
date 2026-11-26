import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PackedSplats, SparkRenderer, SplatMesh, utils } from "@sparkjsdev/spark";

import { createUrlSource } from "./io/url-source.js";
import { decodeSequence } from "./decode/sequence.js";
import { sphericalHarmonicsDegree } from "./splat/sh-radiance.js";

const SH_C0 = 0.28209479177387814;

/**
 * Decode a GSCodec folder and display it with Spark (SplatMesh + SparkRenderer)
 * on a Three.js WebGLRenderer. The page import map must resolve "three",
 * "three/addons/", and "@sparkjsdev/spark".
 */
export class GSCodecViewer {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{
   *   baseUrl?: string,
   *   source?: { resolveAsset: (n: string) => string, readText: (n: string) => Promise<string>, dispose?: () => void },
   *   maxSplats?: number,
   *   frame?: number,
   *   fps?: number,
   *   onFrame?: (index: number, count: number, playing: boolean) => void,
   * }} options
   */
  constructor(canvas, options) {
    this.canvas = canvas;
    this.baseUrl = options.baseUrl ?? "";
    this.maxSplats = options.maxSplats ?? 500_000;
    this.frameIndex = options.frame ?? 0;
    this.fps = options.fps ?? 30;
    this.onFrame = options.onFrame ?? null;
    this.playing = false;
    this._cameraFramed = false;
    this._nextFrameMs = 0;
    if (options.source) {
      this.source = options.source;
    } else if (options.baseUrl) {
      this.source = createUrlSource(options.baseUrl);
    } else {
      throw new Error("GSCodecViewer requires options.source or options.baseUrl");
    }

    this.frames = [];
    this.splats = null;
    this.packed = null;
    this.lastScaleReport = "";

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      alpha: false,
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x0a0b0e, 1);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0b0e);
    this.spark = new SparkRenderer({ renderer: this.renderer });
    this.scene.add(this.spark);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
    this.camera.position.set(0, 0, 2);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;

    this._keys = new Set();
    this._moveRight = new THREE.Vector3();
    this._moveForward = new THREE.Vector3();
    this._moveDelta = new THREE.Vector3();
    this._lastMoveMs = performance.now();

    this._onResize = () => this._resize();
    this._onKeyDown = (event) => this._keyDown(event);
    this._onKeyUp = (event) => this._keyUp(event);
    this._onBlur = () => this._keys.clear();
    window.addEventListener("resize", this._onResize);
    window.addEventListener("keydown", this._onKeyDown);
    window.addEventListener("keyup", this._onKeyUp);
    window.addEventListener("blur", this._onBlur);
    this._resize();
  }

  _keyDown(event) {
    if (isTypingTarget(event.target)) return;
    if (!ARROW_KEYS.has(event.code) && event.code !== "ShiftLeft" && event.code !== "ShiftRight") {
      return;
    }
    this._keys.add(event.code);
    if (ARROW_KEYS.has(event.code)) event.preventDefault();
  }

  _keyUp(event) {
    this._keys.delete(event.code);
  }

  _applyArrowMove() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - this._lastMoveMs) / 1000);
    this._lastMoveMs = now;
    const moving =
      this._keys.has("ArrowLeft") ||
      this._keys.has("ArrowRight") ||
      this._keys.has("ArrowUp") ||
      this._keys.has("ArrowDown");
    if (!moving) return;

    const distance = Math.max(this.camera.position.distanceTo(this.controls.target), 0.05);
    const fast = this._keys.has("ShiftLeft") || this._keys.has("ShiftRight");
    const step = distance * (fast ? 1.8 : 0.6) * dt;

    this.camera.getWorldDirection(this._moveForward);
    this._moveForward.y = 0;
    if (this._moveForward.lengthSq() < 1e-8) this._moveForward.set(0, 0, -1);
    this._moveForward.normalize();
    this._moveRight.crossVectors(this._moveForward, this.camera.up).normalize();

    this._moveDelta.set(0, 0, 0);
    if (this._keys.has("ArrowLeft")) this._moveDelta.addScaledVector(this._moveRight, -step);
    if (this._keys.has("ArrowRight")) this._moveDelta.addScaledVector(this._moveRight, step);
    if (this._keys.has("ArrowUp")) this._moveDelta.addScaledVector(this._moveForward, step);
    if (this._keys.has("ArrowDown")) this._moveDelta.addScaledVector(this._moveForward, -step);

    this.camera.position.add(this._moveDelta);
    this.controls.target.add(this._moveDelta);
  }

  _resize() {
    const w = this.canvas.clientWidth || this.canvas.width || 1;
    const h = this.canvas.clientHeight || this.canvas.height || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /**
   * @param {(msg: string) => void} [onProgress]
   */
  async load(onProgress) {
    this.frames = await decodeSequence(this.source, { onProgress });
    this.setFrame(this.frameIndex);
  }

  /**
   * @param {number} index
   * @param {{ reframe?: boolean }} [options] reframe fits the camera to this frame.
   *   Playback leaves the camera where the user put it.
   */
  setFrame(index, options = {}) {
    if (!this.frames.length) return;
    const next = Math.max(0, Math.min(index, this.frames.length - 1));
    const same = next === this.frameIndex && this.splats;
    this.frameIndex = next;
    if (same && !options.reframe) {
      this._emitFrame();
      return;
    }
    const frame = this.frames[this.frameIndex];
    const reuse = this.packed && packedDegree(this.packed) === frameDegree(frame);

    if (!reuse) {
      if (this.splats) {
        this.scene.remove(this.splats);
        this.splats.dispose?.();
        this.splats = null;
      }
      if (this.packed) {
        this.packed.dispose?.();
        this.packed = null;
      }
    }

    this.packed = frameToPackedSplats(frame, this.maxSplats, reuse ? this.packed : null);
    if (!this.splats) {
      this.splats = new SplatMesh({ packedSplats: this.packed });
      this.scene.add(this.splats);
    } else {
      markShTextures(this.packed);
    }

    if (options.reframe || !this._cameraFramed) {
      this._frameCamera(frame, this.maxSplats);
      this._cameraFramed = true;
      this.lastScaleReport = describeScales(frame);
      console.info(this.lastScaleReport);
    }
    this._emitFrame();
  }

  /** Loop the decoded frames. Holds the current camera. */
  play() {
    if (this.frames.length < 2) {
      this.playing = false;
      this._emitFrame();
      return;
    }
    this.playing = true;
    this._nextFrameMs = performance.now();
    this._emitFrame();
  }

  /** Freeze on the current frame. */
  pause() {
    this.playing = false;
    this._emitFrame();
  }

  /** Pause and return to the first frame. The camera stays put. */
  stopPlayback() {
    this.playing = false;
    if (this.frames.length) this.setFrame(0);
    else this._emitFrame();
  }

  _emitFrame() {
    if (typeof this.onFrame === "function") {
      this.onFrame(this.frameIndex, this.frames.length, this.playing);
    }
  }

  _tickPlayback() {
    if (!this.playing || this.frames.length < 2) return;
    const now = performance.now();
    const frameMs = 1000 / Math.max(1, this.fps);
    if (now < this._nextFrameMs + frameMs) return;
    this._nextFrameMs = now;
    this.setFrame((this.frameIndex + 1) % this.frames.length);
  }

  _frameCamera(frame, maxSplats) {
    const n = frame.count;
    const stride = Math.max(1, Math.ceil(n / maxSplats));
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (let i = 0; i < n; i += stride) {
      const x = frame.means[i * 3];
      const y = frame.means[i * 3 + 1];
      const z = frame.means[i * 3 + 2];
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }
    const center = new THREE.Vector3(
      (minX + maxX) * 0.5,
      (minY + maxY) * 0.5,
      (minZ + maxZ) * 0.5,
    );
    const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5 || 1;
    this.controls.target.copy(center);
    this.camera.position.set(center.x, center.y + radius * 0.15, center.z + radius * 2.4);
    this.camera.near = Math.max(0.01, radius / 200);
    this.camera.far = Math.max(50, radius * 40);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  start() {
    this._resize();
    this._lastMoveMs = performance.now();
    this.renderer.setAnimationLoop(() => {
      this._tickPlayback();
      this._applyArrowMove();
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  stop() {
    this.playing = false;
    this.renderer.setAnimationLoop(null);
  }

  dispose() {
    this.stop();
    window.removeEventListener("resize", this._onResize);
    window.removeEventListener("keydown", this._onKeyDown);
    window.removeEventListener("keyup", this._onKeyUp);
    window.removeEventListener("blur", this._onBlur);
    this._keys.clear();
    if (this.splats) {
      this.scene.remove(this.splats);
      this.splats.dispose?.();
      this.splats = null;
    }
    if (this.packed) {
      this.packed.dispose?.();
      this.packed = null;
    }
    this.scene.remove(this.spark);
    this.spark.dispose?.();
    this.controls.dispose();
    this.renderer.dispose();
    if (typeof this.source.dispose === "function") this.source.dispose();
  }

  get numFrames() {
    return this.frames.length;
  }
}

const ARROW_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]);

function isTypingTarget(target) {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

const SH_C0_COLOR = SH_C0;

function quatToXyzw(frame, i) {
  const o = i * 4;
  const q = frame.quats;
  if (frame.quatConvention === "xyzw") {
    return [q[o], q[o + 1], q[o + 2], q[o + 3]];
  }
  return [q[o + 1], q[o + 2], q[o + 3], q[o]];
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * GSCodec frame → Spark PackedSplats.
 * Scales are the activated linear scales (Spark stores ln(scale) itself).
 * Quaternions are xyzw. Color is the SH DC term. shN is Spark's sh1/sh2/sh3.
 */
export function frameToPackedSplats(frame, maxSplats, reuse = null) {
  const n = frame.count;
  const stride = Math.max(1, Math.ceil(n / Math.max(1, maxSplats)));
  const count = Math.ceil(n / stride);
  const shCoeffs = frame.shN && n > 0 ? frame.shN.length / n / 3 : 0;
  const degree = Number.isFinite(shCoeffs) ? sphericalHarmonicsDegree(shCoeffs) : 0;

  let sh1Max = 1;
  let sh2Max = 1;
  let sh3Max = 1;
  if (degree > 0) {
    const bounds = shAbsMax(frame.shN, n, shCoeffs, stride);
    sh1Max = bounds[0];
    sh2Max = bounds[1];
    sh3Max = bounds[2];
  }

  const packed =
    reuse && packedDegree(reuse) === degree
      ? reuse
      : new PackedSplats({
          maxSplats: count,
          splatEncoding: { sh1Max, sh2Max, sh3Max },
        });
  packed.splatEncoding.sh1Max = sh1Max;
  packed.splatEncoding.sh2Max = sh2Max;
  packed.splatEncoding.sh3Max = sh3Max;
  const array = packed.ensureSplats(count);
  const sh1 = degree >= 1 ? packed.ensureSplatsSh(1, count) : null;
  const sh2 = degree >= 2 ? packed.ensureSplatsSh(2, count) : null;
  const sh3 = degree >= 3 ? packed.ensureSplatsSh(3, count) : null;
  // Spark's SH encoders OR bits into the words, so a reused frame must start at zero.
  if (sh1) sh1.fill(0, 0, count * 2);
  if (sh2) sh2.fill(0, 0, count * 4);
  if (sh3) sh3.fill(0, 0, count * 4);
  const band = new Float32Array(21);

  let j = 0;
  for (let i = 0; i < n; i += stride) {
    const sx = Math.exp(frame.scales[i * 3]);
    const sy = Math.exp(frame.scales[i * 3 + 1]);
    const sz = Math.exp(frame.scales[i * 3 + 2]);
    const [qx, qy, qz, qw] = quatToXyzw(frame, i);
    const shBase = i * 3;
    const r = 0.5 + SH_C0_COLOR * frame.sh0[shBase];
    const g = 0.5 + SH_C0_COLOR * frame.sh0[shBase + 1];
    const b = 0.5 + SH_C0_COLOR * frame.sh0[shBase + 2];
    utils.setPackedSplat(
      array,
      j,
      frame.means[i * 3],
      frame.means[i * 3 + 1],
      frame.means[i * 3 + 2],
      sx,
      sy,
      sz,
      qx,
      qy,
      qz,
      qw,
      sigmoid(frame.opacities[i]),
      r,
      g,
      b,
    );
    if (degree >= 1) {
      copyShBand(frame.shN, i, shCoeffs, 0, 3, band);
      utils.encodeSh1Rgb(sh1, j, band, { sh1Max });
    }
    if (degree >= 2) {
      copyShBand(frame.shN, i, shCoeffs, 3, 5, band);
      utils.encodeSh2Rgb(sh2, j, band, { sh2Max });
    }
    if (degree >= 3) {
      copyShBand(frame.shN, i, shCoeffs, 8, 7, band);
      utils.encodeSh3Rgb(sh3, j, band, { sh3Max });
    }
    j++;
  }
  packed.numSplats = count;
  packed.needsUpdate = true;
  return packed;
}

function frameDegree(frame) {
  const n = frame.count;
  const shCoeffs = frame.shN && n > 0 ? frame.shN.length / n / 3 : 0;
  return Number.isFinite(shCoeffs) ? sphericalHarmonicsDegree(shCoeffs) : 0;
}

function packedDegree(packed) {
  if (packed.extra?.sh3) return 3;
  if (packed.extra?.sh2) return 2;
  if (packed.extra?.sh1) return 1;
  return 0;
}

function markShTextures(packed) {
  for (const key of ["sh1Texture", "sh2Texture", "sh3Texture"]) {
    const texture = packed.extra?.[key]?.value;
    const array = packed.extra?.[key.slice(0, 3)];
    if (!texture) continue;
    const data = texture.image?.data;
    if (array && data && data.buffer === array.buffer) {
      texture.needsUpdate = true;
    } else {
      texture.dispose?.();
      delete packed.extra[key];
    }
  }
}

function copyShBand(shN, gaussian, coeffs, bandOffset, bandCount, out) {
  const base = gaussian * coeffs * 3 + bandOffset * 3;
  const n = bandCount * 3;
  for (let k = 0; k < n; k++) out[k] = shN[base + k];
}

function shAbsMax(shN, n, coeffs, stride) {
  let m1 = 1;
  let m2 = 1;
  let m3 = 1;
  for (let i = 0; i < n; i += stride) {
    const base = i * coeffs * 3;
    for (let k = 0; k < coeffs * 3; k++) {
      const a = Math.abs(shN[base + k]);
      const coeff = Math.floor(k / 3);
      if (coeff < 3) {
        if (a > m1) m1 = a;
      } else if (coeff < 8) {
        if (a > m2) m2 = a;
      } else if (a > m3) m3 = a;
    }
  }
  return [m1, m2, m3];
}

/** Position bbox from the 16-bit means, plus linear scales after exp(log-scale). */
export function describeScales(frame) {
  const n = frame.count;
  let lo = Infinity;
  let hi = -Infinity;
  let logLo = Infinity;
  let logHi = -Infinity;
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = frame.means[i * 3];
    const y = frame.means[i * 3 + 1];
    const z = frame.means[i * 3 + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    for (let c = 0; c < 3; c++) {
      const logS = frame.scales[i * 3 + c];
      const s = Math.exp(logS);
      if (s < lo) lo = s;
      if (s > hi) hi = s;
      if (logS < logLo) logLo = logS;
      if (logS > logHi) logHi = logS;
    }
  }
  return (
    `positions x ${minX.toFixed(3)}..${maxX.toFixed(3)} ` +
    `y ${minY.toFixed(3)}..${maxY.toFixed(3)} ` +
    `z ${minZ.toFixed(3)}..${maxZ.toFixed(3)} (16-bit means_l/means_u). ` +
    `linear scale ${lo.toExponential(2)} .. ${hi.toExponential(2)} ` +
    `(stored log ${logLo.toFixed(2)} .. ${logHi.toFixed(2)}). ` +
    `quaternions ${frame.quatConvention || "wxyz"}. ` +
    `radiance SH degree ${frame.shN ? sphericalHarmonicsDegree(frame.shN.length / n / 3) : 0}.`
  );
}
