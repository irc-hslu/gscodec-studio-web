/**
 * Run: node src/decode/tensor-video.test.js
 * Checks that HEVC padding (video wider than the Gaussian grid) does not scramble scales.
 */
import assert from "node:assert/strict";
import { videoFramesToTensor } from "./tensor-video.js";

// Grid is 2×2. The decoded frame is padded to width 4 (typical HEVC alignment).
// Row-major RGB, only channel 0 is used. Values:
//   (0,0)=0, (1,0)=255, (0,1)=128, (1,1)=64, and the two pad columns are 200.
const W = 2;
const H = 2;
const srcW = 4;
const rgb = new Uint8Array(srcW * H * 3);
function put(x, y, v) {
  const pix = y * srcW + x;
  rgb[pix * 3] = v;
  rgb[pix * 3 + 1] = 0;
  rgb[pix * 3 + 2] = 0;
}
put(0, 0, 0);
put(1, 0, 255);
put(2, 0, 200);
put(3, 0, 200);
put(0, 1, 128);
put(1, 1, 64);
put(2, 1, 200);
put(3, 1, 200);
rgb.width = srcW;
rgb.height = H;

const tensor = videoFramesToTensor([rgb], {
  shape: [1, H, W, 1],
  mins: [0],
  maxs: [1],
});

// Wrong stride (y * gridW) would read the pad (200) as the second row.
assert.equal(tensor.length, 4);
assert.ok(Math.abs(tensor[0] - 0) < 1e-6, `tl ${tensor[0]}`);
assert.ok(Math.abs(tensor[1] - 1) < 1e-6, `tr ${tensor[1]}`);
assert.ok(Math.abs(tensor[2] - 128 / 255) < 1e-6, `bl ${tensor[2]}`);
assert.ok(Math.abs(tensor[3] - 64 / 255) < 1e-6, `br ${tensor[3]}`);

// Log-scale round trip that Spark will log() again: exp(stored) must stay small.
const storedLog = -4.6; // linear ≈ 0.01
const linear = Math.exp(storedLog);
assert.ok(linear > 0.005 && linear < 0.02, linear);
assert.ok(Number.isFinite(Math.log(linear)));

// 16-bit means: low byte + high byte, RGB is xyz, pad columns are ignored.
// Python: raw = (frame_u << 8) + frame_l; value = raw / 65535 * (max-min) + min.
const meansW = 2;
const meansH = 1;
const meansSrcW = 4;
function paint16(dst, x, rgb) {
  const pix = x;
  dst[pix * 3] = rgb[0];
  dst[pix * 3 + 1] = rgb[1];
  dst[pix * 3 + 2] = rgb[2];
}
const meansL = new Uint8Array(meansSrcW * meansH * 3);
const meansU = new Uint8Array(meansSrcW * meansH * 3);
paint16(meansL, 0, [1, 2, 3]);
paint16(meansU, 0, [4, 5, 6]);
paint16(meansL, 1, [7, 8, 9]);
paint16(meansU, 1, [10, 11, 12]);
paint16(meansL, 2, [255, 255, 255]);
paint16(meansU, 2, [255, 255, 255]);
meansL.width = meansSrcW;
meansL.height = meansH;
meansU.width = meansSrcW;
meansU.height = meansH;

const means = videoFramesToTensor(
  [meansL],
  { shape: [1, meansH, meansW, 3], mins: [-2, 0, 10], maxs: [-2, 4, 10] },
  { merge16Bit: true, framesU: [meansU] },
);
function expect16(lo, hi, min, max) {
  return ((hi * 256 + lo) / 65535) * (max - min) + min;
}
assert.ok(Math.abs(means[0] - expect16(1, 4, -2, -2)) < 1e-6);
assert.ok(Math.abs(means[1] - expect16(2, 5, 0, 4)) < 1e-5);
assert.ok(Math.abs(means[2] - expect16(3, 6, 10, 10)) < 1e-6);
assert.ok(Math.abs(means[3] - expect16(7, 10, -2, -2)) < 1e-6);
assert.ok(Math.abs(means[4] - expect16(8, 11, 0, 4)) < 1e-5);
assert.ok(Math.abs(means[5] - expect16(9, 12, 10, 10)) < 1e-6);

console.log("tensor-video padding + 16-bit means tests passed");
