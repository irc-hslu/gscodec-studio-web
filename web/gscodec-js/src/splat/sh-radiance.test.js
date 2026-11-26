import assert from "node:assert/strict";
import { packShRadiance, sphericalHarmonicsDegree } from "./sh-radiance.js";

assert.equal(sphericalHarmonicsDegree(15), 3);
assert.equal(sphericalHarmonicsDegree(0), 0);

const coeffs = 15;
const sh = new Float32Array(coeffs * 3);
sh[0] = 0.5;
sh[1] = -0.5;
sh[2] = 0;
sh[9] = 0.25;

const bands = packShRadiance(sh, 1, 1, 1);
const sh1 = new Uint8Array(bands.sh1.buffer);
const sh2 = new Uint8Array(bands.sh2.buffer);
assert.equal(sh1[0], 192);
assert.equal(sh1[1], 64);
assert.equal(sh1[2], 128);
assert.equal(sh2[0], 160);
assert.equal(bands.sh3.length, 6);

console.log("sh radiance packing tests passed");
