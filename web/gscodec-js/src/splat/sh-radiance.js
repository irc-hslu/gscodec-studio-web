/**
 * Pack GSCodec shN into the byte words Three.js GaussianSplat evaluates
 * as view-dependent radiance (same layout as GaussianSplatPLYLoader).
 *
 * shN is [gaussian, coefficient, rgb], coefficient order
 * sh1_-1, sh1_0, sh1_1, sh2_-2 .. sh3_3 (gsplat / seq_hevc).
 * The shader reads little-endian bytes as (byte - 128) / 128.
 */

const SH_BAND_COMPONENTS = [0, 9, 15, 21];
const SH_BAND_WORDS = [0, 3, 4, 6];
const BAND_OFFSET = [0, 0, 3, 8];

export function sphericalHarmonicsDegree(coefficientCount) {
  if (coefficientCount >= 15) return 3;
  if (coefficientCount >= 8) return 2;
  if (coefficientCount >= 3) return 1;
  return 0;
}

/**
 * @param {Float32Array | null} shN
 * @param {number} gaussianCount source gaussians (before subsampling)
 * @param {number} stride subsample step
 * @param {number} packedCount output splats
 * @returns {Record<string, Uint32Array>}
 */
export function packShRadiance(shN, gaussianCount, stride, packedCount) {
  if (!shN || gaussianCount <= 0) return {};
  const coeffs = shN.length / gaussianCount / 3;
  const degree = sphericalHarmonicsDegree(coeffs);
  if (!Number.isFinite(coeffs) || degree === 0) return {};

  const bands = {};
  const bytes = {};
  for (let d = 1; d <= degree; d++) {
    const packed = new Uint32Array(packedCount * SH_BAND_WORDS[d]);
    packed.fill(0x80808080);
    bands[`sh${d}`] = packed;
    bytes[d] = new Uint8ClampedArray(packed.buffer);
  }

  let out = 0;
  for (let i = 0; i < gaussianCount && out < packedCount; i += stride) {
    writeGaussian(bytes, out, shN, i, coeffs, degree);
    out++;
  }
  return bands;
}

function writeGaussian(bytes, outIndex, shN, srcIndex, coeffs, degree) {
  const base = srcIndex * coeffs * 3;
  for (let d = 1; d <= degree; d++) {
    const target = bytes[d];
    const targetOffset = outIndex * SH_BAND_WORDS[d] * 4;
    const bandOffset = BAND_OFFSET[d];
    const nComp = SH_BAND_COMPONENTS[d];
    for (let j = 0; j < nComp; j++) {
      const coefficient = Math.floor(j / 3);
      const channel = j % 3;
      const value = shN[base + (bandOffset + coefficient) * 3 + channel];
      target[targetOffset + j] = value * 128 + 128;
    }
  }
}
