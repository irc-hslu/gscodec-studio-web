/** @param {number} padLogScale default -9.7 */
export function trimSplatPadding(frame, nOrig, padLogScale = -9.7) {
  const n = frame.count;
  const scales = frame.scales;
  const keep = new Uint8Array(n);
  let kept = 0;
  for (let i = 0; i < n; i++) {
    const sx = scales[i * 3];
    const sy = scales[i * 3 + 1];
    const sz = scales[i * 3 + 2];
    if (sx < padLogScale && sy < padLogScale && sz < padLogScale) continue;
    keep[i] = 1;
    kept++;
  }
  const target =
    nOrig != null && nOrig < kept ? Math.min(nOrig, kept) : kept;
  return compactFrame(frame, keep, target);
}

function compactFrame(frame, keep, maxOut) {
  const keys = ["means", "scales", "opacities", "quats", "sh0"];
  const dims = {
    means: 3,
    scales: 3,
    opacities: 1,
    quats: 4,
    sh0: frame.sh0.length / frame.count / 3,
  };
  const sh0Bands = dims.sh0;
  const hasShN = frame.shN != null;
  const shNBands = hasShN ? frame.shN.length / frame.count / 3 : 0;

  let outCount = 0;
  const out = {
    count: 0,
    means: new Float32Array(maxOut * 3),
    scales: new Float32Array(maxOut * 3),
    opacities: new Float32Array(maxOut),
    quats: new Float32Array(maxOut * 4),
    sh0: new Float32Array(maxOut * sh0Bands * 3),
    shN: hasShN ? new Float32Array(maxOut * shNBands * 3) : null,
  };

  for (let i = 0; i < frame.count && outCount < maxOut; i++) {
    if (!keep[i]) continue;
    copyStride(frame.means, out.means, i, outCount, 3);
    copyStride(frame.scales, out.scales, i, outCount, 3);
    out.opacities[outCount] = frame.opacities[i];
    copyStride(frame.quats, out.quats, i, outCount, 4);
    copyStride(frame.sh0, out.sh0, i, outCount, sh0Bands * 3);
    if (hasShN) copyStride(frame.shN, out.shN, i, outCount, shNBands * 3);
    outCount++;
  }
  out.count = outCount;
  return out;
}

function copyStride(src, dst, from, to, stride) {
  dst.set(src.subarray(from * stride, from * stride + stride), to * stride);
}
