/** Concatenate env + scene splat frames along the Gaussian axis. */
export function concatSplatFrames(parts) {
  if (parts.length === 1) return parts[0];
  const sh0Bands = Math.max(...parts.map((p) => p.sh0.length / p.count / 3));
  const shNBands = Math.max(
    ...parts.map((p) => (p.shN ? p.shN.length / p.count / 3 : 0)),
  );

  let total = 0;
  for (const p of parts) total += p.count;

  const out = {
    count: total,
    means: new Float32Array(total * 3),
    scales: new Float32Array(total * 3),
    opacities: new Float32Array(total),
    quats: new Float32Array(total * 4),
    sh0: new Float32Array(total * sh0Bands * 3),
    shN: shNBands > 0 ? new Float32Array(total * shNBands * 3) : null,
  };

  let offset = 0;
  for (const p of parts) {
    appendPart(out, p, offset, sh0Bands, shNBands);
    offset += p.count;
  }
  out.count = total;
  return out;
}

function appendPart(out, p, offset, sh0Bands, shNBands) {
  out.means.set(p.means, offset * 3);
  out.scales.set(p.scales, offset * 3);
  out.opacities.set(p.opacities, offset);
  out.quats.set(p.quats, offset * 4);

  const pSh0 = p.sh0.length / p.count / 3;
  for (let i = 0; i < p.count; i++) {
    for (let b = 0; b < pSh0; b++) {
      out.sh0.set(
        p.sh0.subarray(i * pSh0 * 3 + b * 3, i * pSh0 * 3 + b * 3 + 3),
        (offset + i) * sh0Bands * 3 + b * 3,
      );
    }
    if (out.shN && p.shN) {
      const pN = p.shN.length / p.count / 3;
      for (let b = 0; b < pN; b++) {
        out.shN.set(
          p.shN.subarray(i * pN * 3 + b * 3, i * pN * 3 + b * 3 + 3),
          (offset + i) * shNBands * 3 + b * 3,
        );
      }
    }
  }
}
