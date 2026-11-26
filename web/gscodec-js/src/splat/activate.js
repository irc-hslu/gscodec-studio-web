/** Match unity/GSCodecUnity GsplatFrameConverter activations. */
export function activateFrame(frame) {
  const n = frame.count;
  const positions = new Float32Array(n * 3);
  const scales = new Float32Array(n * 3);
  const opacities = new Float32Array(n);
  /** w,x,y,z per Gsplat.hlsl */
  const rotations = new Float32Array(n * 4);
  /** raw SH DC in rgb; opacity in a (sigmoid) */
  const colors = new Float32Array(n * 4);

  for (let i = 0; i < n; i++) {
    positions[i * 3] = frame.means[i * 3];
    positions[i * 3 + 1] = frame.means[i * 3 + 1];
    positions[i * 3 + 2] = frame.means[i * 3 + 2];

    scales[i * 3] = Math.exp(frame.scales[i * 3]);
    scales[i * 3 + 1] = Math.exp(frame.scales[i * 3 + 1]);
    scales[i * 3 + 2] = Math.exp(frame.scales[i * 3 + 2]);

    const alpha = sigmoid(frame.opacities[i]);
    opacities[i] = alpha;

    const x = frame.quats[i * 4];
    const y = frame.quats[i * 4 + 1];
    const z = frame.quats[i * 4 + 2];
    const w = frame.quats[i * 4 + 3];
    const q = normalizeQuat(w, x, y, z);
    rotations[i * 4] = q[0];
    rotations[i * 4 + 1] = q[1];
    rotations[i * 4 + 2] = q[2];
    rotations[i * 4 + 3] = q[3];

    const sh0off = i * 3;
    colors[i * 4] = frame.sh0[sh0off];
    colors[i * 4 + 1] = frame.sh0[sh0off + 1];
    colors[i * 4 + 2] = frame.sh0[sh0off + 2];
    colors[i * 4 + 3] = alpha;
  }

  return { count: n, positions, scales, rotations, colors, opacities };
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function normalizeQuat(w, x, y, z) {
  const len = Math.hypot(w, x, y, z) || 1;
  return [w / len, x / len, y / len, z / len];
}
