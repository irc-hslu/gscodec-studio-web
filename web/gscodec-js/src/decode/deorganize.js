/**
 * @param {Record<string, Float32Array>} splatsVideos keys means, scales, ...
 * @param {Record<string, number[]>} shapes per-key shape from meta
 */
export function deorganize(splatsVideos, shapes) {
  const means = splatsVideos.means;
  const shape = shapes.means;
  const T = shape[0];
  const nFrames = T;
  const frames = [];

  const attrs = Object.keys(splatsVideos);
  const flattened = {};
  for (const name of attrs) {
    const sh = shapes[name];
    const nFlat = sh[1] * sh[2];
    const rest = sh.slice(3);
    const per = nFlat * rest.reduce((a, b) => a * b, 1);
    flattened[name] = { data: splatsVideos[name], per, rest, nFlat };
  }

  for (let t = 0; t < nFrames; t++) {
    const count = flattened.means.nFlat;
    const frame = {
      count,
      means: new Float32Array(count * 3),
      scales: new Float32Array(count * 3),
      opacities: new Float32Array(count),
      quats: new Float32Array(count * 4),
      sh0: new Float32Array(
        count * (flattened.sh0.rest.reduce((a, b) => a * b, 1) || 3),
      ),
      shN: null,
    };

    for (const name of attrs) {
      const { data, per, rest, nFlat } = flattened[name];
      const sliceStart = t * per;
      const slice = data.subarray(sliceStart, sliceStart + per);
      if (name === "means" || name === "scales") {
        frame[name].set(slice);
      } else if (name === "opacities") {
        for (let i = 0; i < nFlat; i++) frame.opacities[i] = slice[i];
      } else if (name === "quats") {
        frame.quats.set(slice);
      } else if (name === "sh0") {
        frame.sh0.set(slice);
      } else if (name === "shN") {
        if (rest[0] > 0) {
          frame.shN = new Float32Array(nFlat * rest[0] * 3);
          frame.shN.set(slice);
        }
      }
    }
    frames.push(frame);
  }
  return frames;
}
