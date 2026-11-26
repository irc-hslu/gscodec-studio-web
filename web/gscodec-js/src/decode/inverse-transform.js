import { applyYcbcrToRgb3 } from "../colorspace.js";

/**
 * @param {import('../splat/frame.js').SplatFrame[]} frames
 * @param {import('../meta.js').parseManifest extends Function ? any : never} transformAttributes
 */
export function paramInverseTransform(frames, transformAttributes) {
  if (!transformAttributes.sh0 && !transformAttributes.shN) return frames;

  return frames.map((frame) => {
    const out = { ...frame };
    if (transformAttributes.sh0) {
      out.sh0 = ycbcrBufferToRgb(frame.sh0);
    }
    if (transformAttributes.shN && frame.shN) {
      out.shN = ycbcrBufferToRgb(frame.shN);
    }
    return out;
  });
}

function ycbcrBufferToRgb(buf) {
  const out = new Float32Array(buf.length);
  out.set(buf);
  for (let i = 0; i < out.length; i += 3) {
    applyYcbcrToRgb3(out, i);
  }
  return out;
}
