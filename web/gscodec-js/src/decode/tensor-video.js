import { denorm8, denorm16 } from "../denorm.js";

/** Decoded HEVC frames are often macroblock-padded. Index rows with the real width. */
export function frameWidth(frame, fallback) {
  return frame && frame.width ? frame.width : fallback;
}

function isGrayBuffer(frame) {
  if (frame.gray) return true;
  if (frame.width && frame.height) return frame.length === frame.width * frame.height;
  return false;
}

let sizeWarned = false;
function noteFrameSize(frame, gridW, gridH) {
  if (!frame || sizeWarned) return;
  const w = frame.width || gridW;
  const h = frame.height || gridH;
  if (w !== gridW || h !== gridH) {
    sizeWarned = true;
    console.warn(
      `Attribute video is ${w}×${h} but the Gaussian grid is ${gridW}×${gridH}. ` +
        "Using the video stride and cropping to the grid (HEVC padding).",
    );
  }
}

export function indexTensor(shape, indices) {
  let offset = 0;
  let mult = 1;
  for (let d = shape.length - 1; d >= 0; d--) {
    offset += indices[d] * mult;
    mult *= shape[d];
  }
  return offset;
}

/**
 * @param {Uint8Array[]} frames
 * @param {{ shape: number[], mins: number[], maxs: number[] }} meta
 * @param {{ grayscale?: boolean, merge16Bit?: boolean, framesU?: Uint8Array[] }} opts
 */
export function videoFramesToTensor(frames, meta, opts = {}) {
  const shape = meta.shape;
  const T = shape[0];
  if (frames.length !== T) {
    throw new Error(`Expected ${T} frames, got ${frames.length}`);
  }

  const total = shape.reduce((a, b) => a * b, 1) || 0;
  const out = new Float32Array(Math.max(total, 0));
  const mins = meta.mins;
  const maxs = meta.maxs;
  const H = shape[1];
  const W = shape[2];
  noteFrameSize(frames[0], W, H);

  if (opts.merge16Bit) {
    decode16BitPlanes(frames, opts.framesU, shape, mins, maxs, out);
    return out;
  }

  // [T,H,W] grayscale (opacities)
  if (shape.length === 3 || opts.grayscale) {
    for (let t = 0; t < T; t++) {
      const frame = frames[t];
      const srcW = frameWidth(frame, W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const pix = y * srcW + x;
          const raw = isGrayBuffer(frame) ? frame[pix] : frame[pix * 3];
          out[indexTensor(shape.length === 3 ? shape : [T, H, W], [t, y, x])] =
            denorm8(raw, mins[0], maxs[0]);
        }
      }
    }
    return out;
  }

  // [T,H,W,C]
  if (shape.length === 4) {
    const C = shape[3];
    for (let t = 0; t < T; t++) {
      const frame = frames[t];
      const srcW = frameWidth(frame, W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const pix = y * srcW + x;
          for (let c = 0; c < C; c++) {
            const raw = C === 1 && isGrayBuffer(frame) ? frame[pix] : frame[pix * 3 + c];
            out[indexTensor(shape, [t, y, x, c])] = denorm8(
              raw,
              mins[c] ?? mins[0],
              maxs[c] ?? maxs[0],
            );
          }
        }
      }
    }
    return out;
  }

  // [T,H,W,bands,3] e.g. sh0 with bands=1
  if (shape.length === 5) {
    const bands = shape[3];
    const C = shape[4];
    for (let t = 0; t < T; t++) {
      const frame = frames[t];
      const srcW = frameWidth(frame, W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const pix = y * srcW + x;
          for (let b = 0; b < bands; b++) {
            for (let c = 0; c < C; c++) {
              const minIdx = b * C + c;
              out[indexTensor(shape, [t, y, x, b, c])] = denorm8(
                frame[pix * 3 + c],
                mins[minIdx] ?? mins[c] ?? mins[0],
                maxs[minIdx] ?? maxs[c] ?? maxs[0],
              );
            }
          }
        }
      }
    }
    return out;
  }

  throw new Error(`Unsupported attribute shape: [${shape.join(",")}]`);
}

function planeByte(frame, pix, channel) {
  if (isGrayBuffer(frame)) {
    if (channel !== 0) {
      throw new Error(
        "16-bit means plane decoded as grayscale. xyz needs the RGB bytes of means_l and means_u.",
      );
    }
    return frame[pix];
  }
  return frame[pix * 3 + channel];
}

function decode16BitPlanes(framesL, framesU, shape, mins, maxs, out) {
  const T = shape[0];
  const H = shape[1];
  const W = shape[2];
  const channels = shape[3];
  if (!framesU || framesU.length !== T) {
    throw new Error(`16-bit means needs ${T} high-byte frames, got ${framesU ? framesU.length : 0}`);
  }
  for (let t = 0; t < T; t++) {
    const fl = framesL[t];
    const fu = framesU[t];
    const srcWL = frameWidth(fl, W);
    const srcWU = frameWidth(fu, W);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const pixL = y * srcWL + x;
        const pixU = y * srcWU + x;
        for (let c = 0; c < channels; c++) {
          out[indexTensor(shape, [t, y, x, c])] = denorm16(
            planeByte(fl, pixL, c),
            planeByte(fu, pixU, c),
            mins[c] ?? mins[0],
            maxs[c] ?? maxs[0],
          );
        }
      }
    }
  }
}

/**
 * shN stack: decode 15 band videos into tensor shape [T,H,W,15,3]
 */
export function stackShNBandVideos(bandFrameLists, meta) {
  const shape = meta.shape;
  const T = shape[0];
  const H = shape[1];
  const W = shape[2];
  const bands = shape[3];
  const C = shape[4];
  const out = new Float32Array(T * H * W * bands * C);
  const mins = meta.mins;
  const maxs = meta.maxs;

  for (let b = 0; b < bands; b++) {
    const frames = bandFrameLists[b];
    if (frames.length !== T) throw new Error(`shN band ${b}: expected ${T} frames`);
    for (let t = 0; t < T; t++) {
      const frame = frames[t];
      const srcW = frameWidth(frame, W);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const pix = y * srcW + x;
          for (let c = 0; c < C; c++) {
            const min = mins[b * C + c] ?? mins[0];
            const max = maxs[b * C + c] ?? maxs[0];
            out[indexTensor(shape, [t, y, x, b, c])] = denorm8(
              frame[pix * 3 + c],
              min,
              max,
            );
          }
        }
      }
    }
  }
  return out;
}
