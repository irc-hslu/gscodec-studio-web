import { gopStem } from "../meta.js";
import { decodeMp4ToFrames } from "../video/mp4-frames.js";
import { frameWidth, stackShNBandVideos, videoFramesToTensor } from "./tensor-video.js";
import { deorganize } from "./deorganize.js";
import { paramInverseTransform } from "./inverse-transform.js";
import { SHN_BAND_NAMES } from "../sh-bands.js";

/**
 * @param {import('../io/url-source.js').createUrlSource extends Function ? ReturnType<import('../io/url-source.js').createUrlSource> : any} source
 * @param {number} gopId
 * @param {import('../meta.js').parseGopMeta extends Function ? any : never} meta
 * @param {string} layerPrefix
 */
export async function decompressGop(
  source,
  gopId,
  meta,
  layerPrefix = "",
  onProgress = () => {},
) {
  const ext = (m) => (m.file_extension ?? ".mp4").replace(/^\./, "");
  const url = (name) => source.resolveAsset(name);
  const vidOpts = (paramMeta, extra = {}) => ({
    width: paramMeta.shape[1],
    height: paramMeta.shape[2],
    ...extra,
    onProgress,
  });

  async function loadAttr(paramName, paramMeta) {
    const tag = gopStem(gopId, paramName, layerPrefix);
    const e = ext(paramMeta);
    const T = paramMeta.shape[0];
    if (paramName !== "means") onProgress(`  decoding ${tag}…`);

    if (paramName === "means") {
      // compress_ply_sequence / _compress_video_hevc_16bit writes xyz as
      // lossless low-byte and high-byte videos ({tag}_l, {tag}_u).
      onProgress(`  decoding ${tag}_l + ${tag}_u (16-bit xyz)…`);
      const l = await decodeMp4ToFrames(url(`${tag}_l.${e}`), T, vidOpts(paramMeta));
      const u = await decodeMp4ToFrames(url(`${tag}_u.${e}`), T, vidOpts(paramMeta));
      return videoFramesToTensor(l, paramMeta, { merge16Bit: true, framesU: u });
    }
    if (paramName === "quats") {
      const w = await decodeMp4ToFrames(
        url(`${tag}_w.${e}`),
        T,
        vidOpts(paramMeta, { grayscale: true }),
      );
      const xyz = await decodeMp4ToFrames(url(`${tag}_xyz.${e}`), T, vidOpts(paramMeta));
      return mergeQuatVideos(w, xyz, paramMeta);
    }
    if (paramName === "shN") {
      if (paramMeta.empty || (paramMeta.shape.length >= 2 && paramMeta.shape[3] === 0)) {
        const n = paramMeta.shape.reduce((a, b) => a * b, 1);
        return new Float32Array(Math.max(0, n));
      }
      const bands = [];
      for (const band of SHN_BAND_NAMES) {
        bands.push(
          await decodeMp4ToFrames(url(`${tag}_${band}.${e}`), T, vidOpts(paramMeta)),
        );
      }
      return stackShNBandVideos(bands, paramMeta);
    }
    const grayscale = paramName === "opacities";
    const frames = await decodeMp4ToFrames(
      url(`${tag}.${e}`),
      T,
      vidOpts(paramMeta, { grayscale }),
    );
    return videoFramesToTensor(frames, paramMeta, { grayscale });
  }

  const splats = {};
  const shapes = {};
  const params = ["means", "scales", "opacities", "quats", "sh0", "shN"];
  for (const paramName of params) {
    const paramMeta = meta[paramName];
    if (!paramMeta) continue;
    splats[paramName] = await loadAttr(paramName, paramMeta);
    shapes[paramName] = paramMeta.shape;
  }
  return { splats, shapes };
}

function mergeQuatVideos(wFrames, xyzFrames, meta) {
  const shape = meta.shape;
  const T = shape[0];
  const H = shape[1];
  const W = shape[2];
  const out = new Float32Array(shape.reduce((a, b) => a * b, 1));
  const mins = meta.mins;
  const maxs = meta.maxs;

  for (let t = 0; t < T; t++) {
    const fw = wFrames[t];
    const fxyz = xyzFrames[t];
    const srcW = frameWidth(fxyz, frameWidth(fw, W));
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const pix = y * srcW + x;
        for (let c = 0; c < 4; c++) {
          let raw;
          if (c === 0) raw = fw[pix];
          else raw = fxyz[pix * 3 + (c - 1)];
          const v = (raw / 255) * (maxs[c] - mins[c]) + mins[c];
          const idx = (((t * H + y) * W + x) * 4) + c;
          out[idx] = v;
        }
      }
    }
  }
  return out;
}

export function gopToFrames(decompressed, transformAttributes) {
  const frames = deorganize(decompressed.splats, decompressed.shapes);
  return paramInverseTransform(frames, transformAttributes);
}
