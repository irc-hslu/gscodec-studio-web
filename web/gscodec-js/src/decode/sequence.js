import { loadGopMetaFromSource, loadManifestFromSource } from "../meta.js";
import { decompressGop, gopToFrames } from "./gop.js";
import { concatSplatFrames } from "../splat/concat.js";
import { trimSplatPadding } from "../splat/trim.js";

const PAD_LOG_SCALE = -9.7;

/**
 * Decode a full GSCodec compression folder (browser, no Node.js).
 *
 * @param {ReturnType<import('../io/url-source.js').createUrlSource>} source
 * @param {{ keepPadding?: boolean, onProgress?: (msg: string) => void }} [options]
 * @returns {Promise<import('../splat/frame.js').SplatFrame[]>}
 */
export async function decodeSequence(source, options = {}) {
  const keepPadding = Boolean(options.keepPadding);
  const log = options.onProgress ?? (() => {});

  let manifest = {};
  try {
    manifest = await loadManifestFromSource(source);
  } catch {
    manifest = {
      transform_attributes: {},
      n_orig_per_frame: [],
      layer_encoding: "",
      layer_order: ["env", "scene"],
      n_scene_per_frame: [],
      n_env_per_frame: [],
      num_gop: 1,
    };
  }

  const transformAttributes = {
    means: false,
    scales: false,
    quats: true,
    opacities: false,
    sh0: true,
    shN: true,
    ...manifest.transform_attributes,
  };

  let numGop = manifest.num_gop ?? 0;
  if (!numGop) {
    numGop = await countGops(source);
  }
  const allFrames = [];

  for (let gopId = 0; gopId < numGop; gopId++) {
    log(`Decoding GOP ${gopId}…`);
    const sceneMeta = await loadGopMetaFromSource(source, gopId, "");
    const sceneDec = await decompressGop(source, gopId, sceneMeta, "", log);
    const sceneFrames = gopToFrames(sceneDec, transformAttributes);

    let envFrames = null;
    const split = manifest.layer_encoding === "split";
    if (split) {
      try {
        const envMeta = await loadGopMetaFromSource(source, gopId, "env_");
        const envDec = await decompressGop(source, gopId, envMeta, "env_", log);
        envFrames = gopToFrames(envDec, transformAttributes);
      } catch {
        envFrames = null;
      }
    }

    if (envFrames) {
      const layerOrder = manifest.layer_order ?? ["env", "scene"];
      for (let i = 0; i < sceneFrames.length; i++) {
        const frameIdx = allFrames.length;
        let scene = sceneFrames[i];
        let env = envFrames[i];
        if (!keepPadding) {
          const nScene = manifest.n_scene_per_frame?.[frameIdx];
          const nEnv = manifest.n_env_per_frame?.[frameIdx];
          scene = trimSplatPadding(scene, nScene ?? null, PAD_LOG_SCALE);
          env = trimSplatPadding(env, nEnv ?? null, PAD_LOG_SCALE);
        }
        const ordered = layerOrder.map((layer) => (layer === "env" ? env : scene));
        allFrames.push(concatSplatFrames(ordered));
      }
    } else {
      allFrames.push(...sceneFrames);
    }
  }

  const quatConvention = manifest.quat_convention || "wxyz";
  const stamp = (frame) => {
    frame.quatConvention = quatConvention;
    return frame;
  };

  if (keepPadding) return allFrames.map(stamp);

  const nOrigList = manifest.n_orig_per_frame ?? [];
  return allFrames.map((frame, idx) => {
    const nOrig = nOrigList[idx];
    return stamp(trimSplatPadding(frame, nOrig ?? null, PAD_LOG_SCALE));
  });
}

async function countGops(source) {
  let n = 0;
  for (;; n++) {
    try {
      await source.readText(`gop${n}_meta.json`);
    } catch {
      break;
    }
  }
  return Math.max(1, n);
}
