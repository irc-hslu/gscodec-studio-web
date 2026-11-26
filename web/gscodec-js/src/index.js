/**
 * GSCodec browser library — decode seq_hevc attribute MP4s and render with Spark
 * (https://sparkjs.dev) on a Three.js WebGLRenderer.
 * No Node.js required. The page needs an import map for "three", "three/addons/",
 * and "@sparkjsdev/spark" (see web/gscodec-js/demo/index.html).
 */

export { createUrlSource } from "./io/url-source.js";
export { createFileMapSource } from "./io/file-map-source.js";
export { parseManifest, parseGopMeta, loadManifestFromSource, loadGopMetaFromSource } from "./meta.js";
export { decodeSequence } from "./decode/sequence.js";
export { decompressGop, gopToFrames } from "./decode/gop.js";
export { activateFrame } from "./splat/activate.js";
export { trimSplatPadding } from "./splat/trim.js";
export { concatSplatFrames } from "./splat/concat.js";
export { decodeMp4ToFrames, hevcLikelySupported } from "./video/mp4-frames.js";
export { GSCodecViewer, frameToPackedSplats, describeScales } from "./gscodec-viewer.js";
export { SplatRenderer } from "./render/splat-renderer.js";

import { GSCodecViewer } from "./gscodec-viewer.js";

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{ baseUrl?: string, source?: object, maxSplats?: number, frame?: number, autoStart?: boolean }} options
 */
export async function createViewer(canvas, options) {
  const viewer = new GSCodecViewer(canvas, options);
  await viewer.load();
  if (options.autoStart !== false) viewer.start();
  return viewer;
}
