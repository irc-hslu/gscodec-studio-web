/** @typedef {import('./types.js').GSCodecManifest} GSCodecManifest */
/** @typedef {import('./types.js').GopMeta} GopMeta */
/** @typedef {import('./types.js').ParamMeta} ParamMeta */

const DEFAULT_TRANSFORMS = {
  means: false,
  scales: false,
  quats: true,
  opacities: false,
  sh0: true,
  shN: true,
};

function readFloatFlat(token) {
  const out = [];
  if (!Array.isArray(token)) return out;
  for (const item of token) {
    if (Array.isArray(item)) {
      for (const v of item) out.push(Number(v));
    } else {
      out.push(Number(item));
    }
  }
  return out;
}

function readIntList(token) {
  if (!Array.isArray(token)) return [];
  return token.map((v) => Number(v));
}

function readStringList(token) {
  if (!Array.isArray(token)) return [];
  return token.map((v) => String(v));
}

function readParamMeta(token) {
  const obj = token;
  return {
    shape: readIntList(obj.shape),
    dtype: String(obj.dtype ?? "float32"),
    mins: readFloatFlat(obj.mins),
    maxs: readFloatFlat(obj.maxs),
    file_extension: String(obj.file_extension ?? ".mp4"),
    empty: Boolean(obj.empty ?? false),
  };
}

export function parseManifest(json) {
  const root = JSON.parse(json);
  const transforms = root.transform_attributes ?? {};
  return {
    version: Number(root.version ?? 1),
    compression: String(root.compression ?? "seq_hevc"),
    gop_size: Number(root.gop_size ?? 1),
    frame_num: Number(root.frame_num ?? 1),
    num_gop: Number(root.num_gop ?? 1),
    use_sort: Boolean(root.use_sort ?? true),
    use_all_intra: Boolean(root.use_all_intra ?? false),
    quat_convention: String(root.quat_convention ?? "wxyz"),
    sh_degree: Number(root.sh_degree ?? 3),
    n_orig_per_frame: readIntList(root.n_orig_per_frame),
    layer_encoding: String(root.layer_encoding ?? ""),
    layer_order: readStringList(root.layer_order).length
      ? readStringList(root.layer_order)
      : ["env", "scene"],
    n_scene_per_frame: readIntList(root.n_scene_per_frame),
    n_env_per_frame: readIntList(root.n_env_per_frame),
    transform_attributes: { ...DEFAULT_TRANSFORMS, ...transforms },
  };
}

export function parseGopMeta(json) {
  const root = JSON.parse(json);
  const shN = root.shN;
  const shNMeta =
    shN && !shN.empty && shN.shape?.length ? readParamMeta(shN) : null;
  return {
    means: readParamMeta(root.means),
    scales: readParamMeta(root.scales),
    opacities: readParamMeta(root.opacities),
    quats: readParamMeta(root.quats),
    sh0: readParamMeta(root.sh0),
    shN: shNMeta,
  };
}

export async function loadManifestFromSource(source) {
  const text = await source.readText("unity_manifest.json");
  return parseManifest(text);
}

export async function loadGopMetaFromSource(source, gopId, layerPrefix = "") {
  const suffix = layerPrefix ? `${layerPrefix}meta.json` : "meta.json";
  const text = await source.readText(`gop${gopId}_${suffix}`);
  return parseGopMeta(text);
}

export function gopStem(gopId, paramName, layerPrefix = "") {
  return `gop${gopId}_${layerPrefix}${paramName}`;
}
