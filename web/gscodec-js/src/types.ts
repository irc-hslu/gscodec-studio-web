export const SHN_BAND_NAMES = [
  "sh1_-1",
  "sh1_0",
  "sh1_1",
  "sh2_-2",
  "sh2_-1",
  "sh2_0",
  "sh2_1",
  "sh2_2",
  "sh3_-3",
  "sh3_-2",
  "sh3_-1",
  "sh3_0",
  "sh3_1",
  "sh3_2",
  "sh3_3",
] as const;

export const PAD_LOG_SCALE = -9.7;

export interface ParamMeta {
  shape: number[];
  dtype: string;
  mins: number[];
  maxs: number[];
  file_extension: string;
  empty?: boolean;
}

export interface GopMeta {
  means: ParamMeta;
  scales: ParamMeta;
  opacities: ParamMeta;
  quats: ParamMeta;
  sh0: ParamMeta;
  shN: ParamMeta | null;
}

export interface TransformAttributes {
  means: boolean;
  scales: boolean;
  quats: boolean;
  opacities: boolean;
  sh0: boolean;
  shN: boolean;
}

export interface GSCodecManifest {
  version: number;
  compression: string;
  gop_size: number;
  frame_num: number;
  num_gop: number;
  use_sort: boolean;
  use_all_intra: boolean;
  quat_convention: string;
  sh_degree: number;
  n_orig_per_frame: number[];
  layer_encoding: string;
  layer_order: string[];
  n_scene_per_frame: number[];
  n_env_per_frame: number[];
  transform_attributes: TransformAttributes;
}

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Vec4 {
  x: number;
  y: number;
  z: number;
  w: number;
}

/** One decoded frame (pre-activation, same layout as GSCodec PLY). */
export interface SplatFrame {
  count: number;
  means: Float32Array;
  scales: Float32Array;
  opacities: Float32Array;
  quats: Float32Array;
  sh0: Float32Array;
  /** Flat [count * 15 * 3] when present */
  shN: Float32Array | null;
}

/** GPU-ready frame (post-activation) for rendering. */
export interface RenderableSplatFrame {
  count: number;
  positions: Float32Array;
  scales: Float32Array;
  colors: Float32Array;
  rotations: Float32Array;
}

export interface VideoDecoderBackend {
  /** Decode HEVC attribute MP4 to 8-bit frames (RGB or grayscale). */
  decodeRgb8(
    videoPath: string,
    width: number,
    height: number,
    grayscale?: boolean,
  ): Promise<Uint8Array[]>;
}

export interface CompressionSource {
  /** Directory containing gop*_meta.json and MP4s (absolute on Node). */
  compressionDir: string;
  /** Resolve a file name inside compressionDir (Node: path join; browser: URL join). */
  resolveAsset(fileName: string): string;
  readText(fileName: string): Promise<string>;
}
