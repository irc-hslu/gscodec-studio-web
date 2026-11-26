export function denorm8(value, min, max) {
  return (value / 255) * (max - min) + min;
}

/**
 * Inverse of seq_hevc `_decompress_video_hevc_16bit`:
 *   raw = (upper << 8) + lower
 *   value = raw / 65535 * (max - min) + min
 * Multiplication keeps the 16-bit code in unsigned range (JS `<<` is signed).
 */
export function denorm16(lower, upper, min, max) {
  const raw = upper * 256 + lower;
  return (raw / 65535) * (max - min) + min;
}
