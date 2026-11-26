/** BT.709 YCbCr → RGB (matches gsplat seq_hevc / Unity ColorSpace). */
export function ycbcrToRgb(y: number, cb: number, cr: number): [number, number, number] {
  return [y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb];
}

export function applyYcbcrToRgb3(buf: Float32Array, offset: number): void {
  const [r, g, b] = ycbcrToRgb(buf[offset], buf[offset + 1], buf[offset + 2]);
  buf[offset] = r;
  buf[offset + 1] = g;
  buf[offset + 2] = b;
}
