/** BT.709 YCbCr → RGB (matches gsplat seq_hevc). */
export function ycbcrToRgb(y, cb, cr) {
  return [y + 1.402 * cr, y - 0.344136 * cb - 0.714136 * cr, y + 1.772 * cb];
}

/** In-place on a length-3 slice at offset. */
export function applyYcbcrToRgb3(buf, offset) {
  const [r, g, b] = ycbcrToRgb(buf[offset], buf[offset + 1], buf[offset + 2]);
  buf[offset] = r;
  buf[offset + 1] = g;
  buf[offset + 2] = b;
}
