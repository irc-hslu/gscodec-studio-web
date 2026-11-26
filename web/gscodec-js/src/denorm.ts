export function denorm8(value: number, min: number, max: number): number {
  return (value / 255) * (max - min) + min;
}

export function denorm16(lower: number, upper: number, min: number, max: number): number {
  const raw = (upper << 8) | lower;
  return (raw / 65535) * (max - min) + min;
}
