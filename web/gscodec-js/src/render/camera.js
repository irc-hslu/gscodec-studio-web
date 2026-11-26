export function perspective(fovyRad, aspect, near, far) {
  const f = 1 / Math.tan(fovyRad / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

export function lookAt(eye, target, up) {
  const zx = eye[0] - target[0];
  const zy = eye[1] - target[1];
  const zz = eye[2] - target[2];
  const zlen = Math.hypot(zx, zy, zz) || 1;
  const z0 = zx / zlen, z1 = zy / zlen, z2 = zz / zlen;

  let x0 = up[1] * z2 - up[2] * z1;
  let x1 = up[2] * z0 - up[0] * z2;
  let x2 = up[0] * z1 - up[1] * z0;
  const xlen = Math.hypot(x0, x1, x2) || 1;
  x0 /= xlen; x1 /= xlen; x2 /= xlen;

  const y0 = z1 * x2 - z2 * x1;
  const y1 = z2 * x0 - z0 * x2;
  const y2 = z0 * x1 - z1 * x0;

  return new Float32Array([
    x0, y0, z0, 0,
    x1, y1, z1, 0,
    x2, y2, z2, 0,
    -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]),
    -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]),
    -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]),
    1,
  ]);
}

/** Column-major multiply: out = a * b */
export function multiply4x4(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      out[col * 4 + row] =
        a[row] * b[col * 4] +
        a[4 + row] * b[col * 4 + 1] +
        a[8 + row] * b[col * 4 + 2] +
        a[12 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

export class OrbitCamera {
  /**
   * @param {number[]} center
   * @param {number} radius
   */
  constructor(center, radius) {
    this.center = center;
    this.radius = radius;
    this.yaw = 0.6;
    this.pitch = 0.25;
    this.distance = radius * 2.5;
  }

  viewMatrix() {
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const eye = [
      this.center[0] + this.distance * cp * cy,
      this.center[1] + this.distance * sp,
      this.center[2] + this.distance * cp * sy,
    ];
    return lookAt(eye, this.center, [0, 1, 0]);
  }

  onPointer(dx, dy) {
    this.yaw += dx * 0.005;
    this.pitch = Math.max(-1.2, Math.min(1.2, this.pitch + dy * 0.005));
  }

  onWheel(delta) {
    this.distance *= 1 + delta * 0.001;
    this.distance = Math.max(this.radius * 0.2, Math.min(this.radius * 20, this.distance));
  }
}
