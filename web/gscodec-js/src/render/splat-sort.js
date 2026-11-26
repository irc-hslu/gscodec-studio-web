/**
 * Back-to-front sort (farther splats drawn first for premultiplied alpha).
 * @param {Float32Array} positions
 * @param {Float32Array} view column-major 4x4
 * @param {number[]} splatIds original splat indices to sort
 */
export function sortSplatsBackToFront(positions, view, splatIds) {
  const ids = splatIds.slice();
  ids.sort((a, b) => {
    const za = viewSpaceZ(positions, view, a);
    const zb = viewSpaceZ(positions, view, b);
    // OpenGL view looks down -Z: farther splats have smaller (more negative) z.
    return za - zb;
  });
  return ids;
}

function viewSpaceZ(positions, view, i) {
  const x = positions[i * 3];
  const y = positions[i * 3 + 1];
  const z = positions[i * 3 + 2];
  return view[2] * x + view[6] * y + view[10] * z + view[14];
}
