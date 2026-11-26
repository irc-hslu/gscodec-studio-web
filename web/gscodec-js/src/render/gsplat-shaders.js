/**
 * Gaussian splatting (Jacobian projection) — ported from unityGsplat/Runtime/Shaders/Gsplat.hlsl
 * and Gsplat.shader (PlayCanvas gsplat chunks, MIT).
 */
export const GSPLAT_VS = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aCornerUV;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;
uniform int uInstanceOffset;

layout(std430, binding = 0) buffer SplatsPos { float posData[]; };
layout(std430, binding = 1) buffer SplatsScale { float scaleData[]; };
layout(std430, binding = 2) buffer SplatsRot { float rotData[]; };
layout(std430, binding = 3) buffer SplatsColor { float colorData[]; };
layout(std430, binding = 4) buffer SplatsOrder { uint orderData[]; };

out vec2 vUV;
flat out vec4 vColor;

const vec4 discardVec = vec4(0.0, 0.0, 2.0, 1.0);
const float SH_C0 = 0.28209479177387814;

// HLSL float3x3(...) is row-major; GLSL mat3() is column-major.
// Quaternion is (w, x, y, z), matching unityGsplat Gsplat.hlsl.
void quatToRows(vec4 R, out vec3 r0, out vec3 r1, out vec3 r2) {
  vec4 R2 = R + R;
  float X = R2.x * R.w;
  vec4 Y = R2.y * R;
  vec4 Z = R2.z * R;
  float W = R2.w * R.w;
  float m00 = 1.0 - Z.z - W;
  float m01 = Y.z + X;
  float m02 = Y.w - Z.x;
  float m10 = Y.z - X;
  float m11 = 1.0 - Y.y - W;
  float m12 = Z.w + Y.x;
  float m20 = Y.w + Z.x;
  float m21 = Z.w - Y.x;
  float m22 = 1.0 - Y.y - Z.z;
  r0 = vec3(m00, m01, m02);
  r1 = vec3(m10, m11, m12);
  r2 = vec3(m20, m21, m22);
}

void calcCovariance(vec4 quat, vec3 scale, out vec3 covA, out vec3 covB) {
  vec3 rot0, rot1, rot2;
  quatToRows(quat, rot0, rot1, rot2);
  // HLSL: M = transpose(float3x3(scale.x * rot[0], scale.y * rot[1], scale.z * rot[2]))
  vec3 row0 = vec3(scale.x * rot0.x, scale.y * rot1.x, scale.z * rot2.x);
  vec3 row1 = vec3(scale.x * rot0.y, scale.y * rot1.y, scale.z * rot2.y);
  vec3 row2 = vec3(scale.x * rot0.z, scale.y * rot1.z, scale.z * rot2.z);
  covA = vec3(dot(row0, row0), dot(row0, row1), dot(row0, row2));
  covB = vec3(dot(row1, row1), dot(row1, row2), dot(row2, row2));
}

bool initCorner(vec2 cornerUV, vec3 covA, vec3 covB, vec3 viewCenter, vec4 centerProj, float proj00, mat3 viewRot, out vec2 outUV, out vec2 outOffset) {
  // Symmetric world covariance, same layout as HLSL Vrk.
  mat3 Vrk = mat3(
    vec3(covA.x, covA.y, covA.z),
    vec3(covA.y, covB.x, covB.y),
    vec3(covA.z, covB.y, covB.z)
  );
  float focal = uViewport.x * proj00;
  vec3 v = viewCenter;
  float J1 = focal / v.z;
  vec2 J2 = -J1 / v.z * v.xy;
  // HLSL J rows: (J1, 0, J2.x), (0, J1, J2.y), (0, 0, 0)
  mat3 J = mat3(
    vec3(J1, 0.0, 0.0),
    vec3(0.0, J1, 0.0),
    vec3(J2.x, J2.y, 0.0)
  );
  mat3 T = J * viewRot;
  mat3 cov = T * Vrk * transpose(T);

  float c00 = cov[0][0];
  float c01 = cov[1][0];
  float c11 = cov[1][1];
  float diagonal1 = c00 + 0.3;
  float offDiagonal = c01;
  float diagonal2 = c11 + 0.3;
  float mid = 0.5 * (diagonal1 + diagonal2);
  float radius = length(vec2((diagonal1 - diagonal2) * 0.5, offDiagonal));
  float lambda1 = mid + radius;
  float lambda2 = max(mid - radius, 0.1);
  float vmin = min(1024.0, min(uViewport.x, uViewport.y));
  float l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
  float l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);
  if (l1 < 2.0 && l2 < 2.0) return false;

  vec2 clipScale = centerProj.ww / uViewport;
  float maxL = max(l1, l2);
  if (any(greaterThan(abs(centerProj.xy) - vec2(maxL) * clipScale, centerProj.ww))) return false;

  vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
  vec2 v1 = l1 * diagonalVector;
  vec2 v2 = l2 * vec2(diagonalVector.y, -diagonalVector.x);
  outOffset = (cornerUV.x * v1 + cornerUV.y * v2) * clipScale;
  outUV = cornerUV;
  return true;
}

void main() {
  uint id = orderData[uint(gl_InstanceID) + uint(uInstanceOffset)];
  uint bi = id * 3u;
  vec3 modelCenter = vec3(posData[bi], posData[bi + 1u], posData[bi + 2u]);
  vec4 centerView = uView * vec4(modelCenter, 1.0);
  if (centerView.z > 0.0) {
    gl_Position = discardVec;
    return;
  }
  vec4 centerProj = uProj * centerView;
  centerProj.z = clamp(centerProj.z, -abs(centerProj.w), abs(centerProj.w));
  vec3 viewCenter = centerView.xyz / centerView.w;

  uint ri = id * 4u;
  vec4 quat = vec4(rotData[ri], rotData[ri + 1u], rotData[ri + 2u], rotData[ri + 3u]);
  vec3 scale = vec3(scaleData[bi], scaleData[bi + 1u], scaleData[bi + 2u]);
  vec3 covA, covB;
  calcCovariance(quat, scale, covA, covB);

  vec2 cornerOffset, cornerUV;
  float proj00 = uProj[0][0];
  if (!initCorner(aCornerUV, covA, covB, viewCenter, centerProj, proj00, mat3(uView), cornerUV, cornerOffset)) {
    gl_Position = discardVec;
    return;
  }

  uint ci = id * 4u;
  vec4 color = vec4(colorData[ci], colorData[ci + 1u], colorData[ci + 2u], colorData[ci + 3u]);
  color.rgb = color.rgb * SH_C0 + 0.5;
  color.rgb = max(color.rgb, vec3(0.0));

  vColor = color;
  vUV = cornerUV;
  gl_Position = centerProj + vec4(cornerOffset.x, cornerOffset.y, 0.0, 0.0);
}
`;

/** Fallback VS without SSBO (WebGL2 core only) — uses instanced attributes. */
export const GSPLAT_VS_INSTANCED = `#version 300 es
precision highp float;

layout(location = 0) in vec2 aCornerUV;
layout(location = 1) in vec3 aCenter;
layout(location = 2) in vec3 aScale;
layout(location = 3) in vec4 aRot;
layout(location = 4) in vec4 aColor;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;

out vec2 vUV;
flat out vec4 vColor;

const vec4 discardVec = vec4(0.0, 0.0, 2.0, 1.0);
const float SH_C0 = 0.28209479177387814;

// HLSL float3x3(...) is row-major; GLSL mat3() is column-major.
// Quaternion is (w, x, y, z), matching unityGsplat Gsplat.hlsl.
void quatToRows(vec4 R, out vec3 r0, out vec3 r1, out vec3 r2) {
  vec4 R2 = R + R;
  float X = R2.x * R.w;
  vec4 Y = R2.y * R;
  vec4 Z = R2.z * R;
  float W = R2.w * R.w;
  float m00 = 1.0 - Z.z - W;
  float m01 = Y.z + X;
  float m02 = Y.w - Z.x;
  float m10 = Y.z - X;
  float m11 = 1.0 - Y.y - W;
  float m12 = Z.w + Y.x;
  float m20 = Y.w + Z.x;
  float m21 = Z.w - Y.x;
  float m22 = 1.0 - Y.y - Z.z;
  r0 = vec3(m00, m01, m02);
  r1 = vec3(m10, m11, m12);
  r2 = vec3(m20, m21, m22);
}

void calcCovariance(vec4 quat, vec3 scale, out vec3 covA, out vec3 covB) {
  vec3 rot0, rot1, rot2;
  quatToRows(quat, rot0, rot1, rot2);
  // HLSL: M = transpose(float3x3(scale.x * rot[0], scale.y * rot[1], scale.z * rot[2]))
  vec3 row0 = vec3(scale.x * rot0.x, scale.y * rot1.x, scale.z * rot2.x);
  vec3 row1 = vec3(scale.x * rot0.y, scale.y * rot1.y, scale.z * rot2.y);
  vec3 row2 = vec3(scale.x * rot0.z, scale.y * rot1.z, scale.z * rot2.z);
  covA = vec3(dot(row0, row0), dot(row0, row1), dot(row0, row2));
  covB = vec3(dot(row1, row1), dot(row1, row2), dot(row2, row2));
}

bool initCorner(vec2 cornerUV, vec3 covA, vec3 covB, vec3 viewCenter, vec4 centerProj, float proj00, mat3 viewRot, out vec2 outUV, out vec2 outOffset) {
  // Symmetric world covariance, same layout as HLSL Vrk.
  mat3 Vrk = mat3(
    vec3(covA.x, covA.y, covA.z),
    vec3(covA.y, covB.x, covB.y),
    vec3(covA.z, covB.y, covB.z)
  );
  float focal = uViewport.x * proj00;
  vec3 v = viewCenter;
  float J1 = focal / v.z;
  vec2 J2 = -J1 / v.z * v.xy;
  // HLSL J rows: (J1, 0, J2.x), (0, J1, J2.y), (0, 0, 0)
  mat3 J = mat3(
    vec3(J1, 0.0, 0.0),
    vec3(0.0, J1, 0.0),
    vec3(J2.x, J2.y, 0.0)
  );
  mat3 T = J * viewRot;
  mat3 cov = T * Vrk * transpose(T);

  float c00 = cov[0][0];
  float c01 = cov[1][0];
  float c11 = cov[1][1];
  float diagonal1 = c00 + 0.3;
  float offDiagonal = c01;
  float diagonal2 = c11 + 0.3;
  float mid = 0.5 * (diagonal1 + diagonal2);
  float radius = length(vec2((diagonal1 - diagonal2) * 0.5, offDiagonal));
  float lambda1 = mid + radius;
  float lambda2 = max(mid - radius, 0.1);
  float vmin = min(1024.0, min(uViewport.x, uViewport.y));
  float l1 = 2.0 * min(sqrt(2.0 * lambda1), vmin);
  float l2 = 2.0 * min(sqrt(2.0 * lambda2), vmin);
  if (l1 < 2.0 && l2 < 2.0) return false;

  vec2 clipScale = centerProj.ww / uViewport;
  float maxL = max(l1, l2);
  if (any(greaterThan(abs(centerProj.xy) - vec2(maxL) * clipScale, centerProj.ww))) return false;

  vec2 diagonalVector = normalize(vec2(offDiagonal, lambda1 - diagonal1));
  vec2 v1 = l1 * diagonalVector;
  vec2 v2 = l2 * vec2(diagonalVector.y, -diagonalVector.x);
  outOffset = (cornerUV.x * v1 + cornerUV.y * v2) * clipScale;
  outUV = cornerUV;
  return true;
}

void main() {
  vec4 centerView = uView * vec4(aCenter, 1.0);
  if (centerView.z > 0.0) {
    gl_Position = discardVec;
    return;
  }
  vec4 centerProj = uProj * centerView;
  centerProj.z = clamp(centerProj.z, -abs(centerProj.w), abs(centerProj.w));
  vec3 viewCenter = centerView.xyz / centerView.w;

  vec3 covA, covB;
  calcCovariance(aRot, aScale, covA, covB);
  vec2 cornerOffset, cornerUV;
  if (!initCorner(aCornerUV, covA, covB, viewCenter, centerProj, uProj[0][0], mat3(uView), cornerUV, cornerOffset)) {
    gl_Position = discardVec;
    return;
  }

  vec4 color = aColor;
  color.rgb = color.rgb * SH_C0 + 0.5;
  color.rgb = max(color.rgb, vec3(0.0));
  vColor = color;
  vUV = cornerUV;
  gl_Position = centerProj + vec4(cornerOffset.x, cornerOffset.y, 0.0, 0.0);
}
`;

export const GSPLAT_FS = `#version 300 es
precision highp float;
in vec2 vUV;
flat in vec4 vColor;
out vec4 fragColor;

void main() {
  float A = dot(vUV, vUV);
  if (A > 1.0) discard;
  float alpha = exp(-A * 4.0) * vColor.a;
  if (alpha < 1.0 / 255.0) discard;
  fragColor = vec4(vColor.rgb * alpha, alpha);
}
`;
