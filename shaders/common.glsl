// Shared helpers for the app's shaders. Pixels live in storage buffers as packed RGBA8
// (r | g << 8 | b << 16 | a << 24), the same layout the NR kernels read and write.

vec4 unpackPixel(uint p) {
  return vec4(float(p & 255u), float((p >> 8) & 255u), float((p >> 16) & 255u), float(p >> 24)) * (1.0 / 255.0);
}

uint packPixel(vec4 c) {
  uvec4 u = uvec4(clamp(c, 0.0, 1.0) * 255.0 + 0.5);
  return u.r | (u.g << 8) | (u.b << 16) | (u.a << 24);
}

// Catmull-Rom weights for the 4 taps around a fractional position t in [0,1).
vec4 catmullRom(float t) {
  float t2 = t * t, t3 = t2 * t;
  return vec4(-0.5 * t3 + t2 - 0.5 * t,
              1.5 * t3 - 2.5 * t2 + 1.0,
              -1.5 * t3 + 2.0 * t2 + 0.5 * t,
              0.5 * t3 - 0.5 * t2);
}
