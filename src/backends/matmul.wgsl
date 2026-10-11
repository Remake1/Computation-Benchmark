struct Params { m: u32, n: u32, k: u32, groups_x: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) local: u32) {
  let i = (group.y * params.groups_x + group.x) * 64u + local;
  if (i >= params.m * params.n) { return; }
  let row = i / params.n;
  let col = i % params.n;
  var sum = 0.0;
  for (var k = 0u; k < params.k; k++) {
    sum = sum + a[row * params.k + k] * b[k * params.n + col];
  }
  out[i] = sum;
}
