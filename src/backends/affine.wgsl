struct Params { n: u32, k: u32, groups_x: u32, padding: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) local: u32) {
  let i = (group.y * params.groups_x + group.x) * 64u + local;
  if (i >= params.n) { return; }
  var t = 0.5 * x[i] + y[i];
  for (var step = 0u; step < params.k; step++) {
    t = 0.99609375 * t + 0.001953125;
  }
  out[i] = t;
}
