struct Params { n: u32, b: u32, chunks: u32, groups_x: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> partials: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) local: u32) {
  let linear_group = group.y * params.groups_x + group.x;
  let batch = linear_group / params.chunks;
  let chunk = linear_group % params.chunks;
  let i = chunk * 64u + local;
  var value = 0.0;
  if (batch < params.b && i < params.n) { value = input[batch * params.n + i]; }
  partials[local] = value;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (local < stride) { partials[local] = partials[local] + partials[local + stride]; }
    workgroupBarrier();
  }
  if (local == 0u && batch < params.b) { output[batch * params.chunks + chunk] = partials[0]; }
}
