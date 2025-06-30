__kernel void matrix_vector_multiply_sigmoid(
    __global const float* input,        // 输入向量
    __global const float* weights,      // 权重矩阵 (行优先)
    __global const float* bias,         // 偏置
    __global float*       output,       // 输出
    const int             input_size,
    const int             output_size)
{
    int row = get_global_id(0);
    if (row >= output_size) return;

    float sum = bias[row];
    for (int col = 0; col < input_size; ++col)
        sum += weights[row * input_size + col] * input[col];

    // Sigmoid
    output[row] = 1.0f / (1.0f + exp(-sum));
}

// ──────────────────────────
// 并行 argmax（局部内存版）
__kernel void find_argmax(
    __global const float* input,
    __global       int*   result,
    __local        float* local_values,
    __local        int*   local_indices,
    const int             size)
{
    int lid   = get_local_id(0);
    int gsize = get_local_size(0);
    int gid   = get_global_id(0);

    if (gid < size) {
        local_values[lid]  = input[gid];
        local_indices[lid] = gid;
    } else {
        local_values[lid]  = -INFINITY;
        local_indices[lid] = -1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 归约
    for (int stride = gsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            if (local_values[lid + stride] > local_values[lid]) {
                local_values[lid]  = local_values[lid + stride];
                local_indices[lid] = local_indices[lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) result[get_group_id(0)] = local_indices[0];
}

// ──────────────────────────
// 简化版 argmax（小数组）
__kernel void simple_argmax(
    __global const float* input,
    __global       int*   result,
    const int             size)
{
    if (get_global_id(0) != 0) return;

    int   best_idx = 0;
    float best_val = input[0];

    for (int i = 1; i < size; ++i) {
        if (input[i] > best_val) {
            best_val = input[i];
            best_idx = i;
        }
    }
    result[0] = best_idx;
}
