// neural_network_kernels.cl
#define TILE_SIZE 16

// 一般kernel实现
;__kernel void matrix_vector_multiply_sigmoid(
;    __global const float* input,
;    __global const float* weights,
;    __global const float* bias,
;    __global float*       output,
;    const int             input_size,
;    const int             output_size)
;{
;    int row = get_global_id(0);
;    if (row >= output_size) return;
;
;    float sum = bias[row];
;    for (int col = 0; col < input_size; ++col)
;        sum += weights[row * input_size + col] * input[col];
;
;    // Sigmoid
;    output[row] = 1.0f / (1.0f + exp(-sum));
;}

// 优化kernel实现
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

    int local_id = get_local_id(0);

    // 局部内存缓存输入向量
    __local float local_input[TILE_SIZE];

    float sum = bias[row];

    // 分块处理输入向量
    int num_tiles = (input_size + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < num_tiles; ++tile) {
        int tile_start = tile * TILE_SIZE;
        int tile_end = min(tile_start + TILE_SIZE, input_size);

        // 协作加载数据到局部内存
        if (local_id < (tile_end - tile_start)) {
            local_input[local_id] = input[tile_start + local_id];
        }

        // 同步工作组
        barrier(CLK_LOCAL_MEM_FENCE);

        // 计算这个tile的贡献
        for (int col = 0; col < (tile_end - tile_start); ++col) {
            sum += weights[row * input_size + tile_start + col] * local_input[col];
        }

        // 同步工作组
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Sigmoid
    output[row] = 1.0f / (1.0f + exp(-sum));
}

// ──────────────────────────
// argmax
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
