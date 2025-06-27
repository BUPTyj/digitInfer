// NeuralNetworkOpenCL.cpp
#include <CL/cl.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>      // rand()
#include <random>

#include "cnpy.h"
#include "nets.hpp"

#define KERNEL_FILE "neural_network_kernels.cl"

class NeuralNetworkOpenCL {
    // OpenCL 资源
    cl_platform_id platform{};
    cl_device_id   device{};
    cl_context     context{};
    cl_command_queue command_queue{};
    cl_program     program{};
    cl_kernel      kernel_forward{};
    cl_kernel      kernel_argmax{};

    // 缓冲区
    cl_mem buffer_input{};
    cl_mem buffer_hidden{};
    cl_mem buffer_output{};
    cl_mem buffer_weights_ih{};
    cl_mem buffer_weights_ho{};
    cl_mem buffer_bias_h{};
    cl_mem buffer_bias_o{};
    cl_mem buffer_result{};

    // 网络维度
    static constexpr int INPUT_SIZE  = 784;
    static constexpr int HIDDEN_SIZE = 20;
    static constexpr int OUTPUT_SIZE = 10;

public:
    NeuralNetworkOpenCL() {
        initializeOpenCL();
        createKernels();
        allocateGPUMemory();
    }
    ~NeuralNetworkOpenCL() { releaseResources(); }

    // 加载模型
    void loadModel(const std::vector<float>& w_ih,
                   const std::vector<float>& w_ho,
                   const std::vector<float>& b_h,
                   const std::vector<float>& b_o) const {
        if (w_ih.size() != INPUT_SIZE * HIDDEN_SIZE ||
            w_ho.size() != HIDDEN_SIZE * OUTPUT_SIZE ||
            b_h.size()  != HIDDEN_SIZE ||
            b_o.size()  != OUTPUT_SIZE)
            throw std::runtime_error("模型参数维度不匹配");

        auto wc = [&](const cl_mem& buf, const void* src, const size_t bytes,
                      const char* msg) {
            const cl_int err = clEnqueueWriteBuffer(command_queue, buf, CL_TRUE, 0,
                                              bytes, src, 0, nullptr, nullptr);
            check(err, msg);
        };
        wc(buffer_weights_ih, w_ih.data(), w_ih.size() * sizeof(float),
           "写入输入-隐藏权重失败");
        wc(buffer_weights_ho, w_ho.data(), w_ho.size() * sizeof(float),
           "写入隐藏-输出权重失败");
        wc(buffer_bias_h, b_h.data(), b_h.size() * sizeof(float),
           "写入隐藏层偏置失败");
        wc(buffer_bias_o, b_o.data(), b_o.size() * sizeof(float),
           "写入输出层偏置失败");
    }

    // 前向推理
    size_t predict(const float* image) {
        const cl_int err =
            clEnqueueWriteBuffer(command_queue, buffer_input, CL_FALSE, 0,
                                 INPUT_SIZE * sizeof(float), image, 0, nullptr,
                                 nullptr);
        check(err, "写入输入图像失败");

        // layer 1
        forward(buffer_input, buffer_weights_ih, buffer_bias_h, buffer_hidden,
                INPUT_SIZE, HIDDEN_SIZE);
        // layer 2
        forward(buffer_hidden, buffer_weights_ho, buffer_bias_o, buffer_output,
                HIDDEN_SIZE, OUTPUT_SIZE);

        return static_cast<size_t>(
            argmax(buffer_output, OUTPUT_SIZE));  // 预测类别
    }

private:
    static std::string readKernelFile(const char* path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) throw std::runtime_error("无法打开内核文件: " + std::string(path));
        return { (std::istreambuf_iterator<char>(ifs)),
                 std::istreambuf_iterator<char>() };
    }

    void initializeOpenCL() {
        cl_int err = clGetPlatformIDs(1, &platform, nullptr);
        check(err, "获取平台失败");

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {  // GPU 不可用，回退 CPU
            std::cout << "GPU 不可用，回退到 CPU\n";
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device,
                                 nullptr);
        }
        check(err, "获取设备失败");

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        check(err, "创建上下文失败");

        command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        check(err, "创建命令队列失败");

        // 读取并编译内核文件
        const std::string src = readKernelFile(KERNEL_FILE);
        const char* src_ptr   = src.c_str();
        const size_t      src_len   = src.size();

        program = clCreateProgramWithSource(context, 1, &src_ptr, &src_len, &err);
        check(err, "创建 Program 失败");

        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) { printBuildLog(); throw std::runtime_error("编译失败"); }
    }

    void createKernels() {
        cl_int e;
        kernel_forward = clCreateKernel(program, "matrix_vector_multiply_sigmoid", &e);
        check(e, "创建 forward kernel 失败");
        kernel_argmax  = clCreateKernel(program, "simple_argmax", &e);
        check(e, "创建 argmax kernel 失败");
    }

    void allocateGPUMemory() {
        cl_int e;
        auto mk = [&](cl_mem& buf, const cl_mem_flags f, const size_t sz, const char* msg) {
            buf = clCreateBuffer(context, f, sz, nullptr, &e);
            check(e, msg);
        };
        mk(buffer_input,  CL_MEM_READ_ONLY,
           INPUT_SIZE * sizeof(float), "分配 input 失败");
        mk(buffer_hidden, CL_MEM_READ_WRITE,
           HIDDEN_SIZE * sizeof(float), "分配 hidden 失败");
        mk(buffer_output, CL_MEM_READ_WRITE,
           OUTPUT_SIZE * sizeof(float), "分配 output 失败");
        mk(buffer_weights_ih, CL_MEM_READ_ONLY,
           INPUT_SIZE * HIDDEN_SIZE * sizeof(float), "分配 w_ih 失败");
        mk(buffer_weights_ho, CL_MEM_READ_ONLY,
           HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), "分配 w_ho 失败");
        mk(buffer_bias_h,   CL_MEM_READ_ONLY,
           HIDDEN_SIZE * sizeof(float), "分配 b_h 失败");
        mk(buffer_bias_o,   CL_MEM_READ_ONLY,
           OUTPUT_SIZE * sizeof(float), "分配 b_o 失败");
        mk(buffer_result,   CL_MEM_WRITE_ONLY,
           sizeof(int), "分配 result 失败");
    }

    void forward(cl_mem in, cl_mem w, cl_mem b, cl_mem out,
                 int in_sz, int out_sz) const {
        cl_int e = 0;
        e |= clSetKernelArg(kernel_forward, 0, sizeof(cl_mem), &in);
        e |= clSetKernelArg(kernel_forward, 1, sizeof(cl_mem), &w);
        e |= clSetKernelArg(kernel_forward, 2, sizeof(cl_mem), &b);
        e |= clSetKernelArg(kernel_forward, 3, sizeof(cl_mem), &out);
        e |= clSetKernelArg(kernel_forward, 4, sizeof(int),   &in_sz);
        e |= clSetKernelArg(kernel_forward, 5, sizeof(int),   &out_sz);
        check(e, "设置 forward 参数失败");

        const auto gws = static_cast<size_t>(out_sz);
        constexpr size_t local_work_size = 16;
        const size_t global_work_size = ((out_sz + local_work_size - 1) / local_work_size) * local_work_size;
        e = clEnqueueNDRangeKernel(command_queue, kernel_forward, 1, nullptr,
                                   &global_work_size, &local_work_size, 0, nullptr, nullptr);
        // e = clEnqueueNDRangeKernel(command_queue, kernel_forward, 1, nullptr,
        //                            &gws, nullptr, 0, nullptr, nullptr);
        check(e, "执行 forward 内核失败");
    }

    int argmax(cl_mem in, int size) const {
        cl_int e = 0;
        e |= clSetKernelArg(kernel_argmax, 0, sizeof(cl_mem), &in);
        e |= clSetKernelArg(kernel_argmax, 1, sizeof(cl_mem), &buffer_result);
        e |= clSetKernelArg(kernel_argmax, 2, sizeof(int),    &size);
        check(e, "设置 argmax 参数失败");

        constexpr size_t gws = 1;
        e = clEnqueueNDRangeKernel(command_queue, kernel_argmax, 1, nullptr,
                                   &gws, nullptr, 0, nullptr, nullptr);
        check(e, "执行 argmax 失败");

        int res;
        e = clEnqueueReadBuffer(command_queue, buffer_result, CL_TRUE, 0,
                                sizeof(int), &res, 0, nullptr, nullptr);
        check(e, "读取 argmax 结果失败");
        return res;
    }

    // 错误检查 & 日志
    static void check(const cl_int err, const char* msg) {
        if (err != CL_SUCCESS) {
            std::cerr << msg << " (Error " << err << ")\n";
            throw std::runtime_error(msg);
        }
    }
    void printBuildLog() const {
        size_t sz;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &sz);
        std::string log(sz, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sz, log.data(), nullptr);
        std::cerr << "Build Log:\n" << log << std::endl;
    }

    void releaseResources() const {
        auto rl = [](auto x) { if (x) clReleaseMemObject(x); };
        rl(buffer_input); rl(buffer_hidden); rl(buffer_output);
        rl(buffer_weights_ih); rl(buffer_weights_ho);
        rl(buffer_bias_h); rl(buffer_bias_o); rl(buffer_result);
        if (kernel_forward) clReleaseKernel(kernel_forward);
        if (kernel_argmax)  clReleaseKernel(kernel_argmax);
        if (program)        clReleaseProgram(program);
        if (command_queue)  clReleaseCommandQueue(command_queue);
        if (context)        clReleaseContext(context);
    }
};

static constexpr char shades[] = " .:-=+*#%@";

void ascii_show(const float* img) {
    for (int r=0;r<28;++r){
        for (int c=0;c<28;++c){
            const float val = img[r*28+c];          // 0~1
            const int idx = static_cast<int>(std::lround(val * 9.0f));
            std::cout << shades[idx];
        }
        std::cout << std::endl;
    }
}

// 矩阵运算优化方法：把要算的大数据切成小块，先由同一 work-group 的多个 work-item 把这一块搬进共享内存并同步，随后组内线程反复复用这块数据并并行计算，从而大幅减少全局显存访问、提升效率
int main() {
    try {
        std::cout << "初始化 OpenCL 神经网络...\n";
        NeuralNetworkOpenCL net;

        auto npz_para = cnpy::npz_load("model.npz");
        auto npz_img = cnpy::npz_load("mnist.npz");

        const std::vector<float> w_i_h = to_vec(npz_para["w_i_h"]);
        const std::vector<float> b_i_h = to_vec(npz_para["b_i_h"]);
        const std::vector<float> w_h_o = to_vec(npz_para["w_h_o"]);
        const std::vector<float> b_h_o = to_vec(npz_para["b_h_o"]);

        net.loadModel(w_i_h, w_h_o, b_i_h, b_h_o);

        const auto& imgs = npz_img["x_test"];
        std::cout << "数据导入完成" << std::endl;
        const std::size_t N = imgs.shape[0];
        constexpr std::size_t PITCH = 28 * 28;
        const auto* src = imgs.data<uint8_t>();
        std::vector<std::array<float, 28 * 28>>    images;
        images.resize(N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < PITCH; ++j) {
                images[i][j] = static_cast<float>(src[i * PITCH + j]) / 255.0f;
            }
        }

        const auto& img_types = npz_img["y_test"];
        const auto* type = img_types.data<uint8_t>();
        std::vector<float> img_type;
        img_type.resize(N);
        for (std::size_t i = 0; i < N; ++i) {
            img_type[i] = static_cast<float>(type[i]);
        }

        using clock = std::chrono::steady_clock;
        std::cout << "模型加载完成！输入编号 (0-" << imgs.shape[0]-1
              << "), 负数退出" << std::endl;
        while (true) {
            int idx; std::cout << "index = "; std::cin >> idx;
            if (idx < 0) break;
            if (idx >= N){ std::cout<<"超范围\n"; continue; }
            const auto t0 = clock::now();
            const float* img_data = images[idx].data();
            const size_t pred = net.predict(img_data);
            const auto t1 = clock::now();
            const std::chrono::duration<double, std::milli> dt = t1 - t0;
            std::cout << "预测结果: " << pred << "\n";
            std::cout << "实际结果: " << img_type[idx] << "\n";
            std::cout << "运行时间: " << dt.count() << " ms\n\n";

            ascii_show(img_data);
        }
    //     std::random_device rd;
    //     std::mt19937 gen(rd());
    //     std::uniform_int_distribution<int> dis(0, N - 1);
    //
    //     const int test_count = 1000;
    //     double total_time = 0.0;
    //     int correct_predictions = 0;
    //
    //     std::cout << "开始自动基准测试，运行 " << test_count << " 次...\n";
    //     std::cout << "进度: ";
    //
    //     for (int i = 0; i < test_count; ++i) {
    //         // 显示进度
    //         if (i % 100 == 0) {
    //             std::cout << i << "/" << test_count << " ";
    //             std::cout.flush();
    //         }
    //
    //         // 随机选择索引
    //         int idx = dis(gen);
    //
    //         // 测量预测时间
    //         const auto t0 = clock::now();
    //         const float* img_data = images[idx].data();
    //         const size_t pred = net.predict(img_data);
    //         const auto t1 = clock::now();
    //
    //         // 计算时间
    //         const std::chrono::duration<double, std::milli> dt = t1 - t0;
    //         total_time += dt.count();
    //
    //         // 统计正确率
    //         if (pred == img_type[idx]) {
    //             correct_predictions++;
    //         }
    //     }
    //
    //     // 输出结果
    //     std::cout << "\n\n========== 基准测试结果 ==========\n";
    //     std::cout << "测试次数: " << test_count << "\n";
    //     std::cout << "总运行时间: " << std::fixed << std::setprecision(3) << total_time << " ms\n";
    //     std::cout << "平均运行时间: " << std::fixed << std::setprecision(3) << total_time / test_count << " ms\n";
    //     std::cout << "预测正确率: " << std::fixed << std::setprecision(2) << (double)correct_predictions / test_count * 100 << "%\n";
    //     std::cout << "吞吐量: " << std::fixed << std::setprecision(1) << test_count / (total_time / 1000.0) << " 次/秒\n";
    //     std::cout << "================================\n\n";
    //
    } catch (const std::exception& ex) {
        std::cerr << "错误: " << ex.what() << '\n';
        return -1;
    }
    std::cout << "完成!\n";
    return 0;
}
