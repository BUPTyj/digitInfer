//
// Created by JXMYJ on 25-6-25.
//

#ifndef NETS_HPP
#define NETS_HPP
#include <algorithm>
#include <cstring>

#include "cnpy/cnpy.h"

#endif //NETS_HPP
// nets.hpp
#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>

/* 将 cnpy::NpyArray 映射为 std::vector<float> */
inline std::vector<float> to_vec(const cnpy::NpyArray& a) {
    std::vector<float> v(a.num_vals);

    if (a.word_size == 4) {
        // float32 数据
        const auto* src = a.data<float>();
        std::memcpy(v.data(), src, a.num_vals * sizeof(float));
    } else if (a.word_size == 8) {
        // float64 数据，需要转换为 float32
        const auto* src = a.data<double>();
        std::transform(src, src + a.num_vals, v.begin(),
                       [](const double x) { return static_cast<float>(x); });
    } else {
        throw std::runtime_error("Unsupported dtype: word_size = " + std::to_string(a.word_size));
    }

    return v;
}

struct Net {
    std::vector<float> w_i_h; // 20×784
    std::vector<float> b_i_h; // 20
    std::vector<float> w_h_o; // 10×20
    std::vector<float> b_h_o; // 10
    std::vector<std::array<float, 28 * 28>>    images; // 10000*28*28
    std::vector<float> img_type; // 10000
    size_t             n_imgs = 0;

    size_t predict(const float* img /* 784 elems */) const {
        // ---- 输入层 -> 隐藏层
        std::array<float, 20> h_pre{}, h{};
        for (int i = 0; i < 20; ++i) {
            float sum = b_i_h[i];
            // w_i_h 真实 shape = (784, 20)  —— 每 20 个数是一行，对应同一像素连到 20 个隐层
            for (int j = 0; j < 784; ++j) {
                sum += w_i_h[i * 784 + j] * img[j];   // ← 关键改动
            }
            h_pre[i] = sum;
            h[i]     = 1.f / (1.f + std::exp(-sum));  // sigmoid
        }

        // ---- 隐藏层 -> 输出层
        std::array<float, 10> o_pre{}, o{};
        for (int i = 0; i < 10; i++) {
            float sum = b_h_o[i];
            // w_h_o 真实 shape = (20, 10)
            for (int j = 0; j < 20; ++j) {
                sum += w_h_o[i * 20 + j] * h[j];      // ← 关键改动
            }
            o_pre[i] = sum;
            o[i]     = 1.f / (1.f + std::exp(-sum));
        }

        // ---- argmax
        size_t best = 0;
        for (size_t i = 1; i < 10; ++i)
            if (o[i] > o[best]) best = i;
        return best;
    }
};

/* 从 model.npz 加载网络 */
inline Net load_net() {
    auto npz_para = cnpy::npz_load("model.npz");
    auto npz_img = cnpy::npz_load("mnist.npz");
    Net net;

    net.w_i_h = to_vec(npz_para["w_i_h"]);
    net.b_i_h = to_vec(npz_para["b_i_h"]);
    net.w_h_o = to_vec(npz_para["w_h_o"]);
    net.b_h_o = to_vec(npz_para["b_h_o"]);

    const auto& imgs = npz_img["x_test"];
    const auto& img_types = npz_img["y_test"];

    if (imgs.shape.size()!=3 || imgs.shape[1]!=28 || imgs.shape[2]!=28)
        throw std::runtime_error("images shape mismatch");
    const std::size_t N = imgs.shape[0];          // 样本数
    constexpr std::size_t PITCH = 28 * 28;
    net.images.resize(N);                   // 分配 N 行
    net.img_type.resize(N);                 // 分配 N 行
    net.n_imgs = N;

    const auto* src = imgs.data<uint8_t>();
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < PITCH; ++j) {
            net.images[i][j] = static_cast<float>(src[i * PITCH + j]) / 255.0f;
        }
    }

    const auto* type = img_types.data<uint8_t>();
    for (std::size_t i = 0; i < N; ++i) {
        net.img_type[i] = static_cast<float>(type[i]);
    }
    return net;
}

