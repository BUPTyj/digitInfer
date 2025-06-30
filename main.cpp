// main.cpp
#include <chrono>

#include "nets.hpp"
#include <iostream>
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

int main() {
    const Net net = load_net();
    std::cout << "ģ�ͼ�����ɣ������� (0-" << net.n_imgs-1
              << "), �����˳�\n";
    while (true) {
        int idx; std::cout << "index = "; std::cin >> idx;
        if (idx < 0) break;
        if (idx >= static_cast<int>(net.n_imgs)){ std::cout<<"����Χ\n"; continue; }

        const float* img = net.images[idx].data();
        const size_t pred = net.predict(img);
        std::cout << "Ԥ����: " << pred << "\n";
        std::cout << "ʵ�ʽ��: " << net.img_type[idx] << "\n\n";

        ascii_show(img);
    }
}
