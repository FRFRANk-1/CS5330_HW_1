// TimeBlur.cpp

#include "TimeBlur.h"
#include "Blur_filter.h" // Include the blur functions
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

// filename = "..\\..\\assets\\image\\cathedral.jpeg";

double getTime() {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    return value.count() / 1000.0;
}

void timeBlurProcessing(const std::string& filename) {
    cv::Mat src, dst;
    src = cv::imread(filename);

    if (src.empty()) {
        std::cerr << "Unable to read image " << filename << std::endl;
        return;
    }

    const int Ntimes = 10;
    double startTime, endTime, difference;

    // Timing for blur5x5_1
    startTime = getTime();
    for (int i = 0; i < Ntimes; i++) {
        blur5x5_1(src, dst);
    }
    endTime = getTime();
    difference = (endTime - startTime) / Ntimes;
    std::cout << "Time per image (1): " << difference << " seconds\n";

    // Timing for blur5x5_2
    startTime = getTime();
    for (int i = 0; i < Ntimes; i++) {
        blur5x5_2(src, dst);
    }
    endTime = getTime();
    difference = (endTime - startTime) / Ntimes;
    std::cout << "Time per image (2): " << difference << " seconds\n";
}
