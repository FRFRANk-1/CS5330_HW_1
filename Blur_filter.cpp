/*
a commonly used library for handling input and output operations,  as reading from and writing to the console.
*/ 
#include <iostream> 

/*
is line includes the OpenCV (Open Source Computer Vision) library, 
specifically the main header file, <opencv2/opencv.hpp>. 
OpenCV is a popular open-source computer vision and image processing library. 
It provides a wide range of functions and tools for image and video analysis, 
as well as computer vision tasks.
*/
#include <opencv2/opencv.hpp>
#include "Blur_filter.h"

int blur5x5_1(cv :: Mat &src, cv:: Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    dst = src.clone();

    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    
    int kernelSum = 5; // sum of all kernel elements

    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 2; x < src.cols -2; x++) {
            cv :: Vec3b sum = cv :: Vec3b(0, 0, 0);
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    for (int c = 0; c < 3; c++) {
                        sum[c] += pixel[c] * kernel[ky + 2][kx + 2];
                    }
                }
            }
            for (int c = 0; c < 3; c++) {
                dst.at<cv::Vec3b>(y, x)[c] = sum[c] / kernelSum;
            }
        }
    }
    return 0;
}

int blur5x5_2(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    cv::Mat temp = src.clone();
    dst = src.clone();

    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            cv::Vec3f sum = cv::Vec3f(0,0,0);

            for (int k = -2; k <= 2; k++) {
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y)[x + k];
                for (int c = 0; c < 3; c++) {
                    sum[c] += pixel[c] * kernel[k + 2];
                }
            }

            for (int c = 0; c < 3; c++) {
                temp.ptr<cv::Vec3b>(y)[x][c] = static_cast<uchar>(sum[c] / kernelSum);
            }
        }
    }

    // Vertical pass
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3f sum = cv::Vec3f(0,0,0);

            for (int k = -2; k <= 2; k++) {
                cv::Vec3b pixel = temp.ptr<cv::Vec3b>(y + k)[x];
                for (int c = 0; c < 3; c++) {
                    sum[c] += pixel[c] * kernel[k + 2];
                }
            }

            for (int c = 0; c < 3; c++) {
                dst.ptr<cv::Vec3b>(y)[x][c] = static_cast<uchar>(sum[c] / kernelSum);
            }
        }
    }
    return 0;
}
