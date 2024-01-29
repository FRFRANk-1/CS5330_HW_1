// imageUtils.h
#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <opencv2/opencv.hpp>

int overlayImage(cv::Mat &background, cv::Mat &foreground, cv::Mat &output);

#endif // IMAGE_UTILS_H