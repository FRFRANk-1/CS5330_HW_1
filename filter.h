#ifndef FILTER_H
#define FILTER_H

#include <opencv2/opencv.hpp>

#include "Blur_filter.h"

int greyscale(cv::Mat &src, cv::Mat &dst);

int sepiaTone(cv::Mat &src, cv::Mat &dst);

int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);

int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);

int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);

int colorPopEffect(cv::Mat &src, cv::Mat &dst);
int embossingEffect(cv::Mat &src, cv::Mat &dst);

int adjustBrightnessContrast(cv::Mat &src, cv::Mat &dst, int brightness, int contrast);

int addSparklesToEdges(cv::Mat &src, cv::Mat &edges, const std::vector<cv::Rect> &faces);
int overlayImage(cv::Mat &background, cv::Mat &foreground, cv::Mat &output);

#endif
