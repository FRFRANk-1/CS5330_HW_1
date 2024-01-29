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
#include "imageUtils.h"


int greyscale(cv::Mat &src, cv::Mat &dst) {
    // Ensure the source image is not empty
    if (src.empty()) {
        return -1;
    }

    // Initialize the destination image
    dst = cv::Mat(src.size(), src.type());

    // Process each pixel of the source image
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // Get the pixel's BGR values
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // Apply a custom greyscale transformation (e.g., invert the red channel)
            uchar greyValue = 255 - pixel[2]; // Invert the red channel

            // Set the transformed value to all three channels (BGR) of the destination image
            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(greyValue, greyValue, greyValue);
        }
    }

    return 0; 
}

int sepiaTone(cv :: Mat &src, cv :: Mat &dst) {
    // Ensure the source image is not empty
    if (src.empty()) {
        return -1;
    }

    // Initialize the destination image
    dst = src.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {

            // Get the pixel's BGR values
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            double newRed   = std::min(255.0f, pixel[2] * 0.272f + pixel[1] * 0.534f + pixel[0] * 0.131f);
            double newGreen = std::min(255.0f, pixel[2] * 0.349f + pixel[1] * 0.686f + pixel[0] * 0.168f);
            double newBlue  = std::min(255.0f, pixel[2] * 0.393f + pixel[1] * 0.769f + pixel[0] * 0.189f);

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(newBlue), static_cast<uchar>(newGreen), static_cast<uchar>(newRed));
        }
    }
    return 0;
}

int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    // Check if source image is not empty
    if (src.empty()) {
        return -1;
    }

    // Define Sobel X kernel
    int kernel[3] = {-1, 0, 1};

    // Initialize destination image
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Apply Sobel X filter
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < 3; c++) { // Iterate over color channels
                short sum = 0;
                for (int k = -1; k <= 1; k++) {
                    sum += src.at<cv::Vec3b>(y, x + k)[c] * kernel[k + 1];
                }
                dst.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    return 0;
}

int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    // Check if source image is not empty
    if (src.empty()) {
        return -1;
    }

    // Define Sobel Y kernel
    int kernel[3] = {-1, 0, 1};

    // Initialize destination image
    dst = cv::Mat::zeros(src.size(), CV_16SC3);

    // Apply Sobel Y filter
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            for (int c = 0; c < 3; c++) { // Iterate over color channels
                short sum = 0;
                for (int k = -1; k <= 1; k++) {
                    sum += src.at<cv::Vec3b>(y + k, x)[c] * kernel[k + 1];
                }
                dst.at<cv::Vec3s>(y, x)[c] = sum;
            }
        }
    }

    return 0;
}

int magnitude(cv :: Mat &sx, cv :: Mat &sy, cv :: Mat &dst) {
    if (sx.empty() || sy.empty() || sx.size() != sy.size() || sx.type() != sy.type()) {
        return -1;  // Check for valid input
    }

    // Initialize destination image
    dst = cv::Mat(sx.size(), CV_8UC3);

    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            for (int c = 0; c < 3; c++) {  // Iterate over color channels
                // Compute the magnitude
                double mag = std::sqrt(std::pow(sx.at<cv::Vec3s>(y, x)[c], 2) + std::pow(sy.at<cv::Vec3s>(y, x)[c], 2));
                dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(mag);
            }
        }
    }

    return 0;
}

int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || levels <= 0) {
        return -1;
    }

    // First, apply blur to the source image
    cv::Mat blurred;
    blur5x5_1(src, blurred); // or use blur5x5_2

    // Initialize destination image
    dst = blurred.clone();

    // Calculate the size of each quantization bucket
    int bucketSize = 255 / levels;

    // Quantize each pixel
    for (int y = 0; y < blurred.rows; y++) {
        for (int x = 0; x < blurred.cols; x++) {
            for (int c = 0; c < 3; c++) { // Iterate over color channels
                uchar& pixel = dst.at<cv::Vec3b>(y, x)[c];
                pixel = static_cast<uchar>((pixel / bucketSize) * bucketSize);
            }
        }
    }

    return 0;
}

int colorPopEffect(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(grey, grey, cv::COLOR_GRAY2BGR); // Convert back to BGR format for easy pixel-wise operation

    // Initialize the destination image
    dst = src.clone();

    // Define the color range for red 
    int redLowerBound = 10; // Lower bound of red intensity
    int colorThreshold = 20; // Minimum difference between red and other channels

    // Process each pixel
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // Check if the pixel is not predominantly red
            if ((pixel[2] < redLowerBound) || (pixel[2] <= pixel[1] + colorThreshold) || (pixel[2] <= pixel[0] + colorThreshold)) {
                // Replace it with the corresponding greyscale value
                dst.at<cv::Vec3b>(y, x) = grey.at<cv::Vec3b>(y, x);
            }
        }
    } 
}

int embossingEffect(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Convert the image to greyscale
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Apply Sobel X and Y filters
    cv::Mat grad_x, grad_y;
    cv::Sobel(grey, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(grey, grad_y, CV_32F, 0, 1, 3);
    
    // cv::convertScaleAbs(grad_x, dst);
    // return 0;

    // Initialize the destination image

    // Embossing direction (can be adjusted)
    double embossDirection[2] = {0.7071f, 0.7071f}; // 45-degree angle

    // Compute the emboss effect
    cv::Mat emboss = cv::Mat::zeros(src.size(), CV_32F);
    
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float gx = grad_x.at<float>(y, x);
            float gy = grad_y.at<float>(y, x);

            float embossValue = gx * embossDirection[0] + gy * embossDirection[1];
            emboss.at<float>(y, x) = embossValue;
        }
    }

    // Normalize and convert to a displayable format
    cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
    emboss.convertTo(dst, CV_8UC1);
}

// int brightness = 0; // Range: [-100, 100]
// int contrast = 0;   // Range: [-100, 100]

int adjustBrightnessContrast(cv::Mat &src, cv::Mat &dst, int brightness, int contrast) {
    double alpha = (contrast + 100) / 100.0; // Contrast control
    int beta = brightness;                  // Brightness control

    src.convertTo(dst, -1, alpha, beta);    // Apply adjustments
    return 0;
}

int addSparklesToEdges(cv::Mat &src, cv::Mat &edges, const std::vector<cv::Rect> &faces) {
    std::cout << "test_import img step 1\n" << std::flush;
    
    // Load a sparkle image (preferably a PNG with an alpha channel)
    cv::Mat sparkle = cv::imread("..\\..\\assets\\image\\sparkle_img_3.png", cv::IMREAD_UNCHANGED); // Change file name/extension if necessary
    std::cout << "test_import img step_2 \n" << std::flush;

    if (sparkle.empty()) {
        std::cerr << "Error: Sparkle image not loaded correctly." << std::endl;
        return -1;
    }

    // Debugging: Print information about the loaded image
    std::cout << "Loaded image channels: " << sparkle.channels() << std::endl;

    // Check if the sparkle image has an alpha channel, if not, add one
    cv::Mat sparkle_with_alpha;
    if (sparkle.channels() == 3) {
        cv::Mat alpha_channel(sparkle.size(), CV_8UC1, cv::Scalar(255)); // Fully opaque alpha channel
        cv::Mat channels[] = {sparkle, alpha_channel};
        cv::merge(channels, 4, sparkle_with_alpha);
    } else if (sparkle.channels() == 4) {
        sparkle_with_alpha = sparkle;
    } else {
        std::cerr << "Error: Unexpected number of channels in sparkle image." << std::endl;
        return -1;
    }


    std::cout << "edges type: " << edges.type() << std::endl;
    std::cout << "edges depth: " << edges.depth() << ", channels: " << edges.channels() << std::endl;

    for (const auto& face : faces) {
        for (int y = face.y; y < face.y + face.height; ++y) {
            for (int x = face.x; x < face.x + face.width; ++x) {
                if (y >= edges.rows || x >= edges.cols) {
                    continue; // Check bounds for edges matrix
                }
                if (edges.at<float>(y, x) > 0) { // If it's an edge
                    cv::Rect roi(x, y, sparkle_with_alpha.cols, sparkle_with_alpha.rows);
                    if (x + roi.width > src.cols || y + roi.height > src.rows) {
                        continue; // Check bounds for roi
                    }
                    cv::Mat dstRoi = src(roi);
                    overlayImage(dstRoi, sparkle_with_alpha, dstRoi);
                }
            }
        }
    }
    return 0;
}


// Function to overlay an image with transparency over another image
int overlayImage(cv::Mat &background, cv::Mat &foreground, cv::Mat &output) {
    background.copyTo(output);
    for (int y = std::max(foreground.rows - output.rows, 0); y < foreground.rows; ++y) {
        for (int x = std::max(foreground.cols - output.cols, 0); x < foreground.cols; ++x) {
            double opacity = ((double)foreground.data[y * foreground.step + x * foreground.channels() + 3]) / 255.;
            for (int c = 0; opacity > 0 && c < output.channels(); ++c) {
                unsigned char foregroundPx = foreground.data[y * foreground.step + x * foreground.channels() + c];
                unsigned char backgroundPx = background.data[y * background.step + x * background.channels() + c];
                output.data[y * output.step + output.channels() * x + c] = backgroundPx * (1. - opacity) + foregroundPx * opacity;
            }
        }
    } return 0;
}
