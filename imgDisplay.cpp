#include "imgDisplay.h"
#include "Blur_filter.h"


#include <iostream>
#include <opencv2/opencv.hpp>

void displayImage() {
    cv::Mat image = cv::imread("..\\..\\assets\\image\\cathedral.jpeg");
    cv::Mat blurredImage;

    if (image.empty()) {
        std::cout << "Could not open or find the image!\n";
        std::cin.get(); // wait for any key press
        return;
    }

    cv::namedWindow("Window_1", cv::WINDOW_NORMAL);
    cv::imshow("Window_1", image);

    char key;
    do {
        key = cv::waitKey(0); // wait for infinite time for a keypress

        if (key == '1') {
            // Apply the first blur function
            blur5x5_1(image, blurredImage);
            cv::imshow("Window_1", blurredImage);
        } else if (key == '2') {
            // Apply the second blur function
            blur5x5_2(image, blurredImage);
            cv::imshow("Window_1", blurredImage);
        } else if (key == 'r') {
            // Reset to the original image
            cv::imshow("Window_1", image);
        }

    } while (key != 'q');
}