#include "displayVid.h"
#include "filter.h"
#include "Blur_filter.h"
#include "faceDetect.h"

#include <iostream>
#include <opencv2/opencv.hpp>

enum class EffectMode {
    None, Standard, Custom, SepiaTone, 
    Blur_1, Blur_2, SobelX, SobelY, Magnitude, BlurQuantize,
    FaceDect, ColorPop, Embossing, BrightnessContrast, Sparkles
};

int brightness = 0; // Range: [-100, 100]
int contrast = 0;   // Range: [-100, 100]

void displayVid() {
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened()) {
        std::cerr << "Error: Unable to open video device\n";
        return;
    }

    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Connected to camera (" << refS.width << "x" << refS.height << ")" << std::endl;

    cv::namedWindow("Video", 1);
    cv::Mat frame; 
    EffectMode mode = EffectMode::None;

    for (;;) {
        capdev >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Frame is empty\n";
            continue;
        }
        switch (mode) {
            case EffectMode::Standard: {
                // cv::Mat frame;
                // cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);
                // cv::cvtColor(processedFrame, processedFrame, cv::COLOR_GRAY2BGR);
                cv::imshow("Video", frame);
                break;
            }
            case EffectMode::Custom: {
                cv::Mat processedFrame;
                if (greyscale(frame, processedFrame) == 0) {
                cv::imshow("Video", processedFrame);
                }
                break;
            }
            case EffectMode::SepiaTone: {
                cv::Mat processedFrame;
                if (sepiaTone(frame, processedFrame) == 0) { 
                cv::imshow("Video", processedFrame);
                }
                break;
            }
            case EffectMode::Blur_1: {
                cv::Mat blurredFrame;
                blur5x5_1(frame, blurredFrame); 
                cv::imshow("Video", blurredFrame);
                break;
            }
            case EffectMode::Blur_2: {
                cv::Mat blurredFrame;
                blur5x5_2(frame, blurredFrame); // or use blur5x5_2
                cv::imshow("Video", blurredFrame);
                break;
            }
            case EffectMode::SobelX: {
                cv::Mat sobelXFrame;
                sobelX3x3(frame, sobelXFrame);
                cv::Mat displayFrame;
                cv::convertScaleAbs(sobelXFrame, displayFrame);
                cv::imshow("Video", displayFrame);
                break;
            }
            case EffectMode::SobelY: {
                cv::Mat sobelYFrame;
                sobelY3x3(frame, sobelYFrame);
                cv::Mat displayFrame;
                cv::convertScaleAbs(sobelYFrame, displayFrame);
                cv::imshow("Video", displayFrame);
                break;
            }
            case EffectMode::Magnitude: {
                cv::Mat sobelXFrame, sobelYFrame, magnitudeFrame;
                sobelX3x3(frame, sobelXFrame);
                sobelY3x3(frame, sobelYFrame);
                magnitude(sobelXFrame, sobelYFrame, magnitudeFrame);
                cv::Mat displayFrame;
                cv::convertScaleAbs(magnitudeFrame, displayFrame);
                cv::imshow("Video", displayFrame);
                break;
            }
            case EffectMode::BlurQuantize: {
                cv::Mat blurredQuantizedFrame;
                blurQuantize(frame, blurredQuantizedFrame, 10); // Example: 10 levels
                cv::imshow("Video", blurredQuantizedFrame);
                break;
            }

            case EffectMode::FaceDect: {
                cv:: Mat greyFrame;
                std::vector<cv::Rect> faces;
                cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                detectFaces(greyFrame, faces);
                drawBoxes(frame, faces);
                cv::imshow("Video", frame);
                break;
            }

             case EffectMode::ColorPop: {
                cv::Mat colorPopFrame;
                colorPopEffect(frame, colorPopFrame);
                cv::imshow("Video", colorPopFrame);
                break;
            }
            case EffectMode::Embossing: {
                cv::Mat embossedFrame;
                embossingEffect(frame, embossedFrame);
                cv::imshow("Video", embossedFrame);
                break;
            }
            case EffectMode::BrightnessContrast: {
                cv::Mat processedFrame;
                adjustBrightnessContrast(frame, processedFrame, brightness, contrast);
                cv::imshow("Video", processedFrame);
                break;
            }
            case EffectMode::Sparkles: {
                std::vector<cv::Rect> faces;
                cv::Mat greyFrame, edges;
                cv::cvtColor(frame, greyFrame, cv::COLOR_BGR2GRAY);
                detectFaces(greyFrame, faces);
                cv::Sobel(greyFrame, edges, CV_8U, 1, 1);
                addSparklesToEdges(frame, edges, faces);
                cv::imshow("Video", frame);
                break;
            }
            default:
                cv::imshow("Video", frame);
                break;
        }

        char key = (char)cv::waitKey(10);
        switch (key) {
            case 'q':
                return;
            case 'g':
                mode = EffectMode::Standard;
                break;
            case 'h':
                mode = EffectMode::Custom;
                break;
            case 's':
                mode = EffectMode::SepiaTone;
                break;
            case '7':
                mode = EffectMode::Blur_1;
                break;
            case '8':
                mode = EffectMode::Blur_2;
                break;
            case 'x':
                mode = EffectMode::SobelX;
                break;
            case 'y':
                mode = EffectMode::SobelY;
                break;
            case 'm':
                mode = EffectMode::Magnitude;
                break;
            case 'l':
                mode = EffectMode::BlurQuantize;
                break;
            case 'f':
                mode = EffectMode::FaceDect;
                break;
            case 'c':
                mode = EffectMode::ColorPop;
                break;
            case 'e':
                mode = EffectMode::Embossing;
                break;
            case 'u':
                mode = EffectMode::BrightnessContrast;
                brightness = std::min(brightness + 50, 100);
                break;
            // Bonus, still need to debug why the program dies, unfortunately run out of time 
            case 'p':
                mode = EffectMode::Sparkles;   
            default:
                break;
        }
    }

    capdev.release();
    cv::destroyAllWindows();
}
