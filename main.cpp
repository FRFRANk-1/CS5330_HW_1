#include "imgDisplay.h"
#include "displayVid.h"
#include "timeBlur.h"

#include <iostream>
#include <opencv2/opencv.hpp>



int main(int argc, char** argv) {
    displayImage();
    std:: cout << "Question 1\n" <<std::flush;
    displayVid();
    std::cout << "Hello, world!\n" << std::flush;
    
    // Default image path
    std::string imagePath = "..\\..\\assets\\image\\cathedral.jpeg";
    
    if (argc > 1) {
        imagePath = argv[1];
        
    }     timeBlurProcessing(imagePath);

    return 0;    
}
