#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv)
{      
    int cam_index = 0;
    if(argc == 2)
    {
        cam_index = atoi(argv[1]);
    }
    if(argc > 2)
    {
        std::cout << "please run:\nopen_video num_dev_video  or  open_video" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(cam_index);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FPS, 30);
    cv::Mat img;

    if(!cap.isOpened())
    {   
        std::cout << "please check usb" << std::endl;
        return -1;
    }

    while (true)
    {
        cap >> img;
        cv::resize(img, img, cv::Size(640, 480));
        cv::imshow("img", img);
        if(cv::waitKey(30) == 'q')
            break;
    }
    
    return 0;
}