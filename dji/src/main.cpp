#include <iostream>
#include "carDetect.h"
#include "draw.h"
#include <Windows.h>
#include "mixin.h"

string yolov3Path = "D:\\file\\code\\PROJECTS\\djiDetect\\yolov3";
string curFramePath = "D:\\file\\code\\PROJECTS\\djiDetect\\dji\\resource\\cur.png";
string labelPah = "D:\\file\\code\\PROJECTS\\djiDetect\\yolov3\\runs\\detect\\exp\\labels\\cur.txt";
string carsPath = "D:\\file\\code\\PROJECTS\\djiDetect\\dji\\resource\\cars\\cars";
string illegalStopPath = "D:\\file\\code\\PROJECTS\\djiDetect\\illegalStop";

cv::Mat frameRealTime;

void showOutput() {
    string output = "output";
    cv::namedWindow(output, cv::WINDOW_NORMAL);
    cv::resizeWindow(output, 768, 432);
    cv::moveWindow(output, 200, 0);
    while (true) {
        cv::Mat copyFrame = frameRealTime.clone();
        drawRectangle(copyFrame);
        cv::imshow(output, copyFrame);
        cv::waitKey(1);
    }
}

int main() {
//    cout << a << b << c;
    // string videosAddress = "rtmp://192.168.43.72:1935/live/home";
    string videosAddress = "D:\\file\\code\\PROJECTS\\djiDetect\\100MEDIA\\DJI_0023.MP4";
    cv::VideoCapture capture;
    string realTime = "realTime";
    if (!capture.open(videosAddress)) {
        cout << "can not open ...\n";
        return -1;
    }
    capture.read(frameRealTime);
    thread t(showOutput);
    cv::namedWindow(realTime, cv::WINDOW_NORMAL);
    cv::resizeWindow(realTime, 768, 432);
    cv::moveWindow(realTime, 200, 400);
    while (!frameRealTime.empty()) {
        imshow(realTime, frameRealTime);
        if (cv::waitKey(1) != -1) {
            break;
        }
        capture.read(frameRealTime);
    }
    capture.release();
    return 0;
}
