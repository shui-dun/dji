#ifndef DJI_CARDETECT_H
#define DJI_CARDETECT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
using namespace std;

vector<vector<int>> detect(cv::Mat &frame);

#endif //DJI_CARDETECT_H
