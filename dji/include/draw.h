#ifndef DJI_DRAW_H
#define DJI_DRAW_H

#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

void drawRectangle(cv::Mat &frame, vector<vector<int>> &labels, vector<bool> isIllegal);

void drawRectangle(cv::Mat &frame);

#endif //DJI_DRAW_H
