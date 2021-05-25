#ifndef DJI_ILLEGALSTOP_H
#define DJI_ILLEGALSTOP_H

#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

vector<bool> illegalCars(vector<cv::Mat> &cars);

vector<cv::Mat> cropCars(cv::Mat &frame, vector<vector<int>> &labels);

#endif //DJI_ILLEGALSTOP_H
