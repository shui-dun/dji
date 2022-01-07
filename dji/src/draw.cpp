#include "draw.h"
#include "carDetect.h"
#include "illegalStop.h"
#include <windows.h>

void drawRectangle(cv::Mat &frame, vector<vector<int>> &labels, vector<bool> isIllegal) {
    for (int i = 0; i < labels.size(); ++i) {
        auto &vec = labels[i];
        if (isIllegal[i]) {
            rectangle(frame, cv::Point(vec[0], vec[1]), cv::Point(vec[2], vec[3]), cv::Scalar(0, 0, 255, 0), 6, 8, 0);
        } else {
            rectangle(frame, cv::Point(vec[0], vec[1]), cv::Point(vec[2], vec[3]), cv::Scalar(255, 0, 0, 0), 3, 8, 0);
        }
    }
}

void drawRectangle(cv::Mat &frame) {
    auto detect_results = detect(frame);
    auto cars = cropCars(frame, detect_results);
    auto isIllegal = illegalCars(cars);
    drawRectangle(frame, detect_results, isIllegal);
}
