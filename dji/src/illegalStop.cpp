#include <fstream>
#include "illegalStop.h"
#include <direct.h>
#include "mixin.h"
#include <iomanip>

vector<bool> illegalCars(vector<cv::Mat> &cars) {
    system("wsl rm -r /mnt/d/file/code/PROJECTS/djiDetect/dji/resource/cars/cars/*");
    for (int i = 0; i < cars.size(); ++i) {
        stringstream ss;
        ss << setw(3) << setfill('0') << (i + 1);
        cv::imwrite(carsPath + "\\" + ss.str() + ".jpg", cars[i]);
    }
    string cmd = "activate torch && cd " + illegalStopPath + " && python detect.py";
    system(cmd.c_str());
    ifstream file(illegalStopPath + "/output/result.txt");
    bool tmp;
    vector<bool> ret;
    while (file >> tmp) {
        ret.push_back(tmp);
    }
    return ret;
}

vector<cv::Mat> cropCars(cv::Mat &frame, vector<vector<int>> &labels) {
    vector<cv::Mat> ans;
    for (int i = 0; i < labels.size(); ++i) {
        vector<int> &v = labels[i];
        int expandBig = 28;
        int expandSmall = 22;
        if (v[2] - v[0] < v[3] - v[1])
            swap(expandBig, expandSmall);
        int expand2 = 2;
        int x0 = v[0] - expandBig;
        if (x0 < 0) x0 = 0;
        int y0 = v[1] - expandSmall;
        if (y0 < 0) y0 = 0;
        int x1 = v[2] + expandBig;
        if (x1 > frame.cols) x1 = frame.cols;
        int y1 = v[3] + expandSmall;
        if (y1 > frame.rows) y1 = frame.rows;
        cv::Rect rect(x0, y0, x1 - x0, y1 - y0);
        cv::Mat roi = frame(rect).clone();
        cv::rectangle(roi, cv::Point(v[0] - x0 - expand2, v[1] - y0 - expand2),
                      cv::Point(v[2] - x0 + expand2, v[3] - y0 + expand2),
                      cv::Scalar(0, 0, 0, 0), -1, 8, 0);
        ans.push_back(roi);
    }
    return ans;
}