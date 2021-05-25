#include "carDetect.h"
#include <cstdlib>
#include <fstream>
#include "mixin.h"

vector<vector<int>> detect(cv::Mat &frame) {
    vector<vector<int>> labels;
    cv::imwrite(curFramePath, frame);
    remove(labelPah.c_str());
    string cmd = "activate yolov3 && cd " + yolov3Path + " && python mydetect.py";
    system(cmd.c_str());
    ifstream file(labelPah);
    double val;
    while (file >> val) {
        vector<double> tmp;
        labels.push_back(vector<int>());
        for (int i = 0; i < 4; ++i) {
            file >> val;
            tmp.push_back(val);
        }
        int centerX = (int) (tmp[0] * frame.cols);
        int centerY = (int) (tmp[1] * frame.rows);
        int halfWidth = (int) (tmp[2] * frame.cols) / 2;
        int halfHeight = (int) (tmp[3] * frame.rows) / 2;
        labels.back().push_back(centerX - halfWidth);
        labels.back().push_back(centerY - halfHeight);
        labels.back().push_back(centerX + halfWidth);
        labels.back().push_back(centerY + halfHeight);
    }
    file.close();
    return labels;
}