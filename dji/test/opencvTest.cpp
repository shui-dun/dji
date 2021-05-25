#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
using namespace std;
using namespace cv;

Mat frame;
Point previousPoint, currentPoint;

Rect2i bbox;

void draw_rectangle(int event, int x, int y, int flags, void *) {
    if (event == EVENT_LBUTTONDOWN) {
        previousPoint = Point(x, y);
    } else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON)) {
        Mat tmp;
        frame.copyTo(tmp);
        currentPoint = Point(x, y);
        rectangle(tmp, previousPoint, currentPoint, Scalar(0, 255, 0, 0));
        imshow("output", tmp);
    } else if (event == EVENT_LBUTTONUP) {
        bbox.x = previousPoint.x;
        bbox.y = previousPoint.y;
        bbox.width = abs(previousPoint.x - currentPoint.x);
        bbox.height = abs(previousPoint.y - currentPoint.y);
    } else if (event == EVENT_RBUTTONUP) {
        destroyWindow("output");
    }
}

int main(int argc, char *argv[]) {
    cout << getBuildInformation();
    VideoCapture capture;
//    string address = "rtmp://192.168.43.72:1935/live/home";
    string address = "D:\\file\\code\\PROJECTS\\djiDetect\\100MEDIA\\DJI_0022.MP4";
    if (!capture.open(address)) {
        printf("can not open ...\n");
        return -1;
    }
    capture.read(frame);
    if (!frame.empty()) {
        namedWindow("output", WINDOW_AUTOSIZE);
        imshow("output", frame);
        setMouseCallback("output", draw_rectangle, 0);
        waitKey();
    }
    auto tracker = TrackerKCF::create();
    capture.read(frame);
    tracker->init(frame, bbox);
    namedWindow("output", WINDOW_AUTOSIZE);
    while (capture.read(frame)) {
        bool ret = tracker->update(frame, bbox);
        cout << ret << ' ' << bbox.x << ' ' << bbox.y << '\n';
        rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
        imshow("output", frame);
        if (waitKey(1) == 'q')
            return 0;
    }
    capture.release();
    destroyWindow("output");
    return 0;
}

