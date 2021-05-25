#include <opencv2/opencv.hpp>
#include "torch/script.h"
#include "torch/torch.h"

#include <iostream>
#include <memory>

using namespace std;

// tensor是什么
// 图片预处理与cpp的要一致

// resize并保持图像比例不变
cv::Mat resize_with_ratio(cv::Mat &img) {
    cv::Mat temImage;
    int w = img.cols;
    int h = img.rows;

    float t = 1.;
    float len = t * std::max(w, h);
    int dst_w = 224, dst_h = 224;
    cv::Mat image = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat imageROI;
    if (len == w) {
        float ratio = (float) h / (float) w;
        cv::resize(img, temImage, cv::Size(224, 224 * ratio), 0, 0, cv::INTER_LINEAR);
        imageROI = image(cv::Rect(0, ((dst_h - 224 * ratio) / 2), temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    } else {
        float ratio = (float) w / (float) h;
        cv::resize(img, temImage, cv::Size(224 * ratio, 224), 0, 0, cv::INTER_LINEAR);
        imageROI = image(cv::Rect(((dst_w - 224 * ratio) / 2), 0, temImage.cols, temImage.rows));
        temImage.copyTo(imageROI);
    }

    return image;
}


int main(int argc, const char *argv[]) {
    string address = "D:\\file\\code\\PROJECTS\\djiDetect\\100MEDIA\\DJI_0022.MP4";
    cv::VideoCapture stream;
    if (!stream.open(address)) {
        printf("can not open ...\n");
        return -1;
    }
    cv::namedWindow("Gesture Detect", cv::WINDOW_AUTOSIZE);

    auto module = torch::jit::load("../../resource/best.torchscript.pt");
    module.to(at::kCPU);

    cv::Mat frame;
    cv::Mat image;
    cv::Mat input;

    while (true) {
        stream.read(frame);
        image = resize_with_ratio(frame);

        imshow("resized image", image);    //显示摄像头的数据
        cv::cvtColor(image, input, cv::COLOR_BGR2RGB);

        // 下方的代码即将图像转化为Tensor，随后导入模型进行预测
        torch::Tensor tensor_image = torch::from_blob(input.data, {1, input.rows, input.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.div(255);
        tensor_image = tensor_image.to(torch::kCPU);
//        cout << tensor_image << '\n';
        torch::Tensor result = module.forward({tensor_image}).toTensor();
        cout << result;

        auto max_result = result.max(1, true);
        auto max_index = std::get<1>(max_result).item<float>();
        if (max_index == 0)
            cv::putText(frame, "paper", {40, 50}, cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2);
        else if (max_index == 1)
            cv::putText(frame, "scissors", {40, 50}, cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2);
        else
            cv::putText(frame, "stone", {40, 50}, cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(0, 255, 0), 2);

        imshow("Gesture Detect", frame);    //显示摄像头的数据
        cv::waitKey(1);
    }

}