#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <iostream>
#include <opencv2/ml.hpp>

using namespace cv;



cv::Mat calculateLBP(const cv::Mat& src) {
    cv::Mat lbp_image;
    src.convertTo(lbp_image, CV_32F);

    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            uchar center = src.at<uchar>(i, j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i - 1, j - 1) > center) << 7;
            code |= (src.at<uchar>(i - 1, j) > center) << 6;
            code |= (src.at<uchar>(i - 1, j + 1) > center) << 5;
            code |= (src.at<uchar>(i, j + 1) > center) << 4;
            code |= (src.at<uchar>(i + 1, j + 1) > center) << 3;
            code |= (src.at<uchar>(i + 1, j) > center) << 2;
            code |= (src.at<uchar>(i + 1, j - 1) > center) << 1;
            code |= (src.at<uchar>(i, j - 1) > center) << 0;
            lbp_image.at<float>(i, j) = code;
        }
    }

    return lbp_image;
}



cv::Mat calculateLBPHistogram(const cv::Mat& lbp_image) {
    int histSize = 256; // LBP produces 256 possible values
    float range[] = {0, 256}; // Range of LBP values
    const float* histRange = {range};

    cv::Mat hist;
    cv::calcHist(&lbp_image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}


int predictImage(const cv::Mat& image, cv::Ptr<cv::ml::SVM> svm) {
    cv::Mat lab_image;
    //cv::resize(image, image, cv::Size(256,256));
    cv::cvtColor(image, lab_image, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_planes;
    cv::split(lab_image, lab_planes);

    cv::Mat lbp_image = calculateLBP(lab_planes[0]);
    cv::Mat hist = calculateLBPHistogram(lbp_image);

    hist = hist.reshape(1, 1);
    int response = svm->predict(hist);

    return response;
}

int main() {
    // Cargar el modelo SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load("svm_model.xml");

    // Cargar nueva imagen
    //cv::Mat newImage = cv::imread("./Images/roca_nueva.jpg", cv::IMREAD_UNCHANGED);
    //cv::Mat newImage = cv::imread("./Images/madera2.jpeg", cv::IMREAD_UNCHANGED);
    Mat newImage = imread("./Images/nueva_roca2.png", IMREAD_UNCHANGED);
    // Predecir la categoría de la nueva imagen
    int category = predictImage(newImage, svm);

    std::cout << "La imagen pertenece a la categoría: " << category << std::endl;

    return 0;
}
