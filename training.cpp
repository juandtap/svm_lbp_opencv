
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <limits>
#include <iostream>
#include <opencv2/ml.hpp>

using namespace std;
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


int main() {
    // Cargar imagenes de entrenamiento y sus etiquetas
    vector<Mat> images; // imagenes
    vector<int> labels; // etiquetas

    string rocaDir = "./Images/Rocas/";
    string maderaDir = "./Images/Madera/";

    int numImagesByClass = 20;

    // cargo las imagenes de rocas y sus etiquetas
    for (int i = 1; i <= numImagesByClass; i++){
        string filename = rocaDir + "roca" + to_string(i) + ".png";
        Mat image = imread(filename, IMREAD_UNCHANGED);
        if (image.empty()) {
            cerr << "Error: No se pudo cargar la imagen " << filename << endl;
            continue; // 
        }
        if (image.depth() == CV_16U) {
            cout << "Imagen a 16bits convirtiendo a 8bits " << filename << endl;
            Mat image8bit;
            image.convertTo(image8bit, CV_8U, 255.0 / 65535.0); // Convertir de 16 bits a 8 bits
            image = image8bit;
        }
        images.push_back(image);
        labels.push_back(0); // 0 roca
    }

     // cargo las imagenes de madera y sus etiquetas
    for (int i = 1; i <= numImagesByClass; i++){
        string filename = maderaDir + "madera" + to_string(i) + ".png";
        Mat image = imread(filename, IMREAD_UNCHANGED);
        if (image.empty()) {
            cerr << "Error: No se pudo cargar la imagen " << filename << endl;
            continue; // 
        }
        images.push_back(image);
        labels.push_back(1); // 1 madera
    }
    
    cout << "Imagenes de entrenamiento cargadas " << endl;

    // images.push_back(cv::imread("./Images/roca1.jpg", cv::IMREAD_UNCHANGED));
    // labels.push_back(0); // 0 roca
    // images.push_back(cv::imread("./Images/roca2.jpg", cv::IMREAD_UNCHANGED));
    // labels.push_back(0); // 0 roca
    // images.push_back(cv::imread("./Images/madera1.jpeg", cv::IMREAD_UNCHANGED));
    // labels.push_back(1); //1 madera
    // images.push_back(cv::imread("./Images/madera2.jpeg", cv::IMREAD_UNCHANGED));
    // labels.push_back(1); //1 madera


    cv::Size targetSize(256, 256);
    // Convertir cada imagen al espacio de color CIELAB y calcular LBP
    cout << "Empieza el entrenamiento "<< endl;
    cv::Mat trainingData;
    for (int i = 0; i < images.size(); i++) {
        // Redimensionar la imagen
        cv::Mat resizedImage = Mat::zeros(targetSize, images[0].type());
        cv::resize(images[i], resizedImage, targetSize);

        // Convertir al espacio de color CIELAB
        cv::Mat lab_image;
        cv::cvtColor(resizedImage, lab_image, cv::COLOR_BGR2Lab);

        std::vector<cv::Mat> lab_planes;
        cv::split(lab_image, lab_planes);

        // Calcular LBP y el histograma
        cv::Mat lbp_image = calculateLBP(lab_planes[0]);
        cv::Mat hist = calculateLBPHistogram(lbp_image);

        trainingData.push_back(hist.reshape(1, 1));
    }
    // Crear etiquetas en formato Mat
    cv::Mat labelsMat(labels.size(), 1, CV_32SC1, labels.data());

    // Crear y entrenar el SVM
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

    svm->train(trainingData, cv::ml::ROW_SAMPLE, labelsMat);
    cout << "Entrenamiento terminado " << endl;
    svm->save("svm_model.xml");
     cout << "Modelo SVM guardado " << endl;

    return 0;
}
