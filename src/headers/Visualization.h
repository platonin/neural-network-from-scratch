#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

namespace NN {
    namespace Visualization {
        using namespace Eigen;
        using namespace std;
        using namespace cv;

        // для вывода изображения через opencv
        void showImage(const MatrixXd &image);
        void showImagesRow(const std::vector<Eigen::MatrixXd> &images);
        void showImagesWithLabels(const std::vector<MatrixXd> &images, const std::vector<int> &numbers);
        void showImagesWithTwoNumbers(const std::vector<Eigen::MatrixXd> &images, const std::vector<int> &topNumbers, const std::vector<int> &bottomNumbers);
    }; // namespace Visualization
}; // namespace NeuralNetwork

#endif