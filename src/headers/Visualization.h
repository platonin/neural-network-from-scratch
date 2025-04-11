#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include <QApplication>
#include <QLabel>
#include <QImage>
#include <QPixmap>

#include <Eigen/Dense>
#include <vector>

namespace NN {
    namespace Visualization {
        using namespace Eigen;
        using namespace std;

        QImage matrixToQImage(const MatrixXd& mat);
        void showImageQt(const MatrixXd& image);

    }; // namespace Visualization
}; // namespace NeuralNetwork

#endif