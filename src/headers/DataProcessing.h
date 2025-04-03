#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <vector>
#include <Eigen/Dense>

namespace NeuralNetwork {

    namespace DataProcessing {
        using namespace std;
        using namespace Eigen;

        // преобразование меток в векторы
        vector<Eigen::VectorXd> labelsTransformToVector(const vector<int>& labels_int);

        // преобразование изображений в векторы
        vector<Eigen::VectorXd> imagesTransformToVector(const vector<MatrixXd>& images_matrix);
    };

};

#endif
