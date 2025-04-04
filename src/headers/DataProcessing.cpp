#ifndef DATAPROCESSING_CPP
#define DATAPROCESSING_CPP

#include "DataProcessing.h"

namespace NeuralNetwork {
    namespace DataProcessing {
        using namespace std;
        using namespace Eigen;

        vector<Eigen::VectorXd> labelsTransformToVector(const vector<int>& labels_int) {
            vector<Eigen::VectorXd> labels_vector;
            for (int i = 0; i < labels_int.size(); ++i) {
                Eigen::VectorXd v(10);
                v(labels_int[i]) = 1;
                labels_vector.push_back(v);
            }
            return labels_vector;
        }

        vector<Eigen::VectorXd> imagesTransformToVector(const vector<MatrixXd>& images_matrix) {
            vector<VectorXd> images_vector;
            for (int i = 0; i < images_matrix.size(); ++i) {
                images_vector.push_back(images_matrix[i].reshaped());
            }
            return images_vector;
        }

    }; // namespace DataProcessing
}; // namespace NeuralNetwork

#endif