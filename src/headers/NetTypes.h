#ifndef NET_TYPES_H
#define NET_TYPES_H

#include <Eigen/Dense>
#include "ActivationFunction.h"

namespace NN {

struct forwardData {
    Eigen::VectorXd z;
    Eigen::VectorXd x;
};

struct layerGradData {
    Eigen::MatrixXd grad_W;
    Eigen::VectorXd grad_b;
};

struct layerOptimizerData {
    Eigen::MatrixXd velocity_W;
    Eigen::VectorXd velocity_b;
    Eigen::MatrixXd G_W;
    Eigen::VectorXd G_b;
};

struct LayerParams {
    int inputSize;
    int outputSize;
    ActivationFunc activationType;
};

}; // namespace NeuralNetwork



#endif
