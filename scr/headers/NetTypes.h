#ifndef NET_TYPES_H
#define NET_TYPES_H

#include <Eigen/Dense>

struct forwardData {
    Eigen::VectorXd z;
    Eigen::VectorXd x;
};

struct layerGradData {
    Eigen::MatrixXd grad_W;
    Eigen::VectorXd grad_b;
};

struct layerVelocityData {
    Eigen::MatrixXd velocity_W;
    Eigen::VectorXd velocity_b;
};

struct LayerParams {
    int inputSize;
    int outputSize;
    ActivationFunc activationType;
};



#endif
