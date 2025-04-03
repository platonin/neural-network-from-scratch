#include "LossFunction.h"
#include <iostream>
#include <cmath>

namespace NeuralNetwork {

double LossCreation::MSE(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return 0.5 * (x - y).squaredNorm();
}

Eigen::VectorXd LossCreation::MSE_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return (x - y);
}

double LossCreation::MAE(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return (x - y).norm();
}

Eigen::VectorXd LossCreation::MAE_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    float norm = (x - y).norm();
    if (norm == 0) return Eigen::VectorXd::Zero(x.size());
    return (x - y) / norm;
}

double LossCreation::CrossEntropy(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    double ret = 0;
    for (int i = 0; i < x.size(); ++i) {
        if (x(i) != 0) {
            ret += y(i) * std::log(x(i));
        }
    }
    return -ret;
}

Eigen::VectorXd LossCreation::CrossEntropy_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return -y.array() / x.array();
}

LossFunc LossCreation::getMSE() {
    return {&MSE, &MSE_der, "MSE"};
}

LossFunc LossCreation::getMAE() {
    return {&MAE, &MAE_der, "MAE"};
}

LossFunc LossCreation::getCrossEntropy() {
    return {&CrossEntropy, &CrossEntropy_der, "CrossEntropy"};
}

LossFunc LossCreation::create(std::string type) {
    if (type == "MSE") return getMSE();
    if (type == "MAE") return getMAE();
    if (type == "CrossEntropy") return getCrossEntropy();
    else throw std::invalid_argument("Неизвестный тип функции потерь: " + type);
    return {};
}

}; // namespace NeuralNetwork
