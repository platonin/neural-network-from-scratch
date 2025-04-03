#include "ActivationFunction.h"

namespace NeuralNetwork {

    Eigen::VectorXd ActivationCreation::ReLU(const Eigen::VectorXd& vec) {
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? vec(i) : 0;
        }
        return result;
    }

    Eigen::MatrixXd ActivationCreation::ReLU_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? 1 : 0;
        }
        return result.asDiagonal();
    }

    Eigen::VectorXd ActivationCreation::LeakyReLU(const Eigen::VectorXd& vec) {
        double alpha = 0.01;
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? vec(i) : alpha * vec(i);
        }
        return result;
    }

    Eigen::MatrixXd ActivationCreation::LeakyReLU_der(const Eigen::VectorXd& vec) {
        double alpha = 0.01;
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? 1 : alpha;
        }
        return result.asDiagonal();
    }

    Eigen::VectorXd ActivationCreation::Sigmoid(const Eigen::VectorXd& vec) {
        return (1.0 / (1.0 + (-vec.array()).exp())).matrix();
    }

    Eigen::MatrixXd ActivationCreation::Sigmoid_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd sig = Sigmoid(vec);
        return (sig.array() * (1 - sig.array())).matrix().asDiagonal();
    }

    Eigen::VectorXd ActivationCreation::Softmax(const Eigen::VectorXd& vec) {
        Eigen::VectorXd exp_values = vec.array().exp();
        double sum_exp = exp_values.sum();
        return exp_values / sum_exp;
    }

    Eigen::MatrixXd ActivationCreation::Softmax_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd softmax = Softmax(vec);
        Eigen::MatrixXd diag = softmax.asDiagonal();
        return diag - softmax * softmax.transpose();
    }

    ActivationFunc ActivationCreation::GetReLU() {
        return ActivationFunc{&ReLU, &ReLU_der, "ReLU"};
    }

    ActivationFunc ActivationCreation::GetSigmod() {
        return ActivationFunc{&Sigmoid, &Sigmoid_der, "Sigmoid"};
    }

    ActivationFunc ActivationCreation::GetSoftmax() {
        return ActivationFunc{&Softmax, &Softmax_der, "Softmax"};
    }

    ActivationFunc ActivationCreation::GetLeakyReLU() {
        return ActivationFunc{&LeakyReLU, &LeakyReLU_der, "LeakyReLU"};
    }

    ActivationFunc ActivationCreation::create(std::string type) {
        if (type == "ReLU") return GetReLU();
        if (type == "Sigmoid") return GetSigmod();
        if (type == "Softmax") return GetSoftmax();
        if (type == "LeakyReLU") return GetLeakyReLU();
        else std::cerr << "Неизвестный тип активационной функции: " + type;
        return {};
    }

}; // namespace NeuralNetwork
