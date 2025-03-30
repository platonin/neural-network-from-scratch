#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>

struct ActivationFunc {
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> derivative;
    std::string Type;
};

class ActivationCreation {
private:
    static Eigen::VectorXd ReLU(const Eigen::VectorXd& vec) {
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? vec(i) : 0;
        }
        return result;
    }

    static Eigen::MatrixXd ReLU_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? 1 : 0;
        }
        return result.asDiagonal();
    }

    static Eigen::VectorXd LeakyReLU(const Eigen::VectorXd& vec) {
        double alpha = 0.01;
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? vec(i) : alpha * vec(i);
        }
        return result;
    }
    
    static Eigen::MatrixXd LeakyReLU_der(const Eigen::VectorXd& vec) {
        double alpha = 0.01;
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? 1 : alpha;
        }
        return result.asDiagonal();
    }

    static Eigen::VectorXd Sigmoid(const Eigen::VectorXd& vec) {
        return (1.0 / (1.0 + (-vec.array()).exp())).matrix();
    }

    static Eigen::MatrixXd Sigmoid_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd sig = Sigmoid(vec);
        return (sig.array() * (1 - sig.array())).matrix().asDiagonal();
    }

    static Eigen::VectorXd Softmax(const Eigen::VectorXd& vec) {
        Eigen::VectorXd exp_values = vec.array().exp();
        double sum_exp = exp_values.sum();
        return exp_values / sum_exp;
    }

    static Eigen::MatrixXd Softmax_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd softmax = Softmax(vec);
        Eigen::MatrixXd diag = softmax.asDiagonal();
        return diag - softmax * softmax.transpose();
    }

    
public:

    static ActivationFunc GetReLU() {
        return ActivationFunc{&ReLU, &ReLU_der, "ReLU"};
    }

    static ActivationFunc GetSigmod() {
        return ActivationFunc{&Sigmoid, &Sigmoid_der, "Sigmoid"};
    }

    static ActivationFunc GetSoftmax() {
        return ActivationFunc{&Softmax, &Softmax_der, "Softmax"};
    }

    static ActivationFunc GetLeakyReLU() {
        return ActivationFunc{&LeakyReLU, &LeakyReLU_der, "LeakyReLU"};
    }

    static ActivationFunc create(std::string type) {
        if (type == "ReLU") return GetReLU();
        if (type == "Sigmoid") return GetSigmod();
        if (type == "Softmax") return GetSoftmax();
        if (type == "LeakyReLU") return GetLeakyReLU();
        else std::cerr << "Неизвестный тип активационной функции: " + type;
        return {};
    }
};


#endif
