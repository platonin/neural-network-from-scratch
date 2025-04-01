#ifndef LAYER_H
#define LAYER_H

// #include "DistanceFunction.h"
#include "ActivationFunction.h"
#include <random>
#include <cmath>

namespace NeuralNetwork {

class Layer {
private: // 
    int inputSize_;
    int outputSize_;
    Eigen::MatrixXd W_;
    Eigen::VectorXd b_;

    ActivationFunc activationFunction_;
    
    Eigen::MatrixXd initialize_weights_xavier_normal() {
        double stddev = sqrt(2.0 / inputSize_);
    
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dis(0, stddev);
    
        Eigen::MatrixXd weights = Eigen::MatrixXd::NullaryExpr(outputSize_, inputSize_, [&]() { return dis(gen); });
    
        return weights;
    }

public:

    Layer() = default;

    Layer(int inputSize, int outputSize, ActivationFunc func) : inputSize_(inputSize), outputSize_(outputSize),
    W_(outputSize, inputSize), b_(outputSize), activationFunction_(func) {

        // W_ = initialize_weights_xavier_normal();
        W_.setRandom();

        b_.setZero();
    }

    Layer(Eigen::MatrixXd W, Eigen::VectorXd b, ActivationFunc func) : 
    W_(W), b_(b), inputSize_(W.cols()), outputSize_(W.rows()), activationFunction_(func) {}

    // считает z, получая выходной вектор prev_x из предыдущего слоя
    Eigen::VectorXd CalculateZ(const Eigen::VectorXd& prev_x) {
        return W_ * prev_x + b_;
    }
    // считает x по z применяя activationFunction_
    Eigen::VectorXd CalculateX(const Eigen::VectorXd& z) {
        return activationFunction_.activation(z);
    }

    // пихаем в слой градиенты, чтобы обновить веса 
    void UpdateW(const Eigen::MatrixXd& gradW) {
        W_ -= gradW;
    }
    void UpdateB(const Eigen::MatrixXd& gradB) {
        b_ -= gradB;
    }

    const Eigen::MatrixXd& GetW() {
        return W_;
    }
    const Eigen::VectorXd& GetB() {
        return b_;
    }

    // считает sigma'(z)
    Eigen::MatrixXd getDerActivationFromZ(const Eigen::VectorXd& z) const {
        return activationFunction_.derivative(z); //а если возвращаемое значение не конст то не работает ?
    }

    int GetInputSize() const {
        return inputSize_;
    }
    int GetOutputSize() const {
        return outputSize_;
    }

    std::string GetActivationType() const {
        return activationFunction_.Type;
    }
};

}; // namespace NeuralNetwork
#endif
