#ifndef LAYER_H
#define LAYER_H

#include "ActivationFunction.h"
#include <random>
#include <cmath>
#include <Eigen/Dense>

namespace NeuralNetwork {

class Layer {
private:
    int inputSize_;
    int outputSize_;
    Eigen::MatrixXd W_;
    Eigen::VectorXd b_;
    ActivationFunc activationFunction_;

    Eigen::MatrixXd initialize_weights_xavier_normal();

public:
    Layer();
    Layer(int inputSize, int outputSize, ActivationFunc func);
    Layer(Eigen::MatrixXd W, Eigen::VectorXd b, ActivationFunc func);

    Eigen::VectorXd CalculateZ(const Eigen::VectorXd& prev_x) const;
    Eigen::VectorXd CalculateX(const Eigen::VectorXd& z) const;
    Eigen::VectorXd Forward(const Eigen::VectorXd& prev_x) const;

    void UpdateW(const Eigen::MatrixXd& gradW);
    void UpdateB(const Eigen::MatrixXd& gradB);

    const Eigen::MatrixXd& GetW() const;
    const Eigen::VectorXd& GetB() const;
    Eigen::MatrixXd CalculateActivationDer(const Eigen::VectorXd& z) const;

    int GetInputSize() const;
    int GetOutputSize() const;
    std::string GetActivationType() const;
};

}; // namespace NeuralNetwork

#endif
