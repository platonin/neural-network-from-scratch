#include "Layer.h"

namespace NeuralNetwork {

Eigen::MatrixXd Layer::initialize_weights_xavier_normal() {
    double stddev = sqrt(2.0 / inputSize_);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0, stddev);
    Eigen::MatrixXd weights = Eigen::MatrixXd::NullaryExpr(outputSize_, inputSize_, [&]() { return dis(gen); });
    return weights;
}

Layer::Layer() = default;

Layer::Layer(int inputSize, int outputSize, ActivationFunc func) 
    : inputSize_(inputSize), outputSize_(outputSize), W_(outputSize, inputSize), b_(outputSize), activationFunction_(func) {
    W_.setRandom();
    b_.setZero();
}

Layer::Layer(Eigen::MatrixXd W, Eigen::VectorXd b, ActivationFunc func) 
    : W_(W), b_(b), inputSize_(W.cols()), outputSize_(W.rows()), activationFunction_(func) {}

Eigen::VectorXd Layer::CalculateZ(const Eigen::VectorXd& prev_x) const{
    return W_ * prev_x + b_;
}

Eigen::VectorXd Layer::CalculateX(const Eigen::VectorXd& z) const {
    return activationFunction_.activation(z);
}

Eigen::VectorXd Layer::Forward(const Eigen::VectorXd& prev_x) const {
    return CalculateX(CalculateZ(prev_x));
}

void Layer::UpdateW(const Eigen::MatrixXd& gradW) {
    W_ -= gradW;
}

void Layer::UpdateB(const Eigen::MatrixXd& gradB) {
    b_ -= gradB;
}

const Eigen::MatrixXd& Layer::GetW() const {
    return W_;
}

const Eigen::VectorXd& Layer::GetB() const {
    return b_;
}

Eigen::MatrixXd Layer::CalculateActivationDer(const Eigen::VectorXd& z) const {
    return activationFunction_.derivative(z);
}

int Layer::GetInputSize() const {
    return inputSize_;
}

int Layer::GetOutputSize() const {
    return outputSize_;
}

std::string Layer::GetActivationType() const {
    return activationFunction_.Type;
}

}; // namespace NeuralNetwork
