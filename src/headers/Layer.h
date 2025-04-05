#ifndef LAYER_H
#define LAYER_H

#include "ActivationFunction.h"
#include <random>
#include <cmath>
#include <Eigen/Dense>

namespace NN {

class Layer {
private:
    int inputSize_;
    int outputSize_;
    Eigen::MatrixXd W_;
    Eigen::VectorXd b_;
    ActivationFunc activationFunction_;

    Eigen::MatrixXd initializeWeightsXavierNormal();

public:
    Layer();
    Layer(int inputSize, int outputSize, ActivationFunc func);
    Layer(Eigen::MatrixXd W, Eigen::VectorXd b, ActivationFunc func);

    Eigen::VectorXd calculateZ(const Eigen::VectorXd& prev_x) const;
    Eigen::VectorXd calculateX(const Eigen::VectorXd& z) const;
    Eigen::VectorXd forward(const Eigen::VectorXd& prev_x) const;

    void updateW(const Eigen::MatrixXd& gradW);
    void updateB(const Eigen::MatrixXd& gradB);

    const Eigen::MatrixXd& getW() const;
    const Eigen::VectorXd& getB() const;
    Eigen::MatrixXd calculateActivationDer(const Eigen::VectorXd& z) const;

    int getInputSize() const;
    int getOutputSize() const;
    std::string getActivationType() const;
};

}; // namespace NN

#endif
