#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "Layer.h"
#include "ActivationFunction.h"

namespace NeuralNetwork {

struct LossFunc {
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> lossFunction;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> lossDerivative;
    std::string Type;
};

class LossCreation {
private:
    static double MSE(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    static Eigen::VectorXd MSE_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

    static double MAE(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    static Eigen::VectorXd MAE_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

    static double CrossEntropy(const Eigen::VectorXd& x, const Eigen::VectorXd& y);
    static Eigen::VectorXd CrossEntropy_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y);

public:
    static LossFunc GetMSE();
    static LossFunc GetMAE();
    static LossFunc GetCrossEntropy();
    static LossFunc create(std::string type);
};

}; // namespace NeuralNetwork

#endif
