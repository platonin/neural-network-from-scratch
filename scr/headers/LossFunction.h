#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "Layer.h"
#include "ActivationFunction.h"

struct LossFunc {
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)> lossFunction;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> lossDerivative;
    std::string Type;
};

class LossCreation {
private:
    static double MSE(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        return 0.5 * (x - y).squaredNorm();
    }

    static Eigen::VectorXd MSE_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        return (x-y);
    }

    static double MAE(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        return (x - y).norm();
    }

    static Eigen::VectorXd MAE_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        float norm = (x-y).norm();
        if (norm == 0) return Eigen::VectorXd::Zero(x.size());
        return (x-y)/norm; 
    }

    static double CrossEntropy(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        double ret = 0;
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) != 0) {
                ret += y(i) * std::log(x(i));
            }
        }
        return -ret;
    }

    static Eigen::VectorXd CrossEntropy_der(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        return -y.array() / x.array();
    }

public:
    static LossFunc GetMSE() {
        return {&MSE, &MSE_der, "MSE"};
    }

    static LossFunc GetMAE() {
        return {&MAE, &MAE_der, "MAE"};
    }

    static LossFunc GetCrossEntropy() {
        return {&CrossEntropy, &CrossEntropy_der, "CrossEntropy"};
    }

    static LossFunc create(std::string type) {
        if (type == "MSE") return GetMSE();
        if (type == "MAE") return GetMAE();
        if (type == "CrossEntropy") return GetCrossEntropy();
        else std::cerr << "Неизвестный тип активационной функции: " + type;
        return {};
    }
};

#endif