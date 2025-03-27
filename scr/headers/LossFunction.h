#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "Layer.h"
#include "ActivationFunction.h"

// class LossFunction {
// public:
//     virtual float loss(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, std::shared_ptr<DistanceFunction> distanceFunction) const = 0;
//     virtual float lossSingle(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction) const = 0;
//     //вообще по идее тут не надо передавать размер выборки, тк возможно не для MSE он и не понадобится, но пока пусть так
//     virtual Eigen::VectorXd derevativeLoss(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction, int n) const = 0;
//     virtual ~LossFunction() = default;
// };

// class MSE : public LossFunction {
// public:
//     float loss(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, std::shared_ptr<DistanceFunction> distanceFunction) const override {
//         float ret = 0;
//         for (int i = 0; i < X.size(); ++i) {
//             ret += 1.0/X.size() * distanceFunction->distance(X[i], Y[i]);
//         }
//         return ret;
//     }

//     float lossSingle(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction) const override {
//         return 0.5 * (x - y).squaredNorm();
//     }

//     Eigen::VectorXd derevativeLoss(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction, int n) const override {
//         return (x-y);
//     }

// };

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