#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "Layer.h"

class LossFunction {
public:
    virtual float loss(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, std::shared_ptr<DistanceFunction> distanceFunction) const = 0;
    virtual float lossSingle(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction, int n) const = 0;
    //вообще по идее тут не надо передавать размер выборки, тк возможно не для MSE он и не понадобится, но пока пусть так
    virtual Eigen::VectorXd derevativeLoss(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction, int n) const = 0;
    virtual ~LossFunction() = default;
};

class MSE : public LossFunction {
public:
    float loss(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, std::shared_ptr<DistanceFunction> distanceFunction) const override {
        float ret = 0;
        for (int i = 0; i < X.size(); ++i) {
            ret += 1.0/X.size() * distanceFunction->distance(X[i], Y[i]);
        }
        return ret;
    }

    float lossSingle(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction, int n) const override {
        return 1.0/n * distanceFunction->distance(x, y);
    }

    Eigen::VectorXd derevativeLoss(Eigen::VectorXd& x, Eigen::VectorXd& y, std::shared_ptr<DistanceFunction> distanceFunction, int n) const override {
        return 1.0/n * distanceFunction->gradient(x, y);
    }

};

#endif