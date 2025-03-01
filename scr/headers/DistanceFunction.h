#ifndef DISTANCE_FUNCTION_H
#define DISTANCE_FUNCTION_H

#include "ActivationFunction.h"

class DistanceFunction {
public:
    virtual float distance(Eigen::VectorXd& x, Eigen::VectorXd& y) const = 0;
    //не забывать, что тут мы же градиент–вектор возвращаем, а не скаляр
    virtual Eigen::VectorXd gradient(Eigen::VectorXd& x, Eigen::VectorXd& y) const = 0;

    virtual ~DistanceFunction() = default;
};

class L2Norm : public DistanceFunction {
public:
    float distance(Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
        return (x - y).norm();
    }

    Eigen::VectorXd gradient(Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
        float norm = (x-y).norm();
        if (norm == 0) return Eigen::VectorXd::Zero(x.size());
        return (x-y)/norm; //
    }
};

class SquaredNorm : public DistanceFunction {
    public:
        float distance(Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
            return 0.5 * (x - y).squaredNorm();
        }
    
        Eigen::VectorXd gradient(Eigen::VectorXd& x, Eigen::VectorXd& y) const override {
            return (x-y);
        }
    };

#endif