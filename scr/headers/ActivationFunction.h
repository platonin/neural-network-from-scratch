#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>

// class ActivationFunction {
// public:
//     // virtual double activate(double x) const = 0;
//     // virtual double derivative(double x) const = 0;
//     virtual Eigen::VectorXd activate(const Eigen::VectorXd& vec) const = 0;
//     virtual Eigen::VectorXd derivative(const Eigen::VectorXd& vec) const = 0;
//     virtual std::string getType() const = 0;

//     // virtual Eigen::VectorXd activate(const Eigen::VectorXd vec) const {
//     //     Eigen::VectorXd result(vec.size());
//     //     for (int i = 0; i < vec.size(); ++i) {
//     //         result(i) = activate(vec(i));
//     //     }
//     //     return result;
//     // }

//     // virtual Eigen::VectorXd derivative(const Eigen::VectorXd vec)const {
//     //     Eigen::VectorXd result(vec.size());
//     //     for (int i = 0; i < vec.size(); ++i) {
//     //         result(i) = derivative(vec(i));
//     //     }
//     //     return result;
//     // }

//     virtual ~ActivationFunction() = default;
// };

// class ReLU : public ActivationFunction {
// public:
//     double activate(double x) const {
//         return (x > 0) ? x : 0;
//     }

//     double derivative(double x) const {
//         return (x > 0) ? 1 : 0;
//     }

//     Eigen::VectorXd activate(const Eigen::VectorXd& vec) const override {
//         Eigen::VectorXd result(vec.size());
//         for (int i = 0; i < vec.size(); ++i) {
//             result(i) = activate(vec(i));
//         }
//         return result;
//     }

//     Eigen::VectorXd derivative(const Eigen::VectorXd& vec)const override {
//         Eigen::VectorXd result(vec.size());
//         for (int i = 0; i < vec.size(); ++i) {
//             result(i) = derivative(vec(i));
//         }
//         return result;
//     }

//     std::string getType() const override {
//         return "ReLU";
//     }
// };

// class Sigmoid : public ActivationFunction {
// public:
//     double activate(double x) const {
//         // std::cout << "---- " << 1.0f / (1.0f + std::exp(-x)) << " ----\n";
//         return 1.0 / (1.0 + std::exp(-x));
//     }
    
//     double derivative(double x) const {
//         float sig = activate(x);
//         return sig * (1 - sig);
//     }

//     Eigen::VectorXd activate(const Eigen::VectorXd& vec) const override {
//         Eigen::VectorXd result(vec.size());
//         for (int i = 0; i < vec.size(); ++i) {
//             result(i) = activate(vec(i));
//         }
//         return result;
//     }

//     Eigen::VectorXd derivative(const Eigen::VectorXd& vec) const override {
//         Eigen::VectorXd result(vec.size());
//         for (int i = 0; i < vec.size(); ++i) {
//             result(i) = derivative(vec(i));
//         }
//         return result;
//     }

//     std::string getType() const override {
//         return "Sigmoid";
//     }
// };

// class Softmax : public ActivationFunction {

//     Eigen::VectorXd activate(const Eigen::VectorXd& input) {
//         Eigen::VectorXd exp_values = input.array().exp();
//         double sum_exp = exp_values.sum();
//         return exp_values / sum_exp;
//     }
    
//     Eigen::VectorXd derivative(const Eigen::VectorXd& vec) const override {
//         return Eigen::VectorXd(0);
//     }

//     std::string getType() const override {
//         return "Softmax";
//     }
// };


struct ActivationFunc {
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> derivative;
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

    static Eigen::VectorXd ReLU_der(const Eigen::VectorXd& vec) {
        Eigen::VectorXd result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result(i) = (vec(i) > 0) ? 1 : 0;
        }
        return result;
    }

    static Eigen::VectorXd Sigmoid(const Eigen::VectorXd& vec) {
        // Eigen::VectorXd result(vec.size());
        // for (int i = 0; i < vec.size(); ++i) {
        //     result(i) = 1.0 / (1.0 + std::exp(- vec(i)));
        // }
        // return result;
        return (1.0 / (1.0 + (-vec.array()).exp())).matrix();
    }

    static Eigen::VectorXd Sigmoid_der(const Eigen::VectorXd& vec) {
        // Eigen::VectorXd result(vec.size());
        // for (int i = 0; i < vec.size(); ++i) {
        //     double sig = 1.0 / (1.0 + std::exp(- vec(i)));
        //     result(i) = sig * (1 - sig);
        // }
        // return result;
        Eigen::VectorXd sig = Sigmoid(vec);
        return (sig.array() * (1 - sig.array())).matrix();
    }

    
public:
    // static std::shared_ptr<ActivationFunction> create(const std::string& type) {
    //     if (type == "ReLU") return std::make_shared<ReLU>();
    //     if (type == "Sigmoid") return std::make_shared<Sigmoid>();
    //     throw std::invalid_argument("Неизвестный тип активационной функции: " + type);
    // }

    static ActivationFunc GetReLU() {
        return ActivationFunc{&ReLU, &ReLU_der, "ReLU"};
    }

    static ActivationFunc GetSigmod() {
        return ActivationFunc{&Sigmoid, &Sigmoid_der, "Sigmoid"};
    }

    static ActivationFunc create(std::string type) {
        if (type == "ReLU") return GetReLU();
        if (type == "Sigmoid") return GetSigmod();
        else std::cerr << "Неизвестный тип активационной функции: " + type;
        return {};
    }
};




#endif