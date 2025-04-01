#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>

namespace NeuralNetwork {

    struct ActivationFunc {
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation;
        std::function<Eigen::MatrixXd(const Eigen::VectorXd&)> derivative;
        std::string Type;
    };

    class ActivationCreation {
    private:
        static Eigen::VectorXd ReLU(const Eigen::VectorXd& vec);
        static Eigen::MatrixXd ReLU_der(const Eigen::VectorXd& vec);
        static Eigen::VectorXd LeakyReLU(const Eigen::VectorXd& vec);
        static Eigen::MatrixXd LeakyReLU_der(const Eigen::VectorXd& vec);
        static Eigen::VectorXd Sigmoid(const Eigen::VectorXd& vec);
        static Eigen::MatrixXd Sigmoid_der(const Eigen::VectorXd& vec);
        static Eigen::VectorXd Softmax(const Eigen::VectorXd& vec);
        static Eigen::MatrixXd Softmax_der(const Eigen::VectorXd& vec);

    public:
        static ActivationFunc GetReLU();
        static ActivationFunc GetSigmod();
        static ActivationFunc GetSoftmax();
        static ActivationFunc GetLeakyReLU();
        static ActivationFunc create(std::string type);
    };

}; // namespace NeuralNetwork

#endif
