#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include "Layer.h"
#include "NetTypes.h"

namespace NeuralNetwork {

struct Optimizer {
    double learningRate;
    double beta;
    std::function<void(std::shared_ptr<Layer>&, layerOptimizerData&, double, double, layerGradData&)> update_weights;
    std::string Type;
};

class OptimizerCreation {
private:
    static void SGD(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);
    static void Momentum(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);
    static void RMSProp(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);
    static void Adam(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);

public:
    static Optimizer GetSGD(double learningRate);
    static Optimizer GetMomentum(double learningRate, double beta);
    static Optimizer GetRMSProp(double learningRate, double beta);
    static Optimizer GetAdam(double learningRate, double beta);
    static Optimizer create(std::string type, double learningRate, double beta);
};

}; // namespace NeuralNetwork

#endif