#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <functional>
#include "Layer.h"
#include "NetTypes.h"

namespace NN {

struct Optimizer {
    double learningRate;
    double beta;
    std::function<void(std::shared_ptr<Layer>&, layerOptimizerData&, double, double, layerGradData&)> updateWeights;
    std::string Type;
};

class OptimizerCreation {
private:
    static void SGD(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);
    static void Momentum(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);
    static void RMSProp(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);
    static void Adam(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads);

public:
    static Optimizer getSGD(double learningRate);
    static Optimizer getMomentum(double learningRate, double beta);
    static Optimizer getRMSProp(double learningRate, double beta);
    static Optimizer getAdam(double learningRate, double beta);
    static Optimizer create(std::string type, double learningRate, double beta);
};

}; // namespace NeuralNetwork

#endif