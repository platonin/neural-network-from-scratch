#include "Optimizer.h"

namespace NN {

void OptimizerCreation::SGD(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    layer->updateW(learningRate * grads.grad_W);
    layer->updateB(learningRate * grads.grad_b);
}

void OptimizerCreation::Momentum(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    optimizerData.velocity_b = beta * optimizerData.velocity_b + (1 - beta) * grads.grad_b;
    optimizerData.velocity_W = beta * optimizerData.velocity_W + (1 - beta) * grads.grad_W;
    layer->updateB(learningRate * optimizerData.velocity_b);
    layer->updateW(learningRate * optimizerData.velocity_W);
}

void OptimizerCreation::RMSProp(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    optimizerData.G_W = beta * optimizerData.G_W + (1 - beta) * grads.grad_W.cwiseProduct(grads.grad_W);
    optimizerData.G_b = beta * optimizerData.G_b + (1 - beta) * grads.grad_b.cwiseProduct(grads.grad_b);
    layer->updateW((learningRate / (optimizerData.G_W.array().sqrt() + 1e-8)).matrix().cwiseProduct(grads.grad_W));
    layer->updateB((learningRate / (optimizerData.G_b.array().sqrt() + 1e-8)).matrix().cwiseProduct(grads.grad_b));
}

void OptimizerCreation::Adam(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    optimizerData.velocity_b = beta * optimizerData.velocity_b + (1 - beta) * grads.grad_b;
    optimizerData.velocity_W = beta * optimizerData.velocity_W + (1 - beta) * grads.grad_W;
    optimizerData.G_b = beta * optimizerData.G_b + (1 - beta) * grads.grad_b.cwiseProduct(grads.grad_b);
    optimizerData.G_W = beta * optimizerData.G_W + (1 - beta) * grads.grad_W.cwiseProduct(grads.grad_W);
    layer->updateB((learningRate / (optimizerData.G_b.array().sqrt() + 1e-8)).matrix().cwiseProduct(optimizerData.velocity_b));
    layer->updateW((learningRate / (optimizerData.G_W.array().sqrt() + 1e-8)).matrix().cwiseProduct(optimizerData.velocity_W));
}

Optimizer OptimizerCreation::getSGD(double learningRate) {
    return {learningRate, 0.9, &SGD, "SGD"};
}

Optimizer OptimizerCreation::getMomentum(double learningRate, double beta) {
    return {learningRate, beta, &Momentum, "Momentum"};
}

Optimizer OptimizerCreation::getRMSProp(double learningRate, double beta) {
    return {learningRate, beta, &RMSProp, "RMSProp"};
}

Optimizer OptimizerCreation::getAdam(double learningRate, double beta) {
    return {learningRate, beta, &Adam, "Adam"};
}

Optimizer OptimizerCreation::create(std::string type, double learningRate, double beta) {
    if (type == "SGD") return getSGD(learningRate);
    if (type == "Momentum") return getMomentum(learningRate, beta);
    if (type == "RMSProp") return getRMSProp(learningRate, beta);
    if (type == "Adam") return getAdam(learningRate, beta);
    else throw std::invalid_argument("Неизвестный тип оптимизатора: " + type);
    return getSGD(1); // по умолчанию SGD с learning rate = 1
}

}; // namespace NeuralNetwork
