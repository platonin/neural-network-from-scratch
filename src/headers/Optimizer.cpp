#include "Optimizer.h"

namespace NeuralNetwork {

void OptimizerCreation::SGD(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    layer->UpdateW(learningRate * grads.grad_W);
    layer->UpdateB(learningRate * grads.grad_b);
}

void OptimizerCreation::Momentum(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    optimizerData.velocity_b = beta * optimizerData.velocity_b + (1 - beta) * grads.grad_b;
    optimizerData.velocity_W = beta * optimizerData.velocity_W + (1 - beta) * grads.grad_W;
    layer->UpdateB(learningRate * optimizerData.velocity_b);
    layer->UpdateW(learningRate * optimizerData.velocity_W);
}

void OptimizerCreation::RMSProp(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    optimizerData.G_W = beta * optimizerData.G_W + (1 - beta) * grads.grad_W.cwiseProduct(grads.grad_W);
    optimizerData.G_b = beta * optimizerData.G_b + (1 - beta) * grads.grad_b.cwiseProduct(grads.grad_b);
    layer->UpdateW((learningRate / (optimizerData.G_W.array().sqrt() + 1e-8)).matrix().cwiseProduct(grads.grad_W));
    layer->UpdateB((learningRate / (optimizerData.G_b.array().sqrt() + 1e-8)).matrix().cwiseProduct(grads.grad_b));
}

void OptimizerCreation::Adam(std::shared_ptr<Layer>& layer, layerOptimizerData& optimizerData, double learningRate, double beta, layerGradData& grads) {
    optimizerData.velocity_b = beta * optimizerData.velocity_b + (1 - beta) * grads.grad_b;
    optimizerData.velocity_W = beta * optimizerData.velocity_W + (1 - beta) * grads.grad_W;
    optimizerData.G_b = beta * optimizerData.G_b + (1 - beta) * grads.grad_b.cwiseProduct(grads.grad_b);
    optimizerData.G_W = beta * optimizerData.G_W + (1 - beta) * grads.grad_W.cwiseProduct(grads.grad_W);
    layer->UpdateB((learningRate / (optimizerData.G_b.array().sqrt() + 1e-8)).matrix().cwiseProduct(optimizerData.velocity_b));
    layer->UpdateW((learningRate / (optimizerData.G_W.array().sqrt() + 1e-8)).matrix().cwiseProduct(optimizerData.velocity_W));
}

Optimizer OptimizerCreation::GetSGD(double learningRate) {
    return {learningRate, 0.9, &SGD, "SGD"};
}

Optimizer OptimizerCreation::GetMomentum(double learningRate, double beta) {
    return {learningRate, beta, &Momentum, "Momentum"};
}

Optimizer OptimizerCreation::GetRMSProp(double learningRate, double beta) {
    return {learningRate, beta, &RMSProp, "RMSProp"};
}

Optimizer OptimizerCreation::GetAdam(double learningRate, double beta) {
    return {learningRate, beta, &Adam, "Adam"};
}

Optimizer OptimizerCreation::create(std::string type, double learningRate, double beta) {
    if (type == "SGD") return GetSGD(learningRate);
    if (type == "Momentum") return GetMomentum(learningRate, beta);
    if (type == "RMSProp") return GetRMSProp(learningRate, beta);
    if (type == "Adam") return GetAdam(learningRate, beta);
    else std::cerr << "Неизвестный тип оптимизатора: " << type;
    return GetSGD(1); // по умолчанию SGD с learning rate = 1
}

}; // namespace NeuralNetwork
