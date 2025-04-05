#include "NetBuilder.h"
#include <iostream>

namespace NN {

NetBuilder::NetBuilder(int inputSize) : inputSize_(inputSize), netIsReadyToCreate_(3, false) {}

void NetBuilder::setLoss(LossFunc func) {
    loss_ = func;
    netIsReadyToCreate_[0] = true;
}

void NetBuilder::setLayers(const std::vector<SetLayerParams> setLayers) {
    numbersOfLayers_ = setLayers.size();
    int prev_layer_size = inputSize_;
    for (int i = 0; i < numbersOfLayers_; ++i) {
        int out_size = setLayers[i].layerSize;
        ActivationFunc activation = setLayers[i].activation;
        layers_.push_back(std::make_shared<Layer>(prev_layer_size, out_size, activation));
        prev_layer_size = out_size;
    }
    netIsReadyToCreate_[1] = true;
}

void NetBuilder::setOptimizer(Optimizer optimizer) {
    optimizer_ = optimizer;
    netIsReadyToCreate_[2] = true;
}

bool NetBuilder::isReady() {
    bool flag = true;
    for (int i = 0; i < netIsReadyToCreate_.size(); ++i) {
        flag = flag & netIsReadyToCreate_[i];
    }
    return flag;
}

Net NetBuilder::loadNet(const std::string& path) {
    Net net;
    std::ifstream file(path);
    if (file.is_open()) {
        file >> net;
        file.close();
    } else {
        throw std::runtime_error("Не удалось открыть файл: " + path);
    }
    return net;
}

Net NetBuilder::createNet() {
    if (isReady()) {
        // копирование, а не move, чтобы NetBuilder был многоразовым
        Net net;
        net.loss_ = loss_;
        net.numbersOfLayers_ = numbersOfLayers_;
        net.layers_ = layers_; 
        net.optimizer_ = optimizer_;
        return net;
    }
    // тут мне кажется, что лучше ошибку кидать O_O
    throw std::runtime_error("Не хватает параметров для создания нейросети.");
}

}; // namespace NeuralNetwork
