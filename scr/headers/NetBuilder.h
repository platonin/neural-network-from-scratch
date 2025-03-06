#ifndef NET_BUILDER_H
#define NET_BUILDER_H

#include "Net.h"

// class Net;

struct SetLayerParams {
    int layerSize;
    ActivationFunc activation;
};

class NetBuilder {
private:
    int inputSize_;
    int outputSize_;
    std::vector<LayerParams> layersParams_;
    LossFunc loss_;
    int numbersOfLayers_;
    std::vector<bool> netIsReadyToCreate_;
    std::vector<std::shared_ptr<Layer>> layers_;

    

public:
    NetBuilder(int inputSize) : inputSize_(inputSize), netIsReadyToCreate_(2, false) {}

    void setLoss(LossFunc func) {
        loss_ = func;
        netIsReadyToCreate_[0] = true;
    }

    void setLayers(const std::vector<SetLayerParams> setLayers) {
        numbersOfLayers_ = setLayers.size();
        int prev_layer_size = inputSize_;
        for (int i = 0; i < numbersOfLayers_; ++i) {
            int out_size = setLayers[i].layerSize;
            ActivationFunc activation = setLayers[i].activation;
            // std::cout << "-----: " << activation.activation(Eigen::VectorXd(10));
            layers_.push_back(std::make_shared<Layer>(prev_layer_size, out_size, activation));
            prev_layer_size = out_size;
        }
        netIsReadyToCreate_[1] = true;
    }

    bool isReady() {
        bool flag = true;
        for (int i = 0; i < netIsReadyToCreate_.size(); ++i) {
            flag = flag & netIsReadyToCreate_[i];
        }
        return flag;
    }

    Net createNet() {
        if (isReady()) {
            // копирование, а не move, чтобы NetBuilder был многоразовым
            Net net;
            net.loss_ = loss_;
            net.numbersOfLayers_ = numbersOfLayers_;
            net.layers_ = layers_; 
            return net;

        } else {
            std::cout << "Не хватает параметров для создания нейросети.\n";
        }
        return Net();
    }
};

#endif