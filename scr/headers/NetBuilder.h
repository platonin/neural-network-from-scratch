#ifndef NET_BUILDER_H
#define NET_BUILDER_H

#include "Net.h"

namespace NeuralNetwork {

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
    Optimizer optimizer_;

public:
    NetBuilder(int inputSize);

    void setLoss(LossFunc func);

    void setLayers(const std::vector<SetLayerParams> setLayers);

    void setOptimizer(Optimizer optimizer);

    bool isReady();

    Net createNet();
};

}; // namespace NeuralNetwork

#endif
