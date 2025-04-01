#ifndef NET_H
#define NET_H

#include "LossFunction.h"
#include "DataLoader.h"
#include "NetTypes.h"
#include "ActivationFunction.h"
#include "Optimizer.h"

#include <iomanip> // это для вывода, надо убрать потом будет 
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>

class NetBuilder;

namespace NeuralNetwork {

class Net {
private:
    LossFunc loss_;
    int numbersOfLayers_;
    std::vector<std::shared_ptr<Layer>> layers_;
    Optimizer optimizer_;

public:
    friend class NetBuilder;
    friend std::ostream& operator<<(std::ostream& os, const Net& net);
    friend std::istream& operator>>(std::istream& is, Net& net);

    Net();

    void SaveNet2(std::string path);
    void SaveNet2(std::string path, int num);

    void setLayers(std::vector<std::shared_ptr<Layer>>& layers);

    std::vector<std::vector<forwardData>> forward_propagation(std::vector<Eigen::VectorXd>& X);
    std::vector<std::vector<layerGradData>> back_propagation(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& grads_L_x, std::vector<std::vector<forwardData>>& layers_forward_data);

    void update_weights(std::vector<std::vector<layerGradData>>& gradients_for_batch, std::vector<layerOptimizerData>& optimizerData);

    void train(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, int epochs, int batchSize);

    int predict(Eigen::VectorXd& x0);

    double accuracity(std::vector<Eigen::VectorXd>& X, std::vector<int>& Y);
    double accuracity(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y);

    void print_progress(int percent);
};

std::ostream& operator<<(std::ostream& os, const Net& net);
std::istream& operator>>(std::istream& is, Net& net);

}; // namespace NeuralNetwork

#endif
