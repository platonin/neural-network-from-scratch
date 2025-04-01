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
#include <utility>
#include <span>

class NetBuilder;

namespace NeuralNetwork {

using namespace std;
using namespace Eigen;

class Net {
private:
    LossFunc loss_;
    int numbersOfLayers_;
    vector<shared_ptr<Layer>> layers_;
    Optimizer optimizer_;

public:
    friend class NetBuilder;
    friend ostream& operator<<(ostream& os, const Net& net);
    friend istream& operator>>(istream& is, Net& net);

    Net();

    void SaveNet(string path);
    void SaveNet(string path, int num);

    void setLayers(vector<shared_ptr<Layer>>& layers);

    vector<vector<forwardData>> forward_propagation(vector<VectorXd>& X);
    vector<vector<layerGradData>> back_propagation(vector<VectorXd>& X, vector<VectorXd>& grads_L_x, vector<vector<forwardData>>& layers_forward_data);

    void update_weights(vector<vector<layerGradData>>& gradients_for_batch, vector<layerOptimizerData>& optimizerData);

    void train(vector<VectorXd>& X, vector<VectorXd>& Y, int epochs, int batchSize);

    int predict(VectorXd& x0);

    double accuracity(vector<VectorXd>& X, vector<int>& Y);
    double accuracity(vector<VectorXd>& X, vector<VectorXd>& Y);

    void print_progress(int percent);
};

ostream& operator<<(ostream& os, const Net& net);
istream& operator>>(istream& is, Net& net);

}; // namespace NeuralNetwork

#endif
