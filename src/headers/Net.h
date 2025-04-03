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
#include <algorithm>
#include <random>

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

    vector<vector<forwardData>> forward_propagation(span<VectorXd> X);
    vector<vector<layerGradData>> back_propagation(span<VectorXd> X, const vector<VectorXd>& grads_L_x, const vector<vector<forwardData>>& layers_forward_data);
    void shuffle_train_data(span<VectorXd> X, span<VectorXd> Y);

    void update_weights(vector<vector<layerGradData>>& gradients_for_batch, vector<layerOptimizerData>& optimizerData);

public:
    Net();
    friend class NetBuilder;
    friend ostream& operator<<(ostream& os, const Net& net);
    friend istream& operator>>(istream& is, Net& net);

    

    void SaveNet(const string& path) const;
    void SaveNet(const string& path, int num) const;

    void train(span<VectorXd> X, span<VectorXd> Y, int epochs, int batchSize);

    int predict(const VectorXd& x0) const;
    VectorXd forward(const VectorXd& x) const;

    double accuracity(span<VectorXd> X, span<int> Y) const;
    double accuracity(span<VectorXd> X, span<VectorXd> Y) const;
};

ostream& operator<<(ostream& os, const Net& net);
istream& operator>>(istream& is, Net& net);

namespace Logger {
    static void printProgress(int percent);
    static void printMetrics(Net* net, span<VectorXd> X, span<VectorXd> Y, double metric);
    static void printFinish(Net* net, span<VectorXd> X, span<VectorXd> Y);
    static void printEpoch(int num);
};

}; // namespace NeuralNetwork

#endif
