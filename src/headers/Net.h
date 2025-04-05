#ifndef NET_H
#define NET_H

#include "LossFunction.h"
#include "DataLoader.h"
#include "NetTypes.h"
#include "ActivationFunction.h"
#include "Optimizer.h"
#include "Logger.h"

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

namespace NN {
    using namespace std;
    using namespace Eigen;

    class Net {
    private:
        LossFunc loss_;
        int numbersOfLayers_;
        vector<shared_ptr<Layer>> layers_;
        Optimizer optimizer_;

        Net();

        vector<vector<forwardData>> forwardPropagation(span<VectorXd> X);
        vector<vector<layerGradData>> backPropagation(span<VectorXd> X, const vector<VectorXd>& grads_L_x, const vector<vector<forwardData>>& layers_forward_data);
        void shuffleTrainData(span<VectorXd> X, span<VectorXd> Y);

        void updateWeights(vector<vector<layerGradData>>& gradients_for_batch, vector<layerOptimizerData>& optimizerData);

    public:
        friend class NetBuilder;
        friend ostream& operator<<(ostream& os, const Net& net);
        friend istream& operator>>(istream& is, Net& net);

        

        void saveNet(const string& path) const;
        void saveNet(const string& path, int num) const;

        void train(span<VectorXd> X, span<VectorXd> Y, int epochs, int batchSize);

        // int predictNumber(const VectorXd& x0) const;
        VectorXd forward(const VectorXd& x) const;

        // double accuracity(span<VectorXd> X, span<int> Y) const;
        // double accuracity(span<VectorXd> X, span<VectorXd> Y) const;
    };

    ostream& operator<<(ostream& os, const Net& net);
    istream& operator>>(istream& is, Net& net);

}; // namespace NeuralNetwork

#endif
