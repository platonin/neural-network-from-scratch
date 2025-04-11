#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <span>
#include "Net.h"

namespace NN {
    class Net;
    using namespace std;
    using namespace Eigen;
    namespace Logger {
        void printProgress(int percent);
        void printMetrics(const Net& net, span<VectorXd> X, span<VectorXd> Y, double metric);
        void printFinish(const Net& net, span<VectorXd> X, span<VectorXd> Y);
        void printEpoch(int num);
    };
};

#endif
