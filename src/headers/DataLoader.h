#ifndef DATA_LOADRE_H
#define DATA_LOADRE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <stdexcept>
#include "NetTypes.h"
#include "Layer.h"

class Net;

namespace NeuralNetwork {

    namespace DataLoader {
        using namespace std;
        using namespace Eigen;

        // для чтения 32-битного int
        int readInt(ifstream &file);

        // для загрузки изображений MNIST
        vector<MatrixXd> loadMNISTImages(const string &path);

        // загрузка меток
        vector<int> loadMNISTIlabels(const string &path);
    };

};

#endif
