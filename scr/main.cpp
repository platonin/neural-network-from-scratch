#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include <iostream>
#include <iomanip>

int main() {
    ActivationFunc sigmoid = ActivationCreation::GetSigmod();
    ActivationFunc ReLU = ActivationCreation::GetReLU();
    ActivationFunc softmax = ActivationCreation::GetSoftmax();

    NetBuilder builder(784); // задаем входной размер при создании

    builder.setLoss(LossCreation::GetCrossEntropy()); // устанавливаем функцию ошибки CrossEntropy
    builder.setLayers({{30, sigmoid}, {20, sigmoid}, {10, softmax}});


    Net net = builder.createNet();


    std::string train_images_path = "../train data/MNIST numbers/train-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/train-labels.idx1-ubyte";
    

    auto train_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto train_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto train_labels = DataLoader::labelsTransformToVector(train_labels_int); //вектор VectorXd длины 10
    auto train_images = DataLoader::imagesTransformToVector(train_images_matrixs); // вектор VectorXd длины 784

    int train_size = 6000;
    std::vector<Eigen::VectorXd> X(train_images.begin(), train_images.begin() + train_size);
    std::vector<Eigen::VectorXd> Y(train_labels.begin(), train_labels.begin() + train_size);
    
    net.train_SGD(X, Y, 4, 1, 10);

    net.SaveNet2("../models data");

    return 0;
}
