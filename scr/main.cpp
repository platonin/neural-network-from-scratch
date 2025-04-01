#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include <iostream>
#include <iomanip>

using namespace NeuralNetwork;

int main() {
    ActivationFunc sigmoid = ActivationCreation::GetSigmod();
    ActivationFunc ReLU = ActivationCreation::GetReLU();
    ActivationFunc softmax = ActivationCreation::GetSoftmax();
    ActivationFunc leakyReLU = ActivationCreation::GetLeakyReLU();

    NeuralNetwork::NetBuilder builder(784); // задаем входной размер при создании

    builder.setLoss(LossCreation::GetCrossEntropy()); // устанавливаем функцию ошибки CrossEntropy
    builder.setLayers({{30, sigmoid}, {20, sigmoid}, {10, softmax}}); // добавляем слои с функциями активации
    // builder.setOptimizer(OptimizerCreation::GetSGD(1)); // устанавливаем оптимизатор SGD
    // builder.setOptimizer(OptimizerCreation::GetRMSProp(0.01, 0.9)); // устанавливаем оптимизатор SGD + Momentum
    builder.setOptimizer(OptimizerCreation::GetAdam(0.01, 0.9)); // устанавливаем оптимизатор SGD + Momentum

    NeuralNetwork::Net net = builder.createNet();

    std::string train_images_path = "../train data/MNIST numbers/train-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/train-labels.idx1-ubyte";
    

    auto train_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto train_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto train_labels = DataLoader::labelsTransformToVector(train_labels_int); //вектор VectorXd длины 10
    auto train_images = DataLoader::imagesTransformToVector(train_images_matrixs); // вектор VectorXd длины 784

    int train_size = 6000; // размер тренировочной выборки
    // тренировочные выборки
    std::vector<Eigen::VectorXd> X(train_images.begin(), train_images.begin() + train_size);
    std::vector<Eigen::VectorXd> Y(train_labels.begin(), train_labels.begin() + train_size);
    
    // net.train_SGD(X, Y, 4, 1, 10);
    // net.train_SGD(X, Y, 4, 0.1, 10);
    // net.train(X, Y, 4, 1, 0.9, 100);
    net.train(X, Y, 4, 10);

    // сохранение весов нейросети в файл temporary_weights.txt в переданной папке
    net.SaveNet2("../models data");

    return 0;
}
