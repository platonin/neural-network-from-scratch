#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include "./headers/DataProcessing.h"
#include <iostream>
#include <iomanip>
#include <span>

using namespace NeuralNetwork;

int main() {
    ActivationFunc sigmoid = ActivationCreation::getSigmod();
    ActivationFunc ReLU = ActivationCreation::getReLU();
    ActivationFunc softmax = ActivationCreation::getSoftmax();
    ActivationFunc leakyReLU = ActivationCreation::getLeakyReLU();

    NeuralNetwork::NetBuilder builder(784); // задаем входной размер при создании

    builder.setLoss(LossCreation::getCrossEntropy()); // устанавливаем функцию ошибки CrossEntropy
    builder.setLayers({{30, sigmoid}, {20, sigmoid}, {10, softmax}}); // добавляем слои с функциями активации
    // builder.setOptimizer(OptimizerCreation::getSGD(1)); // устанавливаем оптимизатор SGD
    // builder.setOptimizer(OptimizerCreation::getRMSProp(0.01, 0.9)); // устанавливаем оптимизатор SGD + Momentum
    builder.setOptimizer(OptimizerCreation::getAdam(0.01, 0.9)); // устанавливаем оптимизатор SGD + Momentum

    NeuralNetwork::Net net = builder.createNet();

    std::string train_images_path = "../train data/MNIST numbers/train-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/train-labels.idx1-ubyte";
    

    auto train_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto train_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto train_labels = DataProcessing::labelsTransformToVector(train_labels_int); //вектор VectorXd длины 10
    auto train_images = DataProcessing::imagesTransformToVector(train_images_matrixs); // вектор VectorXd длины 784

    int train_size = 6000; // размер тренировочной выборки
    // тренировочные выборки
    std::vector<Eigen::VectorXd> X(train_images.begin(), train_images.begin() + train_size);
    std::vector<Eigen::VectorXd> Y(train_labels.begin(), train_labels.begin() + train_size);

    // auto start_time = std::chrono::high_resolution_clock::now();
    net.train(X, Y, 5, 10);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_time = end_time - start_time;
    // std::cout << "Training time: " << elapsed_time.count() << " seconds" << std::endl;


    // сохранение весов нейросети в файл temporary_weights.txt в переданной папке
    net.saveNet("../models data");

    return 0;
}
