#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include "./headers/DataProcessing.h"
#include <iostream>
#include <iomanip>
#include <span>

using namespace NN;

int main() {
    ActivationFunc sigmoid = ActivationCreation::getSigmod(); //функция активации сигмоида
    ActivationFunc ReLU = ActivationCreation::getReLU(); // функция активации  ReLU
    ActivationFunc softmax = ActivationCreation::getSoftmax(); // функция активации  softmax
    ActivationFunc leakyReLU = ActivationCreation::getLeakyReLU(); // функция активации LeakyReLU

    NN::NetBuilder builder(784); // задаем входной размер при создании

    builder.setLoss(LossCreation::getCrossEntropy()); // устанавливаем функцию ошибки CrossEntropy
    builder.setLayers({{128, leakyReLU}, {64, leakyReLU}, {10, softmax}}); // добавляем слои с функциями активации
    builder.setOptimizer(OptimizerCreation::getAdam(0.001, 0.8)); // устанавливаем оптимизатор SGD + Momentum

    NN::Net net = builder.createNet(); 

    // можно загрузить уже обученную нейросеть из файла, чтобы дообучить, например
    // NN::Net net = NN::NetBuilder::loadNet("../models data/he_relu_96.24.txt");

    std::string train_images_path = "../train data/MNIST numbers/train-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/train-labels.idx1-ubyte";
    

    auto train_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto train_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto train_labels = DataProcessing::labelsTransformToVector(train_labels_int); //вектор VectorXd размера 10
    auto train_images = DataProcessing::imagesTransformToVector(train_images_matrixs); // вектор векторов VectorXd размера 784

    int train_size = 60000; // размер тренировочной выборки
    // тренировочные выборки
    std::vector<Eigen::VectorXd> X(train_images.begin(), train_images.begin() + train_size);
    std::vector<Eigen::VectorXd> Y(train_labels.begin(), train_labels.begin() + train_size);

    net.train(X, Y, 2, 256);

    // сохранение весов нейросети в файл temporary_weights.txt в переданной папке
    net.saveNet("../models data");

    return 0;
}

