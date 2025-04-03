#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include "./headers/DataProcessing.h"
#include <iostream>
#include <iomanip>

using namespace NeuralNetwork;

int main() {
    // std::string path_to_weigths = "../models data/temporary_weights.txt";
    std::string path_to_weigths = "../models data/weights_93_acc_momentum2.txt"; // файл с конфигурацией на 96% точности

    // загрузка весов нейросети из файла
    NeuralNetwork::Net net = NeuralNetwork::NetBuilder::loadNet(path_to_weigths);

    std::string train_images_path = "../train data/MNIST numbers/t10k-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/t10k-labels.idx1-ubyte";
    auto test_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto test_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto test_images = DataProcessing::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

    std::cout << "Должно быть (первые 30 чисел из выборки): \n";
    for (int i = 0; i < 30; ++i) {
        std::cout << test_labels_int[i] << " ";
    }
    
    std::cout << "\n";

    std::cout << "Предсказано: \n";
    for (int i = 0; i < 30; ++i) {
        std::cout << net.predict(test_images[i]) << " ";
    }
    std::cout << "\n";

    std::cout << "Точность: " << net.accuracity(test_images, test_labels_int);

    return 0;
}
