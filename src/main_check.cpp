#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include "./headers/DataProcessing.h"
#include "./headers/Visualization.h"
#include "./headers/NumberUtils.h"
#include <iostream>
#include <iomanip>

using namespace NN;

int main() {
    std::string path_to_weigths = "../models data/temporary_weights.txt"; // последние веса нейросети
    // std::string path_to_weigths = "../models data/adam_97.14.txt"; // файл с конфигурацией на 97,14% точности

    // загрузка весов нейросети из файла
    NN::Net net = NN::NetBuilder::loadNet(path_to_weigths);

    std::string test_images_path = "../train data/MNIST numbers/t10k-images.idx3-ubyte";
    std::string test_labels_path = "../train data/MNIST numbers/t10k-labels.idx1-ubyte";
    auto test_images_matrixs = DataLoader::loadMNISTImages(test_images_path); //вектор матриц
    auto test_labels_int = DataLoader::loadMNISTIlabels(test_labels_path); //вектор интов

    auto test_images = DataProcessing::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

    // Visualization::showImageQt(test_images_matrixs[0]); 

    std::cout << "Должно быть (первые 30 чисел из выборки): \n";
    for (int i = 0; i < 20; ++i) {
        std::cout << test_labels_int[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Предсказано: \n";
    for (int i = 0; i < 20; ++i) {
        std::cout << NN::NumberUtils::predictNumber(test_images[i], net) << " ";
    }
    std::cout << "\n";
    std::cout << "Точность: " << NN::NumberUtils::accuracityForNumbers(test_images, test_labels_int, net);

    return 0;
}
