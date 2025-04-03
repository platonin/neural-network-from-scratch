#include "./headers/Net.h"
#include "./headers/DataLoader.h"
#include <iostream>
#include <iomanip>

using namespace NeuralNetwork;

int main() {
    std::string path_to_weigths = "../models data/temporary_weights.txt";
    // std::string path_to_weigths = "../models data/weights_96_accurency.txt"; // файл с конфигурацией на 96% точности
    std::ifstream file(path_to_weigths);

    // загрузка весов нейросети из файла
    NeuralNetwork::Net net;
    file >> net;

    std::string train_images_path = "../train data/MNIST numbers/t10k-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/t10k-labels.idx1-ubyte";
    auto test_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto test_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto test_images = DataLoader::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

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
