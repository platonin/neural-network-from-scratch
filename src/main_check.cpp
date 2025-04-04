#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include "./headers/DataProcessing.h"
#include "./headers/Visualization.h"
#include <iostream>
#include <iomanip>

using namespace NeuralNetwork;

int main() {
    std::string path_to_weigths = "../models data/temporary_weights.txt"; // последние веса нейросети
    // std::string path_to_weigths = "../models data/adam_97,14.txt"; // файл с конфигурацией на 97,14% точности

    // загрузка весов нейросети из файла
    NeuralNetwork::Net net = NeuralNetwork::NetBuilder::loadNet(path_to_weigths);

    std::string train_images_path = "../train data/MNIST numbers/t10k-images.idx3-ubyte";
    std::string train_labels_path = "../train data/MNIST numbers/t10k-labels.idx1-ubyte";
    auto test_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto test_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto test_images = DataProcessing::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

    // std::cout << "Должно быть (первые 30 чисел из выборки): \n";
    // for (int i = 0; i < 20; ++i) {
    //     std::cout << test_labels_int[i] << " ";
    // }
    // std::cout << "\n";
    // std::cout << "Предсказано: \n";
    // for (int i = 0; i < 20; ++i) {
    //     std::cout << net.predictNumber(test_images[i]) << " ";
    // }
    // std::cout << "\n";
    // std::cout << "Точность: " << net.accuracity(test_images, test_labels_int);

    int start_index = 10;
    int test_size = 10; 
    
    // картинки из выборки для вывода через opencv
    auto images_for_show = std::vector<Eigen::MatrixXd>(test_images_matrixs.begin() + start_index, test_images_matrixs.begin() + start_index + test_size);
    // предсказанные цифры
    auto num_predicts = std::vector<int>(0);
    for (int i = start_index; i < start_index + test_size; ++i) {
        num_predicts.push_back(net.predictNumber(test_images[i]));
    }
    // цифры, которые должны быть
    auto num_labels = std::vector<int>(test_labels_int.begin() + start_index, test_labels_int.begin() + start_index + test_size);

    // Visualization::showImagesRow(images_for_show); // вывод изображения через opencv
    Visualization::showImagesWithTwoNumbers(images_for_show, num_predicts, num_labels); // вывод изображения через opencv с подписями

    return 0;
}
