#include "Net.h"
#include "DataLoader.h"
#include <iostream>
#include <iomanip>

int main() {
    Net net({{784, 30, "Sigmoid"}, {30, 20, "Sigmoid"}, {20, 10, "Sigmoid"}});

    net.layers_[0]->W_ = DataLoader::loadMatrix("L0W.txt");
    net.layers_[0]->b_ = DataLoader::loadVector("L0b.txt");
    net.layers_[1]->W_ = DataLoader::loadMatrix("L1W.txt");
    net.layers_[1]->b_ = DataLoader::loadVector("L1b.txt");
    net.layers_[2]->W_ = DataLoader::loadMatrix("L2W.txt");
    net.layers_[2]->b_ = DataLoader::loadVector("L2b.txt");

    string train_images_path = "../something/archive-4/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    string train_labels_path = "../something/archive-4/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
    // string train_images_path = "../something/archive-4/train-images-idx3-ubyte/train-images-idx3-ubyte";
    // string train_labels_path = "../something/archive-4/train-labels-idx1-ubyte/train-labels-idx1-ubyte";

    auto test_images_matrixs = DataLoader::loadMNISTImages(train_images_path); //вектор матриц
    auto test_labels_int = DataLoader::loadMNISTIlabels(train_labels_path); //вектор интов

    auto test_images = DataLoader::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

    std::cout << "Должно быть: \n";
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