#include "./headers/Net.h"
#include "./headers/NetBuilder.h"
#include "./headers/DataLoader.h"
#include "./headers/DataProcessing.h"
#include "./headers/NetTypes.h"

#include "./headers/Visualization.h"
#include "./headers/NumberUtils.h"
#include <iostream>
#include <iomanip>
#include <span>

using namespace std;
using namespace Eigen;

void loadAndPredict() {
    cout << "Введите имя модели для загрузки (без .txt): ";
    string model_name;
    cin >> model_name;
    string user_model_path = "../models data/" + model_name + ".txt";
    
    cout << "[*] Загружаем модель...\n";
    NN::NetBuilder builder(784);
    NN::Net net = NN::NetBuilder::loadNet(user_model_path);
    cout << "[+] Модель успешно загружена.\n";


    std::string test_images_path = "../train data/MNIST numbers/t10k-images.idx3-ubyte";
    std::string test_labels_path = "../train data/MNIST numbers/t10k-labels.idx1-ubyte";
    auto test_images_matrixs = NN::DataLoader::loadMNISTImages(test_images_path); //вектор матриц
    auto test_labels_int = NN::DataLoader::loadMNISTIlabels(test_labels_path); //вектор интов

    auto test_images = NN::DataProcessing::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

    cout << "[*] Тестируем точность на тестовой выборке MNIST...\n";
    double acc = NN::NumberUtils::accuracityForNumbers(test_images, test_labels_int, net); 
    cout << "[+] Точность на тестовой выборке: " << std::fixed << std::setprecision(2) << acc * 100 << "%\n";


    while (true) {
        int index;
        cout << "Введите индекс изображения от 0 до 9999 (-1 для выхода): ";
        cin >> index;
        if (index < 0) break;

        auto result = NN::NumberUtils::predictNumber(test_images[index], net);
        cout << "Предсказание: " << result << "\n";
        NN::Visualization::showImage(test_images_matrixs[index]); // вывод изображения через opencv

    }
}

NN::Net configureAndBuildNet() {
    using namespace NN;
    int input_size = 784;
    int output_size = 10;
    int num_layers;

    cout << "Введите количество скрытых слоев: ";
    cin >> num_layers;

    std::vector<SetLayerParams> layers;

    for (int i = 0; i < num_layers; ++i) {
        int neurons, activation_choice;
        cout << "  Количество нейронов в слое " << i + 1 << ": ";
        cin >> neurons;
        cout << "  Выберите функцию активации:\n"
             << "    1. Sigmoid\n"
             << "    2. ReLU\n"
             << "    3. LeakyReLU\n"
             << "Ваш выбор: ";
        cin >> activation_choice;

        ActivationFunc act;
        switch (activation_choice) {
            case 1: act = ActivationCreation::getSigmod(); break;
            case 2: act = ActivationCreation::getReLU(); break;
            case 3: act = ActivationCreation::getLeakyReLU(); break;
            default: cout << "Неверный выбор. Используется Sigmoid.\n"; act = ActivationCreation::getSigmod(); break;
        }

        layers.push_back({neurons, act});
    }

    // последний выходной слой с softmax
    layers.push_back({output_size, ActivationCreation::getSoftmax()});

    int loss_choice;
    cout << "Выберите функцию потерь:\n1. Cross-entropy\n2. MSE\n3. MAE\nВаш выбор: ";
    cin >> loss_choice;
    LossFunc loss;
    switch (loss_choice) {
        case 1: loss = LossCreation::getCrossEntropy(); break;
        case 2: loss = LossCreation::getMSE(); break;
        case 3: loss = LossCreation::getMAE(); break;
        default: loss = LossCreation::getCrossEntropy(); break;
    }

    // Оптимизатор
    int opt_choice;
    double lr, beta;
    cout << "Выберите оптимизатор:\n1. SGD\n2. Momentum\n3. Adam\n4. RMSprop\nВаш выбор: ";
    cin >> opt_choice;
    cout << "Введите скорость обучения: ";
    cin >> lr;
    cout << "Введите beta (0 если не используется): ";
    cin >> beta;

    Optimizer opt;
    switch (opt_choice) {
        case 1: opt = OptimizerCreation::getSGD(lr); break;
        case 2: opt = OptimizerCreation::getMomentum(lr, beta); break;
        case 3: opt = OptimizerCreation::getAdam(lr, beta); break;
        case 4: opt = OptimizerCreation::getRMSProp(lr, beta); break;
        default: opt = OptimizerCreation::getSGD(lr); break;
    }

    // Создание сети
    NN::NetBuilder builder(input_size);
    builder.setLoss(loss);
    builder.setLayers(layers);
    builder.setOptimizer(opt);

    return builder.createNet();
}

void testAccuracy(NN::Net& net) {
    std::string test_images_path = "../train data/MNIST numbers/t10k-images.idx3-ubyte";
    std::string test_labels_path = "../train data/MNIST numbers/t10k-labels.idx1-ubyte";
    auto test_images_matrixs = NN::DataLoader::loadMNISTImages(test_images_path); //вектор матриц
    auto test_labels_int = NN::DataLoader::loadMNISTIlabels(test_labels_path); //вектор интов

    auto test_images = NN::DataProcessing::imagesTransformToVector(test_images_matrixs); // вектор VectorXd длины 784

    cout << "[*] Тестируем точность на тестовой выборке MNIST...\n";
    double acc = NN::NumberUtils::accuracityForNumbers(test_images, test_labels_int, net); 
    cout << "[+] Точность на тестовой выборке: " << std::fixed << std::setprecision(2) << acc * 100 << "%\n";
}

void trainNet() {
    const std::string train_images_path = "../train data/MNIST numbers/train-images.idx3-ubyte";
    const std::string train_labels_path = "../train data/MNIST numbers/train-labels.idx1-ubyte";
    cout << "\n[*] Загружаем тренировочные данные MNIST...\n";
    auto train_images_matrixs = NN::DataLoader::loadMNISTImages(train_images_path);
    auto train_labels_int = NN::DataLoader::loadMNISTIlabels(train_labels_path);
    auto train_labels = NN::DataProcessing::labelsTransformToVector(train_labels_int);
    auto train_images = NN::DataProcessing::imagesTransformToVector(train_images_matrixs);

    int batch_size, epochs, train_size;
    cout << "Введите размер батча: ";
    cin >> batch_size;
    cout << "Введите количество эпох: ";
    cin >> epochs;
    cout << "Введите размер обучающей выборки (макс " << train_images.size() << "): ";
    cin >> train_size;

    vector<VectorXd> X(train_images.begin(), train_images.begin() + train_size);
    vector<VectorXd> Y(train_labels.begin(), train_labels.begin() + train_size);

    NN::Net net = configureAndBuildNet();
    net.train(X, Y, epochs, batch_size);

    string filename;
    cout << "Введите имя для сохранения модели (без расширения): ";
    cin >> filename;
    net.saveNet("../models data", filename);

    cout << "[+] Модель сохранена как " << filename << ".txt\n";
    cout << "Протестировать точность на тестовой выборке? (y/n): ";
    char test_choice;
    cin >> test_choice;
    if (test_choice == 'y' || test_choice == 'Y') {
        testAccuracy(net);
    }
}


int main() {
    while (true) {
        cout << "\n=== Меню ===\n";
        cout << "1 — Обучить нейросеть на MNIST\n";
        cout << "2 — Загрузить модель и сделать предсказание\n";
        cout << "3 — Выйти\n";
        cout << "Выберите пункт: ";

        int choice;
        cin >> choice;

        switch (choice) {
            case 1:

                trainNet();
                break;
            case 2:
                loadAndPredict();
                break;
            case 3:
                cout << "Выход...\n";
                return 0;
            default:
                cout << "Неверный выбор. Повторите попытку.\n";
        }
    }

    return 0;
}

