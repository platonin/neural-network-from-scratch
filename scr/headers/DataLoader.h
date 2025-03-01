#ifndef DATA_LOADRE_H
#define DATA_LOADRE_H

#include <iostream>
#include <fstream>
#include <vector>
// #include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iomanip>

namespace DataLoader {
    using namespace std;
    using namespace Eigen;
    // using namespace cv;

    //для чтения 32-битного int
    int readInt(ifstream &file) {
        unsigned char buffer[4];
        file.read(reinterpret_cast<char*>(buffer), 4);
        return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    }

    //для загрузки изображений MNIST
    vector<MatrixXd> loadMNISTImages(const string &path) {
        ifstream file(path, ios::binary);
        if (!file.is_open()) {
            cerr << "Не удалось открыть " << path << endl;
            exit(1);
        }

        int magic = readInt(file);
        int numImages = readInt(file);
        int rows = readInt(file);
        int cols = readInt(file);

        vector<MatrixXd> images;
        for (int i = 0; i < numImages; i++) {
            MatrixXd img(rows, cols);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    img(r, c) = file.get() / 255.0; // Нормализация

            images.push_back(img);
        }

        return images;
    }

    // загрузка меток
    vector<int> loadMNISTIlabels(const string &path) {
        ifstream file(path, ios::binary);
        if (!file.is_open()) {
            cerr << "Не удалось открыть " << path << endl;
            exit(1);
        }

        int magic = readInt(file);
        int numLabels = readInt(file);
        vector<int> labels(0);
        if (magic != 2049) {
            cout << "Неправильный формат лейблов";
        } else {
            for (int i = 0; i < numLabels; ++i) {
                int label = file.get();
                labels.push_back(label);
            }
        }
        return labels;
    }

    vector<Eigen::VectorXd> labelsTransformToVector(const vector<int>& labels_int) {
        vector<Eigen::VectorXd> labels_vector;
        for (int i = 0; i < labels_int.size(); ++i) {
            Eigen::VectorXd v(10);
            v(labels_int[i]) = 1;
            labels_vector.push_back(v);
        }
        return labels_vector;
    }

    // для передачи в нейросеть изображения как вектор
    vector<Eigen::VectorXd> imagesTransformToVector(const vector<MatrixXd>& images_matrix) {
        vector<VectorXd> images_vector;
        for (int i = 0; i < images_matrix.size(); ++i) {
            images_vector.push_back(images_matrix[i].reshaped());
        }
        return images_vector;
    }

    // для тестирования, потом уберу этот метод
    void printMatrix(const MatrixXd &mat) {
        for (int i = 0; i < mat.rows(); ++i) {
            for (int j = 0; j < mat.cols(); ++j) {
                double val = mat(i, j);
                if (val == 0) { // Проверяем, является ли число целым
                    std::cout << std::fixed << std::setprecision(0) << val << "    ";
                } else {
                    std::cout << std::fixed << std::setprecision(2) << val << " ";
                }
            }
            std::cout << std::endl;
            std::cout << "\n";
        }
    }

    // для вывода изображения через opencv
    // void showImage(const MatrixXd &image) {
    //     MatrixXd mt = image.transpose();
    //     Mat img(28, 28, CV_64F, (void*)mt.data()); // Создание OpenCV матрицы
    //     Mat img8bit;
    //     img.convertTo(img8bit, CV_8U, 255); // Конвертация в 8-битное изображение
    //     imshow("MNIST Image", img8bit);
    //     waitKey(0); // Ожидание нажатия клавиши
    // }

    //--------------------------------------------------------

    // наверное разумно вынести в какой-нибудь namespace DataSaver
    // сохранение матрицы просто в текстовом формате (в строке числа разделены через пробел, сами строки через \n)
    void saveMatrix(const Eigen::MatrixXd& matrix, const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << matrix.rows() << " " << matrix.cols() << "\n";
            file << matrix << "\n";
            file.close();
        } else {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи.\n";
        }
    }
    
    void saveVector(const Eigen::VectorXd& vector, const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << vector.size() << "\n";
            file << vector.transpose() << "\n";
            file.close();
        } else {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << " для записи.\n";
        }
    }

    Eigen::MatrixXd loadMatrix(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << "\n";
            return Eigen::MatrixXd();
        }
        
        int rows, cols;
        file >> rows >> cols; 
    
        Eigen::MatrixXd matrix(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                file >> matrix(i, j);
    
        file.close();
        return matrix;
    }
    
    Eigen::VectorXd loadVector(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Ошибка: не удалось открыть файл " << filename << "\n";
            return Eigen::VectorXd();
        }
    
        int size;
        file >> size;
    
        Eigen::VectorXd vector(size);
        for (int i = 0; i < size; ++i)
            file >> vector(i);
    
        file.close();
        return vector;
    }
};

#endif