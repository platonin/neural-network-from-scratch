#include "DataLoader.h"

namespace NeuralNetwork {

namespace DataLoader {
    using namespace std;
    using namespace Eigen;

    int readInt(ifstream &file) {
        unsigned char buffer[4];
        file.read(reinterpret_cast<char*>(buffer), 4);
        return (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
    }

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

    vector<Eigen::VectorXd> imagesTransformToVector(const vector<MatrixXd>& images_matrix) {
        vector<VectorXd> images_vector;
        for (int i = 0; i < images_matrix.size(); ++i) {
            images_vector.push_back(images_matrix[i].reshaped());
        }
        return images_vector;
    }
};

};
