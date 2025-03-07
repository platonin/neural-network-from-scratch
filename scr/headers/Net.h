#ifndef NET_H
#define NET_H

#include "LossFunction.h"
#include "DataLoader.h"
#include "NetTypes.h"
#include "ActivationFunction.h"


#include <iomanip> //это для вывода, надо убрать потом будет 
class NetBuilder;

class Net {
private:
    LossFunc loss_;
    int numbersOfLayers_;
    std::vector<std::shared_ptr<Layer>> layers_;


public:
    friend class NetBuilder;
    friend std::ostream& operator<<(std::ostream& os, const Net& net);
    friend std::istream& operator>>(std::istream& is, Net& net);
    

    Net() : loss_(), numbersOfLayers_(0), layers_() {}

    // Net(std::vector<LayerParams> layersParams) : loss_(LossCreation::GetMSE()), numbersOfLayers_(layersParams.size()) {
    //     for (int i = 0; i < numbersOfLayers_; ++i) {
    //         int in_size = layersParams[i].inputSize;
    //         int out_size = layersParams[i].outputSize;
    //         ActivationFunc activation = layersParams[i].activationType;
    //         layers_.push_back(std::make_shared<Layer>(in_size, out_size, activation));
    //     }
    // }

    Net(std::vector<LayerParams> layersParams, std::vector<std::shared_ptr<Layer>> layers) : 
    loss_(LossCreation::GetMSE()),
    numbersOfLayers_(layersParams.size()), 
    layers_(layers) {}

    // сохранение конфигурации сети, всех матриц весов и сдвигов в txt файлы с названиями "L" + "<номер слоя>" + "<W или b>.txt" 
    // void SaveNet(std::string path) {
    //     std::string config = std::to_string(numbersOfLayers_) + "\n";
    //     for (int l = 0; l < numbersOfLayers_; ++l) {
    //         config += std::to_string(layers_[l]->GetInputSize()) + " ";
    //         config += std::to_string(layers_[l]->GetOutputSize()) + " ";
    //         config += layers_[l]->GetActivationType() + "\n";

    //         std::string name_W = "/L" + std::to_string(l) + "W.txt"; 
    //         std::string name_b = "/L" + std::to_string(l) + "b.txt";
    //         DataLoader::saveMatrix(layers_[l]->GetW(), path + name_W);
    //         DataLoader::saveMatrix(layers_[l]->GetB(), path + name_b);
    //     }
    //     std::string configFileName = path + "/config.txt";
    //     std::ofstream fileConfig(configFileName);
    //     fileConfig << config;
    //     fileConfig.close();
    // }

    void SaveNet2(std::string path) {
        std::string fileName = path + "/temporary_weights.txt";
        std::ofstream file(fileName);
        file << *this; //это правильно?
        file.close();
    }

    void SaveNet2(std::string path, int num) {
        std::string fileName = path + "/weights_" + std::to_string(num) + ".txt";
        std::ofstream file(fileName);
        file << *this; //это правильно?
        file.close();
    }


    void setLayers(std::vector<std::shared_ptr<Layer>>& layers) {
        if (layers.size() != numbersOfLayers_) {
            std::cout << "Некорректные данные слоев.\n";
            return;
        }

        layers_ = std::move(layers); // или тут не надо move ?
    }

    std::vector<std::vector<forwardData>> forward_propagation(std::vector<Eigen::VectorXd>& X) {
        int batchSize = X.size();
        std::vector<std::vector<forwardData>> layers_data(batchSize, std::vector<forwardData>(numbersOfLayers_));
        for (int i = 0; i < batchSize; ++i) {
            Eigen::VectorXd x_i_l = X[i];
            Eigen::VectorXd z_i_l;

            // l – номер слоя
            for (int l = 0; l < numbersOfLayers_; ++l) {
                z_i_l = layers_[l]->CalculateZ(x_i_l); // тут еще x_i_l – x из предыдущего слоя
                x_i_l = layers_[l]->CalculateX(z_i_l); // тут уже из нашего слоя l

                layers_data[i][l].z = z_i_l;
                layers_data[i][l].x = x_i_l;
            }
        }
        return layers_data;
    }

    std::vector<std::vector<layerGradData>> back_propagation(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& grads_L_x, std::vector<std::vector<forwardData>>& layers_forward_data) {
        int batchSize = grads_L_x.size();
        std::vector<std::vector<layerGradData>> layers_back_data(batchSize, std::vector<layerGradData>(numbersOfLayers_));
        for (int i = 0; i < batchSize; ++i) {
            // Eigen::VectorXd grad_L_x_i = grads_L_x[i];
            int L = numbersOfLayers_-1;

            layers_back_data[i][L].grad_b = layers_[L]->getDerActivationFromZ(layers_forward_data[i][L].z).cwiseProduct(grads_L_x[i]);
            layers_back_data[i][L].grad_W = layers_back_data[i][L].grad_b * layers_forward_data[i][L-1].x.transpose();
            // std::cout << "ok ok\n"; 

            for (int l = L-1; l > 0; --l) {
                layers_back_data[i][l].grad_b = layers_[l]->getDerActivationFromZ(layers_forward_data[i][l].z).cwiseProduct(layers_[l+1]->GetW().transpose() * layers_back_data[i][l+1].grad_b);
                layers_back_data[i][l].grad_W = layers_back_data[i][l].grad_b * layers_forward_data[i][l-1].x.transpose();
            }
            layers_back_data[i][0].grad_b = layers_[0]->getDerActivationFromZ(layers_forward_data[i][0].z).cwiseProduct(layers_[1]->GetW().transpose() * layers_back_data[i][1].grad_b);
            layers_back_data[i][0].grad_W = layers_back_data[i][0].grad_b * X[i].transpose();
        }
        return layers_back_data;
    }

    void update_weights(std::vector<std::vector<layerGradData>>& gradients_for_batch, int learningRate) {
        int batchSize = gradients_for_batch.size();
        std::vector<layerGradData> joint_grad(numbersOfLayers_);

        for (int l = 0; l < numbersOfLayers_; ++l) {
            joint_grad[l].grad_b = Eigen::VectorXd::Zero(layers_[l]->GetOutputSize());
            joint_grad[l].grad_W = Eigen::MatrixXd::Zero(layers_[l]->GetOutputSize(), layers_[l]->GetInputSize());
            for (int i = 0; i < batchSize; ++i) {
                joint_grad[l].grad_b += 1.0/batchSize * gradients_for_batch[i][l].grad_b;
                joint_grad[l].grad_W += 1.0/batchSize * gradients_for_batch[i][l].grad_W;
            }
        }

        for (int l = 0; l < numbersOfLayers_; ++l) {
            // std::cout << "размер слоя: " << layers_[l]->GetOutputSize() << " размер градиента: " << joint_grad[l].grad_b.size();
            layers_[l]->UpdateB(joint_grad[l].grad_b, learningRate);
            layers_[l]->UpdateW(joint_grad[l].grad_W, learningRate);
        }
    }

    void train_SGD(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y, int epochs, double learningRate, int batchSize) {
        int numberOfBatch = X.size()/batchSize; // надо сделать, чтобы если нацело не делится, то захватывался последний неполноценный батч
        std::cout << "Количество батчей: " << numberOfBatch << "\n";
        for (int numberEpoch = 1; numberEpoch <= epochs; ++numberEpoch) {
            std::cout << "\nЭпоха номер: " << numberEpoch << "\n";
            double metric = 0;
            for (int i = 0; i < numberOfBatch; ++i) {
                std::vector<Eigen::VectorXd> batch_x_i(X.begin() + batchSize*i, X.begin() + batchSize*(i+1)); // вообще надо избавиться от копирования
                std::vector<Eigen::VectorXd> batch_y_i(Y.begin() + batchSize*i, Y.begin() + batchSize*(i+1)); // вообще надо избавиться от копирования

                std::vector<std::vector<forwardData>> layers_forward_data_batch_i = forward_propagation(batch_x_i); // в [i][j] хранятся параметры для i-ого элемента в батче и (j+1)-ого слоя

                std::vector<Eigen::VectorXd> grads_L_x(batchSize);
                for (int num_batch = 0; num_batch < batchSize; ++num_batch) {
                    grads_L_x[num_batch] = loss_.lossDerivative(layers_forward_data_batch_i[num_batch][2].x, batch_y_i[num_batch]);
                }

                // вообще это для вывода ошибки после каждой эпохи, но оно как-то криво считается, потом поправлю
                // for (int num_batch = 0; num_batch < batchSize; ++num_batch) {
                //     metric += (1.0/X.size()) * loss_.lossFunction(layers_forward_data_batch_i[num_batch][2].x, batch_y_i[num_batch]);
                // }

                std::vector<std::vector<layerGradData>> gradients_for_batch = back_propagation(batch_x_i, grads_L_x, layers_forward_data_batch_i);

                update_weights(gradients_for_batch, learningRate);

                print_progress(round((double)i/(numberOfBatch-1) * 100));
            }
            // SaveNet("../models data/temporary weights"); // пока после каждой эпохи сохраняются веса, но вообще надо сделать это опциональным аргументом, чтобы можно было выбрать сохранять или нет
        }
        
        std::cout << "\nОбучение завершено. Точность на тренировочной выборке: " << accuracity(X, Y) * 100.0 << "%\n";
    }
    
    //только для чисел из mnist пока (а так в общем виде надо возвращать вектор выходной длины)
    int predict(Eigen::VectorXd& x0) {
        Eigen::VectorXd x_i = x0;
        Eigen::VectorXd z_i;
        for (int l = 0; l < numbersOfLayers_; ++l) {
            // std::cout << "в predict слой " << l << "\n";
            layers_[0];
            // std::cout << "------" << "\n";
            z_i = layers_[l]->CalculateZ(x_i);
            // std::cout << "------" << "\n";
            x_i = layers_[l]->CalculateX(z_i);
            // std::cout << "------" << "\n";
        }
        
        int mx_ind = 0;
        double mx = 0;

        for (int i = 0; i < x_i.size(); ++i) {
            if (x_i(i) > mx) {
                mx = x_i(i);
                mx_ind = i;
            }
        }
        return mx_ind;
    }

    //тоже пока только для mnist
    double accuracity(std::vector<Eigen::VectorXd>& X, std::vector<int>& Y) {
        if (X.size() != Y.size()) {
            std::cout << "Некорректные данные\n";
            return -1;
        }
        int count = 0;
        for (int i = 0; i < X.size(); ++i) {
            if (predict(X[i]) == Y[i]) {
                count++;
            }
        }
        return ((double)count) / X.size();
    }

    // принимает массив входных векторов и соответствующих правильынх выходных векторов
    double accuracity(std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Y) {
        if (X.size() != Y.size()) {
            std::cout << "Некорректные данные\n";
            return -1;
        }

        int count = 0;
        for (int i = 0; i < X.size(); ++i) {
            double mx = Y[i].maxCoeff();
            int fact_val = 0;
            for (int j = 0; j < Y[i].size(); ++j) {
                if(Y[i](j) == mx) {
                    fact_val = j;
                    break;
                }
            }
            
            if (predict(X[i]) == fact_val) {
                count++;
            }
        }
        return ((double)count) / X.size();
    }

    // для красивого вывода полосочек прогресса во время обчуения (наверное, разумно в отдельный класс вынести аля Visualizer)
    void print_progress(int percent) {
        std::cout << "\r";
        for (int i = 0; i < percent/2; ++i) std::cout << "█";
        for (int i = percent/2; i < 50; ++i) std::cout << "░";
        std::cout << " " << percent << "%";
        std::cout .flush();
    }
};


std::ostream& operator<<(std::ostream& os, const Net& net) {
    os << std::to_string(net.numbersOfLayers_) << " " << net.loss_.Type << "\n";
    //наверное размеры слоев отдельно не нужны
    // for (int l = 0; l < net.numbersOfLayers_; ++l) {
    //     os << std::to_string(net.layers_[l]->GetInputSize()) + " ";
    //     os << std::to_string(net.layers_[l]->GetOutputSize()) + " ";
    //     os << net.layers_[l]->GetActivationType() + "\n";
    // }

    for (int l = 0; l < net.numbersOfLayers_; ++l) {
        const Eigen::MatrixXd& W = net.layers_[l]->GetW();
        os << W.rows() << " " << W.cols() << " " << net.layers_[l]->GetActivationType() << "\n" << W << "\n";

        const Eigen::VectorXd& B = net.layers_[l]->GetB();
        os << B.rows() << " " << B.cols() << "\n" << B << "\n";
    }
    return os;
}

std::istream& operator>>(std::istream& is, Net& net) {
    int numberOfLayers;
    std::string lossType;
    is >> numberOfLayers >> lossType;
    std::vector<std::shared_ptr<Layer>> layers(numberOfLayers);
    // for (int l = 0; l < numberOfLayers; ++l) {
    //     int inSize, outSize;
    //     std::string activation;
    //     is >> inSize >> outSize >> activation;
    //     layers[l]->inputSize_ = inSize;
    //     layers[l]->outputSize_ = outSize;
    //     layers[l]->activationFunction_ = ActivationCreation::create(activation);
    // }
    
    for (int l = 0; l < numberOfLayers; ++l) {
        int inSize, outSize;
        std::string activation;
        is >> outSize >> inSize >> activation;

        Eigen::MatrixXd W(outSize, inSize); // с размерами все норм?
        for (int i = 0; i < outSize; ++i)
            for (int j = 0; j < inSize; ++j)
                is >> W(i, j);
        
        is >> outSize >> inSize;
        Eigen::VectorXd b(outSize); // с размерами все норм?
        for (int i = 0; i < outSize; ++i)
            is >> b(i);

        // тут по идее move (ну оно в конструкторе должно быть вроде)
        layers[l] = std::make_shared<Layer>(W, b, ActivationCreation::create(activation));
    }

    net.numbersOfLayers_ = numberOfLayers;
    net.loss_ = LossCreation::create(lossType);
    net.layers_ = std::move(layers);
    return is;
}

#endif