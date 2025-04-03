#include "Net.h"

namespace NeuralNetwork {

Net::Net() : loss_(), numbersOfLayers_(0), layers_(), optimizer_(OptimizerCreation::getSGD(1)) {}

void Net::saveNet(const string& path) const {
    string fileName = path + "/temporary_weights.txt";
    ofstream file(fileName);
    file << *this; //это правильно?
    file.close();
}

void Net::saveNet(const string& path, int num) const {
    string fileName = path + "/weights_" + to_string(num) + ".txt";
    ofstream file(fileName);
    file << *this; //это правильно?
    file.close();
}

vector<vector<forwardData>> Net::forwardPropagation(span<VectorXd> X) {
    int batchSize = X.size();
    vector<vector<forwardData>> layers_data(batchSize, vector<forwardData>(numbersOfLayers_));
    for (int i = 0; i < batchSize; ++i) {
        VectorXd x_i_l = X[i];
        VectorXd z_i_l;

        // l – номер слоя
        for (int l = 0; l < numbersOfLayers_; ++l) {
            z_i_l = layers_[l]->calculateZ(x_i_l); // тут еще x_i_l – x из предыдущего слоя
            x_i_l = layers_[l]->calculateX(z_i_l); // тут уже из нашего слоя l

            layers_data[i][l].z = z_i_l;
            layers_data[i][l].x = x_i_l;
        }
    }
    return layers_data;
}

vector<vector<layerGradData>> Net::backPropagation(span<VectorXd> X, const vector<VectorXd>& grads_L_x, const vector<vector<forwardData>>& layers_forward_data) {
    int batchSize = grads_L_x.size();
    vector<vector<layerGradData>> layers_back_data(batchSize, vector<layerGradData>(numbersOfLayers_));
    for (int i = 0; i < batchSize; ++i) {
        // VectorXd grad_L_x_i = grads_L_x[i];
        int L = numbersOfLayers_-1;

        // layers_back_data[i][L].grad_b = layers_[L]->calculateActivationDer(layers_forward_data[i][L].z).cwiseProduct(grads_L_x[i]);
        layers_back_data[i][L].grad_b = layers_[L]->calculateActivationDer(layers_forward_data[i][L].z) * grads_L_x[i]; // стало matrix[KxK] * matrix[Kx1] = matrix[Kx1]
        layers_back_data[i][L].grad_W = layers_back_data[i][L].grad_b * layers_forward_data[i][L-1].x.transpose();
        // cout << "ok ok\n"; 

        for (int l = L-1; l > 0; --l) {
            // layers_back_data[i][l].grad_b = layers_[l]->calculateActivationDer(layers_forward_data[i][l].z).cwiseProduct(layers_[l+1]->getW().transpose() * layers_back_data[i][l+1].grad_b);
            layers_back_data[i][l].grad_b = layers_[l]->calculateActivationDer(layers_forward_data[i][l].z) * layers_[l+1]->getW().transpose() * layers_back_data[i][l+1].grad_b;
            layers_back_data[i][l].grad_W = layers_back_data[i][l].grad_b * layers_forward_data[i][l-1].x.transpose();
        }
        layers_back_data[i][0].grad_b = layers_[0]->calculateActivationDer(layers_forward_data[i][0].z) * layers_[1]->getW().transpose() * layers_back_data[i][1].grad_b;
        layers_back_data[i][0].grad_W = layers_back_data[i][0].grad_b * X[i].transpose();
    }
    return layers_back_data;
}

void Net::updateWeights(vector<vector<layerGradData>>& gradients_for_batch, vector<layerOptimizerData>& optimizerData) {
    int batchSize = gradients_for_batch.size();
    vector<layerGradData> joint_grad(numbersOfLayers_);

    for (int l = 0; l < numbersOfLayers_; ++l) {
        joint_grad[l].grad_b = VectorXd::Zero(layers_[l]->getOutputSize());
        joint_grad[l].grad_W = MatrixXd::Zero(layers_[l]->getOutputSize(), layers_[l]->getInputSize());
        for (int i = 0; i < batchSize; ++i) {
            joint_grad[l].grad_b += 1.0/batchSize * gradients_for_batch[i][l].grad_b;
            joint_grad[l].grad_W += 1.0/batchSize * gradients_for_batch[i][l].grad_W;
        }
    }

    for (int l = 0; l < numbersOfLayers_; ++l) {
        optimizer_.updateWeights(layers_[l], optimizerData[l], optimizer_.learningRate, optimizer_.beta, joint_grad[l]);
    }
}

// чтобы перемешивать тренировочную выборку перед каждой эпохой
void Net::shuffleTrainData(span<VectorXd> X, span<VectorXd> Y) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<size_t> indices(X.size());
    for (size_t i = 0; i < indices.size(); ++i)
        indices[i] = i;

    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<VectorXd> X_shuffled(X.size());
    std::vector<VectorXd> Y_shuffled(Y.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        X_shuffled[i] = X[indices[i]];
        Y_shuffled[i] = Y[indices[i]];
    }

    std::copy(X_shuffled.begin(), X_shuffled.end(), X.begin());
    std::copy(Y_shuffled.begin(), Y_shuffled.end(), Y.begin());
}

void Net::train(span<VectorXd> X, span<VectorXd> Y, int epochs, int batchSize) {
    int numberOfBatch = X.size()/batchSize; // надо сделать, чтобы если нацело не делится, то захватывался последний неполноценный батч
    for (int numberEpoch = 1; numberEpoch <= epochs; ++numberEpoch) {
        Logger::printEpoch(numberEpoch);
        double metric = 0;

        // инициализация optimizerData нулевыми значениями для матриц весов и сдвигов каждого слоя
        vector<layerOptimizerData> optimizerData(numbersOfLayers_);
        for (int l = 0; l < numbersOfLayers_; ++l) {
            optimizerData[l].velocity_W = MatrixXd::Zero(layers_[l]->getOutputSize(), layers_[l]->getInputSize());
            optimizerData[l].velocity_b = VectorXd::Zero(layers_[l]->getOutputSize());
            optimizerData[l].G_W = MatrixXd::Zero(layers_[l]->getOutputSize(), layers_[l]->getInputSize());
            optimizerData[l].G_b = VectorXd::Zero(layers_[l]->getOutputSize());
        }

        shuffleTrainData(X, Y); // перемешиваем выборку 

        for (int i = 0; i < numberOfBatch; ++i) {
            span<VectorXd> batch_x_i(X.begin() + batchSize*i, X.begin() + batchSize*(i+1));
            span<VectorXd> batch_y_i(Y.begin() + batchSize*i, Y.begin() + batchSize*(i+1));

            vector<vector<forwardData>> layers_forward_data_batch_i = forwardPropagation(batch_x_i); // в [i][j] хранятся параметры для i-ого элемента в батче и (j+1)-ого слоя

            vector<VectorXd> grads_L_x(batchSize);
            for (int num_batch = 0; num_batch < batchSize; ++num_batch) {
                grads_L_x[num_batch] = loss_.lossDerivative(layers_forward_data_batch_i[num_batch][numbersOfLayers_-1].x, batch_y_i[num_batch]);
            }

            // для вывода статистики
            for (int num_batch = 0; num_batch < batchSize; ++num_batch) {
                metric += (1.0/X.size()) * loss_.lossFunction(layers_forward_data_batch_i[num_batch][2].x, batch_y_i[num_batch]);
            }

            vector<vector<layerGradData>> gradients_for_batch = backPropagation(batch_x_i, grads_L_x, layers_forward_data_batch_i);

            updateWeights(gradients_for_batch, optimizerData);

            Logger::printProgress(round((double)i/(numberOfBatch-1) * 100));
        }
        
        Logger::printMetrics(this, X, Y, metric);
    }
    
    Logger::printFinish(this, X, Y);
}

//только для чисел из mnist пока (а так в общем виде надо возвращать вектор выходной длины)
int Net::predict(const VectorXd& x0) const {
    VectorXd x_i = x0;
    VectorXd z_i;
    for (int l = 0; l < numbersOfLayers_; ++l) {
        layers_[0];
        z_i = layers_[l]->calculateZ(x_i);
        x_i = layers_[l]->calculateX(z_i);
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

VectorXd Net::forward(const VectorXd& x) const {
    VectorXd x_i = x;
    VectorXd z_i;
    for (int l = 0; l < numbersOfLayers_; ++l) {
        x_i = layers_[l]->forward(x_i);
    }
    return x_i;
}

//тоже пока только для mnist
double Net::accuracity(span<VectorXd> X, span<int> Y) const {
    if (X.size() != Y.size()) {
        cout << "Некорректные данные\n";
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
double Net::accuracity(span<VectorXd> X, span<VectorXd> Y) const {
    if (X.size() != Y.size()) {
        cout << "Некорректные данные\n";
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

ostream& operator<<(ostream& os, const Net& net) {
    os << to_string(net.numbersOfLayers_) << " " << net.loss_.Type << "\n";
    os << net.optimizer_.Type << " " << to_string(net.optimizer_.learningRate) << " " << to_string(net.optimizer_.beta) << "\n";

    for (int l = 0; l < net.numbersOfLayers_; ++l) {
        const MatrixXd& W = net.layers_[l]->getW();
        os << W.rows() << " " << W.cols() << " " << net.layers_[l]->getActivationType() << "\n" << W << "\n";

        const VectorXd& B = net.layers_[l]->getB();
        os << B.rows() << " " << B.cols() << "\n" << B << "\n";
    }
    return os;
}

istream& operator>>(istream& is, Net& net) {
    int numberOfLayers;
    string lossType;
    string optimizerType;
    double learningRate, beta;
    is >> numberOfLayers >> lossType;
    is >> optimizerType >> learningRate >> beta;
    vector<shared_ptr<Layer>> layers(numberOfLayers);
    
    for (int l = 0; l < numberOfLayers; ++l) {
        int inSize, outSize;
        string activation;
        is >> outSize >> inSize >> activation;

        MatrixXd W(outSize, inSize); // с размерами все норм?
        for (int i = 0; i < outSize; ++i) {
            for (int j = 0; j < inSize; ++j) {
                string t;
                is >> t;
                W(i, j) = std::stod(t);
            }
        }
        
        is >> outSize >> inSize;
        VectorXd b(outSize); // с размерами все норм?
        for (int i = 0; i < outSize; ++i)
            is >> b(i);

        // тут по идее move (ну оно в конструкторе должно быть вроде)
        layers[l] = make_shared<Layer>(W, b, ActivationCreation::create(activation));
    }

    net.numbersOfLayers_ = numberOfLayers;
    net.loss_ = LossCreation::create(lossType);
    net.optimizer_ = OptimizerCreation::create(optimizerType, learningRate, beta);
    net.layers_ = std::move(layers);
    return is;
}

static void Logger::printProgress(int percent) {
    cout << "\r";
    for (int i = 0; i < percent/2; ++i) cout << "█";
    for (int i = percent/2; i < 50; ++i) cout << "░";
    cout << " " << percent << "%";
    cout .flush();
}

void Logger::printMetrics(Net* net, span<VectorXd> X, span<VectorXd> Y, double metric) {
    cout << "\nТочность: " << net->accuracity(X, Y) * 100.0 << "% ";
    cout << "Ошибка: " << metric << "\n";
}

void Logger::printFinish(Net* net, span<VectorXd> X, span<VectorXd> Y) {
    cout << "\nОбучение завершено. Точность на тренировочной выборке: " << net->accuracity(X, Y) * 100.0 << "%\n";
}

void Logger::printEpoch(int num) {
    cout << "\nЭпоха номер: " << num << "\n";
}

}; // namespace NeuralNetwork
