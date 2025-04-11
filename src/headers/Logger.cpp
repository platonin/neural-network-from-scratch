#include "Logger.h"
#include "NumberUtils.h"

namespace NN {
    void Logger::printProgress(int percent) {
        cout << "\r";
        for (int i = 0; i < percent/2; ++i) cout << "█";
        for (int i = percent/2; i < 50; ++i) cout << "░";
        cout << " " << percent << "%";
        cout .flush();
    }
    
    void Logger::printMetrics(const Net& net, span<VectorXd> X, span<VectorXd> Y, double metric) {
        cout << "\nТочность: " << NumberUtils::accuracityForNumbers(X, Y, net) * 100.0 << "% ";
        cout << "Ошибка: " << metric << "\n";
    }
    
    void Logger::printFinish(const Net& net, span<VectorXd> X, span<VectorXd> Y) {
        cout << "\nОбучение завершено. Точность на тренировочной выборке: " << NumberUtils::accuracityForNumbers(X, Y, net) * 100.0 << "%\n";
    }
    
    void Logger::printEpoch(int num) {
        cout << "\nЭпоха номер: " << num << "\n";
    }
}; 
