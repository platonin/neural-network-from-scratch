#ifndef NUMBERUTILS_H
#define NUMBERUTILS_H
#include <Eigen/Dense>
#include <vector>
#include "Net.h"


namespace NN {
    namespace NumberUtils {
        using namespace std;
        using namespace Eigen;

        //возвращает число, значение нейрона на котором наибольшее значение
        int predictNumber(const VectorXd& x0, const Net& net);

        double accuracityForNumbers(span<VectorXd> X, span<int> Y, const Net& net);

        double accuracityForNumbers(span<VectorXd> X, span<VectorXd> Y, const Net& net);
    }; // namespace NumberUtils
}; // namespace NN

#endif
