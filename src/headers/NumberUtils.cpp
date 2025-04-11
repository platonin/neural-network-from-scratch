#include "NumberUtils.h"

namespace NN {
    namespace NumberUtils {
        using namespace std;
        using namespace Eigen;

        int predictNumber(const VectorXd& x0, const Net& net) {
            VectorXd x = net.forward(x0); 
            int mx_ind = 0;
            double mx = 0;

            for (int i = 0; i < x.size(); ++i) {
                if (x(i) > mx) {
                    mx = x(i);
                    mx_ind = i;
                }
            }
            return mx_ind;
        }

        double accuracityForNumbers(span<VectorXd> X, span<int> Y, const Net& net) {
            if (X.size() != Y.size()) {
                cout << "Некорректные данные\n";
                return -1;
            }
            int count = 0;
            for (int i = 0; i < X.size(); ++i) {
                if (predictNumber(X[i], net) == Y[i]) {
                    count++;
                }
            }
            return ((double)count) / X.size();
        }

        double accuracityForNumbers(span<VectorXd> X, span<VectorXd> Y, const Net& net) {
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
                
                if (predictNumber(X[i], net) == fact_val) {
                    count++;
                }
            }
            return ((double)count) / X.size();
        }
    }; // namespace NumberUtils
}; // namespace NN
