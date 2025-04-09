#include "Visualization.h"

namespace NN {
    namespace Visualization {

        QImage matrixToQImage(const MatrixXd& matrix) {
            MatrixXd mat = matrix.transpose();
            int width = mat.cols();
            int height = mat.rows();
            QImage image(width, height, QImage::Format_Grayscale8);
        
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    double val = std::clamp(mat(y, x), 0.0, 1.0);
                    uint8_t pixel = static_cast<uint8_t>(val * 255.0);
                    image.setPixel(x, y, qRgb(pixel, pixel, pixel));
                }
        
            return image;
        }

        void showImageQt(const MatrixXd& image) {
            int argc = 0;
            char* argv[] = { nullptr };
            QApplication app(argc, argv);
        
            MatrixXd mt = image.transpose();
            QImage img = matrixToQImage(mt);
        
            QLabel label;
            label.setPixmap(QPixmap::fromImage(img.scaled(280, 280)));
            label.setWindowTitle("MNIST Image");
            label.show();
        
            app.exec();
        }
        
    }; // namespace Visualization
}; // namespace NeuralNetwork
