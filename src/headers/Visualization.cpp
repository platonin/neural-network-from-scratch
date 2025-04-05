#include "Visualization.h"

namespace NN {
    namespace Visualization {
        void showImage(const MatrixXd &image) {
            MatrixXd mt = image.transpose();
            Mat img(28, 28, CV_64F, (void*)mt.data());
            Mat img8bit;
            img.convertTo(img8bit, CV_8U, 255);
            imshow("MNIST Image", img8bit);
            waitKey(0); 
        }

        void showImagesRow(const vector<MatrixXd> &images) {
            vector<Mat> mats;
        
            for (const auto& image : images) {
                MatrixXd mt = image.transpose();
                Mat img(28, 28, CV_64F, (void*)mt.data());
                Mat img8bit;
                img.convertTo(img8bit, CV_8U, 255); 
                mats.push_back(img8bit);
            }
        
            Mat result;
            hconcat(mats, result);
        
            imshow("MNIST Row", result);
            waitKey(0);
        }

        void showImagesWithLabels(const vector<MatrixXd> &images, const vector<int> &numbers) {
            vector<Mat> labeledImages;
        
            for (size_t i = 0; i < images.size(); ++i) {
                MatrixXd mt = images[i].transpose();
                Mat img(28, 28, CV_64F, (void*)mt.data());
                Mat img8bit;
                img.convertTo(img8bit, CV_8U, 255);
        
                Mat resizedImg;
                resize(img8bit, resizedImg, Size(56, 56), 0, 0, INTER_NEAREST);
        
                Mat labelArea(20, resizedImg.cols, CV_8U, Scalar(255));
        
                string label = to_string(numbers[i]);
                putText(labelArea, label, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0), 1);
        
                Mat fullBlock;
                vconcat(resizedImg, labelArea, fullBlock);
        
                labeledImages.push_back(fullBlock);
            }
            Mat finalImage;
            hconcat(labeledImages, finalImage);

            imshow("Images with Labels", finalImage);
            waitKey(0);
        }

        void showImagesWithTwoNumbers(const vector<MatrixXd> &images, const vector<int> &topNumbers, const vector<int> &bottomNumbers) {
            vector<Mat> labeledImages;
        
            for (size_t i = 0; i < images.size(); ++i) {
                MatrixXd mt = images[i].transpose();
                Mat img(28, 28, CV_64F, (void*)mt.data());
                Mat img8bit;
                img.convertTo(img8bit, CV_8U, 255);
        
                Mat resizedImg;
                resize(img8bit, resizedImg, Size(56, 56), 0, 0, INTER_NEAREST);
        
                Mat topLabel(20, resizedImg.cols, CV_8U, Scalar(255));
                string topText = to_string(topNumbers[i]);
                putText(topLabel, topText, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0), 1);
        
                Mat bottomLabel(20, resizedImg.cols, CV_8U, Scalar(255));
                string bottomText = to_string(bottomNumbers[i]);
                putText(bottomLabel, bottomText, Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0), 1);
        
                Mat block;
                vconcat(topLabel, resizedImg, block);
                vconcat(block, bottomLabel, block);
        
                labeledImages.push_back(block);
            }
        
            Mat finalImage;
            hconcat(labeledImages, finalImage);
        
            imshow("Images with Two Numbers", finalImage);
            waitKey(0);
        }
        
        
    }; // namespace Visualization
}; // namespace NeuralNetwork
