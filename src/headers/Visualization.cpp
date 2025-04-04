#include "Visualization.h"

namespace NeuralNetwork {
    namespace Visualization {
        void showImage(const MatrixXd &image) {
            MatrixXd mt = image.transpose();
            Mat img(28, 28, CV_64F, (void*)mt.data());
            Mat img8bit;
            img.convertTo(img8bit, CV_8U, 255);
            imshow("MNIST Image", img8bit);
            waitKey(0); 
        }

        void showImagesRow(const std::vector<Eigen::MatrixXd> &images) {
            std::vector<cv::Mat> mats;
        
            for (const auto& image : images) {
                Eigen::MatrixXd mt = image.transpose();
                cv::Mat img(28, 28, CV_64F, (void*)mt.data());
                cv::Mat img8bit;
                img.convertTo(img8bit, CV_8U, 255); 
                mats.push_back(img8bit);
            }
        
            cv::Mat result;
            cv::hconcat(mats, result);
        
            cv::imshow("MNIST Row", result);
            cv::waitKey(0);
        }

        void showImagesWithLabels(const std::vector<MatrixXd> &images, const std::vector<int> &numbers) {
            std::vector<Mat> labeledImages;
        
            for (size_t i = 0; i < images.size(); ++i) {
                MatrixXd mt = images[i].transpose();
                Mat img(28, 28, CV_64F, (void*)mt.data());
                Mat img8bit;
                img.convertTo(img8bit, CV_8U, 255);
        
                Mat resizedImg;
                resize(img8bit, resizedImg, Size(56, 56), 0, 0, INTER_NEAREST);
        
                Mat labelArea(20, resizedImg.cols, CV_8U, Scalar(255));
        
                std::string label = std::to_string(numbers[i]);
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

        void showImagesWithTwoNumbers(const std::vector<Eigen::MatrixXd> &images, const std::vector<int> &topNumbers, const std::vector<int> &bottomNumbers) {
            std::vector<cv::Mat> labeledImages;
        
            for (size_t i = 0; i < images.size(); ++i) {
                // --- Преобразуем MatrixXd в cv::Mat ---
                Eigen::MatrixXd mt = images[i].transpose();
                cv::Mat img(28, 28, CV_64F, (void*)mt.data());
                cv::Mat img8bit;
                img.convertTo(img8bit, CV_8U, 255);
        
                cv::Mat resizedImg;
                cv::resize(img8bit, resizedImg, cv::Size(56, 56), 0, 0, cv::INTER_NEAREST);
        
                cv::Mat topLabel(20, resizedImg.cols, CV_8U, cv::Scalar(255));
                std::string topText = std::to_string(topNumbers[i]);
                cv::putText(topLabel, topText, cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0), 1);
        
                cv::Mat bottomLabel(20, resizedImg.cols, CV_8U, cv::Scalar(255));
                std::string bottomText = std::to_string(bottomNumbers[i]);
                cv::putText(bottomLabel, bottomText, cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0), 1);
        
                cv::Mat block;
                cv::vconcat(topLabel, resizedImg, block);
                cv::vconcat(block, bottomLabel, block);
        
                labeledImages.push_back(block);
            }
        
            cv::Mat finalImage;
            cv::hconcat(labeledImages, finalImage);
        
            cv::imshow("Images with Two Numbers", finalImage);
            cv::waitKey(0);
        }
        
        
    }; // namespace Visualization
}; // namespace NeuralNetwork
