#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator>  // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using std::vector;

/**
 * @brief 进行特征提取及跟踪，提取FAST角点
 * @param [in] img_1 前一帧图像
 * @param [in] img_2 第二帧图像
 * @param [in] points1 前一帧图像的特征点
 * @param [in] points2 第二帧图像的特征点
 * @param [in] status 状态
 */
void featureTracking(Mat img_1, Mat img_2, vector<Point2f> &points1, vector<Point2f> &points2, vector<uchar> &status)
{
    vector<float> err;
    Size winSize = Size(21, 21);
    // opencv迭代终止条件类，参数分别为迭代终止类型、迭代最大次数和特定的阈值
    // COUNT 最大次数终止 EPS 阈值终止 COUNT+EPS 满足二者之一即终止
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

    // opencv 自带的LK光流法函数
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    // 这一步删除KLT跟踪失败的点或超出框架的点
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++){
        Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)){
            if ((pt.x < 0) || (pt.y < 0)){
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

/**
 * @brief 进行图像FAST特征点的提取
 * @param [in] img_1 要提取特征点的图像
 * @param [in] points 提取到的特征点
 */
void featureDetection(Mat img_1, vector<Point2f> &points1)
{
    vector<KeyPoint> keypoints_1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
    KeyPoint::convert(keypoints_1, points1, vector<int>());
}