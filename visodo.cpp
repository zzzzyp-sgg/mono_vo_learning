#include "vo_features.h"

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal)
{
    std::string line;
    int i = 0;
    std::ifstream myfile("/home/zyp/DATA/KITTI_DATA/dataset/poses/00.txt");   // 真实轨迹的文件地址
    double x = 0, y = 0, z = 0;
    double x_prev, y_prev, z_prev;
    if (myfile.is_open())
    {
        while ((getline(myfile, line)) && (i <= frame_id))
        {
            z_prev = z;
            y_prev = y;
            x_prev = x;
            std::istringstream in(line);
            // [R|t] 每一行的最后一列数据为平移的t的数据
            // 把位移t提取出来
            for (int j = 0; j < 12; j++){
                in >> z;
                if (j == 7) y = z;
                if (j == 3) x = z;
            }
            i++;
        }
        myfile.close();
    }

    else{
        std::cout << "Unable to open file";
        return 0;
    }
    
    // 当前帧的（x，y，z）减去上一帧的（x，y，z）作为真实距离
    return sqrt((x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev)
                + (z - z_prev) * (z - z_prev));
}

int main(int argc, char **argv)
{
    Mat img_1, img_2;
    Mat R_f, t_f;   // 最终的旋转矩阵和平移向量

    std::ofstream myfile;
    myfile.open("result1_1.txt");

    double scale = 1.00;
    char filename1[200];
    char filename2[200];
    sprintf(filename1, "/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/%010d.png", 0);
    sprintf(filename2, "/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/%010d.png", 1);

    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    // 读取数据集中前两个框架
    // Mat img_tmp = imread("/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000000000.png");
    Mat img_1_c = imread(filename1);
    Mat img_2_c = imread(filename2);

    if (!img_1_c.data || !img_2_c.data){
        std::cout << "--(!) Error reading images " << std::endl;
        return -1; 
    }

    // 处理的是灰度图像
    cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
    cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

    // 特征点提取
    vector<Point2f> points1, points2;
    featureDetection(img_1, points1);
    // 特征追踪
    vector<uchar> status;
    featureTracking(img_1, img_2, points1, points2, status);

    // 这里是设定相机参数，不同的数据集需要设置不同的参数
    double focal = 718.8560;
    Point2d pp(607.1928, 185.2157);
    // 恢复姿态和本质矩阵E
    Mat E, R, t, mask;
    E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points2, points1, R, t, focal, pp, mask);

    Mat prevImage = img_2;
    Mat currImage;
    vector<Point2f> prevFeatures = points2;
    vector<Point2f> currFeatures;

    char filename[100];

    R_f = R.clone();
    t_f = t.clone();

    clock_t begin = clock();

    namedWindow( "Road facing camera", WINDOW_AUTOSIZE ); // 这个窗口展示图像
    namedWindow( "Trajectory", WINDOW_AUTOSIZE );         // 这个窗口展示轨迹

    Mat traj = Mat::zeros(600, 600, CV_8UC3);

    for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++)
    {
        sprintf(filename, "/home/zyp/DATA/KITTI_DATA/00/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/%010d.png", numFrame);
        Mat currImage_c = imread(filename);
        cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
        vector<uchar> status;
        featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

        E= findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

        Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2, currFeatures.size(), CV_64F); 

        for (int i = 0; i < prevFeatures.size(); i++){
            prevPts.at<double>(0, i) = prevFeatures.at(i).x;
            prevPts.at<double>(1, i) = prevFeatures.at(i).y;

            currPts.at<double>(0, i) = currFeatures.at(i).x;
            currPts.at<double>(0, i) = currFeatures.at(i).y;
        }

        scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

        if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            t_f = t_f + scale * (R_f * t);      // 先旋转后平移再加上原来的平移
            R_f = R * R_f;
        }
        else {
        //cout << "scale below 0.1, or incorrect translation" << endl;
    }
    
    // lines for printing results
    myfile << t_f.at<double>(0) << " " << t_f.at<double>(1) << " " << t_f.at<double>(2) << std::endl;

    // 如果被追踪的图像数量低于一个特定的阈值，就会触发重新检测
        if (prevFeatures.size() < MIN_NUM_FEAT){
            std::cout << "Number of tracked features reduced to " << prevFeatures.size() << std::endl;
            std::cout << "trigerring redection " << std::endl;
            featureDetection(prevImage, prevFeatures);
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }

        prevImage = currImage.clone();
        prevFeatures = currFeatures;

        int x = int(t_f.at<double>(0)) + 300;
        int y = int(t_f.at<double>(2)) + 100;
        // 半径为1，线宽为2，画出来就成了实心的圆。然后按照trajectory绘制轨迹
        circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);
        // 画出显示的矩形
        rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0), CV_FILLED);
        sprintf(text, "Coordinate: x = %02fm y = %02fm z = %02fm", t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
        putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

        imshow("Road facing camera", currImage_c);
        imshow("Trajectory", traj);

        waitKey(1);
    }

    myfile.close();
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "Total time taken: " << elapsed_secs << "s" << std::endl;

    return 0;
}