//
// Created by yitaowei on 11/18/22.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <vector>
#include <string>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <boost/format.hpp>

using namespace std;
using namespace cv;

#define CV_PI   3.1415926535897932384626433832795
// image size
int width = 11968;
int height = 5984;

typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 2, 3> Matrix23d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;

void calcPixelUV(double x, double y, double z, double pointDistance2Cam, double &u, double &v) //Use before  calcPixelm and calcPixeln. all in camera coords
{
    double alpha = atan2(y, x); //alpha is  radian , atan2 belongs to -pi-pi
    double omega = asin(z / pointDistance2Cam);
    u = 0.5 - alpha / (2 * CV_PI);
    v = 0.5 - omega / CV_PI;
}

Sophus::SO3d CalibEstimate(const Mat &dist_img,
                           const VecVector3d &init_points,
                           Sophus::SO3d &R21);

int main() {

    Eigen::Matrix<double, 3, 1> Trans;
    Trans << 0.048250, -0.000045, -0.155001;
    VecVector3d pcd_points;
    ifstream ifs;
    ifs.open("../Second_Floor/depth_unconsis_point.txt");
    assert(ifs.is_open());

    string buff;
    int i = 0;
    while(getline(ifs, buff))
    {
        stringstream ss(buff);
        double x,y,z;
        char ch;
        ss >> x >> ch >> y >> ch >> z;
        Eigen::Vector3d point;
        point << x , y , z;
        pcd_points.push_back(point);
    }
    cout << pcd_points.size() << endl;

    // generate distance img for calib estimate
    Mat ori_img = imread("../Second_Floor/IMG_20221125_000213_00_019.jpg", 0);
    Mat img_blur;
    GaussianBlur(ori_img, img_blur, Size(3, 3), 0);
    Mat edge_img;
    Canny(img_blur, edge_img, 150, 250, 3, false);
    Mat img_threshold;
    Mat distance_img;
    threshold(edge_img, img_threshold, 150, 255, THRESH_BINARY);
    bitwise_not(img_threshold, img_threshold);
    distanceTransform(img_threshold, distance_img, DIST_L2, 3, DIST_LABEL_PIXEL);
    /*Rect rect(1150, 1770, 730, 590);
    distance_img = distance_img(rect);*/

    // *** SO3d optimization
    Sophus::SO3d R_init = Sophus::SO3d(Eigen::Quaterniond(1, 0, 0, 0));
    Sophus::SO3d R_update= CalibEstimate (distance_img, pcd_points, R_init);

    // *** show opi result
    Mat img_show;
    cv::cvtColor(distance_img, img_show, CV_GRAY2BGR);
    for (size_t j = 0; j < pcd_points.size(); j++){
        Eigen::Matrix<double, 3, 1> org_pt = pcd_points[j];
        Eigen::Matrix<double, 3, 1> dst_pt = R_update * pcd_points[j] + Trans;
        double org_x = org_pt[0];
        double org_y = org_pt[1];
        double org_z = org_pt[2];
        double org_dis = sqrt(org_x * org_x + org_y * org_y + org_z * org_z);
        double org_u, org_v;
        calcPixelUV(org_x, org_y, org_z, org_dis, org_u,org_v);
        int org_m = (int)(org_u * width + 0.5);
        int org_n = (int)(org_v * height + 0.5);

        double dst_x = dst_pt[0];
        double dst_y = dst_pt[1];
        double dst_z = dst_pt[2];
        double dst_dis = sqrt(dst_x * dst_x + dst_y * dst_y + dst_z * dst_z);
        double dst_u, dst_v;
        calcPixelUV(dst_x, dst_y, dst_z, dst_dis, dst_u,dst_v);
        int dst_m = (int)(dst_u * width + 0.5);
        int dst_n = (int)(dst_v * height + 0.5);

        cv::circle(img_show, cv::Point2f(dst_m, dst_n), 1, cv::Scalar(0, 0, 250), 3);
        //cv::circle(img_show, cv::Point2f(org_m, org_n), 1, cv::Scalar(255, 0, 0), 3);
        cv::line(img_show, cv::Point2f(org_m, org_n), cv::Point2f(dst_m, dst_n), cv::Scalar(0, 255, 0));
    }
    cv::namedWindow("current", WINDOW_FREERATIO);
    cv::imshow("current", img_show);
    cv::waitKey();
    cv::imwrite("../Second_Floor/opi.jpg", img_show);
    return 0;
}

Sophus::SO3d CalibEstimate(const Mat &dist_img,
                           const VecVector3d &init_points,
                           Sophus::SO3d &R21){
    Eigen::Matrix<double, 3, 1> Translation;
    Translation << 0.048250, -0.000045, -0.155201;
    int iterations = 20;
    double lastCost = 0;
    for (int iter =0; iter < iterations; iter++)
    {
        double cost = 0;
        int count = 0;
        // Define Hessian and bias
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();  // 3x3 Hessian
        Eigen::Vector3d b = Eigen::Vector3d::Zero();  // 3x1 bias

        for (size_t i = 0; i < init_points.size(); i++)
        {
            // compute the projection in distance image
            Eigen::Matrix<double, 3, 1> pw2 = R21 * init_points[i] + Translation;
            double X = pw2[0], Y = pw2[1], Z = pw2[2];
            double distance = sqrt(X * X + Y * Y + Z * Z);
            double size_u, size_v;
            calcPixelUV(X, Y, Z, distance, size_u,size_v);
            int pixelm = (int)(size_u * width + 0.5);
            int pixeln = (int)(size_v * height + 0.5);
            count++;

            /*if (pixelm < 250 || pixelm > 3050 || pixeln < 1000 || pixeln > 2000) continue;
            count++;*/

            double error = dist_img.at<float>(pixeln, pixelm);

            double hori_dis = sqrt(X * X + Y * Y);
            double derive_m2x = (width * Y) / (2 * CV_PI * hori_dis * hori_dis);
            double derive_m2y = -(width * X) / (2 * CV_PI * hori_dis * hori_dis);
            double derive_n2x = (height * X) / (CV_PI * hori_dis * distance * distance);
            double derive_n2y = (height * Y) / (CV_PI * hori_dis * distance * distance);
            double derive_n2z = -(height * hori_dis) / (CV_PI * distance * distance);

            Matrix23d J_pixeltoT; // derive from pixel to SE3d T
            J_pixeltoT(0, 0) = -Z * derive_m2y;
            J_pixeltoT(0, 1) = Z * derive_m2x;
            J_pixeltoT(0, 2) = -Y * derive_m2x + X * derive_m2y;
            J_pixeltoT(1, 0) = -Z * derive_n2y + Y * derive_n2z;
            J_pixeltoT(1, 1) = Z * derive_n2x - X * derive_n2z;
            J_pixeltoT(1, 2) = -Y * derive_n2x + X * derive_n2y;

            Eigen::Vector2d J_pixel_grad;    // image gradients
            J_pixel_grad = Eigen::Vector2d(
                    0.5 * (dist_img.at<float>(pixeln + 1, pixelm) - dist_img.at<float>(pixeln - 1, pixelm)),
                    0.5 * (dist_img.at<float>(pixeln, pixelm + 1) - dist_img.at<float>(pixeln, pixelm - 1))
            );
            // Total Jacobin
            Eigen::Vector3d J = -1.0 * (J_pixel_grad.transpose() * J_pixeltoT).transpose();

            H += J * J.transpose();
            b += -error * J;
            cost += error;
        }
        // solve update and put it into estimation
        Eigen::Vector3d update = H.ldlt().solve(b);
        R21 = Sophus::SO3d::exp(update) * R21;

        if (isnan(update[0])) {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }

        cost = cost/count;

        if (iter > 0 && cost > lastCost) {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            // cout << "R21 = \n" << R21.matrix() << endl;
            break;
        }
        lastCost = cost;
        cout << "iteration index " << iter << "---cost = " << cost << "---valid point is " << count << endl;
    }
    cout << "R21 = \n" << R21.matrix() << endl;
    cout << R21.matrix().eulerAngles(2,1,0) << endl;
    return R21;
}
