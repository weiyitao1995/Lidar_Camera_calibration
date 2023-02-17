//
// Created by yitaowei on 11/14/22.
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
using namespace Eigen;

#define CV_PI   3.1415926535897932384626433832795
// image size
int width = 3840;
int height = 1920;

class edgeMsg {
public:
    int pointIndex;
    Eigen::Vector2d measure;
};

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

void calcPixelUV(double x, double y, double z, double pointDistance2Cam, double &u, double &v) //Use before  calcPixelm and calcPixeln. all in camera coords
{
    double alpha = atan2(y, x); //alpha is  radian , atan2 belongs to -pi-pi
    double omega = asin(z / pointDistance2Cam);
    u = 0.5 - alpha / (2 * CV_PI);
    v = 0.5 - omega / CV_PI;
}

Sophus::SE3d CalibEstimate(const VecVector2d &pixel_truth,
                           const VecVector3d &init_points,
                           Sophus::SE3d &R21);

class CalibVertex : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CalibVertex() {}

    ~CalibVertex() {}

    bool read(std::istream &is) {}

    bool write(std::ostream &os) const {}

    virtual void setToOriginImpl() {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update_) {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> update(update_);
        setEstimate(Sophus::SE3d::exp(update) * estimate());
    }
};

class calibEdge: public g2o::BaseUnaryEdge<2, Vector2d , CalibVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    calibEdge(Eigen::Vector3d &pt) {
        this->orgpt = pt;
        // this->pixel = px;
    }
    ~calibEdge() {}

    virtual void computeError() override
    {
        const CalibVertex *v = static_cast<const CalibVertex*> (_vertices[0]);
        Eigen::Vector3d dstpt = v->estimate() * orgpt;
        double X = dstpt[0], Y = dstpt[1], Z = dstpt[2];
        double distance = sqrt(X * X + Y * Y + Z * Z);
        double alpha = atan2(Y, X); //alpha is  radian , atan2 belongs to -pi-pi
        double omega = asin(Z / distance);
        double size_u = 0.5 - alpha / (2 * CV_PI);
        double size_v = 0.5 - omega / CV_PI;
        double pixelm = size_u * width;
        double pixeln = size_v * height;
        Vector2d proj (pixeln,pixelm);
        _error= _measurement - proj;
    }

    /*virtual void linearizeOplus() override
    {
        Matrix26d J_pixeltoT; // derive from pixel to SE3d T
        const CalibVertex *v = static_cast<const CalibVertex*> (_vertices[0]);
        Eigen::Vector3d dstpt = v->estimate() * orgpt;
        double X = dstpt[0], Y = dstpt[1], Z = dstpt[2];
        double distance = sqrt(X * X + Y * Y + Z * Z);
        double hori_dis = sqrt(X * X + Y * Y);
        double alpha = atan2(Y, X); //alpha is  radian , atan2 belongs to -pi-pi
        double omega = asin(Z / distance);
        double size_u = 0.5 - alpha / (2 * CV_PI);
        double size_v = 0.5 - omega / CV_PI;
        int pixelm = (int)(size_u * width + 0.5);
        int pixeln = (int)(size_v * height + 0.5);
        double derive_m2x = (width * Y) / (2 * CV_PI * hori_dis * hori_dis);
        double derive_m2y = -(width * X) / (2 * CV_PI * hori_dis * hori_dis);
        double derive_n2x = (height * X) / (CV_PI * hori_dis * distance * distance);
        double derive_n2y = (height * Y) / (CV_PI * hori_dis * distance * distance);
        double derive_n2z = -(height * hori_dis) / (CV_PI * distance * distance);

        J_pixeltoT(0, 0) = derive_m2x;
        J_pixeltoT(0, 1) = derive_m2y;
        J_pixeltoT(0, 2) = 0.0;
        J_pixeltoT(0, 3) = -Z * derive_m2y;
        J_pixeltoT(0, 4) = Z * derive_m2x;
        J_pixeltoT(0, 5) = -Y * derive_m2x + X * derive_m2y;

        J_pixeltoT(1, 0) = derive_n2x;
        J_pixeltoT(1, 1) = derive_n2y;
        J_pixeltoT(1, 2) = derive_n2z;
        J_pixeltoT(1, 3) = -Z * derive_n2y + Y * derive_n2z;
        J_pixeltoT(1, 4) = Z * derive_n2x - X * derive_n2z;
        J_pixeltoT(1, 5) = -Y * derive_n2x + X * derive_n2y;

        Eigen::Vector2d J_pixel_grad;    // image gradients
        J_pixel_grad = Eigen::Vector2d(
                0.5 * (targetImg.at<float>(pixeln + 1, pixelm) - targetImg.at<float>(pixeln - 1, pixelm)),
                0.5 * (targetImg.at<float>(pixeln, pixelm + 1) - targetImg.at<float>(pixeln, pixelm - 1))
        );

        // Total Jacobin
        _jacobianOplusXi = -1.0 * (J_pixel_grad.transpose() * J_pixeltoT).transpose();
    };*/

    virtual bool read( istream& in ) {}
    virtual bool write( ostream& out ) const {}

private:
    Eigen::Vector3d orgpt;
    // Eigen::Vector2d pixel;  // the target image
};
Sophus::SE3d init_calib(
        Eigen::Quaterniond(1, 0, 0, 0),
        Eigen::Vector3d(-0.048, 0, -0.1891));

int main() {

    /*Eigen::Matrix<double, 3, 1> Trans;
    Trans << -0.046250, -0.000045, -0.125001;*/
    VecVector3d pcd_points;
    ifstream ifs;
    ifs.open("../theta_calib/3D_points.txt");
    assert(ifs.is_open());
    string buff;
    int i = 0;
    while (getline(ifs, buff)) {
        stringstream ss(buff);
        double x, y, z;
        char ch;
        ss >> x >> ch >> y >> ch >> z;
        Eigen::Vector3d point;
        point << x, y, z;
        pcd_points.push_back(point);
    }
    cout << pcd_points.size() << endl;

    // load pixel value
    VecVector2d pixel_points;
    ifstream ifs_pixel;
    ifs_pixel.open("../theta_calib/2D_pixels.txt");
    assert(ifs_pixel.is_open());
    string buff2;
    int j = 0;
    while (getline(ifs_pixel, buff2)) {
        stringstream sss(buff2);
        double u,v;
        char ch;
        sss >> u >> ch >> v;
        Eigen::Vector2d pixel;
        pixel << u, v;
        pixel_points.push_back(pixel);
    }
    cout << pixel_points.size() << endl;


    // *** use g20 to optimize
    // build solver
    g2o::SparseOptimizer optimizer;
    // build optimization problem
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock;  // 求解的向量是6＊1的
    typedef g2o::LinearSolverDense<DirectBlock::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<DirectBlock>(g2o::make_unique<LinearSolverType>()));

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true); // allow debug output

    // add vertexes
    CalibVertex* vertex = new CalibVertex();
    vertex->setId(0);
    vertex->setEstimate(Sophus::SE3d(
            Eigen::Quaterniond(1, 0, 0, 0),
            Eigen::Vector3d(-0.044, 0.0016, -0.1884)));
    optimizer.addVertex(vertex);


    // add edges
    for (int j = 0; j < 15; j++)
    {
        calibEdge *edge = new calibEdge(pcd_points[j]);
        edge->setId(j);
        edge->setVertex(0,vertex);

        // set measurement
        edge->setMeasurement(pixel_points[j]);

        edge->setInformation(Eigen::Matrix2d::Identity());
        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
        rk->setDelta(1.0);
        edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
    }

    // carry out the optimizer
    optimizer.initializeOptimization();
    optimizer.optimize(100);

    Sophus::SE3d Tcw = vertex->estimate();
    cout << Tcw.matrix() << endl;
    cout << Tcw.angleX() << " " << Tcw.angleY() << " " << Tcw.angleZ() << endl;

    // use theta updata result
    /*double theta_update = 0.0468185;
    Eigen::Matrix3d R_update;
    R_update << cos(theta_update), -sin(theta_update), 0,
            sin(theta_update), cos(theta_update), 0,
            0, 0, 1;
    Sophus::SE3d dest_calib(
            R_update,
            Eigen::Vector3d(-0.0423, 0, -0.1532));*/

    // Calculate the reprojection error
    double cost = 0;
    for (size_t j = 0; j < 15; j++)
    {
        Eigen::Matrix<double, 3, 1> dst_pt = Tcw * pcd_points[j] ;
        double org_x = dst_pt[0];
        double org_y = dst_pt[1];
        double org_z = dst_pt[2];
        double org_dis = sqrt(org_x * org_x + org_y * org_y + org_z * org_z);
        double org_u, org_v;
        calcPixelUV(org_x, org_y, org_z, org_dis, org_u,org_v);
        int org_m = (int)(org_u * width + 0.5);
        int org_n = (int)(org_v * height + 0.5);
        double var_m = org_m - pixel_points[j][1];
        double var_n = org_n - pixel_points[j][0];
        double var = sqrt(var_m * var_m + var_n * var_n);
        cost += var;
    }
    cout << cost << endl;

    // show the reprojection points
    /*Mat ori_img = imread("../calib_image/001.jpg", 1);
    for (size_t j = 0; j < 3; j++){
        Eigen::Matrix<double, 3, 1> org_pt = init_calib * pcd_points[j];
        Eigen::Matrix<double, 3, 1> dst_pt = Tcw * pcd_points[j] ;
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

        cv::circle(ori_img, cv::Point2f(dst_m, dst_n), 3, cv::Scalar(0, 0, 255), 2);
        cv::line(ori_img, cv::Point2f(org_m, org_n), cv::Point2f(dst_m, dst_n), cv::Scalar(0, 255, 0));
    }
    cv::namedWindow("current", WINDOW_FREERATIO);
    cv::imshow("current", ori_img);
    cv::waitKey();
    cv::imwrite("001_reprojection.jpg", ori_img);*/

    return 0;

}