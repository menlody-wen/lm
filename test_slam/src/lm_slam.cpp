#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <iostream>
#include <vector>
//#include <io.h>
#include <dirent.h> 
#include <fstream>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

const double DERIV_STEP = 1e-5;
const int MAX_ITER = 100;

Eigen::Matrix4f pose_to_Matrix(std::string filename)
{
    std::fstream file_a;
    file_a.open(filename.c_str());
    std::string str;
    std::getline(file_a, str);
    std::vector<float> data;
    data.resize(12, 0);
    std::istringstream iss(str);
    for (unsigned int count = 0; count < 12; count++)
    {
        std::string sub;
        iss >> sub;
        double value = ::atof(sub.c_str());
        data[count] = value;
    }
   Eigen::Matrix4f output;
   output = Eigen::Matrix4f::Identity();
   Eigen::Matrix3f rotate_Matrix;
    rotate_Matrix(0,0) = data[3];
    rotate_Matrix(0,1) = data[4];
    rotate_Matrix(0,2) = data[5];
    rotate_Matrix(1,0) = data[6];
    rotate_Matrix(1,1) = data[7];
    rotate_Matrix(1,2) = data[8];
    rotate_Matrix(2,0) = data[9];
    rotate_Matrix(2,1) = data[10];
    rotate_Matrix(2,2) = data[11];
    Eigen::Vector3f translate;
    translate<<data[0],data[1],data[2];
    output.block(0,0,3,3) = rotate_Matrix;
    output.block(0,3,3,1) = translate;
    return output;
}

Eigen::Matrix4f vec_pose_to_matrix(std::vector<float> & vec_pose){
        ///generate transform matrix according to current pose
        Eigen::AngleAxisf current_rotation_x( vec_pose[3], Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf current_rotation_y( vec_pose[4], Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf current_rotation_z( vec_pose[5], Eigen::Vector3f::UnitZ());
        Eigen::Translation3f current_translation(vec_pose[0], vec_pose[1], vec_pose[2]);
        Eigen::Matrix4f transform_matrix =
                (current_translation * current_rotation_z * current_rotation_y * current_rotation_x).matrix();

        return transform_matrix ;
    }


Eigen::Matrix4f get_imu_pose(std::string filename)
    {
        std::fstream file;
        file.open(filename.c_str());
        std::string str;
        std::getline(file, str);
        std::vector<float> data,data_2;
        data.resize(6, 0);
        data_2.resize(7,0);
        std::istringstream iss(str);

        for (unsigned int count = 0; count < 7; count++)
        {
            std::string sub;
            iss >> sub;
            double value = ::atof(sub.c_str());
            data_2[count] = value;
        }

        tf::Quaternion q(data_2[3], data_2[4], data_2[5], data_2[6]);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        data[0] = data_2[0];
        data[1] = data_2[1];
        data[2] = data_2[2];
        data[3] = roll;
        data[4] = pitch;
        data[5] = yaw;
        Eigen::Matrix4f pose_trans;
        pose_trans=vec_pose_to_matrix(data);
        return pose_trans;
    }

void GetPoses(std::string path, std::vector<std::string> &vector_filename)
{
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    vector_filename.resize(4000);
    int idx = 0;
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            std::string tmp_str(ptr->d_name);
            std::string tmp2_str(ptr->d_name);
            std::string str2 = tmp_str.substr(0, tmp_str.find("."));
            int int_temp = atoi(str2.c_str());
            if (int_temp > 3999)
                continue;
            if (tmp2_str.find("pose") != std::string::npos)
            {
                vector_filename[int_temp] = path + "/" + tmp2_str;
                idx++;
            }
        }
    }
    closedir(pDir);
    vector_filename.resize(idx);
}

void matrix_to_pose(Eigen::Matrix4f transform_matrix, double &pose_x, double &pose_y, double &pose_z,
                    double &pose_roll, double &pose_pitch, double &pose_yaw)
{
    ///get current pose according to ndt transform matrix
    tf::Matrix3x3 mat_rotate;
    mat_rotate.setValue(static_cast<double>(transform_matrix(0, 0)), static_cast<double>(transform_matrix(0, 1)),
                        static_cast<double>(transform_matrix(0, 2)), static_cast<double>(transform_matrix(1, 0)),
                        static_cast<double>(transform_matrix(1, 1)), static_cast<double>(transform_matrix(1, 2)),
                        static_cast<double>(transform_matrix(2, 0)), static_cast<double>(transform_matrix(2, 1)),
                        static_cast<double>(transform_matrix(2, 2)));

    pose_x = transform_matrix(0, 3);
    pose_y = transform_matrix(1, 3);
    pose_z = transform_matrix(2, 3);
    mat_rotate.getRPY(pose_roll, pose_pitch, pose_yaw, 1);
}

float cost_distance(const Eigen::Matrix4f input,const Eigen::Matrix4f output,const cv::Mat params,const Eigen::Matrix4f init_pose)
{
    double init_x, init_y, init_z, init_roll, init_pitch, init_yaw;

    matrix_to_pose(init_pose, init_x, init_y, init_z, init_roll, init_pitch, init_yaw);
    Eigen::AngleAxisf current_rotation_x(params.at<double>(2,0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf current_rotation_y(params.at<double>(1,0), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf current_rotation_z(params.at<double>(0,0), Eigen::Vector3f::UnitZ());
    Eigen::Translation3f current_translation(params.at<double>(3,0), params.at<double>(4,0), init_z);
    Eigen::Matrix4f transform_matrix =
        (current_translation * current_rotation_z * current_rotation_y * current_rotation_x).matrix();
    Eigen::Matrix4f tmp_lidar_matrix = input;
    Eigen::Matrix4f imu_lidar_matrix = output*transform_matrix;
    Eigen::Matrix4f diff_matrix = tmp_lidar_matrix.inverse()*imu_lidar_matrix;
    float distance = fabs(diff_matrix(0,3))+fabs(diff_matrix(1,3))+fabs(diff_matrix(2,3));
     Eigen::Matrix3f rotate_matrix = diff_matrix.block(0,0,3,3);
    double pose_roll,pose_pitch,pose_yaw;
    tf::Matrix3x3 mat_rotate;
    mat_rotate.setValue(static_cast<double>(rotate_matrix(0, 0)), static_cast<double>(rotate_matrix(0, 1)),
                        static_cast<double>(rotate_matrix(0, 2)), static_cast<double>(rotate_matrix(1, 0)),
                        static_cast<double>(rotate_matrix(1, 1)), static_cast<double>(rotate_matrix(1, 2)),
                        static_cast<double>(rotate_matrix(2, 0)), static_cast<double>(rotate_matrix(2, 1)),
                        static_cast<double>(rotate_matrix(2, 2)));
    mat_rotate.getRPY(pose_roll, pose_pitch, pose_yaw, 1);
    // pose_roll *= 180./3.1415926;
    // pose_pitch *= 180./3.1415926;
    // pose_yaw *=180./3.1415926;
    // std::cout<<"the rotate matrix is "<<distance<<","<<pose_roll<<","<<pose_pitch<<","<<pose_yaw<<std::endl;
    distance +=fabs(pose_roll)+fabs(pose_pitch)+fabs(pose_yaw);
    return distance;
}


float Deriv(const Eigen::Matrix4f input,const Eigen::Matrix4f output,const cv::Mat params,const Eigen::Matrix4f init_pose,int n)
{
     // Returns the derivative of the nth parameter
    cv::Mat params1 = params.clone();
    cv::Mat params2 = params.clone();

    // Use central difference  to get derivative
    params1.at<double>(n, 0) -= DERIV_STEP;
    params2.at<double>(n, 0) += DERIV_STEP;

    double p1 = cost_distance(input,output,params1,init_pose);
    double p2 = cost_distance(input,output,params2,init_pose);

    double d = (p2 - p1) / (2 * DERIV_STEP);

    return d;
}

int main()
{
    std::string g2o_path,raw_path;
    g2o_path = "/home/wen/ros_seven/L_M/src/test_slam/g2o/";
    raw_path = "/home/wen/ros_seven/L_M/src/test_slam/raw_data/";
    std::vector<std::string> vector_g2o_files,vector_raw_files;
    std::vector<Eigen::Matrix4f> vector_g2o_pose,vector_raw_pose;
    GetPoses(raw_path,vector_g2o_files);
    GetPoses(raw_path,vector_raw_files);
    for(int i =0;i<vector_g2o_files.size();++i)
    {
    //    Eigen::Matrix4f g2o_pose = pose_to_Matrix(vector_g2o_files[i]);
    Eigen::Matrix4f g2o_pose = get_imu_pose(vector_g2o_files[i]);
    Eigen::Matrix4f raw_pose = get_imu_pose(vector_raw_files[i]);
    vector_g2o_pose.push_back(g2o_pose);
    vector_raw_pose.push_back(raw_pose);
    // std::cout<<"the imu pose is "<<raw_pose<<std::endl;
    }

    Eigen::Matrix4f init_traslate;
    // init_traslate = vector_g2o_pose[0];
    // double init_x, init_y, init_z, init_roll, init_pitch, init_yaw;
    // matrix_to_pose(init_traslate, init_x, init_y, init_z, init_roll, init_pitch, init_yaw);
    std::vector<float> init_pose;
    init_pose.resize(6,0);
    init_pose[0] = 0.3;
    init_pose[1] = 0.2;
    init_pose[3] = 0.3;
    init_pose[4] = 0.2;
    init_pose[5] = 1.9;
    init_traslate = vec_pose_to_matrix(init_pose);
    std::cout<<"the init traslate is "<<init_traslate<<std::endl;
    for (int i = 0; i < vector_g2o_pose.size(); ++i)
    {
        // vector_g2o_pose[i] =init_traslate.inverse()*vector_g2o_pose[i];
        vector_g2o_pose[i] =vector_g2o_pose[i]*init_traslate;
    }

    int num_params = 5;
    int total_data = vector_g2o_pose.size();

    // {
        cv::Mat inputs(total_data, 1, CV_32S);
        //load observation data
        for (int i = 0; i < total_data; i++)
        {
            inputs.at<int>(i, 0) = i; 
        }
        // Guess the parameters, it should be close to the true value, else it can fail for very sensitive functions!
        cv::Mat params(num_params, 1, CV_64F);

        //init gues
        params.at<double>(0, 0) = 1.5;
        params.at<double>(1, 0) = 0.1;
        params.at<double>(2, 0) = 0.5;
        params.at<double>(3, 0) = 0.1;
        params.at<double>(4, 0) = 0.5;
        int m = inputs.rows;
        int n = inputs.cols;

        cv::Mat r(m, 1, CV_64F); // residual Matrix
        cv::Mat r_tmp(m, 1, CV_64F);
        cv::Mat Jf(m, num_params, CV_64F); // Jacobian of Func()
        cv::Mat input(1, n, CV_32S);       // single row input
        cv::Mat params_tmp = params.clone();

        double last_mse = 0;
        float u = 1, v = 2;
        cv::Mat I = cv::Mat::eye(num_params, num_params, CV_64F); //construct identity Matrix
        int i = 0;
        for (i = 0; i < MAX_ITER; i++)
        {
            double mse = 0;
            double mse_temp = 0;

            for (int j = 0; j < m; j++)
            {
                r.at<double>(j, 0) = cost_distance(vector_g2o_pose[j],vector_raw_pose[j],params,init_traslate); //diff between previous esticv::Mate and observation population
                mse += 0.5*r.at<double>(j, 0) * r.at<double>(j, 0);
                for (int k = 0; k < num_params; k++)
                {
                    Jf.at<double>(j, k) = Deriv(vector_g2o_pose[j],vector_raw_pose[j], params,init_traslate, k); //construct jacobian Matrix
                    // std::cout<<"the jk is "<<j<<","<<k<<":"<<Jf.at<double>(j, k)<<std::endl;
                }
            }
            
            mse /= m;
            std::cout<<"the raw mse is "<<mse<<std::endl;
            params_tmp = params.clone();
            std::cout<<"the parameter is "<<params_tmp.at<double>(0,0)<<","<<params_tmp.at<double>(1,0)<<","<<params_tmp.at<double>(2,0)<<","<<params_tmp.at<double>(3,0)<<","<<params_tmp.at<double>(4,0)<<std::endl;
            cv::Mat hlm = (Jf.t() * Jf + u * I).inv() * (-Jf.t() * r);
            cv::Mat before_inv = Jf.t() * Jf;
            std::cout<<"the last roll of before is "<<before_inv.at<double>(0,0)<<","<<before_inv.at<double>(0,1)<<","<<before_inv.at<double>(0,2)<<","<<before_inv.at<double>(0,3)<<","<<before_inv.at<double>(0,4)<<std::endl;
            std::cout<<"the last roll of before is "<<before_inv.at<double>(1,0)<<","<<before_inv.at<double>(1,1)<<","<<before_inv.at<double>(1,2)<<","<<before_inv.at<double>(1,3)<<","<<before_inv.at<double>(1,4)<<std::endl;
            std::cout<<"the last roll of before is "<<before_inv.at<double>(2,0)<<","<<before_inv.at<double>(2,1)<<","<<before_inv.at<double>(2,2)<<","<<before_inv.at<double>(2,3)<<","<<before_inv.at<double>(2,4)<<std::endl;
            std::cout<<"the last roll of before is "<<before_inv.at<double>(3,0)<<","<<before_inv.at<double>(3,1)<<","<<before_inv.at<double>(3,2)<<","<<before_inv.at<double>(3,3)<<","<<before_inv.at<double>(3,4)<<std::endl;
            std::cout<<"the last roll of before is "<<before_inv.at<double>(4,0)<<","<<before_inv.at<double>(4,1)<<","<<before_inv.at<double>(4,2)<<","<<before_inv.at<double>(4,3)<<","<<before_inv.at<double>(4,4)<<std::endl;
           
            params_tmp += hlm;
            params_tmp.at<double>(0,0) = atan2f(sinf(params_tmp.at<double>(0,0)),cosf(params_tmp.at<double>(0,0)));
            for (int j = 0; j < m; j++)
            {
                r_tmp.at<double>(j, 0) = cost_distance(vector_g2o_pose[j],vector_raw_pose[j],params_tmp,init_traslate);; //diff between current esticv::Mate and observation population
                mse_temp += 0.5*r_tmp.at<double>(j, 0) * r_tmp.at<double>(j, 0);
            }

            mse_temp /= m;
        
            cv::Mat q(1, 1, CV_64F);
            q = (mse - mse_temp) / (0.5 * hlm.t() * (u * hlm - Jf.t() * r));
            double q_value = q.at<double>(0, 0);
            std::cout<<"the q value is "<<q_value<<std::endl;
            if (q_value > 0)
            {
                double s = 1.0 / 3.0;
                v = 2;
                mse = mse_temp;
                params = params_tmp;
                double temp = 1 - pow(2 * q_value - 1, 3);
                if (s > temp)
                {
                    u = u * s;
                }
                else
                {
                    u = u * temp;
                }
            }
            else
            {
                u = u * v;
                v = 2 * v;
            }

            // // The difference in mse is very small, so quit
            // if (fabs(mse - last_mse) < 1e-8)
            // {
            //     break;
            // }

            //printf("%d: mse=%f\n", i, mse);
            printf("%d %lf\n", i, mse);
            last_mse = mse;
        }
    // }
}