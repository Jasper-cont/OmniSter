#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include "opencv2/core.hpp"
#include "stereocam.h"
#include <string>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

using namespace cv;
using namespace std;

bool has_suffix(const std::string &str, const std::string &suffix);
int savePointCloud(string file, Mat pointCloud);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr generatePointCloud(Mat undistImg, Mat dispMap, StereoCam cam, float fs, OutputArray pointCloud, string type = "gray");

#endif
