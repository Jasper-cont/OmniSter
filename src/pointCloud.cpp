#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <string>

#include "stereocam.h"
#include "calib_img_gen.h"

using namespace cv;
using namespace std;

bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr generatePointCloud(Mat rectImgC, Mat dispMap, StereoCam cam, float fs, OutputArray pointCloud, string type = "gray")
{
  vector<Mat> rgb;

  // default grayscale
  bool grayscale = 1;
  if (type=="color") {
    //cout << "Outputting Colored Point Cloud..." << endl;
    grayscale = 0;
    split(rectImgC, rgb);
  }

  dispMap.convertTo(dispMap, CV_32F);

  // some regions of image1 is black, the corresponding regions of disparity map is also invalid.
  Mat grayImg, binaryImg, idx, _dispMap;
  if (rectImgC.channels() == 3)
  {
      cvtColor(rectImgC, grayImg, COLOR_RGB2GRAY);
  }
  else
  {
      grayImg = rectImgC;
  }

  binaryImg = (grayImg <= 0);
  findNonZero(binaryImg, idx);
  for (int i = 0; i < (int)idx.total(); ++i)
  {
      Vec2i _idx = idx.at<Vec2i>(i);
      dispMap.at<float>(_idx[1], _idx[0]) = 0.0f;
  }

  // norm may end up flipping y-axis
  float baseline = norm(cam.tvec);

  vector<Vec3f> _pointCloud;
  vector<Vec6f> _pointCloudColor;
  // PCL variable declaration
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  for (int i=0; i< dispMap.size().width; ++i) {
    for (int j=0; j< dispMap.size().height; ++j) {
      Vec3f point; // (x,y,z) coordinate
      Vec6f pointColor; // (x,y,z,r,g,b)

      // Arbritrary Threshold - First try without
      if (dispMap.at<float>(j,i) >= 1){
        float theta_top = j / fs;
        float phi_top = i / fs;

        float phi_bot = (i + dispMap.at<float>(j,i))/fs;

        // trig in radians
        float rad_disp = (dispMap.at<float>(j,i))/fs;
        //float rho_top = (sin(phi_bot) / sin(rad_disp)) * baseline;
        float rho_top = baseline * fs / dispMap.at<float>(j, i);

        float x = rho_top * sin(phi_top) * cos(theta_top);
        float y = -rho_top * cos(phi_top);
        float z = rho_top * sin(phi_top) * sin(theta_top);

        point = Vec3f(x,y,z);

        if (grayscale) {
          _pointCloud.push_back(point);
        } else {
          pointColor[0] = point[0];
          pointColor[1] = point[1];
          pointColor[2] = point[2];
          pointColor[3] = rgb[2].at<uchar>(j,i);
          pointColor[4] = rgb[1].at<uchar>(j,i);
          pointColor[5] = rgb[0].at<uchar>(j,i);

          _pointCloudColor.push_back(pointColor);


          //PCL Allocation
          pcl::PointXYZRGB pointer;
		  pointer.x = point[0];
          pointer.y = point[1];
          pointer.z = point[2];
          uint8_t r = rgb[2].at<uchar>(j,i);
          uint8_t g = rgb[1].at<uchar>(j,i);
          uint8_t b = rgb[0].at<uchar>(j,i);
          std::uint32_t rgb = (static_cast<std::uint32_t>(r) << 16 | static_cast<std::uint32_t>(g) << 8 | static_cast<std::uint32_t>(b));
          pointer.rgb = *reinterpret_cast<float*>(&rgb);

          cloud->points.push_back(pointer);
          //cout << pointColor<< endl;
        }
      }
    }  // End height sweep
  } // End width sweep

  if (grayscale) {
    Mat(_pointCloud).convertTo(pointCloud, CV_MAKE_TYPE(CV_32F, 3));
  } else {
    Mat(_pointCloudColor).convertTo(pointCloud, CV_MAKE_TYPE(CV_32F, 6));
  }
  return cloud;
  cout << "Here" << endl;
}

int savePointCloud(string file, Mat pointCloud)
{

  if (!has_suffix(file, ".ply")) {
    cout << "Point Cloud Generation Failed" << endl;
    cout << "... File must be of type .ply" << endl;
    return 0;
  }

  else {
    ofstream fs(file);
    if (fs.is_open())
    {
      //cout << "Saving Point Cloud" << endl;

      // Header stuff
      fs << "ply" << endl;
      fs << "format ascii 1.0" << endl;
      fs << "element vertex " << int2str(pointCloud.rows * pointCloud.cols) << endl;
      fs << "property float x" << endl;
      fs << "property float y" << endl;
      fs << "property float z" << endl;
      fs << "property uchar red" << endl;
      fs << "property uchar green" << endl;
      fs << "property uchar blue" << endl;
      fs << "end_header" << endl;

      for (int i=0; i < pointCloud.rows; ++i)
      {

        // TODO: need to change to factor float xyz and uchar rgb!
        // TODO: find out how to access each channel in a MAT cuz whatever below definetly aint right.
        for (int j=0; j < 6; ++j)
        {
          fs << pointCloud.at<float>(i,j);
          if (j < 5)
          {
            fs << " ";
          }
        }
        fs << endl;
      }
      fs.close();
      //cout << "Point Cloud saved to " << file << endl;
      return 1;
    } else {
      cout << "Could not open file!" << endl;
      return 0;
    }
  }
}
