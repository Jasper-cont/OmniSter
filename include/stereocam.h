#ifndef STEREOCAM_H
#define STEREOCAM_H

#include "opencv2/core.hpp"
#include <string>
#include <vector>

using namespace cv;
using namespace std;

class StereoCam
{
public:
  Mat K1, K2;
  Mat D1, D2;
  double xi1, xi2;
  Mat rvec, tvec;
  int flags;
  double rms;
  Mat idx;

  Size imgSize_1, imgSize_2;

  vector<string> detect_list_1, detect_list_2, detect_list;
  vector<Mat> imgPoints_1, imgPoints_2;
  vector<Mat> objPoints;
  vector<Vec3d> rvecs, tvecs;
  vector<double> reprojError;
};

#endif
