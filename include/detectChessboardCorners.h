#ifndef DETECTCHESSBOARDCORNERS_H
#define DETECTCHESSBOARDCORNERS_H

#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <time.h>

using namespace std;
using namespace cv;

bool detectChessboardCorners(const vector<string>& list1,
  vector<string>& list_detected_1, const vector<string>& list2,
  vector<string>& list_detected_2, vector<Mat>& image_points_1,
  vector<Mat>& image_points_2, Size boardSize, Size& imageSize1,
  Size& imageSize2, bool show_img);

#endif
