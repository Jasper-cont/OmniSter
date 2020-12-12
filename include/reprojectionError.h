#ifndef REPROJECTIONERROR_H
#define REPORJECTIONERROR_H

#include "opencv2/core.hpp"
#include <vector>

using namespace cv;

double reprojectionError(Mat imgPoints, Mat projPoints);

#endif
