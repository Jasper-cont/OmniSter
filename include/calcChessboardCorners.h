#ifndef CALCCHESSBOARDCORNERS_H
#define CALCCHESSBOARDCORNERS_H

#include "opencv2/core.hpp"

using namespace cv;

void calcChessboardCorners(Size boardSize, double square_width, double square_height,
    Mat& corners);

#endif
