#include "opencv2/core.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

double reprojectionError(Mat imgPoints, Mat projPoints)
{
  double rms = 0.0;

  Mat errorI = imgPoints.reshape(2, imgPoints.rows * imgPoints.cols) -
              projPoints.reshape(2, projPoints.rows * projPoints.cols);

  Vec2d* ptr_err = errorI.ptr<Vec2d>();
  for (int i=0; i < (int)errorI.total(); i++)
  {
    rms += sqrt(ptr_err[i][0] * ptr_err[i][0] + ptr_err[i][1] * ptr_err[i][1]);
  }

  rms /= (double)errorI.total();

  return rms;
}
