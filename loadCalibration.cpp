#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <time.h>

#include "stereocam.h"

using namespace cv;
using namespace std;

int loadCalibration(const string& filename, StereoCam& camera)
{
  FileStorage fs(filename, FileStorage::READ);
  if (!fs.isOpened())
  {
    return fprintf(stderr, "Could not open calibration file\n"), -1;
  }

  fs["camera_matrix_1"] >> camera.K1;
  fs["camera_matrix_2"] >> camera.K2;
  fs["distortion_coefficients_1"] >> camera.D1;
  fs["distortion_coefficients_2"] >> camera.D2;
  fs["xi_1"] >> camera.xi1;
  fs["xi_2"] >> camera.xi2;
  fs["rvec"] >> camera.rvec;
  fs["tvec"] >> camera.tvec;
  fs["idx"] >> camera.idx;

  fs.release();
  return 0;
}
