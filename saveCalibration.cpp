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

void saveCalibration(const string& filename, StereoCam& camera)
{
  FileStorage fs(filename, FileStorage::WRITE);

  time_t tt;
  time(&tt);
  struct tm *t2 = localtime(&tt);
  char buf[1024];
  strftime(buf, sizeof(buf)-1, "%c", t2);

  fs << "calibration_time" << buf;

  fs << "camera_matrix_1" << camera.K1;
  fs << "distortion_coefficients_1" << camera.D1;
  fs << "xi_1" << camera.xi1;

  fs << "camera_matrix_2" << camera.K2;
  fs << "distortion_coefficients_2" << camera.D2;
  fs << "xi_2" << camera.xi2;

  fs << "rvec" << camera.rvec;
  fs << "tvec" << camera.tvec;

  fs << "rvecs" << camera.rvecs;
  fs << "tvecs" << camera.tvecs;

  fs << "rms" << camera.rms;

  fs << "idx" << camera.idx;

  fs << "rmss" << camera.reprojError;

  if ( !camera.imgPoints_1.empty() )
  {
      Mat imageMat((int)camera.imgPoints_1.size(), (int)camera.imgPoints_1[0].total(), CV_64FC2);
      for (int i = 0; i < (int)camera.imgPoints_1.size(); ++i)
      {
          Mat r = imageMat.row(i).reshape(2, imageMat.cols);
          Mat imagei(camera.imgPoints_1[i]);
          imagei.copyTo(r);
      }
      fs << "image_points_1" << imageMat;
  }

  if ( !camera.imgPoints_2.empty() )
  {
      Mat imageMat((int)camera.imgPoints_2.size(), (int)camera.imgPoints_2[0].total(), CV_64FC2);
      for (int i = 0; i < (int)camera.imgPoints_2.size(); ++i)
      {
          Mat r = imageMat.row(i).reshape(2, imageMat.cols);
          Mat imagei(camera.imgPoints_2[i]);
          imagei.copyTo(r);
      }
      fs << "image_points_2" << imageMat;
  }

  if ( !camera.objPoints.empty() )
  {
      Mat imageMat((int)camera.objPoints.size(), (int)camera.objPoints[0].total(), CV_64FC3);
      for (int i = 0; i < (int)camera.objPoints.size(); ++i)
      {
          Mat r = imageMat.row(i).reshape(3, imageMat.cols);
          Mat imagei(camera.objPoints[i]);
          imagei.copyTo(r);
      }
      fs << "object_points" << imageMat;
  }
  //fs << "image_points_1" << camera.imgPoints_1;
  //fs << "image_points_2" << camera.imgPoints_2;
  //fs << "object_points" << camera.objPoints;

  fs << "detect_list_1" << camera.detect_list_1;
  fs << "detect_list_2" << camera.detect_list_2;
}
