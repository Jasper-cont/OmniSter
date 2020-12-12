#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/rgbd.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <glob.h>
#include <algorithm>
#include <memory>
#include <thread>

#include "omp.h"
#include "yocto_api.h"
#include "yocto_gps.h"
#include "stereocam.h"
#include "calib_img_gen.h"
#include "globVector.h"
#include "detectChessboardCorners.h"
#include "calcChessboardCorners.h"
#include "saveCalibration.h"
#include "loadCalibration.h"
#include "reprojectionError.h"
#include "pointCloud.h"
#include "sgbm_param.h"
#include "SPSStereo.h"
#include "defParameter.h"

using namespace cv;
using namespace std;

static void help()
{
  printf("\n"
    "-g                        # Generate Images for Calibration\n"
    "-c                        # Run Calibration\n"
    "-m                        # Manual Calibration Confirmation\n"
    "-i <calibration_img_dir>  # Calibration Image Directory\n"
    "-w <board_width>          # The Number of Inner Corners x-direction\n"
    "-h <board_height>         # The Number of Inner Corners y-direction\n"
    "-sw <square_width>        # The Width of the Square\n"
    "-sh <square_height>       # The Height of the Square\n"
    "-fs <fix_skew>            # Fix Skew\n"
    "-fp                       # Fix the Principal Point\n"
    "-p <parameter_dir>        # Directory to Save Calibration to\n"
    "-s                        # Stream from Cameras\n"
    "-pc                       # Output Point Cloud\n"
    "-l                        # Record, Log and Save Camera Stream and GPS\n"
    "-gps                      # Use GPS\n"
    "-fps                      # Record Frame Rate\n"
    "-export                   # Export Merged Video\n"
    "-o                        # Use Odometry\n"
  );
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

bool sgbmChange(char key) {
  switch (key) {
    case 'q':
    case 'a':
    case 'w':
    case 's':
    case 'e':
    case 'd':
    case 'r':
    case 'f':
    case 't':
    case 'g':
    case 'y':
    case 'h':
    case 'u':
    case 'j':
    case 'i':
    case 'k':
    case 'o':
    case 'l':
    case 'p':
    case ';':
    case 'z':return 1;break;
    default: return 0; break;
  }
}

void sgbmHelp() {
  cout << "\nTo Configure Stereo Parameters Dynamically: " << endl;
  printf( "Number of outer Iterarions: q/a\n"
          "Number of inner Iterations: w/s\n"
          "Position Weight: e/d\n"
          "Depth Weight: r/f\n"
          "Boundary Length Weight: t/g\n"
          "Smoothness Weight: y/h\n"
          "Inlier Threshold: u/j\n"
          "Hinge Penalty Value: i/k\n"
          "Occlusion Penalty Value: o/l\n"
          "Impossible Penalty Value: p/;\n"
          "Print All Parameters: z\n\n"
  );
}


void rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, pcl::visualization::PCLVisualizer::Ptr viewer, int frame_num)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  const char* id = "Frame number: ";
  string point_cloud_id = id + to_string(frame_num);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, point_cloud_id);
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, point_cloud_id);
  viewer->addCoordinateSystem (1.0);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_merging(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud, pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloudB, const Mat & pcLength, const Mat & pcBLength, int frame_num, int frame_difference, Mat & accumulatedTranMatrix)
{
    // Concatenating Forward view and Backward View Point Clouds into an Omnidirectional Point Cloud and then Adding that onto the large point cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr aggregated_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    float movement_x_direction = 0, movement_z_direction = 0;
    if (frame_num >= frame_difference)
    {
    	movement_x_direction = accumulatedTranMatrix.at<float>(frame_num, 0) - accumulatedTranMatrix.at<float>(frame_num - frame_difference, 0);
        movement_z_direction = accumulatedTranMatrix.at<float>(frame_num, 1) - accumulatedTranMatrix.at<float>(frame_num - frame_difference, 1);
    }
		for (int y=0; y<pcLength.rows; y++)
		{
			  pcl::PointXYZRGB pointer;
			  pointer.x = movement_x_direction + pointcloud->points[y].x;
			  pointer.y = pointcloud->points[y].y;
			  pointer.z = movement_z_direction + pointcloud->points[y].z;
			  pointer.r = pointcloud->points[y].r;
			  pointer.g = pointcloud->points[y].g;
			  pointer.b = pointcloud->points[y].b;

			  aggregated_cloud->points.push_back(pointer);
		}
		for (int y=0; y<pcBLength.rows; y++)
		{
			  pcl::PointXYZRGB pointer;
			  pointer.x = -pointcloudB->points[y].x + movement_x_direction;
			  pointer.y = pointcloudB->points[y].y;
			  pointer.z = -pointcloudB->points[y].z + movement_z_direction;
			  pointer.r = pointcloudB->points[y].r;
			  pointer.g = pointcloudB->points[y].g;
			  pointer.b = pointcloudB->points[y].b;

			  aggregated_cloud->points.push_back(pointer);
		}
    return aggregated_cloud;
}

int main(int argc, char** argv)
{

  /*************************** Calibration Parameters ************************/
  Size boardSize;
  double square_width, square_height;
  int calib_flags = 0;

  const char* calibDir = "../data/calibration_parameters";
  const char* calibImgDir = "../data/calibration_images";

  StereoCam frontCameras, backCameras;

  // Calibration Image Prefixes
  const string imgTF = "/top_front/tf_";
  const string imgTB = "/top_back/tb_";
  const string imgBF = "/bottom_front/bf_";
  const string imgBB = "/bottom_back/bb_";
  // Calibration Image Format (Use either png or bmp)
  const string imgFormat = ".bmp";

  // Calibration Output Filenames
  const string calibFrontOut = "/front_camera_params.xml";
  const string calibBackOut = "/back_camera_params.xml";
  /***************************************************************************/

  /************************** Run-Time Flags *********************************/
  bool CALIBRATION = 0;
  bool IMG_GEN = 0;
  bool SHOW_CALIB_IMG = 0;
  bool CAMERA_STREAM = 0;
  bool USE_SGBM = 1;
  bool POINT_CLOUD = 0;
  bool LOG_DATA = 0;
  bool USE_GPS = 0;
  bool USE_PSEUDO = 0;
  bool MEDIA_OUT = 0;
  bool LOG_FPS = 0;
  bool ODOMETRY = 0;
  /***************************************************************************/

  // Vertical and Horizontal FOV set to 190 deg
  float fov_r = 190.0 /*DEG*/ * 3.1415926 * 2.0 / 360;

  // Assuming image height = image width = 640 => fs is the same for both
  float fs = 640.0 / fov_r;

  // Parsing Command Line Parameters
  for (int i = 1; i < argc; i++)
  {
    const char* s = argv[i];

    if (strcmp(s, "-g") == 0)
    {
      IMG_GEN = 1;
    }
    else if (strcmp(s,"-l") == 0)
    {
      LOG_DATA = 1;
    }
    else if (strcmp(s, "-c") == 0)
    {
      CALIBRATION = 1;
    }
    else if (strcmp(s, "-i") == 0)
    {
      calibImgDir = argv[++i];
    }
    else if (strcmp(s, "-w") == 0)
    {
      if (sscanf(argv[++i], "%u", &boardSize.width) != 1 || boardSize.width <= 0)
        return fprintf(stderr, "Invalid Board Width\n"), -1;
    }
    else if (strcmp(s, "-h") == 0)
    {
      if (sscanf(argv[++i], "%u", &boardSize.height) != 1 || boardSize.height <= 0)
        return fprintf(stderr, "Invalid Board Height\n"), -1;
    }
    else if (strcmp(s, "-sw") == 0)
    {
      if (sscanf(argv[++i], "%lf", &square_width) != 1 || square_width <= 0)
        return fprintf(stderr, "Invalid Square Width\n"), -1;
    }
    else if (strcmp(s, "-sh") == 0)
    {
      if (sscanf(argv[++i], "%lf", &square_height) != 1 || square_height <= 0)
        return fprintf(stderr, "Invalid Square Height\n"), -1;
    }
    else if (strcmp(s, "-fs") == 0)
    {
      calib_flags |= omnidir::CALIB_FIX_SKEW;
    }
    else if (strcmp(s, "-fp") == 0)
    {
      calib_flags |= omnidir::CALIB_FIX_CENTER;
    }
    else if (strcmp(s, "-p") == 0)
    {
      calibDir = argv[++i];
    }
    else if (strcmp(s, "-pc") == 0)
    {
      POINT_CLOUD = 1;
    }
    else if (strcmp(s, "-m") == 0)
    {
      SHOW_CALIB_IMG = 1;
    }
    else if (strcmp(s, "-sgbm") == 0)
    {
      USE_SGBM = 1;
    }
    else if (strcmp(s, "-s") == 0)
    {
      CAMERA_STREAM = 1;
    }
    else if (strcmp(s, "-gps") == 0)
    {
      USE_GPS = 1;
    }
    else if (strcmp(s, "-fps") == 0)
    {
      LOG_FPS = 1;
    }
    else if (strcmp(s, "-export") == 0)
    {
      MEDIA_OUT = 1;
    }
    else if (strcmp(s, "-o") == 0)
	{
	  ODOMETRY = 1;
	}
    else
    {
      return fprintf(stderr, "Unknown Option %s\n", s), -1;
    }
  } // END Parsing


  if (LOG_DATA)
  {
    // Stores the streamed data from the cameras along with GPS data
    // for further analysis

    // Files to store data to:
    string gps_settings = "../data/logging/gps_settings.csv";
    string gps_data = "../data/logging/gps_data.csv";

    ofstream gps_settings_out (gps_settings);
    ofstream gps_data_out (gps_data);

    // Setup Camera stream
    VideoCapture top_stream;
    VideoCapture bot_stream;
    Mat top_cap, bot_cap;

    cout << "Streaming from Cameras ... " << endl;

    top_stream.open(4);
    bot_stream.open(2);

    if (!top_stream.isOpened() || !bot_stream.isOpened())
    {
      cerr << "Unable to open cameras.\n";
      return -1;
    }

    char flipKey = 0;
    int baseLine = 0;
    const Scalar GREEN(0,255,0);

    while (!(flipKey == 'n') && !(flipKey == 'y'))
    {
      top_stream.read(top_cap);
      bot_stream.read(bot_cap);

      string msg = "Flip Cameras? (y/n)";
      Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
      Point textOrigin(top_cap.cols - 2*textSize.width -10, top_cap.rows
                          -2*baseLine - 10);
      putText(top_cap, msg, textOrigin, 1, 1, GREEN);
      putText(bot_cap, msg, textOrigin, 1, 1, GREEN);

      imshow("Top", top_cap);
      imshow("Bottom", bot_cap);
      flipKey = (char)waitKey(5);
    }

    if (flipKey == 'y')
    {
      top_stream.release();
      bot_stream.release();
      top_stream.open(2);
      bot_stream.open(4);
    }

    destroyWindow("Top");
    destroyWindow("Bottom");

    cout << "Cameras Opened" << endl;

    double dTopWidth = top_stream.get(CAP_PROP_FRAME_WIDTH);
    double dTopHeight = top_stream.get(CAP_PROP_FRAME_HEIGHT);
    double dBotWidth = bot_stream.get(CAP_PROP_FRAME_WIDTH);
    double dBotHeight = bot_stream.get(CAP_PROP_FRAME_HEIGHT);

    Size top_frame_size(static_cast<int>(dTopWidth), static_cast<int>(dTopHeight));
    Size bot_frame_size(static_cast<int>(dBotWidth), static_cast<int>(dBotHeight));

    // Working out camera fps
    cout << "Estimating FPS ... " << endl;
    int num_frames = 120;
    double fps = 0.0;
    Mat test_frame;
    int64 tickFreq = getTickFrequency();
    int64 startTime = getTickCount();

    for (int i =0; i < num_frames; ++i)
    {
      top_stream >> test_frame;
    }

    int64 endTime = getTickCount();

    fps = num_frames / ((endTime - startTime)/tickFreq);
    cout << "Estimated FPS = " << fps << endl;

    VideoWriter top_write("../data/logging/top_view.avi", VideoWriter::fourcc('H','2','6','4'), fps, top_frame_size, true);
    VideoWriter bot_write("../data/logging/bot_view.avi", VideoWriter::fourcc('H','2','6','4'), fps, bot_frame_size, true);

    // Setup for GPS
    YGps *gps;
    if (USE_GPS)
    {
      cout << "\nSetting up GPS ..." << endl;
      string errmsg, target;

      if (yRegisterHub("usb", errmsg) != YAPI_SUCCESS)
      {
        cerr << "RegisterHub Error: " << errmsg << endl;
        return 1;
      }

      gps = yFirstGps();
      if (gps == NULL)
      {
        cout << "GPS Device Not Found. (Check USB Connection)" << endl;
      }

      // Storing GPS Settings
      //gps_settings_out << "serial: " << gps->get_serialNumber() << endl;
      //gps_settings_out << "logical name: " << gps->get_logicalName() << endl;
    }

    cout << "Start Logging ..." << endl;

    bool GPS_ONLINE = 1;
    bool GPS_FIXED = 1;

    // Main Logging Loop
    while(1) {
      if (USE_GPS)
      {
        if (!gps->isOnline()){
          gps_data_out << 0.0 << "," << 0.0 << "," << 0.0 << "," <<gps->get_satCount()<< "," <<
            0.0 << endl;
          if(GPS_ONLINE){
            GPS_ONLINE = 0;
            cout << "GPS Lost Connection." << endl;
          }
        }
        else if (!gps->isFixed()) {
          gps_data_out << 0.0 << "," << 0.0 << "," << 0.0 << "," << gps->get_satCount() << "," <<
            0.0 << endl;
            if(GPS_FIXED){
              GPS_FIXED = 0;
              cout << "GPS Fixing ... " << endl;
            }
        }
        else {
          GPS_ONLINE = 1;
          GPS_FIXED = 1;
          cout << "GPS Save: " << gps->get_unixTime() << endl;
          // GPS has locked and ready to go
          gps_data_out << gps->get_unixTime() << "," << gps->get_longitude() <<
            "," << gps->get_latitude() << "," << gps->get_satCount() << "," <<
            gps->get_altitude() << endl;
        }
      }

      // Grab Camera frames
      top_stream.grab();
      bot_stream.grab();
      top_stream.retrieve(top_cap);
      bot_stream.retrieve(bot_cap);

      top_write.write(top_cap);
      bot_write.write(bot_cap);

      imshow("Top Live", top_cap);
      imshow("Bot Live", bot_cap);

      if(waitKey(5) == 27) {
        cout << "Logging Ended" << endl;
        gps_settings_out.close();
        gps_data_out.close();
        return 0;
      }

    }

  }

  if (IMG_GEN)
  {
    calib_img_gen(calibImgDir, imgTF, imgTB, imgBF, imgBB, imgFormat);
  }

  if (CALIBRATION)
  {
    vector<string> tfImgList, tbImgList, bfImgList, bbImgList;

    cout << "Reading Calibration Image Directory: " << calibImgDir << endl;

    tfImgList = globVector(calibImgDir + imgTF + "*" + imgFormat);
    tbImgList = globVector(calibImgDir + imgTB + "*" + imgFormat);
    bfImgList = globVector(calibImgDir + imgBF + "*" + imgFormat);
    bbImgList = globVector(calibImgDir + imgBB + "*" + imgFormat);

    if (tfImgList.empty() || tbImgList.empty() || bfImgList.empty() ||
        bbImgList.empty())
    {
      fprintf(stderr, "Image Files Not Found in Directory: %s\n", calibImgDir);
      return -1;
    } else
    {
      cout << "Image Lists Generated!\n" << endl;
    }

    cout << "Detecting Chessboard Corners ... " << endl;

    if (!detectChessboardCorners(tfImgList, frontCameras.detect_list_1,
         bfImgList, frontCameras.detect_list_2,
         frontCameras.imgPoints_1, frontCameras.imgPoints_2,
         boardSize, frontCameras.imgSize_1, frontCameras.imgSize_2,
         SHOW_CALIB_IMG))
      return fprintf(stderr, "Not enough corner detected front images\n"), -1;

    if (!detectChessboardCorners(tbImgList, backCameras.detect_list_1,
         bbImgList, backCameras.detect_list_2,
         backCameras.imgPoints_1, backCameras.imgPoints_2,
         boardSize, backCameras.imgSize_1, backCameras.imgSize_2,
         SHOW_CALIB_IMG))
      return fprintf(stderr, "Not enough corner detected back images\n"), -1;

    cout << "Top Front Camera: " << frontCameras.detect_list_1.size() <<
            "/" << tfImgList.size() << " Detected" << endl;
    cout << "Bottom Front Camera: " << frontCameras.detect_list_2.size() <<
            "/" << bfImgList.size() << " Detected" << endl;
    cout << "Top Back Camera: " << backCameras.detect_list_1.size() <<
            "/" << tbImgList.size() << " Detected" << endl;
    cout << "Bottom Back Camera: " << backCameras.detect_list_2.size() <<
            "/" << bbImgList.size() << " Detected" << endl;

    cout << "Chessboard Corner Detection Finished!\n" << endl;

    cout << "Calculating Object Coordinates of Chessboard Corners ... ";

    Mat objectF, objectB;
    calcChessboardCorners(boardSize, square_width, square_height, objectF);
    for (int i=0; i < (int)frontCameras.detect_list_1.size(); ++i)
    {
      frontCameras.objPoints.push_back(objectF);
    }

    calcChessboardCorners(boardSize, square_width, square_height, objectB);
    for (int i=0; i < (int)backCameras.detect_list_1.size(); ++i)
    {
      backCameras.objPoints.push_back(objectB);
    }
    cout << "Done" << endl;

    Mat _xiTF, _xiBF, _xiTB, _xiBB;

    cout << "Front Camera Calibration ... ";
    // criteria(3,200,0.0001) was the default which worked.
    TermCriteria criteriaA(3, 100, 0.0001);
    frontCameras.rms = omnidir::stereoCalibrate(frontCameras.objPoints,
                        frontCameras.imgPoints_1, frontCameras.imgPoints_2,
                        frontCameras.imgSize_1, frontCameras.imgSize_2,
                        frontCameras.K1, _xiTF, frontCameras.D1,
                        frontCameras.K2, _xiBF, frontCameras.D2,
                        frontCameras.rvec, frontCameras.tvec,
                        frontCameras.rvecs, frontCameras.tvecs,
                        calib_flags, criteriaA, frontCameras.idx);
    cout << "Done" << endl;

    frontCameras.xi1 = _xiTF.at<double>(0);
    frontCameras.xi2 = _xiBF.at<double>(0);

    cout << "Back Camera Calibration ... ";
    TermCriteria criteriaB(3, 100, 0.0001);
    backCameras.rms = omnidir::stereoCalibrate(backCameras.objPoints,
                        backCameras.imgPoints_1, backCameras.imgPoints_2,
                        backCameras.imgSize_1, backCameras.imgSize_2,
                        backCameras.K1, _xiTB, backCameras.D1,
                        backCameras.K2, _xiBB, backCameras.D2,
                        backCameras.rvec, backCameras.tvec,
                        backCameras.rvecs, backCameras.tvecs,
                        calib_flags, criteriaB, backCameras.idx);

    backCameras.xi1 = _xiTB.at<double>(0);
    backCameras.xi2 = _xiBB.at<double>(0);
    cout << "Done" << endl;

    for (int i=0; i < frontCameras.idx.total(); ++i)
    {
      double reprojErrT, reprojErrB, reprojErr;
      Mat projectedPointsT, projectedPointsB;
      Mat _idx = frontCameras.idx;
      omnidir::projectPoints(frontCameras.objPoints[i],
                             projectedPointsT,
                             frontCameras.rvecs[i],
                             frontCameras.tvecs[i],
                             frontCameras.K1, frontCameras.xi1,
                             frontCameras.D1, noArray());
      Mat _omR, _RR, _TR, _RL, _TL, _R;
      Rodrigues(frontCameras.rvec, _R);
      Rodrigues(frontCameras.rvecs[i], _RL);
      _TL = Mat(frontCameras.tvecs[i]).reshape(1,3);
      _RR = _R * _RL;
      _TR = _R * _TL + frontCameras.tvec;
      Rodrigues(_RR, _omR);
      omnidir::projectPoints(frontCameras.objPoints[i],
                             projectedPointsB, _omR, _TR,
                             frontCameras.K2, frontCameras.xi2,
                             frontCameras.D2, noArray());

      reprojErrT = reprojectionError(frontCameras.imgPoints_1[_idx.at<int>(i)],
                                    projectedPointsT);
      reprojErrB = reprojectionError(frontCameras.imgPoints_2[_idx.at<int>(i)],
                                    projectedPointsB);

      reprojErr = (reprojErrT + reprojErrB) / 2.0;
      frontCameras.reprojError.push_back(reprojErr);
    }
    for (int i=0; i < backCameras.idx.total(); ++i)
    {
      double reprojErrT, reprojErrB, reprojErr;
      Mat projectedPointsT, projectedPointsB;
      Mat _idx = backCameras.idx;
      omnidir::projectPoints(backCameras.objPoints[i],
                             projectedPointsT,
                             backCameras.rvecs[i],
                             backCameras.tvecs[i],
                             backCameras.K1, backCameras.xi1,
                             backCameras.D1, noArray());
      Mat _omR, _RR, _TR, _RL, _TL, _R;
      Rodrigues(backCameras.rvec, _R);
      Rodrigues(backCameras.rvecs[i], _RL);
      _TL = Mat(backCameras.tvecs[i]).reshape(1,3);
      _RR = _R * _RL;
      _TR = _R * _TL + backCameras.tvec;
      Rodrigues(_RR, _omR);
      omnidir::projectPoints(backCameras.objPoints[i],
                             projectedPointsB, _omR, _TR,
                             backCameras.K2, backCameras.xi2,
                             backCameras.D2, noArray());
      cout << endl << backCameras.detect_list_1[_idx.at<int>(i)] << endl;
      reprojErrT = reprojectionError(backCameras.imgPoints_1[_idx.at<int>(i)],
                                    projectedPointsT);
      reprojErrB = reprojectionError(backCameras.imgPoints_2[_idx.at<int>(i)],
                                    projectedPointsB);

      reprojErr = (reprojErrT + reprojErrB) / 2.0;
      backCameras.reprojError.push_back(reprojErr);
    }

    cout << "Saving longitude-latitude projection of used calibration images ...";
    string rectDir = "/rectified_images";
    for (int i=0; i < frontCameras.idx.total(); ++i)
    {
      Mat uFrame;
      Mat frame = imread(frontCameras.detect_list_1[frontCameras.idx.at<int>(i)]);

      Size size = frame.size();

      Mat Knew = (Mat_<double>(3,3) << size.width / 3.1415926,0,0,
                                 0, size.height / 3.1415926, 0,
                                 0, 0, 1);

      omnidir::undistortImage(frame, uFrame, frontCameras.K1, frontCameras.D1,
                              frontCameras.xi1, omnidir::RECTIFY_LONGLATI,
                              Knew, size);

      string fname = calibImgDir+rectDir+imgTF+int2str(i)+imgFormat;
      cout << fname << endl;
      imwrite(fname, uFrame);
    }

    cout << "Saving Calibration ... ";
    saveCalibration(calibDir+calibFrontOut, frontCameras);
    saveCalibration(calibDir+calibBackOut, backCameras);
    cout << "Done" << endl;

  } // END CALIBRATION

  if (!CALIBRATION)
  {
    loadCalibration(calibDir + calibFrontOut, frontCameras);
    loadCalibration(calibDir + calibBackOut, backCameras);

    if (frontCameras.rvec.empty() ||backCameras.rvec.empty())
    {
      cerr << "Failed to Load Calibration Data\n";
      return -1;
    }
  }

  // Main Stereo Algorithm

  Mat top_frame, bot_frame, top_frameB, bot_frameB;
  Mat top_cap, bot_cap;
  Mat tf_template, bf_template, tb_template, bb_template;

  VideoCapture top_stream;
  VideoCapture bot_stream;

  Mat mapTF1, mapTF2, mapBF1, mapBF2, mapTB1, mapTB2, mapBB1, mapBB2;



  if (CAMERA_STREAM)
  {
    cout << "Streaming from Cameras ... " << endl;

    top_stream.open(4);
    bot_stream.open(2);

    if (!top_stream.isOpened() || !bot_stream.isOpened())
    {
      cout << "Unable to open cameras." << endl;
      cout << "Trying to open Psuedo-Live Camera ... " << endl;

      top_stream.open("../data/live/top_view.avi");
      bot_stream.open("../data/live/bot_view.avi");

      if (!top_stream.isOpened() || !bot_stream.isOpened())
      {
        cerr << "Cannot Open Pseudo-Live Camera. Program Terminating\n";
        return -1;
      } else {
        cout << "Pseudo Cameras Opened" << endl;
        USE_PSEUDO = 1;
        int top_fps = top_stream.get(CAP_PROP_FPS);
        int bot_fps = bot_stream.get(CAP_PROP_FPS);
        top_stream.set(CAP_PROP_FPS, top_fps);
        bot_stream.set(CAP_PROP_FPS, bot_fps);

        // Used to change Start Time of Loaded in Pseudo Live Feed
        //top_stream.set(CV_CAP_PROP_POS_MSEC, 902000);
        //bot_stream.set(CV_CAP_PROP_POS_MSEC, 902000);
      }

    }

    char flipKey = 0;
    int baseLine = 0;
    const Scalar GREEN(0,255,0);

    while (!(flipKey == 'n') && !(flipKey == 'y'))
    {
      top_stream.read(top_cap);
      bot_stream.read(bot_cap);

      string msg = "Flip Cameras? (y/n)";
      Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
      Point textOrigin(top_cap.cols - 2*textSize.width -10, top_cap.rows
                          -2*baseLine - 10);
      putText(top_cap, msg, textOrigin, 1, 1, GREEN);
      putText(bot_cap, msg, textOrigin, 1, 1, GREEN);

      imshow("Top", top_cap);
      imshow("Bottom", bot_cap);
      flipKey = (char)waitKey(5);

      if (USE_PSEUDO)
      {
        flipKey = 'n';
      }
    }

    if (flipKey == 'y')
    {
      top_stream.release();
      bot_stream.release();
      top_stream.open(2);
      bot_stream.open(4);
    }

    bf_template = Mat(bot_cap, Rect(Point(0, 0), Point((bot_cap.cols / 2), 640)));
    bb_template = Mat(bot_cap, Rect(Point((bot_cap.cols / 2), 0), Point(bot_cap.cols, 640)));
    tf_template = Mat(top_cap, Rect(Point(0, 0), Point(top_cap.cols / 2, 640)));
    tb_template = Mat(top_cap, Rect(Point((top_cap.cols / 2), 0), Point(top_cap.cols, 640)));

    destroyWindow("Top");
    destroyWindow("Bottom");

    // Testing Cameras
    transpose(tf_template, top_frame);
    flip(top_frame, top_frame, 1);
    transpose(bf_template, bot_frame);
    flip(bot_frame, bot_frame, 0);
    transpose(tb_template, top_frameB);
    flip(top_frameB, top_frameB, 0);
    transpose(bb_template, bot_frameB);
    flip(bot_frameB, bot_frameB, 1);
  }
  else
  {
    /********************* Stereo Set for Testing ******************************/
    top_frame = imread(calibImgDir + imgTF + "12" +imgFormat, IMREAD_COLOR);
    bot_frame = imread(calibImgDir + imgBF + "12" +imgFormat, IMREAD_COLOR);
    top_frameB = imread(calibImgDir + imgTB + "10" +imgFormat, IMREAD_COLOR);
    bot_frameB = imread(calibImgDir + imgBB + "10" +imgFormat, IMREAD_COLOR);
    /***************************************************************************/
    if (top_frame.empty() || bot_frame.empty())
    {
      return fprintf(stderr, "Could not open image files\n"), -1;
    }
  }

  /*************************** Undistortion Setup ****************************/
  Size imgSize = top_frame.size();

  // Longitude-Latitude Projection Matrix
  Mat Knew = (Mat_<double>(3,3) << imgSize.width / fov_r,0,0,
                               0, imgSize.height / fov_r, 0,
                               0, 0, 1);
  Mat dispMap, dispMap_8U, rect_top, rect_bot, dispColor;
  Mat dispMapB, dispMap_8UB, rect_topB, rect_botB, dispColorB;
  Mat R1F, R2F, R1B, R2B;

  omnidir::stereoRectify(frontCameras.rvec, frontCameras.tvec, R1F, R2F);
  omnidir::stereoRectify(backCameras.rvec, backCameras.tvec, R1B, R2B);

  omnidir::initUndistortRectifyMap(frontCameras.K1, frontCameras.D1,
                frontCameras.xi1, R1F, Knew, imgSize, CV_16SC2,
                mapTF1, mapTF2, omnidir::RECTIFY_LONGLATI);
  omnidir::initUndistortRectifyMap(frontCameras.K2, frontCameras.D2,
                frontCameras.xi2, R2F, Knew, imgSize, CV_16SC2,
                mapBF1, mapBF2, omnidir::RECTIFY_LONGLATI);

  omnidir::initUndistortRectifyMap(backCameras.K1, backCameras.D1,
                backCameras.xi1, R1B, Knew, imgSize, CV_16SC2,
                mapTB1, mapTB2, omnidir::RECTIFY_LONGLATI);
  omnidir::initUndistortRectifyMap(backCameras.K2, backCameras.D2,
                backCameras.xi2, R2B, Knew, imgSize, CV_16SC2,
                mapBB1, mapBB2, omnidir::RECTIFY_LONGLATI);

  // Initially populate rectify Mat so that ROI of disp can be set
  remap(top_frame, rect_top, mapTF1, mapTF2, INTER_LINEAR, BORDER_CONSTANT);
  remap(bot_frame, rect_bot, mapBF1, mapBF2, INTER_LINEAR, BORDER_CONSTANT);
  remap(top_frameB, rect_topB, mapTB1, mapTB2, INTER_LINEAR, BORDER_CONSTANT);
  remap(bot_frameB, rect_botB, mapBB1, mapBB2, INTER_LINEAR, BORDER_CONSTANT);

  Mat rect_tf, rect_bf, rect_bb, rect_tb, dispColorRot, dispColorBRot;
  rect_tf = Mat(rect_top, Rect(Point(20,0),Point(370,640)));
  rect_bf = Mat(rect_bot, Rect(Point(20,0),Point(370,640)));
  rect_tb = Mat(rect_topB, Rect(Point(40,0),Point(390,640)));
  rect_bb = Mat(rect_botB, Rect(Point(40,0),Point(390,640)));

  /***************************************************************************/

  /****************************** SGBM Setup *********************************/
  Ptr<StereoSGBM> sgbm, sgbmB;
  int frame_channels = top_frame.channels();
  SGBM_Param sm;
  bool sgbm_changed = 0;
  /***************************************************************************/

  /****************************** SPS Setup *********************************/
  int height_ = rect_tf.cols, width_ = rect_tf.rows, dims[] = {width_, height_, 3};
  int height_B = rect_tb.cols, width_B = rect_tb.rows;
  Mat segmentImageB( width_B, height_B , CV_8U), disparityImageB( width_B, height_B, CV_8U);
  Mat segmentImage( width_, height_ , CV_8U), disparityImage( width_, height_, CV_8U);
  SPSStereo sps;
  SPSStereo spsB;
  // Cropping the output to remove the data caused by the casing
  //Rect croppingRectangle = Rect(0, 340, 650, 340);
  /***************************************************************************/

  /****************************** Odometry Setup *********************************/
  std::shared_ptr<rgbd::Odometry> odom;
  Mat grayImage0, grayImage1, depthFlt0, depthFlt1;
  bool isFirst = true;
  Mat rigidTransform;
  Mat rotationMatrix, tranlslationMatrix;
  float baseline = norm(frontCameras.tvec);
  const float MILLIMETER_TO_METER_SCALEFACTOR = 0.001f;
  // Visualize trajectory
  const float WINDOW_SIZE = 1100;
  const float VISUALIZATION_SCALE_FACTOR = 45.0f;    // Originally 60.0f
  Mat traj = Mat::zeros(WINDOW_SIZE, WINDOW_SIZE, CV_8UC3);
  if (ODOMETRY){
  namedWindow("RGBD Color", WINDOW_AUTOSIZE);
  namedWindow("Normalized RGBD Depth", WINDOW_AUTOSIZE);
  namedWindow("RGBD Trajectory", WINDOW_AUTOSIZE);
  }
  char text[100];
  int fontFace = FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 1;
  cv::Point textOrg(10, 50);
  int translationSaveMatSize = 1e5;
  if (USE_PSEUDO == 1) { translationSaveMatSize = top_stream.get(CAP_PROP_FRAME_COUNT); }
  Mat accumulatedTranMatrix( translationSaveMatSize, 3, CV_32F);
  //For rotating the images
  Mat imageR, dispMapR;
  /***************************************************************************/

  /*********************** Point Cloud Display Setup *************************/
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr displayPC;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr displayPCB;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr displayOmni;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr displayOmni2;
  pcl::visualization::PCLVisualizer::Ptr viewerO (new pcl::visualization::PCLVisualizer ("Omnidirectional View Point Cloud"));
  viewerO->setBackgroundColor (255, 255, 255);
  int frame_difference = 20;
  int num_pc_displayed = 3;
  /***************************************************************************/


  // SGBM on by default
  if (USE_SGBM){
    /************************ SGBM Parameters *******************************/
    sm.min_disparity = 0;
    sm.max_disparity = 16 * 5;
    sm.SADWindowSize = 3;
    sm.smooth_p1 = 8; // Implicit: * frame_channels * sm.SADWindowSize * sm.SADWindowSize;
    sm.smooth_p2 = 64; // Implicit: * frame_channels * sm.SADWindowSize * sm.SADWindowSize;
    sm.maxLRdisparity = 160;
    sm.preFilterCap = 32;
    sm.thres = 8;
    sm.speckWindowSize = 200;
    sm.speckRange = 2;
    /*************************************************************************/

    sgbmHelp();
  }
  /************************ SPS Parameters *******************************/
  int outerIterationTotalMe = 3;
  int innerIterationTotalMe = 3;
  double lambda_pos_me = 500.0;
  double lambda_depth_me = 2000.0;
  double lambda_bou_me = 1000.0;
  double lambda_smo_me = 1600.0;
  double lambda_d_me = 3.0;
  double lambda_hinge_me = 10.0;
  double lambda_occ_me = 30.0;
  double lambda_pen_me = 60.0;

  sps.setIterationTotal(outerIterationTotal, innerIterationTotal);
  sps.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
  sps.setInlierThreshold(lambda_d);
  sps.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);

  spsB.setIterationTotal(outerIterationTotal, innerIterationTotal);
  spsB.setWeightParameter(lambda_pos, lambda_depth, lambda_bou, lambda_smo);
  spsB.setInlierThreshold(lambda_d);
  spsB.setPenaltyParameter(lambda_hinge, lambda_occ, lambda_pen);
  /*************************************************************************/
  char key = 'q';

  sgbm = StereoSGBM::create(sm.min_disparity, sm.max_disparity,
                            sm.SADWindowSize,
                            sm.smooth_p1*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                            sm.smooth_p2*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                            sm.maxLRdisparity,sm.preFilterCap, sm.thres,
                            sm.speckWindowSize,sm.speckRange);

  sgbmB = StereoSGBM::create(sm.min_disparity, sm.max_disparity,
                            sm.SADWindowSize,
                            sm.smooth_p1*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                            sm.smooth_p2*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                            sm.maxLRdisparity,sm.preFilterCap, sm.thres,
                            sm.speckWindowSize,sm.speckRange);


/*****************************ODOMETRY Parameters**************************************/
  float MIN_DEPTH = 0.5f, MAX_DEPTH = 13.0f, MAX_DEPTH_DIFF = 0.7f, MAX_POINTS_PART = 0.3f;

  vector<int> iterCounts(4);
  iterCounts[0] = 7;
  iterCounts[1] = 7;
  iterCounts[2] = 7;
  iterCounts[3] = 10;

  vector<float> minGradMagnitudes(4);
  minGradMagnitudes[0] = 12;
  minGradMagnitudes[1] = 5;
  minGradMagnitudes[2] = 3;
  minGradMagnitudes[3] = 1;

  odom = std::make_shared<rgbd::RgbdOdometry>(frontCameras.K1, MIN_DEPTH, MAX_DEPTH, MAX_DEPTH_DIFF, iterCounts,
			                                                  minGradMagnitudes, MAX_POINTS_PART,
									  rgbd::Odometry::RIGID_BODY_MOTION);

/************************************************************************************/

  int frame_num = 0;
  string out_dir = "../data/temp/";
  cout << top_stream.get(CAP_PROP_FRAME_COUNT) << endl;


  ofstream fps_data("../data/frame_rate/fps_data.csv");
  ofstream timesync("../data/timesync.csv");

  // Variables to merge 6 videos into 1
  Mat merged_frame(Size(1920,1280),CV_8UC3);
  Mat roi_TF(merged_frame,  Rect(0,0,640,640));
  Mat roi_BF(merged_frame, Rect(0,640,640,640));
  Mat roi_TB(merged_frame, Rect(640,0,640,640));
  Mat roi_BB(merged_frame, Rect(640,640,640,640));
  Mat roi_DF(merged_frame, Rect(1280,145,640,350));
  Mat roi_DB(merged_frame, Rect(1280,785,640,350));

  Mat show_frame;

  string merge_str = "";
  if (MEDIA_OUT)
  {
    merge_str = "";
  } else {
    merge_str = "1";
  }

  VideoWriter merge_write("../data/logging/merged"+merge_str+".avi", VideoWriter::fourcc('H','2','6','4'), 13, Size(1920,1280), true);

  int64 timeSGBMstart, timeSGBMstop;
  Mat pointCloud, pointCloudB, pointCloud_omnidir;


  if (POINT_CLOUD)
  {
    cout << "\nPress '.' to capture Point Cloud of shown frame." << endl;
  }
/****************Coverage Data Saving Setup*************************************/
string cov_data = "../data/logging/coverage_data.csv";
ofstream out_cov;
out_cov.open(cov_data);
/*********************************************************************************/
  while (1)
  {

    if (USE_SGBM && sgbmChange(key)){
      sgbm = StereoSGBM::create(sm.min_disparity, sm.max_disparity,
                                sm.SADWindowSize,
                                sm.smooth_p1*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                                sm.smooth_p2*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                                sm.maxLRdisparity,sm.preFilterCap, sm.thres,
                                sm.speckWindowSize,sm.speckRange);
    sgbmB = StereoSGBM::create(sm.min_disparity, sm.max_disparity,
                              sm.SADWindowSize,
                              sm.smooth_p1*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                              sm.smooth_p2*frame_channels * sm.SADWindowSize * sm.SADWindowSize,
                              sm.maxLRdisparity,sm.preFilterCap, sm.thres,
                              sm.speckWindowSize,sm.speckRange);
    }
    key = 0;

    int64 timeStart = getTickCount();
    int64 timeGrab, timeDecode;


    if (CAMERA_STREAM)
    {
      top_stream.grab();
      timeGrab = getTickCount();
      bot_stream.grab();
      timeDecode = getTickCount();

      //timesync << ((timeDecode - timeGrab)/getTickFrequency()) * 1000 << endl;

      top_stream.retrieve(top_cap);
      bot_stream.retrieve(bot_cap);
    }

    #pragma omp parallel sections
    {
      {if (CAMERA_STREAM)
      {
        transpose(tf_template, top_frame);
        flip(top_frame, top_frame, 1);
        transpose(bf_template, bot_frame);
        flip(bot_frame, bot_frame, 0);
        top_frame.copyTo(roi_TF);
        bot_frame.copyTo(roi_BF);
      }

      remap(top_frame, rect_top, mapTF1, mapTF2, INTER_LINEAR, BORDER_CONSTANT);
      remap(bot_frame, rect_bot, mapBF1, mapBF2, INTER_LINEAR, BORDER_CONSTANT);

int sgbm_size = 0, sps_size = 0, image_size = 0;
float sgbm_coverage = 0, sps_coverage = 0;

      if(USE_SGBM) {
        timeSGBMstart = getTickCount();
		    sgbm -> compute(rect_tf,rect_bf,dispMap);
		    filterSpeckles(dispMap, 0.0, sm.speckWindowSize, sm.max_disparity);


			// Returned by factor of 16 for some reason...
		    Mat dispMapMapping = dispMap / 16.0f;

			std::vector< std::vector<double> > disparityPlaneParameters;
			std::vector< std::vector<int> > boundaryLabels;
			sps.compute(superpixelTotal, rect_tf, width_, height_, dispMapMapping, segmentImage, disparityImage, disparityPlaneParameters, boundaryLabels);
			timeSGBMstop = getTickCount();

			//Computing the Coverage for this frame and saving it
			for(int i = 0; i<rect_tf.rows; i++)
			{
				for(int j = 0; j<rect_tf.cols; j++)
				{
					if (dispMap.at<float>(i,j) == 0) { sgbm_size++ ;}
					if (disparityImage.at<float>(i,j) == 0) { sps_size++ ;}
				}
			}
			image_size = rect_tf.rows * rect_tf.cols;
			sgbm_coverage = 100*sgbm_size/image_size;
			sps_coverage = 100*sps_size/image_size;
                        out_cov << sgbm_coverage << " | " << sps_coverage << endl;

      }


      normalize(disparityImage, dispMap_8U, 0, 255, NORM_MINMAX, CV_8U);

      applyColorMap(dispMap_8U, dispColor, COLORMAP_HSV);
      transpose(dispColor,dispColorRot);
      flip(dispColorRot,dispColorRot,1);
      dispColorRot.copyTo(roi_DF);


    }

      #pragma omp section
      {
        if (CAMERA_STREAM)
        {
          transpose(tb_template, top_frameB);
          flip(top_frameB, top_frameB, 0);
          transpose(bb_template, bot_frameB);
          flip(bot_frameB, bot_frameB, 1);
          top_frameB.copyTo(roi_TB);
          bot_frameB.copyTo(roi_BB);
        }

        remap(top_frameB, rect_topB, mapTB1, mapTB2, INTER_LINEAR, BORDER_CONSTANT);
        remap(bot_frameB, rect_botB, mapBB1, mapBB2, INTER_LINEAR, BORDER_CONSTANT);

        if(USE_SGBM) {
  		    sgbmB -> compute(rect_tb,rect_bb,dispMapB);

  		  filterSpeckles(dispMapB, 0.0, sm.speckWindowSize, sm.max_disparity);

  		  Mat dispMapMappingB = dispMapB / 16.0f;
		  std::vector< std::vector<double> > disparityPlaneParametersB;
		  std::vector< std::vector<int> > boundaryLabelsB;

		  spsB.compute(superpixelTotal, rect_tb, width_B, height_B, dispMapMappingB, segmentImageB, disparityImageB, disparityPlaneParametersB, boundaryLabelsB);

        }


        normalize(disparityImageB,dispMap_8UB, 0, 255, NORM_MINMAX, CV_8U);

        applyColorMap(dispMap_8UB, dispColorB, COLORMAP_HSV);

        transpose(dispColorB,dispColorBRot);
        flip(dispColorBRot,dispColorBRot,1);

        dispColorBRot.copyTo(roi_DB);

      }

    }

    // Rotating the inputs for Odometry as they are rotated 90 degrees anti-clockwise
	rotate(rect_tf, imageR, ROTATE_90_CLOCKWISE);
	rotate(dispMap, dispMapR, ROTATE_90_CLOCKWISE);

    if (frame_num == 0)
    {
    	cvtColor(imageR, grayImage0, COLOR_BGR2GRAY);
    	Mat depth0 = fs * baseline / dispMapR;
    	depth0.convertTo(depthFlt0, CV_32FC1, MILLIMETER_TO_METER_SCALEFACTOR);
    }
    if(ODOMETRY && frame_num > 0)
    {
    	Mat depth1 = fs * baseline / dispMapR;
    	depth1.convertTo(depthFlt1, CV_32FC1, MILLIMETER_TO_METER_SCALEFACTOR);
    	cvtColor(imageR, grayImage1, COLOR_BGR2GRAY);
		bool isSuccess = odom->compute(grayImage0, depthFlt0, Mat(), grayImage1,
									   depthFlt1, Mat(), rigidTransform);

		Mat rotationMat = rigidTransform(cv::Rect(0, 0, 3, 3)).clone();
		Mat translateMat = rigidTransform(cv::Rect(3, 0, 1, 3)).clone();

		// If computed successfully, then update rotationMatrix and tranlslationMatrix
		if (isSuccess == true) {
		  if (isFirst == true) {
			rotationMatrix = rotationMat.clone();
			tranlslationMatrix = translateMat.clone();
			isFirst = false;
			continue;
		  }

		  // Update Rt
		  tranlslationMatrix = tranlslationMatrix + (rotationMatrix * translateMat);
		  rotationMatrix = rotationMat * rotationMatrix;
		}
		grayImage0 = grayImage1.clone();
		depthFlt0 = depthFlt1.clone();

	    if (isFirst == false) {
	      int x =
	          int(VISUALIZATION_SCALE_FACTOR * (tranlslationMatrix.at<double>(0))) +
	          WINDOW_SIZE / 2;
	      int y =
	          int(VISUALIZATION_SCALE_FACTOR * (-tranlslationMatrix.at<double>(2))) +
	          WINDOW_SIZE / 2;


	      circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);
	      rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0),
	                cv::FILLED);
	      if (isSuccess == true) {
	        sprintf(text, "Coordinates: x = %04fm y = %04fm z = %04fm",
	                25*tranlslationMatrix.at<double>(0),
	                25*tranlslationMatrix.at<double>(2),
	                25*tranlslationMatrix.at<double>(1));
	        		accumulatedTranMatrix.at<float>(frame_num, 0) = -25*tranlslationMatrix.at<double>(0);
	        		accumulatedTranMatrix.at<float>(frame_num, 1) = -25*tranlslationMatrix.at<double>(2);
	        		accumulatedTranMatrix.at<float>(frame_num, 2) = -25*tranlslationMatrix.at<double>(1);
	        // These values are approximately divided by a factor of 25
	      } else {
	        sprintf(text, "Fail to compute odometry");
	      }

	      putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255),
	              thickness, 8);
	    }
	    imshow("RGBD Trajectory", traj);
	    imshow("RGBD Color", imageR);

	    cv::Mat normalizeDepth;
	    depthFlt1.convertTo(normalizeDepth, CV_8UC1, 255.0 / MAX_DEPTH);
	    imshow("Normalized RGBD Depth", normalizeDepth);

	    const Mat distCoeff(1, 5, CV_32FC1, Scalar(0));

	    //Code for saving Odometry Data in A CSV file 

	    if(USE_PSEUDO){
		if (frame_num == top_stream.get(CAP_PROP_FRAME_COUNT) - 2)
		{
	    	  string odo_data = "../data/logging/odometry_data.csv";
	    	  ofstream out_odo;
	    	  out_odo.open(odo_data);
	    	  out_odo << "Odometry Data Logging:\n Format \n Double x; \n Double y; \n Double z; \n\n" << accumulatedTranMatrix << endl;
	    	  out_odo.close();
		}
    	    }

    }

    //Making the point cloud concatenation with Odometry Data
    if (POINT_CLOUD && frame_num%frame_difference == 0 && ODOMETRY)
    {

		displayPC = generatePointCloud(rect_top, dispMap_8U, frontCameras, fs, pointCloud, "color");
		displayPCB = generatePointCloud(rect_topB, dispMap_8UB, backCameras, fs, pointCloudB, "color");
		displayOmni = point_cloud_merging(displayPC, displayPCB, pointCloud, pointCloudB, frame_num, frame_difference, accumulatedTranMatrix);

		rgbVis (displayOmni, viewerO, frame_num);

		if (frame_num >= frame_difference*num_pc_displayed)
		{
	      const char* id = "Frame number: ";
		  string remove_point_cloud_id = id + to_string(frame_num-(frame_difference*num_pc_displayed));
		  viewerO->removePointCloud (remove_point_cloud_id);
		}

    }
    //Making the point cloud concatenation without Odometry Data
    else if(POINT_CLOUD && frame_num%frame_difference == 0)
    {
	  displayPC = generatePointCloud(rect_top, dispMap_8U, frontCameras, fs, pointCloud, "color");
	  displayPCB = generatePointCloud(rect_topB, dispMap_8UB, backCameras, fs, pointCloudB, "color");
	  displayOmni = point_cloud_merging(displayPC, displayPCB, pointCloud, pointCloudB, frame_num, frame_difference, accumulatedTranMatrix);
	  rgbVis (displayOmni, viewerO, frame_num);
    }
	viewerO->spinOnce (50);
    if(MEDIA_OUT)
    {
      merge_write.write(merged_frame);
    }

    int64 timeEnd = getTickCount();
    if (LOG_FPS)
    {
      fps_data << (timeEnd-timeStart)/getTickFrequency() * 1000 <<","<<
            (timeSGBMstop-timeSGBMstart)/getTickFrequency() * 1000 << endl;
    }

    resize(merged_frame,show_frame,Size(), 0.8, 0.8);
    imshow("Output", show_frame);

    key = waitKey(5);
    frame_num++;

    if (POINT_CLOUD && key == '.')
    {
    	displayPC = generatePointCloud(rect_top, dispMap_8U, frontCameras,
                        fs, pointCloud, "color");
      // TODO: Add in back cameras as well
      string fname = "../data/point_cloud/" + int2str(frame_num) + ".ply";
      savePointCloud(fname, pointCloud);
      cout << "Point Cloud of Frame " << int2str(frame_num) << " Saved to: "
           << fname << endl;
    }

    else if (key == 27 || top_cap.empty())
    {
      fps_data.close();
      cout << "Total Time: " << ((timeEnd - timeStart)/getTickFrequency()) * 1000 << "ms" << endl;
      timesync.close();
      if(merge_str == "1")
      {
        remove("../data/logging/merged1.avi");
      }
      break;
    }
    // Dynamic reconfiguration or sgbm parameters
    else if (sgbmChange(key) && USE_SGBM)
    {
      switch(key) {
        case 'q': {cout << "Outer Iteration Total = " << ++outerIterationTotalMe << endl;
        	sps.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
        	spsB.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
        	break;
        }
        case 'a': {cout << "Outer Iteration Total = " << --outerIterationTotalMe << endl;
        	sps.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
        	spsB.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
        	break;
        }
        case 'w': {cout << "Inner Iteration Total = " << ++innerIterationTotalMe << endl;
    		sps.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
    		spsB.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
    		break;
        }
        case 's': {cout << "Inner Iteration Total = " << --innerIterationTotalMe << endl;
			sps.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
			spsB.setIterationTotal(outerIterationTotalMe, innerIterationTotalMe);
			break;
		}

        case 'e': {lambda_pos_me += 25.0;
        	cout << "Position Weight = " << lambda_pos_me << endl;
        	sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
        	spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'd': {lambda_pos_me -= 25.0;
			cout << "Position Weight = " << lambda_pos_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'r': {lambda_depth_me += 25.0;
			cout << "Depth Weight = " << lambda_depth_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'f': {lambda_depth_me -= 25;
			cout << "Depth Weight = " << lambda_depth_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 't': {lambda_bou_me += 25.0;
			cout << "Boudary Length Weight = " << lambda_bou_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'g': {lambda_bou_me -= 25.0;
			cout << "Boudary Length Weight = " << lambda_bou_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'y': {lambda_smo_me += 25.0;
			cout << "Smoothness Weight = " << lambda_smo_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'h': {lambda_smo_me += 25.0;
			cout << "Smoothness Weight = " << lambda_smo_me << endl;
			sps.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			spsB.setWeightParameter(lambda_pos_me, lambda_depth_me, lambda_bou_me, lambda_smo_me);
			break;
		}
        case 'u': {lambda_d_me += 0.5;
			cout << "Inlier Threshold = " << lambda_d_me << endl;
			sps.setInlierThreshold(lambda_d_me);
			spsB.setInlierThreshold(lambda_d_me);
			break;
		}
        case 'j': {lambda_d_me -= 0.5;
			cout << "Inlier Threshold = " << lambda_d_me << endl;
			sps.setInlierThreshold(lambda_d_me);
			spsB.setInlierThreshold(lambda_d_me);
			break;
		}
        case 'i': {lambda_hinge_me += 0.5;
			cout << "Hinge Penalty Value = " << lambda_hinge_me << endl;
			sps.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			spsB.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			break;
		}
        case 'k': {lambda_hinge_me -= 0.5;
			cout << "Hinge Penalty Value = " << lambda_hinge_me << endl;
			sps.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			spsB.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			break;
		}
        case 'o': {lambda_occ_me += 1.0;
			cout << "Occlusion Penalty Value = " << lambda_occ_me << endl;
			sps.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			spsB.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			break;
		}
        case 'l': {lambda_occ_me -= 1.0;
			cout << "Occlusion Penalty Value = " << lambda_occ_me << endl;
			sps.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			spsB.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			break;
        }
        case 'p': {lambda_pen_me += 1.0;
			cout << "Impossible Penalty Value = " << lambda_pen_me << endl;
			sps.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			spsB.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			break;
		}
        case ';': {lambda_pen_me -= 1.0;
			cout << "Impossible Penalty Value = " << lambda_pen_me << endl;
			sps.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			spsB.setPenaltyParameter(lambda_hinge_me, lambda_occ_me, lambda_pen_me);
			break;
		}
        case 'z': {
          cout << "\n\n\nOuter Iteration Total = " << outerIterationTotalMe << endl;
          cout << "Inner Iteration Total = " << innerIterationTotalMe << endl;
          cout << "Position Weight = " << lambda_pos_me << endl;
          cout << "Depth Weight = " << lambda_depth_me << endl;
          cout << "Boundary Length Weight = " << lambda_bou_me << endl;
          cout << "Smoothness Weight = " << lambda_smo_me << endl;
          cout << "Inlier Threshold = " << lambda_d_me << endl;
          cout << "Hinge Penalty Value = " << lambda_hinge_me << endl;
          cout << "Occulsion Penalty Value = " << lambda_occ_me << endl;
          cout << "Impossible Penalty Value = " << lambda_pen_me << endl;
          break;
        }
      }
    }
  }
  return 0;
}
