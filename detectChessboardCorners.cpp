#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <time.h>

using namespace cv;
using namespace std;

bool detectChessboardCorners(const vector<string>& list1,
  vector<string>& list_detected_1, const vector<string>& list2,
  vector<string>& list_detected_2, vector<Mat>& image_points_1,
  vector<Mat>& image_points_2, Size boardSize, Size& imageSize1,
  Size& imageSize2, bool show_img)
{
  image_points_1.resize(0);
  image_points_2.resize(0);
  list_detected_1.resize(0);
  list_detected_2.resize(0);

  int n_img = (int)list1.size();

  Mat img_top, img_bot;

  for(int i = 0; i < n_img; ++i)
  {
    // Image Points (2D) for Individual Top and Bottom Image
    Mat points_top, points_bot;

    img_top = imread(list1[i], IMREAD_GRAYSCALE);
    img_bot = imread(list2[i], IMREAD_GRAYSCALE);
    //cout << list1[i] << "\t" << list2[i] << endl;

    bool found_top = findChessboardCorners(img_top, boardSize, points_top);
    bool found_bot = findChessboardCorners(img_bot, boardSize, points_bot);

    if (found_top && found_bot)
    {
      char key;
      if (show_img)
      {
        drawChessboardCorners(img_top, boardSize, points_top, found_top);
        drawChessboardCorners(img_bot, boardSize, points_bot, found_bot);
        imshow("Top", img_top);
        imshow("Bottom", img_bot);
        key = waitKey(0);
      } else {
        key = 'y';
      }

      if (key == 'y')
      {
        // corners have type CV_64FC2 <- i.e. 2D
        if (points_top.type() != CV_64FC2)
          points_top.convertTo(points_top, CV_64FC2);
        if (points_bot.type() != CV_64FC2)
          points_bot.convertTo(points_bot, CV_64FC2);

        image_points_1.push_back(points_top);
        image_points_2.push_back(points_bot);
        list_detected_1.push_back(list1[i]);
        list_detected_2.push_back(list2[i]);

        /*cout << list1[i] << endl;
        cout << points_top << endl << endl;
        cout << list2[i] << endl;
        cout << points_bot << endl << endl;*/
      }
    }

  }

  if (!img_top.empty())
    imageSize1 = img_top.size();
	if (!img_bot.empty())
		imageSize2 = img_bot.size();

  /*destroyWindow("Top");
  destroyWindow("Bottom");*/

  if (image_points_1.size() < 3)
    return false;
  else
    return true;
}
