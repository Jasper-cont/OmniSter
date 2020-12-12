#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

/* c - capture images
   f - flip to back camera
   esc - exit program
*/

string int2str(int a) {
  ostringstream temp;
  temp << a;
  return temp.str();
}

int calib_img_gen(string calibImgDir,
                  string imgTF, string imgTB, string imgBF, string imgBB,
                  string imgFormat)
{
    Mat top_frame;
    Mat bot_frame;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture top_cap;
    VideoCapture bot_cap;

    top_cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('H', '2', '6', '4'));
    bot_cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('H', '2', '6', '4'));

    // open the default camera using default API
    top_cap.open(4);
    bot_cap.open(2);

    // check if we succeeded
    if (!top_cap.isOpened() || !bot_cap.isOpened()) {
        cerr << "ERROR! Unable to open camera(s)\n";
        return -1;
    }

    char flipKey = 0;
    int baseLine = 0;
    const Scalar GREEN(0,255,0);

    while (!(flipKey == 'n') && !(flipKey == 'y')) {
      top_cap.read(top_frame);
      bot_cap.read(bot_frame);

      string msg = "Flip Cameras? (y/n)";
      Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
      Point textOrigin(top_frame.cols - 2*textSize.width - 10, top_frame.rows - 2*baseLine - 10);
      putText( top_frame, msg, textOrigin, 1, 1, GREEN);
      putText(bot_frame, msg, textOrigin, 1, 1, GREEN);

      imshow("Top", top_frame);
      imshow("Bottom", bot_frame);
      flipKey = (char)waitKey(5);
    }

    if (flipKey == 'y') {
      top_cap.release();
      bot_cap.release();
      top_cap.open(2);
      bot_cap.open(4);
    }

    destroyWindow("Top");
    destroyWindow("Bottom");

    // Splits the raw Image into the two halves
    Mat bot_front (bot_frame, Rect(Point(0, 0), Point((bot_frame.cols / 2), 640)));
    Mat bot_back (bot_frame, Rect(Point((bot_frame.cols / 2), 0), Point(bot_frame.cols, 640)));
    Mat top_front (top_frame, Rect(Point(0, 0), Point(top_frame.cols / 2, 640)));
    Mat top_back (top_frame, Rect(Point((top_frame.cols / 2), 0), Point(top_frame.cols, 640)));

    Mat top_back_r;
    Mat top_front_r;
    Mat bot_back_r;
    Mat bot_front_r;

    int frame_num = 0;
    int frame_dir = 0; // 0 - front, 1 - back

    cout << "Press 'c' on the image window to capture (default on front side).";
    cout << "\nPress 'f' to flip active camera capture side.";
    cout << "\nPress 'ESC' to exit.\n\n";

    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        top_cap.grab();
        bot_cap.grab();
        top_cap.retrieve(top_frame);
        bot_cap.retrieve(bot_frame);
        // check if we succeeded
        if (top_frame.empty()) {
            cerr << "ERROR! blank top_frame grabbed\n";
            break;
        }
        if (bot_frame.empty()) {
          cerr << "ERROR! blank bot_frame grabbed\n";
          break;
        }
        // Rotate CW 90 degrees
        transpose(top_front, top_front_r);
        flip(top_front_r, top_front_r,1);
        transpose(bot_back, bot_back_r);
        flip(bot_back_r, bot_back_r,1);

        //Rotate CCW 90 degrees
        transpose(top_back, top_back_r);
        flip(top_back_r, top_back_r,0);
        transpose(bot_front, bot_front_r);
        flip(bot_front_r, bot_front_r,0);

        // show live and wait for a key with timeout long enough to show images
        imshow("Top Front", top_front_r);
        imshow("Top Back", top_back_r);
        imshow("Bottom Front", bot_front_r);
        imshow("Bottom Back", bot_back_r);

        // save images to folder

        string top_name, bot_name;

        char key = (char)waitKey(5);
        if (key == 'c') {
          if (!frame_dir) {
            top_name = calibImgDir + imgTF + int2str(frame_num) + imgFormat;
            bot_name = calibImgDir + imgBF + int2str(frame_num) + imgFormat;
            imwrite(top_name, top_front_r);
            imwrite(bot_name, bot_front_r);
            cout << top_name << "," << bot_name << " Saved." << endl;
            frame_num++;
          } else {
            top_name = calibImgDir + imgTB + int2str(frame_num) + imgFormat;
            bot_name = calibImgDir + imgBB + int2str(frame_num) + imgFormat;
            imwrite(top_name, top_back_r);
            imwrite(bot_name, bot_back_r);
            cout << top_name << " " << bot_name << endl;
            frame_num++;
          }
        }

        if (key == 'f') {
          frame_num = 0;
          frame_dir = !frame_dir;

          if (!frame_dir) {
            cout << "Capturing Front Images" << endl;
          } else {
            cout << "Capturing Back Images" << endl;
          }
        }

        if (key == 27) {
            break;
          }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
