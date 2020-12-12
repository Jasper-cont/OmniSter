#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include <vector>

using namespace cv;
using namespace std;

void calcChessboardCorners(Size boardSize, double square_width, double square_height,
    Mat& corners)
{
    // corners has type of CV_64FC3
    corners.release();

    // n is the total number of corners in one image
    int n = boardSize.width * boardSize.height;
    corners.create(n, 1, CV_64FC3);

    // ptr used to access elements of type Mat
    Vec3d *ptr = corners.ptr<Vec3d>();

    for (int i = 0; i < boardSize.height; ++i)
    {
        for (int j = 0; j < boardSize.width; ++j)
        {
            ptr[i*boardSize.width + j] =
              Vec3d(double(j * square_width), double(i * square_height), 0.0);
        }
    }
}
