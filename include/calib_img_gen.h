#ifndef CALIB_IMG_GEN_H
#define CALIB_IMG_GEN_H

#include <string>

using namespace std;

string int2str(int a);

int calib_img_gen(string calibImgDir,
                  string imgTF, string imgTB, string imgBF, string imgBB,
                  string imgFormat);

#endif
