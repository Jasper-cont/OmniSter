#ifndef SGBM_PARAM_H
#define SGBM_PARAM_H

using namespace std;

class SGBM_Param
{
public:
  int min_disparity;
  int max_disparity;
  int SADWindowSize;
  int smooth_p1;
  int smooth_p2;
  int maxLRdisparity;
  int preFilterCap;
  int thres;
  int speckWindowSize;
  int speckRange;
};

#endif
