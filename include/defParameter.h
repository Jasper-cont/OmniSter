// Parameters for KITTI dataset

// The number of superpixel
int superpixelTotal = 1000;

// The number of iterations
int outerIterationTotal = 3;
int innerIterationTotal = 3;

// Weight parameters
double lambda_pos = 500.0;
double lambda_depth = 2000.0;
double lambda_bou = 1000.0;
double lambda_smo = 1600.0;

// Inlier threshold
double lambda_d = 3.0;

// Penalty values
double lambda_hinge = 10.0;
double lambda_occ = 30.0;
double lambda_pen = 60.0;
