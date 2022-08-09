
#include <hls_stream.h>
#include "ap_axi_sdata.h"
#include <stdint.h>
#include "ap_int.h"




typedef unsigned char PIXEL;
typedef ap_axiu<8, 0, 0, 0> trans_pkt;

typedef int DTYPE;
typedef unsigned char PIXEL;


void filter2d_accel(DTYPE* img_in, DTYPE* kernel, DTYPE* img_out, int rows, int cols);

