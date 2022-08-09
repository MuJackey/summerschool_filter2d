#include <iostream>
using namespace std;
#include <math.h>
#include "filter2d.h"
DTYPE c[128][128];
DTYPE core[3][3];
static DTYPE my3x3_kernel(DTYPE WB[3][3],DTYPE core[3][3])
{
	int i,j,k;
	DTYPE out_pix;
	out_pix=0;
	k=0;
	for(i=0;i<3;i++)
	{
		for(j=0;j<3;j++)
		{
			out_pix=out_pix+core[i][j]*WB[i][j];
			k++;
		}
	}
	if(out_pix < 0)
		{
			out_pix = 0;
		}
		else if(out_pix > 255)
		{
			out_pix = 255;
		}
	return (DTYPE) out_pix;
}

void filter2d_accel(DTYPE *img_in, DTYPE *kernel, DTYPE *img_out, int rows, int cols)
{

#pragma HLS INTERFACE m_axi port=img_in  offset=slave depth=16384
#pragma HLS INTERFACE m_axi port=img_out offset=slave depth=15876
#pragma HLS INTERFACE m_axi port=kernel  offset=slave depth=9
#pragma HLS INTERFACE s_axilite port=rows  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=cols  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL


	DTYPE _sobel;

	DTYPE LineBuffer[3][128];
#pragma HLS ARRAY_PARTITION variable=LineBuffer complete dim=1

	DTYPE WindowBuffer[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
#pragma HLS ARRAY_PARTITION variable=WindowBuffer complete dim=0

	DTYPE row, col;
	DTYPE lb_r_i;
	DTYPE top, mid, btm;//line buffer row index
	int m=0;
	for (int i =0; i < rows; i++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS pipeline
		for(int j=0; j<cols; j++)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS pipeline
		  c[i][j] = img_in[m++];
		}
	}
	int n=0;
	for (int i =0; i < 3; i++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
#pragma HLS pipeline
		for(int j=0; j<3; j++)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
#pragma HLS pipeline
		  core[i][j] = kernel[n++];
		}
	}

	for(col = 0; col < cols; col++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS pipeline

		LineBuffer[0][col] = c[0][col];
		LineBuffer[1][col] = c[1][col];
	}

	lb_r_i = 2;
	for(row = 2; row < rows; row++)
	{
#pragma HLS LOOP_TRIPCOUNT min=1 max=126
		if(lb_r_i == 2)
		{
			top = 0; mid = 1; btm = 2;
		}
		else if(lb_r_i == 0)
		{
			top = 1; mid = 2; btm = 0;
		}
		else if(lb_r_i == 1)
		{
			top = 2; mid = 0; btm = 1;
		}

		WindowBuffer[top][0] = c[row+top-2][0];
		WindowBuffer[mid][0] = c[row+mid-2][0];
		WindowBuffer[btm][0] = c[row+btm-2][0];
		WindowBuffer[top][1] = c[row+top-2][1];
		WindowBuffer[mid][1] = c[row+mid-2][1];
		WindowBuffer[btm][1] = c[row+btm-2][1];
		for(col = 2; col < cols; col++)
		{
#pragma HLS LOOP_TRIPCOUNT min=1 max=126
#pragma HLS pipeline
			if(row < rows)
			{
				LineBuffer[btm][col] =  c[row][col];
			}
			else
				LineBuffer[btm][col] = 0;

			WindowBuffer[0][2] = LineBuffer[top][col];
			WindowBuffer[1][2] = LineBuffer[mid][col];
			WindowBuffer[2][2] = LineBuffer[btm][col];
			_sobel = my3x3_kernel(WindowBuffer, core);
			WindowBuffer[0][0] = WindowBuffer[0][1];
			WindowBuffer[1][0] = WindowBuffer[1][1];
			WindowBuffer[2][0] = WindowBuffer[2][1];
			WindowBuffer[0][1] = WindowBuffer[0][2];
			WindowBuffer[1][1] = WindowBuffer[1][2];
			WindowBuffer[2][1] = WindowBuffer[2][2];


			*img_out++ = _sobel;
		}
		lb_r_i++;
		if(lb_r_i == 3) lb_r_i = 0;
	}
}
