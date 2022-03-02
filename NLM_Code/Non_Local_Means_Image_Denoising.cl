#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> 
#endif

//////////////////////////////////////////////////////////////////////////////
// kernel implementation of NonLocal Means Algorithm
//////////////////////////////////////////////////////////////////////////////

// Global variables
#define sdev 10
#define h 0.3
#define smallwindow   3
#define searchwindow     ( (2 * smallwindow + 1) * (2 * smallwindow + 1) )
#define weightmax  0.10f


inline float CalcDist(float3 a, float3 b)
{
	return (
		(b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z)
		);
}

inline float getIndex(float a, float b, float c)
{
	return a + (b - a) * c;
}

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/////////////////Kernel for greyscale image denoising/////////////////////////////
__kernel void NonLocalMeansFilter_Greyscale(__global float* input_image, __global float* input_Temp1, __global float* input_Temp2, __global int* width, __global int* height)
{
	int image_width;
	image_width = *width;

	const int xdir = get_global_id(0);
	const int ydir = get_global_id(1);
	const int local_x = get_local_id(0);

	//Search window loop to find the difference between two pixels over the image
	for (int local_y = 0; local_y < image_width; local_y++)
	{
		float dist = 0;
		float Finaldist = 0;
		for (int m = local_x; m < local_x + smallwindow; m++)
		{
			for (int n = local_y; n < local_y + smallwindow; n++)
			{

				dist = (input_image[(xdir + m - local_x) * image_width + ydir + n - local_y] - input_image[m * image_width + n]);
				Finaldist += (dist * dist);
			}
		}

		// Computing weight for similarities between pixels
		float weight = exp(-Finaldist / (h * h));


		// Multiplying the weights with original input pixels
		input_Temp1[xdir * image_width + ydir] += weight * input_image[local_x * image_width + local_y];
		input_Temp2[xdir * image_width + ydir] += weight;
	}
}


/////////////////Kernel for color image denoisng/////////////////////////////
__kernel void NonLocalMeansFilter_Color(read_only image2d_t d_input,
	write_only image2d_t d_output,
	__global int* width,
	__global int* height,
	__global float* filt,
	__global float* sigma)
{

	const int xdir = get_global_id(0);
	const int ydir = get_global_id(1);
	const int lx_size = get_local_size(0);
	const int ly_size = get_local_size(1);

	const int t_x = lx_size * xdir + smallwindow;
	const int t_y = ly_size * ydir + smallwindow;

	//Search window loop to find the difference between two pixels over the image
	if (xdir < *width && ydir < *height) {
		float weight = 0;

		for (int n = -smallwindow; n <= smallwindow; n++) {
			for (int m = -smallwindow; m <= -smallwindow; m++) {
				weight += CalcDist(
					(float3)read_imagef(d_input, sampler, (int2)(t_x + m, t_y + n)).xyz,
					(float3)read_imagef(d_input, sampler, (int2)(xdir + m, ydir + n)).xyz

				);
			}
		}

		// Computing weight for similarities between pixels
		weight = exp(-weight * (*sigma) / (sdev * sdev));


		float wmax = 0;
		float sumWeights = 0;
		float3 color = { 0,0,0 };

		//Multiplying the weights with original input pixels on each channel
		for (int i = -smallwindow; i <= smallwindow + 1; i++) {
			for (int j = -smallwindow; j <= smallwindow + 1; j++) {
				float fweight = weight;

				float3 icolor = read_imagef(d_input, sampler, (int2)(xdir + j, ydir + i)).xyz;
				color.x += icolor.x * fweight;
				color.y += icolor.y * fweight;
				color.z += icolor.z * fweight;

				sumWeights += fweight;

				wmax += (fweight > weightmax) ? fweight : 0;
			}
		}

		// Dividing each pixel by the weight sum
		color.x = color.x / sumWeights;
		color.y = color.y / sumWeights;
		color.z = color.z / sumWeights;

		float average = (wmax > 0) ? (*filt) : 1.0f - *filt;

		//Writing output image
		float3 ocolor = read_imagef(d_input, sampler, (int2)(xdir, ydir)).xyz;
		color.x = getIndex(color.x, ocolor.x, average);
		color.y = getIndex(color.y, ocolor.y, average);
		color.z = getIndex(color.z, ocolor.z, average);
		write_imagef(d_output, (int2)(xdir, ydir), (float4)(color.x, color.y, color.z, 1.0f));
	}

}
