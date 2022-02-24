//////////////////////////////////////////////////////////////////////////////
//			Non local means for image denoising								//
//////////////////////////////////////////////////////////////////////////////

// includes
#include <time.h>
#include <stdio.h>
#include <fstream>
#include <istream>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <CL/opencl.h>
#include<Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/opencv.hpp>
//#ifdef _APPLE_
//#include <OpenCL/cl.h>
//#else
//#include <CL\cl.h>
////#include <CL\cl.hpp>
//#endif
#include <opencv2/photo.hpp>
#pragma warning( disable : 4996 )

#define GreyDenoise
//#define RGBDenoise

//global declarations
using namespace std;
static float sigma = 0.15f;
#define string_length (0x100000)

//////////////////////////////////////////////////////////////////////////////
// CPU implementation of Non-local means Algorithm
/////////////////////////////////////////////////////////////////////////////
#ifdef GreyDenoise
float Grey_CPU_time;
void NLM_Grey_CPU(string fileName)
{
	cout << "\n\nExecuting Greyscale Host implementation " << endl;
	clock_t host_t1, host_t2;
	host_t1 = clock();

	//Reading input greyscale image
	cv::Mat raw_image = cv::imread(fileName);
	cv::imshow("CPU- Grey Raw Image", raw_image);

	//Calling OpenCV inbuilt NLM filtering function  and displaying the output image
	cv::Mat denoised_image;
	cv::fastNlMeansDenoising(raw_image, denoised_image, 10, 3, 7);
	cv::imshow("CPU- Grey NLMDenosing", denoised_image);

	host_t2 = clock();

	//CPU profiling
	Grey_CPU_time = host_t2 - host_t1;
	cout << "Time required for CPU operation for grey image: " << Grey_CPU_time << "ms" << endl;
	cout << "Host Greyscale implementation successful" << endl;
}
#endif

#ifdef RGBDenoise
float RGB_CPU_time;
void NLM_RGB_CPU(string fileName)
{
	cout << "\n\nExecuting RGB Host implementation " << endl;
	clock_t host_t1, host_t2;
	host_t1 = clock();

	//Reading input RGB image
	cv::Mat raw_image = cv::imread(fileName);
	cv::imshow("CPU- RGB Raw Image", raw_image);

	//Calling OpenCV inbuilt NLM filtering function  and displaying the output image
	cv::Mat denoised_image;
	cv::fastNlMeansDenoisingColored(raw_image, denoised_image, 10, 7, 7, 21);
	cv::imshow("CPU-RGB NLMDenosing", denoised_image);

	host_t2 = clock();
	//CPU profiling
	RGB_CPU_time = host_t2 - host_t1;
	cout << "Time required for CPU operation for color image: " << RGB_CPU_time << "ms" << endl;
	cout << "Host color implementation successful" << endl;

}

/*
//Function for reading the color image from file and creating OpenCL image object
*/
cl_mem GetRGBImage(cl_context context, std::string fileName, int& width, int& height)
{
	cv::Mat rd_image = cv::imread(fileName);
	width = rd_image.cols;
	height = rd_image.rows;
	char* buffer = new char[width * height * 4];

	int z = 0;
	for (int x = height - 1; x >= 0; x--)
	{
		for (int y = 0; y < width; y++)
		{
			buffer[z++] = rd_image.at<cv::Vec3b>(x, y)[0];
			buffer[z++] = rd_image.at<cv::Vec3b>(x, y)[1];
			buffer[z++] = rd_image.at<cv::Vec3b>(x, y)[2];
			z++;
		}
	}

	// Create OpenCL image  
	cl_image_format clImageFormat;
	clImageFormat.image_channel_order = CL_RGBA;
	clImageFormat.image_channel_data_type = CL_UNORM_INT8;

	cl_int errNum;
	cl_mem cl_Image;
	cl_Image = clCreateImage2D(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		&clImageFormat,
		width,
		height,
		0,
		buffer,
		&errNum);

	if (errNum != CL_SUCCESS)
	{
		std::cerr << "Error creating CL color image object" << std::endl;
		return 0;
	}

	return cl_Image;
}

/*
//Function for creating OpenCL output color image object
*/
cl_mem GetDestinationCLImage(cl_context context, int width, int height)
{
	cl_int errNum;
	cl_mem	wr_image = 0;
	cl_image_format clImageFormat;
	clImageFormat.image_channel_order = CL_RGBA;
	clImageFormat.image_channel_data_type = CL_UNORM_INT8;
	wr_image = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &clImageFormat, width, height, 0, NULL, &errNum);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error creating CL output color image object" << endl;
		return NULL;
	}
	return wr_image;
}

#endif


//////////////////////////////////////////////////////////////////////////////
// Main function
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	//Predefining variables
	cl_int errNum;
	cl_uint platforms;
	cl_platform_id platformId;
	cl_context context = NULL;
	cl_command_queue commandQueue = 0;
	cl_program program = 0;
	cl_device_id device = 0;
	cl_kernel kernel = 0;
	cl_sampler sampler = 0;

	//kernel file
	const char* cl_kernel_file = "C:/Users/gsupr/Downloads/Opencl-Basics-Windows (1)/Opencl-Basics-Windows/Opencl-ex1/src/Non_Local_Means_Image_Denoising.cl";

	//Get platform details
	errNum = clGetPlatformIDs(1, &platformId, &platforms);
	if (errNum != CL_SUCCESS || platforms <= 0)
	{
		std::cerr << "No platforms found" << std::endl;
		return NULL;
	}

	// Create context
	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);

	// Get a device of the context
	cl_device_id* devices;
	size_t deviceNr = -1;
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceNr);
	if (deviceNr <= 0)
	{
		std::cerr << "No devices are available";
		return NULL;
	}
	devices = new cl_device_id[deviceNr / sizeof(cl_device_id)];
	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceNr, devices, NULL);
	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
	device = devices[0];
	delete[] devices;

	//Check if the image is supported
	cl_bool ifsupport = CL_FALSE;
	clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &ifsupport, NULL);
	if (ifsupport != CL_TRUE)
		cerr << "OpenCL device does not support the image" << endl;


/////////////////////////////////Greyscale Implementation ////////////////////////////////
#ifdef  GreyDenoise

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 30;
	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof(float); // Size of data in bytes

	// Allocate space for output data from CPU and GPU on the host
	std::vector<float> h_input(count);
	std::vector<float> h_outputCpu(count);
	std::vector<float> h_outputGpu(count);

	// Initialize memory to 0xff (useful for debugging because otherwise GPU memory will contain information from last execution)
	memset(h_input.data(), 255, size);
	memset(h_outputGpu.data(), 255, size);
	memset(h_outputCpu.data(), 255, size);

	// Allocate space for input and output data on the device
	std::size_t image_width = 640, image_Height = 480;
	std::vector<float> inputData;
	int patchWidth = 3;

	cl_mem g_image = clCreateBuffer(context, CL_MEM_READ_WRITE, image_width * image_Height * sizeof(float), NULL, &errNum);
	cl_mem g_imageTemp1 = clCreateBuffer(context, CL_MEM_READ_WRITE, image_width * image_Height * sizeof(float), NULL, &errNum);
	cl_mem g_imageTemp2 = clCreateBuffer(context, CL_MEM_READ_WRITE, image_width * image_Height * sizeof(float), NULL, &errNum);

	// GPU Write Buffer
	errNum = clEnqueueWriteBuffer(commandQueue, g_imageTemp1, CL_TRUE, 0, image_width * image_Height * sizeof(float), h_outputGpu.data(), 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, g_imageTemp2, CL_TRUE, 0, image_width * image_Height * sizeof(float), h_outputGpu.data(), 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error writing buffer" << endl;
		return 1;
	}

	// Load input data 
	string src = "C:/Users/gsupr/Downloads/Opencl-Basics-Windows (1)/Opencl-Basics-Windows/Opencl-ex1/src/Greyscale_raw_noisy_image.pgm";
	Core::readImagePGM(src, inputData, image_width, image_Height);
	for (size_t j = 0; j < image_Height; j++) {
		for (size_t i = 0; i < image_width; i++) {
			h_input[i + j * image_width] = inputData[(i % image_width) + (j % image_Height) * image_width];
		}
	}

	// Copy input data to device
	errNum = clEnqueueWriteBuffer(commandQueue, g_image, CL_TRUE, 0, image_width * image_Height * sizeof(float), h_input.data(), 0, NULL, NULL);

	// Reinitialize output memory to 0xff
	memset(h_outputGpu.data(), 1.2, image_width * image_Height * sizeof(float));

	// Copy input data to device
	errNum = clEnqueueWriteBuffer(commandQueue, g_imageTemp2, true, 0, image_width * image_Height * sizeof(float), h_outputGpu.data(), 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, g_imageTemp1, true, 0, image_width * image_Height * sizeof(float), h_outputGpu.data(), 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error copying data to device" << endl;
		return 1;
	}

	// Load the source code
	clock_t kernel_t1, kernel_t2;
	kernel_t1 = clock();
	FILE* fp;
	char* source_str;
	size_t source_size;
	fp = fopen(cl_kernel_file, "r");
	if (!fp) {
		cerr << "Failed to load kernel\n" << stderr << endl;
		exit(1);
	}
	source_str = (char*)malloc(string_length);
	source_size = fread(source_str, 1, string_length, fp);
	fclose(fp);
	cout << "Kernel loaded successfully" << endl;

	//compile the source code.
	program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &errNum);

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// determine the reason for the error  
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}
	
	//create kernel object
	cout << "\nExecuting Greyscale device implementation" << endl;
	kernel = clCreateKernel(program, "NonLocalMeansFilter_Greyscale", NULL);
	if (kernel == NULL)
	{
		cerr << "Failed to create kernel" << endl;
		return 1;
	}

	// Setting kernel arguments
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &g_image);
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_imageTemp1);
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &g_imageTemp2);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error setting kernel arguments" << endl;
		return 1;
	}

	//Launching kernel on device
	size_t globalWorkOffset[2] = { image_Height, image_width };
	size_t localWorkOffset[2] = { 8 , 8};
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkOffset, localWorkOffset, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error queuing kernel for execution" << endl;
		return 1;
	}

	kernel_t2 = clock();

	// Read the output buffer back to the Host  
	clock_t copy_t1, copy_t2;
	copy_t1 = clock();
	std::vector<float> h_imageTemp1(count);
	std::vector<float> h_imageTemp2(count);
	errNum = clEnqueueReadBuffer(commandQueue, g_imageTemp1, CL_TRUE, 0, image_width * image_Height * sizeof(float), h_imageTemp1.data(), 0, NULL, NULL);
	errNum = clEnqueueReadBuffer(commandQueue, g_imageTemp2, CL_TRUE, 0, image_width * image_Height * sizeof(float), h_imageTemp2.data(), 0, NULL, NULL);

	for (std::size_t localI = 0; localI < image_Height - patchWidth + 1; localI++)
	{
		for (std::size_t localJ = 0; localJ < image_width - patchWidth + 1; localJ++)
		{
			h_outputGpu[localI * image_width + localJ] = (h_imageTemp1[localI * image_width + localJ]) / (h_imageTemp2[localI * image_width + localJ]);
		}
	}
	copy_t2 = clock();

	// GPU profiling
	float GPU_time = kernel_t2 - kernel_t1;
	float copy_time = copy_t2 - copy_t1;
	cout << "Time required for GPU operation: " << GPU_time << " ms" << endl;
	cout << "Time required for memory copy back to the host: " << copy_time << " ms" << endl;
	cout << "Time required for GPU operation with memory copy: " << GPU_time + copy_time << " ms" << endl;

	// Store GPU output image
	Core::writeImagePGM("Denoise_Greyscale_image.pgm", h_outputGpu, image_width, image_Height);
	cout << "Device Greyscale implementation successful" << endl; //endif


	// Greyscale Host implementation call
	NLM_Grey_CPU(src);
	cv::waitKey(0);

	//Comparing the performance of GPU and CPU   //if
	float Speed_up = Grey_CPU_time / GPU_time;
	cout << "\nSpeed Up for Greyscale: " << Speed_up << "ms" << endl;

#endif //  GreyDenoise


#ifdef RGBDenoise
	// Declare some values
	cl_mem source_image = 0;
	cl_mem	dest_image = 0;
	cl_mem cl_width = NULL;
	cl_mem cl_height = NULL;
	cl_mem cl_filt = NULL;
	cl_mem cl_sigma = NULL;

	float nlmsigma = 1.0f / (sigma * sigma);
	float filt = 0.2f;

	//Load image from file
	int width, height;
	string src = "C:/images/Color_raw_noisy_image.bmp";		

	//GPU function call for loading color image
	source_image = GetRGBImage(context, src, width, height);
	if (source_image == 0)
		cerr << "Error loading color image: " << string(src) << endl;

	dest_image = GetDestinationCLImage(context, width, height);
	if (dest_image == 0)
		cerr << "Error loading GPU color output image: " << string(src) << endl;

	// Create sampler for sampling image object  
	sampler = clCreateSampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &errNum);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error creating CL sampler object." << endl;
		return 1;
	}

	// Load the source code
	clock_t kernel_t1, kernel_t2;
	kernel_t1 = clock();
	FILE* fp;
	char* source_str;
	size_t source_size;
	fp = fopen(cl_kernel_file, "r");
	if (!fp) {
		cerr << "Failed to load Kernel\n" << stderr << endl;
		exit(1);
	}
	source_str = (char*)malloc(string_length);
	source_size = fread(source_str, 1, string_length, fp);
	fclose(fp);
	cout << "Kernel loaded successfully" << endl;

	//Compile the source code.
	program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &errNum);

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		// Determine the reason for the error  
		char buildLog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			sizeof(buildLog), buildLog, NULL);

		std::cerr << "Error in kernel: " << std::endl;
		std::cerr << buildLog;
		clReleaseProgram(program);
		return NULL;
	}

	//Create kernel object
	cout << "\nExecuting Device implementation" << endl;

	//kernel = clCreateKernel(program, "NonLocalMeansAlgo", NULL);
	kernel = clCreateKernel(program, "NonLocalMeansFilter_Color", NULL);
	if (kernel == NULL)
	{
		cerr << "Failed to create kernel" << endl;
		return 1;
	}

	//Create Buffers for color images
	cl_width = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_height = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_sigma = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);
	cl_filt = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_mem), NULL, &errNum);

	//Enqueue writing buffers for color images 
	errNum = clEnqueueWriteBuffer(commandQueue, cl_width, CL_TRUE, 0, sizeof(int), (void*)&width, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, cl_height, CL_TRUE, 0, sizeof(int), (void*)&height, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, cl_sigma, CL_TRUE, 0, sizeof(float), (void*)&nlmsigma, 0, NULL, NULL);
	errNum = clEnqueueWriteBuffer(commandQueue, cl_filt, CL_TRUE, 0, sizeof(float), (void*)&filt, 0, NULL, NULL);

	//Calling kernel arguments for color image 
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &source_image);
	errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dest_image);
	errNum = clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_width);
	errNum = clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_height);
	errNum = clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_filt);
	errNum = clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_sigma);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error setting kernel arguments" << endl;
		return 1;
	}

	//Enqueue the images to the kernel
	size_t localWorkSize[2] = { 8, 8 };
	size_t globalWorkSize[2] = { width,height };
	errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error queuing kernel for execution" << endl;
		return 1;
	}
	kernel_t2 = clock();

	// Read the output buffer back to the Host  
	clock_t copy_t1, copy_t2;
	copy_t1 = clock();
	char* buffer = new char[width * height * 4];
	char* buffer1 = new char[width * height * 1];
	size_t origin[3] = { 0, 0, 0 };
	size_t region[3] = { width, height, 1 };
	errNum = clEnqueueReadImage(commandQueue, dest_image, CL_TRUE, origin, region, 0, 0, buffer, 0, NULL, NULL);
	//errNum = clEnqueueReadImage(commandQueue, dest_image_grey, CL_TRUE, origin, region, 0, 0, buffer1, 0, NULL, NULL);
	if (errNum != CL_SUCCESS)
	{
		cerr << "Error reading result back to the host" << endl;
		return 1;
	}
	copy_t2 = clock();

	//GPU profiling
	float GPU_time = kernel_t2 - kernel_t1;
	float copy_time = copy_t2 - copy_t1;
	cout << "Time required for GPU operation: " << GPU_time << " ms" << endl;
	cout << "Time required for memory copy back to the host: " << copy_time << " ms" << endl;
	cout << "Time required for GPU operation with memory copy: " << GPU_time + copy_time << " ms" << endl;

	cout << "Device implementation successful" << endl;

	// CPU function calls RGB images
	NLM_RGB_CPU(src);
	
	//Creating output color images
	cv::Mat gpuColor = cv::imread(src);
	cv::Mat gpuColor1 = cv::imread(src);
	cv::Mat gpuColor2;
	gpuColor2.create(gpuColor.rows, gpuColor.cols, gpuColor1.type());
	int z = 0;
	for (int x = gpuColor2.rows - 1; x >= 0; x--)
	{
		for (int y = 0; y < gpuColor2.cols; y++)
		{
			gpuColor2.at<cv::Vec3b>(x, y)[0] = buffer[z++];
			gpuColor2.at<cv::Vec3b>(x, y)[1] = buffer[z++];
			gpuColor2.at<cv::Vec3b>(x, y)[2] = buffer[z++];
			z++;
		}
	}
	cv::imshow("OpenCL-NLMDenosing", gpuColor2);
	cv::waitKey(0);
	delete[] buffer;

	//Comparing the performance of GPU and CPU
	float RGB_speed_up = RGB_CPU_time / GPU_time;
	cout << "\nSpeed Up for Color: " << RGB_speed_up << "ms" << endl;


#endif // RGBDenoise


	std::cout << "\nSuccess" << std::endl;

	return 0;


}