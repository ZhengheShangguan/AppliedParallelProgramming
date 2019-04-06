// Histogram Equalization
#include <wb.h>

#define HISTOGRAM_LENGTH 256
typedef unsigned int uint_t;
typedef unsigned char uint8_t;
#define TILE_WIDTH 32

//@@ insert code here
//@@ Cast the image from float to unsigned char
__global__ void floatToUInt8(float* input, uint8_t* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = blockIdx.z * (width * height) + y * width + x;
    output[index] = (uint8_t) ((HISTOGRAM_LENGTH - 1) * input[index]);
  }
}

//@@ Convert the image from RGB to GrayScale
__global__ void rgbToGrayScale(uint8_t* input, uint8_t* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int index = y * width + x;
    uint8_t r = input[3 * index + 0];
    uint8_t g = input[3 * index + 1];
    uint8_t b = input[3 * index + 2];

    output[index] = (uint8_t)(0.21 * r + 0.71 * g + 0.07 * b);
  }
}

//@@ Compute the histogram of grayImage
__global__ void grayScaleToHistogram(uint8_t *input, uint_t *output, int width, int height) {
  // Use shared memory to make the atomicAdd faster
  __shared__ uint_t histogram[HISTOGRAM_LENGTH];

  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
  if (tIdx < HISTOGRAM_LENGTH) {
    histogram[tIdx] = 0;
  }
  __syncthreads();

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * width + x;
    uint8_t val = input[idx];
    atomicAdd(&(histogram[val]), 1);
  }
  __syncthreads();

  if (tIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tIdx]), histogram[tIdx]);
  }
}

//@@ Compute the Cumulative Distribution Function of histogram
__global__ void histogramToCDF(uint_t* input, float* output, int width, int height) {
  __shared__ uint_t cdf[HISTOGRAM_LENGTH];
  int x = threadIdx.x;
  cdf[x] = input[x];

  // Scan part 1:
  for (unsigned int stride = 1; stride <= HISTOGRAM_LENGTH / 2; stride *= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx < HISTOGRAM_LENGTH) {
      cdf[idx] += cdf[idx - stride];
    }
  }

  // Scan part: 2
  for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2) {
    __syncthreads();
    int idx = (x + 1) * 2 * stride - 1;
    if (idx + stride < HISTOGRAM_LENGTH) {
      cdf[idx + stride] += cdf[idx];
    }
  }

  __syncthreads();
  output[x] = cdf[x] / ((float) (width * height));
}

//@@ Define the histogram equalization function, Apply the histogram equalization function
__global__ void equalizeImage(uint8_t* input, float* cdf, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * width * height + width * y + x;
    uint8_t val = input[idx];
    float equalized = 255.0f * (cdf[val] - cdf[0]) / (1.0f - cdf[0]);
    float clamped = min(max(equalized, 0.0f), 255.0f);

    input[idx] = (uint8_t)clamped;
  }
}

//@@ Cast back to float
__global__ void uInt8ToFloat(uint8_t* input, float* output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int idx = blockIdx.z * (width * height) + y * width + x;
    output[idx] = (float) (input[idx] / 255.0f);
  }
}




int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImageFloat;
  float *deviceImageCDF;
  uint8_t *deviceImageUChar;
  uint8_t *deviceImageUCharGrayScale;
  uint_t *deviceImageHistogram;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  hostInputImageData = wbImage_getData(inputImage); 
  hostOutputImageData = wbImage_getData(outputImage); 
  
  //@@ insert code here
  cudaMalloc((void**) &deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceImageCDF, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void**) &deviceImageUChar, imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
  cudaMalloc((void**) &deviceImageUCharGrayScale, imageWidth * imageHeight * sizeof(uint8_t));
  cudaMalloc((void**) &deviceImageHistogram, HISTOGRAM_LENGTH * sizeof(uint_t));
  cudaMemset((void**) deviceImageHistogram, 0, HISTOGRAM_LENGTH * sizeof(uint_t));

  //@@ Cast the image from float to unsigned char
  cudaMemcpy(deviceImageFloat, hostInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid = dim3(ceil(1.0 * imageWidth / TILE_WIDTH), ceil(1.0 * imageHeight / TILE_WIDTH), imageChannels);
  dim3 dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  floatToUInt8<<<dimGrid, dimBlock>>>(deviceImageFloat, deviceImageUChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ Convert the image from RGB to GrayScale
  dimGrid = dim3(ceil(1.0 * imageWidth / TILE_WIDTH), ceil(1.0 * imageHeight / TILE_WIDTH), 1);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  rgbToGrayScale<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageUCharGrayScale, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ Compute the histogram of grayImage
  grayScaleToHistogram<<<dimGrid, dimBlock>>>(deviceImageUCharGrayScale, deviceImageHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ Compute the Cumulative Distribution Function of histogram
  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  histogramToCDF<<<dimGrid, dimBlock>>>(deviceImageHistogram, deviceImageCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ Compute the minimum value of the CDF. The maximal value of the CDF should be 1.0. 
  //@@ Define the histogram equalization function, Apply the histogram equalization function
  dimGrid = dim3(ceil(1.0 * imageWidth / TILE_WIDTH), ceil(1.0 * imageHeight / TILE_WIDTH), imageChannels);
  dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  equalizeImage<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ Cast back to float
  uInt8ToFloat<<<dimGrid, dimBlock>>>(deviceImageUChar, deviceImageFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  //@@ Copy back to CPU
  cudaMemcpy(hostOutputImageData, deviceImageFloat, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceImageFloat);
  cudaFree(deviceImageCDF);
  cudaFree(deviceImageUChar);
  cudaFree(deviceImageUCharGrayScale);
  cudaFree(deviceImageHistogram);

  return 0;
}