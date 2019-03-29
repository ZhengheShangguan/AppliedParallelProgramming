// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// My Kernel function: Add Scanned BLock Sum to Scanned blocks
__global__ void add(float *input, float *output, float *sum, int len) {
  int tx = threadIdx.x, bx = blockIdx.x;
  int x = tx + (bx * BLOCK_SIZE * 2);

  __shared__ float increment;
  if (tx == 0) {
      increment = (bx == 0) ? 0 : sum[bx - 1];
  }
  __syncthreads();

  if (x < len) {
      output[x] = input[x] + increment;
  }
  if (x + BLOCK_SIZE < len) {
      output[x + BLOCK_SIZE] = input[x + BLOCK_SIZE] + increment;
  }
}

// My Kernel function: Scan BLock Sums 
__global__ void scanSum(float* input, float* output, int len) {  
  __shared__ float T[BLOCK_SIZE * 2];
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // 1. load into shared memory
  int loadIdx = (tx + 1) * BLOCK_SIZE * 2 - 1;
  T[tx] = (loadIdx < len) ? input[loadIdx]:0;

  // 2. prefix sum for block-wise: Reduction STEP
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
      __syncthreads();
      int index = (tx + 1) * stride * 2 - 1;
      if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
          T[index] += T[index - stride];
      }
      stride = stride * 2;
  }

  // 3. prefix sum for block-wise: Reverse Reduction STEP
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
      __syncthreads();
      int index = (tx + 1) * stride * 2 - 1;
      if ((index + stride) < 2 * BLOCK_SIZE) {
          T[index + stride] += T[index];
      }
      stride = stride / 2;
  }

  // 4. store the prefix sum results into output
  __syncthreads();
  int storeIdx = tx + bx * BLOCK_SIZE * 2;
  if (storeIdx < len) {
      output[storeIdx] = T[tx];
  }
  if (storeIdx + BLOCK_SIZE < len) {
      output[storeIdx + BLOCK_SIZE] = T[tx + BLOCK_SIZE];
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[BLOCK_SIZE * 2];
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  // 1. load into shared memory
  int loadIdx = bx * BLOCK_SIZE * 2 + tx;
  T[tx] = (loadIdx < len) ? input[loadIdx]:0;
  T[tx + BLOCK_SIZE] = (loadIdx + BLOCK_SIZE < len) ? input[loadIdx + BLOCK_SIZE]:0;

  // 2. prefix sum for block-wise: Reduction STEP
  int stride = 1;
  while (stride < 2 * BLOCK_SIZE) {
      __syncthreads();
      int index = (tx + 1) * stride * 2 - 1;
      if (index < 2 * BLOCK_SIZE && (index - stride) >= 0) {
          T[index] += T[index - stride];
      }
      stride = stride * 2;
  }

  // 3. prefix sum for block-wise: Reverse Reduction STEP
  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
      __syncthreads();
      int index = (tx + 1) * stride * 2 - 1;
      if ((index + stride) < 2 * BLOCK_SIZE) {
          T[index + stride] += T[index];
      }
      stride = stride / 2;
  }

  // 4. store the prefix sum results into output
  __syncthreads();
  int storeIdx = tx + bx * BLOCK_SIZE * 2;
  if (storeIdx < len) {
      output[storeIdx] = T[tx];
  }
  if (storeIdx + BLOCK_SIZE < len) {
      output[storeIdx + BLOCK_SIZE] = T[tx + BLOCK_SIZE];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  // @@ My buffer scan memory
  float *deviceScanBuffer;
  float *deviceScanSum;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  // @@ My buffer scan memory
  wbCheck(cudaMalloc((void **)&deviceScanBuffer, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanSum, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(1.0 * numElements / (BLOCK_SIZE * 2)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  //@@ Stage 1: Scan Kernel into deviceScanBuffer
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceScanBuffer, numElements);
  cudaDeviceSynchronize();

  //@@ Stage 2: Scan Block Sums
  dim3 dimSumGrid(1, 1, 1);
  dim3 dimSumBlock(BLOCK_SIZE, 1, 1);
  scanSum<<<dimSumGrid, dimSumBlock>>>(deviceScanBuffer, deviceScanSum, numElements);
  cudaDeviceSynchronize();

  //@@ Stage 3: Add Scanned BLock Sum to all corresponding values of scanned block
  add<<<dimGrid, dimBlock>>>(deviceScanBuffer, deviceOutput, deviceScanSum, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  //@@ Free the buffer and the sum cuda memory
  cudaFree(deviceScanBuffer);
  cudaFree(deviceScanSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
