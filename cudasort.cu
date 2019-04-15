#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>

#define THREADS 512
#ifdef __cplusplus
extern "C"
{
#endif

__global__ void
gpu_sort(int N, float *input, float *tmp, int x)
{
  // get the index of the current thread
  int y = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (y < N && (y % (1 << (x+1)) == 0))
  {
    unsigned width = 1 << x;

    int left, middle, right;

  left = y;
  middle = y + width;
  right = y + 2*width;

  // merge function
  int i, j, k;
  i = left;
  j = middle;
  k = left;

  while(i < middle || j < right)
  {
    if (i < middle && j < right)
    {
      if (input[i] < input[j])
      {
        tmp[k++] = input[i++];
      }
      else
      {
        tmp[k++] = input[j++];
      }
    }
    else if (i == middle)
    {
      tmp[k++] = input[j++];
    }
    else if (j == right)
    {
      tmp[k++] = input[i++];
    }
  }

  // copy tmp back into input
  for(i = left; i < right; i++)
  {
    input[i] = tmp[i];
  }
  }
}

int cuda_sort(int number_of_elements, float *a)
{
  float *input_buf, *tmp_buf;

  // allocate device memory
  cudaMalloc( (void **) &input_buf, sizeof(float) * number_of_elements );
  cudaMalloc( (void **) &tmp_buf, sizeof(float) * number_of_elements);

  // move elements of a to the CUDA device
  cudaMemcpy( input_buf, a, sizeof(float) * number_of_elements, cudaMemcpyHostToDevice );

  unsigned blocks_per_grid = (number_of_elements + THREADS - 1) / THREADS;
  
  // determine what log2(N) is
  unsigned num_widths = 0;
  unsigned N = number_of_elements;
  while (N >>= 1) ++num_widths;

  // launch the kernel log2(N) times, each time setting a different
  // value for the 3rd kernel arguments. This value will be used to determine
  // the width of subarrays that the kernel should merge
  for (unsigned i = 0; i < num_widths; ++i)
  {
    gpu_sort<<<blocks_per_grid, THREADS>>>(number_of_elements, input_buf, tmp_buf, i);
  }
  
  cudaMemcpy( a, input_buf, sizeof(float) * number_of_elements, cudaMemcpyDeviceToHost);

  cudaFree(input_buf);
  cudaFree(tmp_buf);

  return 0;
}

#ifdef __cplusplus
}
#endif