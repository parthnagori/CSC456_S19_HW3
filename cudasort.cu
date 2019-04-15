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


__global__ void bitonic_sort(float *arr, int j, int k)
{
  unsigned int i, ij; 
    i = threadIdx.x + blockDim.x * blockIdx.x;
  ij = i^j;

  if ((ij)>i) {
    if ((i&k)==0) {
      if (arr[i]>arr[ij]) {
        float temp = arr[i];
        arr[i] = arr[ij];
        arr[ij] = temp;
      }
    }
    if ((i&k)!=0) {
      if (arr[i]<arr[ij]) {
        float temp = arr[i];
        arr[i] = arr[ij];
        arr[ij] = temp;
      }
    }
  }
}


int cuda_sort(int number_of_elements, float *a)
{
  
  float *arr;
  
  cudaEvent_t event;
  cudaEventCreate(&event);

  cudaMalloc((void**) &arr, number_of_elements * sizeof(float));
  cudaMemcpy(arr, a, number_of_elements * sizeof(float), cudaMemcpyHostToDevice);
  
  
  dim3 dimGrid(512,1);    
  dim3 dimBlock(number_of_elements/512,1);

  int l, m;
  for (l = 2; l <= number_of_elements; l <<= 1) {
    for (m=l>>1; m>0; m=m>>1) {
      bitonic_sort<<<dimGrid, dimBlock>>>(arr, m, l);
    }
  }

  cudaMemcpy(a, arr, number_of_elements * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(arr);
  cudaThreadSynchronize();
  cudaEventSynchronize(event);

  return 0;
}

#ifdef __cplusplus
}
#endif

