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

__device__ void merge(float* arr, float* final, int start, int mid, int end)
{
    int i = start;
    int j = mid;
    int k = start;
    printf("start : %d mid: %d end: %d", start, mid, end);
    while (k < end)
    {
      if (i==mid){
        final[k] = arr[j++];
      }
      else if (j == end){
        final[k] = arr[i++];
      }
      else if (arr[i] < arr[j]){
        final[k] = arr[i++];
      }
      else{
        final[k] = arr[j++];
      }
      k++;
    }

    // for(i = start; i < end; i++)
    // {
    //   arr[i] = final[i];
    // }

}

__global__ void merge_sort(float* arr, float* final, int numberOfBlocks, int elementsPerBlock, int partition){

    int block_id = blockIdx.x;
    int start = block_id * partition;
    int n = numberOfBlocks*elementsPerBlock;
    int end = min(start + partition,n);
    int mid = min(start + partition/2,n);
    merge(arr, final, start, mid, end);
}

int cuda_sort(int number_of_elements, float *a)
{
  
  float *arr;
  float *final;
  // int n;
  // int part = 0;

  int numberOfBlocks = 512;
  int elementsPerBlock = number_of_elements/numberOfBlocks;

  cudaEvent_t event;
  cudaEventCreate(&event);

  cudaMalloc((void **) &arr, sizeof(float)*number_of_elements);
  cudaMalloc((void **) &final, sizeof(float)*number_of_elements);
  cudaMemcpy(arr, a, sizeof(float)*number_of_elements, cudaMemcpyHostToDevice);

  dim3 dimGrid(numberOfBlocks);
  dim3 dimBlock(1);

  int partition;
  // int partitions;

  int cnt = 0;
  // n = number_of_elements;
  // while (n != 0){
  //   ++partitions;
  //   n/=2;
  // } 


  // for (part = 0; part < partitions; part++){
  //   int part_size = part << 1;
  //   merge_sort<<<dimGrid, dimBlock>>>(arr, final, numberOfBlocks, elementsPerBlock, part); 
  // } 


  for (partition = 2; partition < 2*number_of_elements; partition*=2) {
    if (cnt%2 == 0)
      merge_sort<<<dimGrid, dimBlock>>>(arr, final, numberOfBlocks, elementsPerBlock, partition); 
    else
      merge_sort<<<dimGrid, dimBlock>>>(final, arr, numberOfBlocks, elementsPerBlock, partition);
    cnt+=1; 
  }

  printf("cnt: %d", cnt);
  cudaMemcpy(a, arr, sizeof(float)*number_of_elements, cudaMemcpyDeviceToHost);
  // cudaFree(gpu_arr);
  cudaThreadSynchronize();
  cudaEventSynchronize(event);
  
  return 0;
}

#ifdef __cplusplus
}
#endif
