#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "cuda_add.h"

__global__ void add( int *a, int *b, int *c, int N) {
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] +b[tid];
}

int cuda_add(int* a, int* b, int *c, int N){
	int *dev_a, *dev_b, *dev_c;

	// Allocate memory on the GPU
	cudaMalloc((void**) &dev_a, N*sizeof(int));
	cudaMalloc((void**) &dev_b, N*sizeof(int));
	cudaMalloc((void**) &dev_c, N*sizeof(int));
	
	// Move the arrays onto the GPU
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
				
	cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
				
	add<<<N,1>>>(dev_a, dev_b, dev_c, N);
	
	// copy the array back
	cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
	
    // Free memory allocated on device
    cudaFree (dev_a);
    cudaFree (dev_b);
    cudaFree (dev_c);
    
	for (int i=0;i<N;i++){
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
		}
		
	return 1;
}

