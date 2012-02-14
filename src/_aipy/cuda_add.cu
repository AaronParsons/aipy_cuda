#include <stdio.h>
#include <cuda.h>


__global__ void add( int *a, int *b, int *c, int N) {
	int tid = blockIdx.x;
	if (tid < N)
		c[tid] = a[tid] +b[tid];
}

int cuda_add(int* a, int* b, int N){
	int *dev_a, *dev_b, *dev_c;
	int c[N];

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
	
	for (int i=0;i<N;i++){
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
		}
		
	return 1;
}

