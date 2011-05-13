
__global__ void {{ function_name }}(float t{{ params }}, float *block_sums, \
    unsigned int *count)
{
    // This ordering of the global index variables (i, j, and k) effectively
    // transforms the normal column-major CUDA thread ordering into a 
    // row-major ordering.
    float i = {{ offset[0] }} + {{ spacing[0] }} * (blockIdx.{{ custom_blockID[0] }} * blockDim.z + threadIdx.z); 
    float j = {{ offset[1] }} + {{ spacing[1] }} * (blockIdx.{{ custom_blockID[1] }} * blockDim.y + threadIdx.y);
    float k = {{ offset[2] }} + {{ spacing[2] }} * (blockIdx.{{ custom_blockID[2] }} * blockDim.x + threadIdx.x); 
    float *thread_sums = (float*) sum_array;

    // Index into shared variable (vals).
    int ind = threadIdx.x + blockDim.x * threadIdx.y + 
        blockDim.x * blockDim.y * threadIdx.z;

    thread_sums[ind] = 0; // Initialize to 0.

    for(; (i < {{ limit[0] }}) && (j < {{ limit[1] }}) && (k < {{ limit[2] }}); 
        {{ ijk_step }} += ({{ step_spacing }} * blockDim.{{ zyx_step }})) {
        // Sum up!
        thread_sums[ind] += {{ source }};
    }

    __syncthreads();

    // Choose the first thread to calculate the partial sum within the block.
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0)) {
       float partial_sum = 0;
       for (int l=0 ; l<(blockDim.x*blockDim.y*blockDim.z) ; l++) 
           partial_sum += thread_sums[l];

       // Write out the partial sum.
       block_sums[blockIdx.x + gridDim.x * blockIdx.y] = partial_sum;

        // Thread 0 makes sure its result is visible to all other threads.
        __threadfence(); 

        // Thread 0 of each block signals that it is done. 
        unsigned int value = atomicInc(count, gridDim.x * gridDim.y); 

        // Thread 0 of each block determines if its block is the 
        // last block to be done. 
        if (value == ((gridDim.x * gridDim.y) - 1)) {
            // The last block sums the partial sums 
            // stored in result[0 .. gridDim.x-1]. 
            float totalSum = 0;
            for (int l=0 ; l<(gridDim.x * gridDim.y) ; l++)
                totalSum += block_sums[l];

            // Thread 0 of last block stores total sum to global memory 
            // and resets count so that next kernel call works properly. 
            {{ result_field }} = totalSum; 
            count[0] = 0; 
        }
    }
}



