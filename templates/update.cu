
__global__ void {{ function_name }}(float t{{ params }})
{
    // This ordering of the global index variables (i, j, and k) effectively
    // transforms the normal column-major CUDA thread ordering into a 
    // row-major ordering.
    float i = {{ offset[0] }} + {{ spacing[0] }} * (blockIdx.{{ custom_blockID[0] }} * blockDim.z + threadIdx.z); 
    float j = {{ offset[1] }} + {{ spacing[1] }} * (blockIdx.{{ custom_blockID[1] }} * blockDim.y + threadIdx.y);
    float k = {{ offset[2] }} + {{ spacing[2] }} * (blockIdx.{{ custom_blockID[2] }} * blockDim.x + threadIdx.x); 

    for(; (i < {{ limit[0] }}) && (j < {{ limit[1] }}) && (k < {{ limit[2] }}); 
        {{ ijk_step }} += ({{ step_spacing }} * blockDim.{{ zyx_step }})) {
{{ source }}
    }
}


