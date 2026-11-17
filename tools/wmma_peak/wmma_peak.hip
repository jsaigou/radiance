#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef int i8v __attribute__((ext_vector_type(8)));
typedef int i4v __attribute__((ext_vector_type(4)));
__global__ void kf16(float* out,int iters){
  h16 a,b; for(int i=0;i<16;i++){a[i]=(_Float16)(threadIdx.x*0.001f+i);b[i]=(_Float16)(i*0.01f);}
  f8 c0={},c1={},c2={},c3={};
  for(int it=0;it<iters;it++){
    c0=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c0);
    c1=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c1);
    c2=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c2);
    c3=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,c3);
  }
  float s=0; for(int i=0;i<8;i++) s+=c0[i]+c1[i]+c2[i]+c3[i]; out[blockIdx.x*blockDim.x+threadIdx.x]=s;
}
__global__ void ki8(int* out,int iters){
  i4v a,b; for(int i=0;i<4;i++){a[i]=threadIdx.x+i;b[i]=i*3;}
  i8v c0={},c1={},c2={},c3={};
  for(int it=0;it<iters;it++){
    c0=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c0,false);
    c1=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c1,false);
    c2=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c2,false);
    c3=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,a,true,b,c3,false);
  }
  int s=0; for(int i=0;i<8;i++) s+=c0[i]+c1[i]+c2[i]+c3[i]; out[blockIdx.x*blockDim.x+threadIdx.x]=s;
}
int main(){
  int dev=0; hipDeviceProp_t p; hipGetDeviceProperties(&p,dev);
  int cus=p.multiProcessorCount; printf("%s CUs=%d\n",p.gcnArchName,cus);
  int blocks=cus*8, threads=256, iters=20000;
  float* of; int* oi; hipMalloc(&of,blocks*threads*4); hipMalloc(&oi,blocks*threads*4);
  hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for(int rep=0;rep<2;rep++){
   hipEventRecord(e0); kf16<<<blocks,threads>>>(of,iters); hipEventRecord(e1); hipEventSynchronize(e1);
   float ms; hipEventElapsedTime(&ms,e0,e1);
   double ops=2.0*16*16*16*4.0*iters*(double)blocks*(threads/32);
   printf("f16  %.1f TFLOPs (%.1f ms)\n",ops/ms/1e9,ms);
   hipEventRecord(e0); ki8<<<blocks,threads>>>(oi,iters); hipEventRecord(e1); hipEventSynchronize(e1);
   hipEventElapsedTime(&ms,e0,e1);
   printf("iu8  %.1f TOPS   (%.1f ms)\n",ops/ms/1e9,ms);
  }
}
