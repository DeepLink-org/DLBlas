#include <cuda.h>
#include <cuda_fp16.h>
// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include<stdio.h>
#ifdef USE_MACA
typedef _Float16 __FLOAT16__;
// typedef half __FLOAT16__;
#else//!USE_MACA
typedef half __FLOAT16__;
#endif//USE_MACA

template <typename T>
struct ViaTypeMap {
    typedef T ViaT;
};
#define MIN(a,b) ((a) < (b) ? (a):(b))

#define CUDA_INIT() \
  cudaSetDevice(0); \
  cudaStream_t stream; \
  cudaStreamCreate(&stream);

#define MAX_DIMENSION 7 // should be acquired from ppl.common
#include <vector>
#include <stdint.h>
#include <assert.h>
/*
  GArray: an small array for transferring parameters to GPU(device)
*/
template <typename T, int32_t capacity = MAX_DIMENSION>
struct GArray {
    GArray()
        : size_(0)
        , data_()
    {
    }

    GArray(int32_t size)
        : size_(size)
        , data_()
    {
        assert(size >= 0 && size <= capacity);
    }

    GArray(const std::vector<T>& vec)
        : GArray(static_cast<int32_t>(vec.size()))
    {
// std::is_trivially_copyable is not implemented in older versions of GCC
#if !defined(__GNUC__) || __GNUC__ >= 5
        static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
#endif
        memcpy(data_, vec.data(), vec.size() * sizeof(T));
    }

    void SetSize(int32_t size)
    {
        assert(size >= 0 && size <= capacity);
        size_ = size;
    }

    __host__ __device__ int32_t Size() const
    {
        return size_;
    }

    __host__ __device__ T& operator[](int32_t index)
    {
        return data_[index];
    }

    __host__ __device__ __forceinline__ const T& operator[](int32_t index) const
    {
        return data_[index];
    }

    __host__ __device__ T* Data()
    {
        return data_;
    }

    __host__ __device__ const T* Data() const
    {
        return data_;
    }

    static constexpr int32_t Capacity()
    {
        return capacity;
    };

private:
    int32_t size_;
    T data_[capacity];
};

template<typename T>
bool checkresult(T *origin, T*dst,int num_elements){
    int diff_nums = 0;
    // printf("this branch :%d\n",sizeof(T));
    for(int i = 0; i < num_elements; i++){
        if(sizeof(T)==4){
            if(abs((float)dst[i] - (float)origin[i]) > 0.1){
                diff_nums++;
                if(diff_nums < 10)
                {
                    if(sizeof(T)==4)
                    {
                        printf(" a Tv:%.8f,Fv:%.8f,index:%d\n",origin[i],dst[i],i);
                    }else{
                        printf(" b Tv:%d,Fv:%d,index:%d\n",origin[i],dst[i],i);
                    }
                }
            }
        } else if(sizeof(T) == 1) {
            if(abs((int)dst[i] - (int)origin[i]) > 0.0000001)
            {
                diff_nums++;
                if(diff_nums < 10)
                {
                    printf("Tv:%d,Fv:%d,index:%d\n",origin[i],dst[i],i);
                }
            }
        } else {
            if(abs((float)(dst[i]) - (float)(origin[i])) > 0.0001){
                diff_nums++;
                if(diff_nums < 10)
                {
                    printf("Tv:%f,Fv:%f,index:%d\n",(float)(origin[i]),(float)(dst[i]),i);
                }
            }
        }

    }
    if(diff_nums > 0){
        printf("result is not right\n");
        return false;
    }else{
        printf("result is right\n");
        return true;
        // for(int i = 0; i < 10; i++) {
        //   printf("data:%f\n",(float)origin[i]);
        // }
    }
}

template<>
bool checkresult(int32_t *origin, int32_t*dst,int num_elements){
    int diff_nums = 0;
    for(int i = 0; i < num_elements; i++){

      if(abs(dst[i] - origin[i]) > 0){
          diff_nums++;
          if(diff_nums < 16)
          {
              printf("Tv:%d,Fv:%d,index:%d\n",origin[i],dst[i],i);
          }
      }
    }
    if(diff_nums > 0){
        printf("result is not right\n");
        return false;
    }else{
        printf("result is right\n");
        return true;
    }
}


template <typename InT, typename OutT>
__device__ __inline__ OutT ppl_scalar_cast(const InT &a)
{
    const bool any_float16 = std::is_same<half, InT>::value || std::is_same<half, OutT>::value;
    typedef typename std::conditional<any_float16, half, OutT>::type T;
    typedef typename ViaTypeMap<T>::ViaT ViaT;
    return (OutT)((ViaT)a);
}

template <typename InT, typename OutT>
__global__ void ppl_cukernel_cast_any(
    const uint64_t num_elems,
    const void *input,
    void *output)
{
    uint16_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_elems)
        return;
    const InT *input_ptr = static_cast<const InT *>(input);
    InT in_val           = input_ptr[index];
    OutT *output_ptr     = static_cast<OutT *>(output);
    output_ptr[index]    = ppl_scalar_cast<InT, OutT>(in_val);
}

template <typename Int,typename OuT>
void datacast(cudaStream_t stream,const uint64_t num_elems,const void* input,void* output)
{
    int blockSize = 256;
    int grid = (num_elems + blockSize - 1) / blockSize;
    ppl_cukernel_cast_any<Int,OuT><<<grid,blockSize,0,stream>>>(num_elems,input,output);
}

struct DivModFast {
    DivModFast(int d = 1)
    {
        d_ = (d == 0) ? 1 : d;
        for (l_ = 0;; ++l_) {
            if ((1U << l_) >= d_)
                break;
        }
        uint64_t one = 1;
        uint64_t m   = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        m_           = static_cast<uint32_t>(m);
    }

    __device__ __inline__ int div(int idx) const
    {
        uint32_t tm = __umulhi(m_, idx); // get high 32-bit of the product
        return (tm + idx) >> l_;
    }

    __device__ __inline__ int mod(int idx) const
    {
        return idx - d_ * div(idx);
    }

    __device__ __inline__ void divmod(int idx, int &quo, int &rem)
    {
        quo = div(idx);
        rem = idx - quo * d_;
    }

    uint32_t d_; // divisor
    uint32_t l_; // ceil(log2(d_))
    uint32_t m_; // m' in the papaer
};

enum CVTFormatMode {
    CVTFormatUnknown = 0,

    NDARRAY_N4CX = 2,
    N4CX_NDARRAY = 11,

    NDARRAY_NHWC  = 31,
    NHWC_NDARRAY  = 32,
    NHWC8_NHWC16  = 33,
    NHWC16_NHWC8  = 34,
};

enum CVTTypeMode {
    CVTTypeUnknown  = 0,
    INT8_FLOAT32    = 1,
    FLOAT32_INT8    = 2,
    FLOAT32_INT4B   = 3,
    INT4B_FLOAT32   = 4,
    INT8_FLOAT16    = 5,
    FLOAT16_INT8    = 6,
    FLOAT32_FLOAT16 = 7,
    FLOAT16_FLOAT32 = 8,
    INT4B_FLOAT16   = 9,
    FLOAT16_INT4B   = 10,
    INT8_INT4B      = 11,
    INT4B_INT8      = 12,
    INT8_INT8       = 13,
    INT4B_INT4B     = 14,
    INT32_INT16     = 15,
    INT16_INT32     = 16,
    FLOAT32_INT16   = 17,
    INT16_FLOAT32   = 18,
    UINT8_FLOAT32   = 19,
};

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifndef USE_ROCM
  f = __half2float(*(__half*)&h);
#else
asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
#ifndef USE_ROCM
  uint16_t lo, hi;
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = v;
  lo = tmp.u16[0];
  hi = tmp.u16[1];
  return make_float2(half_to_float(lo), half_to_float(hi));
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = v;
  float2 ret;
  ret.x = half_to_float(tmp.u16[0]);
  ret.y = half_to_float(tmp.u16[1]);
  return ret;
#endif
}

struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
{
{
unsigned short __a=(a);
unsigned short __b=(b);
__half __d=__hadd(*(__half*)&__a,*(__half*)&__b);
(c)=*(unsigned short*)&__d;
}
}

#else
asm volatile("v_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
{
{
unsigned int __a=(a);
unsigned int __b=(b);
__half2 __d=__hadd2(*(__half2*)&__a,*(__half2*)&__b);
(c)=*(unsigned int*)&__d;
}
}

#else
asm volatile("v_pk_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}
inline __device__ float add(float a, float b) {
  return a + b;
}

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}


inline __device__ float2 add(uint32_t a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}
template<typename T>
inline __device__ float sum(T v);
template<>
inline __device__ float sum(float v) {
  return v;
}

template<>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template<>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template<>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template<>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

struct timeval start;
struct timeval end;
float time_use=0;
unsigned long throughput;
float bandwidth;

#define PRINT_BANDWIDTH()\
printf("throughput : %f G, time_use:%f,bandwidth:%f TB/S\n",throughput * 1.0f / 1024 / 1024 / 1024,time_use,throughput * 1.0f / 1024 / 1024 / 1024 /time_use);

#define PRINT_THROUGHTPUT()\
printf("throughput : %f G\n",throughput * 1.0f / 1024 / 1024 / 1024);
#define Align(x, y) (((x) + (y)-1) / (y) * (y))
#define DivUp(x, y) (((x) + (y)-1) / (y))
