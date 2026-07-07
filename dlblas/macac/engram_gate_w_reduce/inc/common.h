#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <iostream>
#include<stdio.h>
#ifdef USE_MACA
typedef _Float16 __FLOAT16__;
#else
typedef half __FLOAT16__;
#endif

template <typename T>
struct ViaTypeMap {
    typedef T ViaT;
};
#define MIN(a,b) ((a) < (b) ? (a):(b))

#define CUDA_INIT()   cudaSetDevice(0);   cudaStream_t stream;   cudaStreamCreate(&stream);

#define MAX_DIMENSION 7
#include <vector>
#include <stdint.h>
#include <assert.h>

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
