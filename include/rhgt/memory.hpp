#pragma once

#include "error_handling.hpp"

//Allocate `count` items in page-locked memory
template<class T>
T* PageLockedMalloc(const size_t count, const T *const host_data=nullptr){
    T *temp;
    RCHECKCUDAERROR(cudaMallocHost(&temp, count*sizeof(T), cudaHostAllocDefault));
    if(host_data)
        RCHECKCUDAERROR(cudaMemcpy(temp, host_data, count*sizeof(T), cudaMemcpyHostToHost));
    return temp;
}



//Allocate `count` items on device memory
template<class T>
T* DeviceMalloc(const size_t count, const T *const host_data=nullptr){
    T *temp;
    RCHECKCUDAERROR(cudaMalloc(&temp, count*sizeof(T)));
    if(host_data)
        RCHECKCUDAERROR(cudaMemcpy(temp, host_data, count*sizeof(T), cudaMemcpyHostToDevice));
    return temp;
}
