#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>

namespace rhgt {

#define RCHECKCUDAERROR(error) rhgt::CudaErrorHandler(error, __FILE__, __LINE__)

inline void CudaErrorHandler(const cudaError_t &err, const std::string &file, const size_t line){
  if (err==cudaSuccess)
    return;

  const std::string msg = std::string("[RHGT CUDA ERROR:] ")
    + cudaGetErrorString(err)
    + " (CUDA error no.="+std::to_string(err)+")"
    + " in "+file+":"+std::to_string(line);

  throw std::runtime_error(msg);
}

}