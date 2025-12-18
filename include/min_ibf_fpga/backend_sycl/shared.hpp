#pragma once

#if __INTEL_LLVM_COMPILER < 20230000
  #include <CL/sycl.hpp>
  #include <CL/sycl/INTEL/ac_types/ac_int.hpp>
#else
  #include <sycl/sycl.hpp>
  #include <sycl/ext/intel/ac_types/ac_int.hpp>
#endif

// Note: This header contains code that is shared between host and device.

#define INTEGER_DIVISION_CEIL(lhs, rhs) ((lhs + rhs - 1) / rhs)
#define MIN_QUERY_LENGTH 50  // Helps to allocate reasonable sized memory on the device
#define MAX_QUERY_LENGTH 250 // Upper bound for THRESHOLDS_CACHE_SIZE
#define THRESHOLDS_CACHE_SIZE MAX_QUERY_LENGTH
#define HOST_SIZE_TYPE_BITS 64
#define MAX_BUS_WIDTH 512

#if __INTEL_LLVM_COMPILER < 20230100
  #ifdef FPGA_EMULATOR
  sycl::ext::intel::fpga_emulator_selector device_selector;
  #else
  sycl::ext::intel::fpga_selector device_selector;
  #endif
#else
  #ifdef FPGA_EMULATOR
  auto device_selector = sycl::ext::intel::fpga_emulator_selector_v;
  #else
  auto device_selector = sycl::ext::intel::fpga_selector_v;
  #endif
#endif

namespace min_ibf_fpga::backend_sycl
{

using HostSizeType = ac_int<HOST_SIZE_TYPE_BITS, false>;

struct kernelData {
  HostSizeType numberOfQueries;
  HostSizeType binSize;
  HostSizeType hashShift;
  HostSizeType minimalNumberOfMinimizers;
  HostSizeType maximalNumberOfMinimizers;
};

} // namespace min_ibf_fpga::backend_sycl
