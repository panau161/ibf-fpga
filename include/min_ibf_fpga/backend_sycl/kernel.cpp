#include <sycl/ext/intel/fpga_extensions.hpp>

// Utilities
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

// Kernel includes
#include "kernel.hpp"
#include "kernel_ibf.hpp"
#include "kernel_minimizer.hpp"


namespace min_ibf_fpga::backend_sycl
{

// Forward declaration of the kernel names. FPGA best practice to reduce compiler name mangling in the optimization reports.
class Distributor;
template <std::size_t id> class MinimizerKernel;
template <std::size_t id> class IbfKernel;
class Collector;

void RunKernel(sycl::queue& queue,
	const char* queries_ptr,
	const HostSizeType* querySizes_ptr,
	const Chunk* ibfData_ptr,
	const HostSizeType* thresholds_ptr,
	kernelData* kData_ptr,
	Chunk* result_ptr,
	std::vector<sycl::event>& kernelEvents)
{
	using DistributorPipes = fpga_tools::PipeArray<class DistributorPipe, DistributorToMinimizerData, 2, KERNEL_COPYS>;
	using MinimizerToIBFPipes = fpga_tools::PipeArray<class MinimizerToIBFPipe, MinimizerToIBFData, 25, KERNEL_COPYS>;
	using CollectorPipes = fpga_tools::PipeArray<class CollectorPipe, Chunk, 25, KERNEL_COPYS>;

	using PrefetchingLSU = sycl::ext::intel::lsu<sycl::ext::intel::prefetch<true>, sycl::ext::intel::statically_coalesce<false>>;

	const HostSizeType numberOfQueries = kData_ptr->numberOfQueries;
	const HostSizeType binSize = kData_ptr->binSize;
	const HostSizeType hashShift = kData_ptr->hashShift;
	const HostSizeType minimalNumberOfMinimizers = kData_ptr->minimalNumberOfMinimizers;
	const HostSizeType maximalNumberOfMinimizers = kData_ptr->maximalNumberOfMinimizers;

	kernelEvents.push_back( queue.submit([&](sycl::handler &handler)
	{
		#include "distributor.cpp"
	}) );

	fpga_tools::UnrolledLoop<KERNEL_COPYS>([&](auto id)
	{
		QueryIndex localNumberOfQueries = numberOfQueries / KERNEL_COPYS;
		QueryIndex remainder = numberOfQueries % KERNEL_COPYS;

		if (remainder > id) localNumberOfQueries++;

		kernelEvents.push_back( queue.submit([&](sycl::handler &handler)
		{
			#include "kernel_minimizer.cpp"
		}) );

		kernelEvents.push_back( queue.submit([&](sycl::handler &handler)
		{
			#include "kernel_ibf.cpp"
		}) );
	});

	kernelEvents.push_back( queue.submit([&](sycl::handler &handler)
	{
		#include "collector.cpp"
	}) );
}

} // namespace min_ibf_fpga::backend_sycl
