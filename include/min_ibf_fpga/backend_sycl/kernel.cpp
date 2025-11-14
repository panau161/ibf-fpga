#include <sycl/ext/intel/fpga_extensions.hpp>

// Utilities
#include "autorun.hpp"
#include "pipe_utils.hpp"

// Kernel includes
#include "kernel.hpp"
#include "kernel_ibf.hpp"
#include "kernel_minimizer.hpp"


namespace min_ibf_fpga::backend_sycl
{

struct InterfaceToDistributorData
{
	HostSizeType numberOfQueries;
	char* queries_ptr;
	HostSizeType* querySizes_ptr;
};

struct InterfaceToIBFData
{
	kernelData kData;
	HostSizeType* thresholds_ptr;
	Chunk* ibfData_ptr;
};

struct InterfaceToCollectorData
{
	HostSizeType numberOfQueries;
	Chunk* result_ptr;
};


// Forward declaration of the kernel names. FPGA best practice to reduce compiler name mangling in the optimization reports.
class Interface;
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
	using InterfaceToDistributorPipe = sycl::pipe<class I2D, InterfaceToDistributorData, 1>;
	using InterfaceToMinimizerPipe = sycl::pipe<class I2M, HostSizeType, 1>;
	using InterfaceToIBFPipe = sycl::pipe<class I2IBF, InterfaceToIBFData, 1>;
	using InterfaceToCollectorPipe = sycl::pipe<class I2C, InterfaceToCollectorData, 1>;
	using CollectorToInterfacePipe = sycl::pipe<class C2I, bool, 1>;

	using DistributorPipes = fpga_tools::PipeArray<class DistributorPipe, DistributorToMinimizerData, 2, KERNEL_COPYS>;
	using MinimizerToIBFPipes = fpga_tools::PipeArray<class MinimizerToIBFPipe, MinimizerToIBFData, 25, KERNEL_COPYS>;
	using CollectorPipes = fpga_tools::PipeArray<class CollectorPipe, Chunk, 25, KERNEL_COPYS>;

	using PrefetchingLSU = sycl::ext::intel::lsu<sycl::ext::intel::prefetch<true>, sycl::ext::intel::statically_coalesce<false>>;

	const auto id = 0;

	kernelEvents.push_back( queue.submit([&](sycl::handler &handler)
	{
		handler.single_task<Interface>([=]() [[intel::kernel_args_restrict]]
		{
			// kData_ptr has been created using malloc_shared, but we pretend it's a device_ptr
			sycl::ext::intel::device_ptr<const kernelData> kData_ptr_casted(kData_ptr);
			kernelData kData = *kData_ptr_casted;

			InterfaceToDistributorData distributorData;
			distributorData.numberOfQueries = kData.numberOfQueries;
			distributorData.queries_ptr = (char*)queries_ptr;
			distributorData.querySizes_ptr = (HostSizeType*)querySizes_ptr;

			InterfaceToDistributorPipe::write(distributorData);

			InterfaceToMinimizerPipe::write(kData.numberOfQueries);

			InterfaceToIBFData ibfData;
			ibfData.kData = kData;
			ibfData.thresholds_ptr = (HostSizeType*)thresholds_ptr;
			ibfData.ibfData_ptr = (Chunk*)ibfData_ptr;

			InterfaceToIBFPipe::write(ibfData);

			InterfaceToCollectorData collectorData;
			collectorData.numberOfQueries = kData.numberOfQueries;
			collectorData.result_ptr = (Chunk*)result_ptr;

			InterfaceToCollectorPipe::write(collectorData);

			// Wait for the collector to finish
			CollectorToInterfacePipe::read();
		});
	}) );

	#include "distributor.cpp"
	#include "kernel_minimizer.cpp"
	#include "kernel_ibf.cpp"
	#include "collector.cpp"

	fpga_tools::Autorun<Distributor> d_kernel{device_selector, Distributor{}};
	fpga_tools::Autorun<MinimizerKernel> m_kernel{device_selector, MinimizerKernel{}};
	fpga_tools::Autorun<IbfKernel> ibf_kernel{device_selector, IbfKernel{}};
	fpga_tools::Autorun<Collector> c_kernel{device_selector, Collector{}};
}

} // namespace min_ibf_fpga::backend_sycl
