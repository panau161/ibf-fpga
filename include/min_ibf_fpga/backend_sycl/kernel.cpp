#include <sycl/ext/intel/fpga_extensions.hpp>

// Utilities
#include "autorun.hpp"
#include "pipe_utils.hpp"

// Kernel includes
#include "kernel.hpp"
#include "collector.hpp"
#include "distributor.hpp"
#include "kernel_ibf.hpp"
#include "kernel_minimizer.hpp"

namespace min_ibf_fpga::backend_sycl
{

// Forward declaration of the kernel names. FPGA best practice to reduce compiler name mangling in the optimization reports.
class InterfaceID;
class DistributorID;
template <std::size_t id> class MinimizerKernelID;
template <std::size_t id> class IbfKernelID;
class CollectorID;

void RunKernel(sycl::queue& queue,
	const char* queries_ptr,
	const HostSizeType* querySizes_ptr,
	const Chunk* ibfData_ptr,
	const HostSizeType* thresholds_ptr,
	kernelData* kData_ptr,
	Chunk* result_ptr,
	std::vector<sycl::event>& kernelEvents)
{
	kernelEvents.push_back( queue.submit([&](sycl::handler &handler)
	{
		handler.single_task<InterfaceID>([=]() [[intel::kernel_args_restrict]]
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

	fpga_tools::Autorun<DistributorID> d_kernel{device_selector, Distributor{}};
	fpga_tools::Autorun<MinimizerKernelID<id>> m_kernel{device_selector, MinimizerKernel{}};
	fpga_tools::Autorun<IbfKernelID<id>> ibf_kernel{device_selector, IbfKernel{}};
	fpga_tools::Autorun<CollectorID> c_kernel{device_selector, Collector{}};
}

} // namespace min_ibf_fpga::backend_sycl
