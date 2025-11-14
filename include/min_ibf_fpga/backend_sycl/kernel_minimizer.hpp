#include "kernel.hpp"

namespace min_ibf_fpga::backend_sycl
{

Element inline translateCharacterToElement(const char character)
{
	return character == 'A'? 0
		: character == 'C'? 1
		: character == 'G'? 2
		: 3;
}

Hash inline extractHash(const char* buffer) //, sycl::stream out) // DEBUG
{
	Hash kmer = 0;
	Hash kmerComplement = 0;

	#pragma unroll
	for (unsigned char elementIndex = 0; elementIndex < MIN_IBF_K; ++elementIndex)
	{
		const Element value = translateCharacterToElement(buffer[elementIndex]);

		//out << elementIndex << ": " << static_cast<unsigned>(kmer) << " - " << static_cast<unsigned>(kmerComplement) << sycl::endl;
		//	out << elementIndex << ": ";
		//for (unsigned i = 0; i < 2*MIN_IBF_K; i++)
		//	out << (unsigned)kmer[i];
		//out << " - ";
		//for (unsigned i = 0; i < 2*MIN_IBF_K; i++)
		//	out << (unsigned)kmerComplement[i];
		//out << sycl::endl;

		//out << "char " << static_cast<unsigned>(buffer[elementIndex]) << " value " << static_cast<unsigned>(value) << " ~value " << static_cast<unsigned>((Element)~value) << sycl::endl;

		kmer           |= (Hash)(value) << (2 * (MIN_IBF_K - 1) - elementIndex * 2);
		kmerComplement |= (Hash)((Element)~value) << elementIndex * 2;
	}

	//out << "Extracted Hash " << static_cast<unsigned>(kmer) << " Complement " << static_cast<unsigned>(kmerComplement) << sycl::endl;
	//out << "XH ";
	//for (unsigned i = 0; i < 2*MIN_IBF_K; i++)
	//	out << (unsigned)kmer[i];
	//out << " C ";
	//for (unsigned i = 0; i < 2*MIN_IBF_K; i++)
	//	out << (unsigned)kmerComplement[i];
	//out << sycl::endl;

	kmer ^= (Hash)MINIMIZER_SEED_ADJUSTED;
	kmerComplement ^= (Hash)MINIMIZER_SEED_ADJUSTED;

	return kmer < kmerComplement? kmer : kmerComplement;
}

Minimizer inline findMinimizer(const Hash* hashBuffer)
{
	Minimizer minimizer = {~(Hash)0, 0};

	#pragma unroll
	for (unsigned char kmerIndex = 0; kmerIndex < NUMBER_OF_KMERS_PER_WINDOW; ++kmerIndex)
	{
		// Position 0 indicates leaving the window, so we add 1.
		// A new minimizer cannot leave the window in the same iteration as it is found.
		const Minimizer current = {hashBuffer[kmerIndex], static_cast<unsigned char>(kmerIndex + 1)};

		// Prefer right most minimizer
		if (current.hash <= minimizer.hash)
			minimizer = current;
	}

	return minimizer;
}

struct MinimizerKernel
{
	void operator()() const
	{
		QueryIndex numberOfQueries = InterfaceToMinimizerPipe::read();

		for (QueryIndex queryIndex = 0; queryIndex < numberOfQueries; queryIndex++)
		{
			DistributorToMinimizerData query;
			query = DistributorPipes::PipeAt<id>::read();

			const QueryIndex iterations = query.size;

			char queryBuffer[MIN_IBF_K] = {0};
			Hash hashBuffer[NUMBER_OF_KMERS_PER_WINDOW] = {0};

			// Set initial element's position to 0, so the first real element will never be skipped
			Minimizer lastMinimizer = {0, 0};

			for (QueryIndex iteration = 0; iteration <= iterations; iteration++)
			{
				// Shift register: Query buffer
				#pragma unroll
				for (unsigned char i = 0; i < MIN_IBF_K - 1; ++i)
					queryBuffer[i] = queryBuffer[i + 1];

				// Query as long as elements are left, then only do calculations (end phase)
				if (iteration < query.size)
					queryBuffer[MIN_IBF_K - 1] = query.query[iteration];

				// Shift register: hash buffer
				#pragma unroll
				for (ushort i = 0; i < NUMBER_OF_KMERS_PER_WINDOW - 1; ++i)
					hashBuffer[i] = hashBuffer[i + 1];

				hashBuffer[NUMBER_OF_KMERS_PER_WINDOW - 1] = extractHash(queryBuffer); // , out); // DEBUG

				const Minimizer minimizer = findMinimizer(hashBuffer);

				// After WINDOW_SIZE many iterations, we have to write the first minimizer.
				// Initialise lastMinimizer here such that skipMinimizer will be false for the next iteration.
				if (iteration == WINDOW_SIZE - 1)
				{
					lastMinimizer = minimizer;
				}

				// If false, a new minimizer was found
				const bool skipMinimizer = lastMinimizer.position != 0 && lastMinimizer.hash == minimizer.hash;
				// If true, we are at the last element
				const bool lastElement = iteration > iterations - 1;
						
				if (iteration >= WINDOW_SIZE && (!skipMinimizer || lastElement))
				{
					MinimizerToIBFData data;
					data.isLastElement = lastElement;
					data.hash = lastMinimizer.hash;

					MinimizerToIBFPipes::PipeAt<id>::write(data);
					lastMinimizer = minimizer;
				}
				if (lastMinimizer.position != 0)
					--lastMinimizer.position;
			}
		}
	}
};

} // namespace min_ibf_fpga::backend_sycl
