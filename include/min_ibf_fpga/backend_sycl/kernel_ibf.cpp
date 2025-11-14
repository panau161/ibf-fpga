struct IbfKernel
{
	void operator()() const
	{
		InterfaceToIBFData ibfData = InterfaceToIBFPipe::read();

		sycl::ext::intel::device_ptr<const HostSizeType> thresholds_ptr_casted(ibfData.thresholds_ptr);
		sycl::ext::intel::device_ptr<const Chunk> ibfData_ptr_casted(ibfData.ibfData_ptr);

		HostSizeType thresholds[THRESHOLDS_CACHE_SIZE];

		const HostSizeType thresholdsMaxIndex = ibfData.kData.maximalNumberOfMinimizers - ibfData.kData.minimalNumberOfMinimizers;

		for (ushort i = 0; i <= thresholdsMaxIndex; i++)
		{
			thresholds[i] = thresholds_ptr_casted[i];
		}

		for (QueryIndex queryIndex = 0; queryIndex < ibfData.kData.numberOfQueries; queryIndex++)
		{
			[[intel::fpga_register]] Counter counters[CHUNKS][CHUNK_BITS];

			bool countersInitialized = 0;

			QueryIndex numberOfHashes = 0;

			MinimizerToIBFData data;

			do
			{
				data = MinimizerToIBFPipes::PipeAt<id>::read();

				const QueryIndex localNumberOfHashes = ++numberOfHashes;

				Counter threshold = 0;

				if (data.isLastElement)
				{
					threshold = getThreshold(localNumberOfHashes, ibfData.kData.minimalNumberOfMinimizers, ibfData.kData.maximalNumberOfMinimizers, thresholds);
				}

				HostSizeType binOffsets[HASH_COUNT];

				#pragma unroll
				for (unsigned char seedIndex = 0; seedIndex < HASH_COUNT; ++seedIndex)
				{
					binOffsets[seedIndex] = calculateBinIndex(data.hash, seedIndex, ibfData.kData.hashShift, ibfData.kData.binSize) * CHUNKS_PER_BIN;
				}

				for (unsigned char chunkIndex = 0; chunkIndex < CHUNKS; chunkIndex++)
				{
					[[intel::fpga_register]] Chunk bitvector = ~(Chunk)0;
					[[intel::fpga_register]] Chunk localResult = 0;

					// Unroll: Burst-coalesced over chunks per seed
					#pragma unroll
					for (unsigned char seedIndex = 0; seedIndex < HASH_COUNT; ++seedIndex)
					{
						bitvector &= //__burst_coalesced_cached_load(
							/*&*/ibfData_ptr_casted[static_cast<size_t>(binOffsets[seedIndex]) + chunkIndex];//,
							//1048576); // 1 MiB = 8 megabit
							//65536); // 65536 byte = 512 kilobit (default)
					}

					#pragma unroll
					for (ushort bitOffset = 0; bitOffset < CHUNK_BITS; ++bitOffset)
					{
						const Counter counter =
							// Avoid additional port for init
							(!countersInitialized? 0 : counters[chunkIndex][bitOffset])
							+ bitvector[bitOffset];

						counters[chunkIndex][bitOffset] = counter;

						localResult[bitOffset] = counter >= threshold;
					}

					if (data.isLastElement)
					{
						CollectorPipes::PipeAt<id>::write(localResult);
					}
				}

				countersInitialized = 1;
			}
			while(!data.isLastElement);
		}
	}
};
