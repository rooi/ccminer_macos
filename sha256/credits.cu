/* creditscoin SHA256 djm34 implementation - 2015 */

extern "C" {
#include "sph/sph_types.h"
#include "sph/sph_sha2.h"
}

#include "miner.h"
#include "cuda_helper.h"

static uint32_t *d_hash[MAX_GPUS] ;

extern void bitcredit_setBlockTarget(uint32_t * data,const uint32_t * midstate, const void *ptarget);
extern void bitcredit_cpu_init(uint32_t thr_id, int threads, uint32_t* hash);

extern uint32_t bitcredit_cpu_hash(uint32_t thr_id, int threads, uint32_t startNounce, int order);

// CPU HASH
void creditshash(void *state, const void *input)
{
	uint32_t hash[8], hash2[8];
	sph_sha256_context sha1,sha2;

	sph_sha256_init(&sha1);
	sph_sha256(&sha1, input, 168);
	sph_sha256_close(&sha1, hash);

	sph_sha256_init(&sha2);
	sph_sha256(&sha2, hash, 32);
	sph_sha256_close(&sha2, hash2);

	memcpy(state, hash2, 32);
}

static bool init[MAX_GPUS] = { 0 };

extern "C" int scanhash_credits(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
	const uint32_t *midstate, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t endiandata[42];
	const uint32_t first_nonce = pdata[35];

	uint32_t throughput = device_intensity(thr_id, __func__, 1U << 25); // 256*256*64*8
	throughput = min(throughput, (max_nonce - first_nonce));

	if (opt_benchmark)
		((uint32_t*)ptarget)[7] = 0x0000ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		bitcredit_cpu_init(thr_id, throughput, d_hash[thr_id]);

		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], throughput * 32));

		init[thr_id] = true;
	}

	for (int k = 0; k < 42; k++)
		be32enc(&endiandata[k], ((uint32_t*)pdata)[k]);

	bitcredit_setBlockTarget(pdata,midstate,ptarget);

	// uint64_t nloop = max_nonce/throughput + 1;

	do {
		int order = 0;
		uint32_t foundNonce = bitcredit_cpu_hash(thr_id, throughput, pdata[35], order++);
		if (foundNonce != UINT32_MAX)
		{
			pdata[35] = foundNonce;
			*hashes_done = foundNonce - first_nonce + 1;
			return 1;
		}
		if ((uint64_t)pdata[35] + throughput > UINT32_MAX) {
			pdata[35]=  UINT32_MAX;
			*hashes_done = pdata[35] - first_nonce + 1; return 0;
		} else {
			pdata[35] += throughput;
		}

	} while (pdata[19] < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = pdata[35] - first_nonce + 1;
	return 0;
}
