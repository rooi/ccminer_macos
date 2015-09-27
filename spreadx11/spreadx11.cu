/**
 * From the SpreadX11 spreadminer Cuda Implementation by tsiv
 *
 * Adapted to common ccminer by tpruvot@github - 2015
 */

#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include <openssl/bn.h>
#include <openssl/sha.h>
}

#include "miner.h"
#include "spreadx11.h"
#include "cuda_helper.h"

static uint32_t *d_sha256hash[MAX_GPUS];
static uint32_t *d_signature[MAX_GPUS];
static uint32_t *d_hashwholeblock[MAX_GPUS];
static uint32_t *d_hash[MAX_GPUS];
static uint32_t *d_wholeblockdata[MAX_GPUS];

extern void spreadx11_sha256double_cpu_hash_88(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash);
extern void spreadx11_sha256double_setBlock_88(void *data);
extern void spreadx11_sha256_cpu_init( int thr_id, uint32_t threads);

extern void spreadx11_sha256_cpu_hash_wholeblock(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_wholeblock);
extern void spreadx11_sha256_setBlock_wholeblock(struct work *work, uint32_t *d_wholeblock);

extern void spreadx11_sign_cpu_init(int thr_id, uint32_t threads);
extern void spreadx11_sign_cpu_setInput(struct work *work);
extern void spreadx11_sign_cpu(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature);

extern void blake_cpu_init(int thr_id, uint32_t threads);
extern void blake_cpu_setBlock_185(void *pdata);
extern void blake_cpu_hash_185(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_hashwholeblock);

extern void quark_bmw512_cpu_init(int thr_id, uint32_t threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_groestl512_cpu_init(int thr_id, uint32_t threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, uint32_t threads);
extern void quark_skein512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_keccak512_cpu_init(int thr_id, uint32_t threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_jh512_cpu_init(int thr_id, uint32_t threads);
extern void quark_jh512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, uint32_t threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_cubehash512_cpu_init(int thr_id, uint32_t threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_shavite512_cpu_init(int thr_id, uint32_t threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_simd512_cpu_init(int thr_id, uint32_t threads);
extern void x11_simd512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

//extern void spreadx11_echo512_cpu_init(int thr_id, uint32_t threads);
//extern void spreadx11_echo512_cpu_setTarget(const void *ptarget);
//extern uint32_t spreadx11_echo512_cpu_hash_64_final(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, uint32_t *hashidx);

extern void x11_echo512_cpu_init(int thr_id, uint32_t threads);
extern void x11_echo512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

#define PROFILE 0
#if PROFILE == 1
#define PRINTTIME(s) do { \
	double duration; \
	gettimeofday(&tv_end, NULL); \
	duration = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec); \
	printf("%s: %.2f ms, %.2f MH/s\n", s, duration*1000.0, (double)throughput / 1000000.0 / duration); \
	} while(0)
#define TRACE(algo) { \
	if (nonce == first_nonce) { \
		uint32_t* debugbuf = NULL; \
		cudaMallocHost(&debugbuf, 8*sizeof(uint32_t)); \
		cudaMemcpy(debugbuf, d_hash[thr_id], 32, cudaMemcpyDeviceToHost); \
		printf(" %s %08x %08x %08x %08x...\n", algo, swab32(debugbuf[0]), swab32(debugbuf[1]), \
			swab32(debugbuf[2]), swab32(debugbuf[3])); \
		cudaFreeHost(debugbuf); \
	} \
}
#else
#define PRINTTIME(s)
#define TRACE(s)
#endif

void hextobin(unsigned char *p, const char *hexstr, size_t len)
{
	char hex_byte[3];
	char *ep;

	hex_byte[2] = '\0';

	while (*hexstr && len) {
		if (!hexstr[1]) {
			applog(LOG_ERR, "hex2bin str truncated");
			return;
		}
		hex_byte[0] = hexstr[0];
		hex_byte[1] = hexstr[1];
		*p = (unsigned char) strtol(hex_byte, &ep, 16);
		if (*ep) {
			applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
			return;
		}
		p++;
		hexstr += 2;
		len--;
	}
}

static void spreadx11_hash(void *output, struct work *work, uint32_t nonce)
{
	SHA256_CTX ctx_sha;
	sph_blake512_context ctx_blake;
	sph_bmw512_context ctx_bmw;
	sph_groestl512_context ctx_groestl;
	sph_jh512_context ctx_jh;
	sph_keccak512_context ctx_keccak;
	sph_skein512_context ctx_skein;
	sph_luffa512_context ctx_luffa;
	sph_cubehash512_context ctx_cubehash;
	sph_shavite512_context ctx_shavite;
	sph_simd512_context ctx_simd;
	sph_echo512_context ctx_echo;

	unsigned char mod[32] = {
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
		0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36, 0x41, 0x41
	};
	unsigned char tmp[185];
	unsigned char hashforsig[32];
	unsigned char wholeblock[MAX_BLOCK_SIZE];
	unsigned char finalhash[64];
	BIGNUM *bn_hash, *bn_privkey, *bn_kinv, *bn_mod, *bn_res;
	BN_CTX *bn_ctx;

	struct spreadwork* swork = (struct spreadwork*) work->extradata;

	uint32_t *nonceptr = (uint32_t *)&tmp[84];
	unsigned char *hashwholeblockptr = &tmp[88];
	unsigned char *signatureptr = &tmp[153];

	memcpy(tmp, swork->data, 185);
	nonceptr[0] = nonce & 0xffffffc0;

	SHA256_Init(&ctx_sha);
	SHA256_Update(&ctx_sha, tmp, 88);
	SHA256_Final(hashforsig, &ctx_sha);
	SHA256_Init(&ctx_sha);
	SHA256_Update(&ctx_sha, hashforsig, 32);
	SHA256_Final(hashforsig, &ctx_sha);

	bn_ctx = BN_CTX_new();
	bn_hash = BN_new();
	bn_privkey = BN_new();
	bn_kinv = BN_new();
	bn_mod = BN_new();
	bn_res = BN_new();

	BN_bin2bn(hashforsig, 32, bn_hash);
	BN_bin2bn(swork->privkey, 32, bn_privkey);
	BN_bin2bn(swork->kinv, 32, bn_kinv);
	BN_bin2bn(mod, 32, bn_mod);

	BN_mod_add_quick(bn_privkey, bn_privkey, bn_hash, bn_mod);
	BN_mod_mul(bn_res, bn_privkey, bn_kinv, bn_mod, bn_ctx);
	int nBitsS = BN_num_bits(bn_res);
	memset(signatureptr, 0, 32);
	BN_bn2bin(bn_res, &signatureptr[32-(nBitsS+7)/8]);

	BN_CTX_free(bn_ctx);
	BN_clear_free(bn_hash);
	BN_clear_free(bn_privkey);
	BN_clear_free(bn_kinv);
	BN_clear_free(bn_mod);
	BN_clear_free(bn_res);

	memcpy(wholeblock+0, tmp+84, 4); // nNonce
	memcpy(wholeblock+4, tmp+68, 8); // nTime
	memcpy(wholeblock+12, tmp+120, 65); // MinerSignature
	memcpy(wholeblock+77, tmp+0, 4); // nVersion
	memcpy(wholeblock+81, tmp+4, 32); // hashPrevBlock
	memcpy(wholeblock+113, tmp+36, 32); // HashMerkleRoot
	memcpy(wholeblock+145, tmp+76, 4); // nBits
	memcpy(wholeblock+149, tmp+80, 4); // nHeight
	memcpy(wholeblock+153, swork->tx, swork->txsize); // tx

	// the total amount of bytes in our data
	int blocksize = 153 + swork->txsize;

	// pad the block with 0x07 bytes to make it a multiple of uint32_t
	while( blocksize & 3 ) wholeblock[blocksize++] = 0x07;

	// figure out the offsets for the padding
	uint32_t *pFillBegin = (uint32_t*)&wholeblock[blocksize];
	uint32_t *pFillEnd = (uint32_t*)&wholeblock[MAX_BLOCK_SIZE]; // FIXME: isn't this out of bounds by one... but it seems to work out...
	uint32_t *pFillFooter = pFillBegin > pFillEnd - 8 ? pFillBegin : pFillEnd - 8;

	memcpy(pFillFooter, tmp+4, (pFillEnd - pFillFooter)*4);
	for (uint32_t *pI = pFillFooter; pI < pFillEnd; pI++)
		*pI |= 1;

	for (uint32_t *pI = pFillFooter - 1; pI >= pFillBegin; pI--)
		pI[0] = pI[3]*pI[7];

	SHA256_Init(&ctx_sha);
	SHA256_Update(&ctx_sha, wholeblock, MAX_BLOCK_SIZE);
	SHA256_Update(&ctx_sha, wholeblock, MAX_BLOCK_SIZE);
	SHA256_Final(hashwholeblockptr, &ctx_sha);
	//applog_hash(hashwholeblockptr);

	nonceptr[0] = nonce;

	sph_blake512_init(&ctx_blake);
	sph_blake512 (&ctx_blake, tmp, 185);
	sph_blake512_close(&ctx_blake, (void*) finalhash);
	//applog_hash(finalhash);

	sph_bmw512_init(&ctx_bmw);
	sph_bmw512 (&ctx_bmw, (const void*) finalhash, 64);
	sph_bmw512_close(&ctx_bmw, (void*) finalhash);

	sph_groestl512_init(&ctx_groestl);
	sph_groestl512 (&ctx_groestl, (const void*) finalhash, 64);
	sph_groestl512_close(&ctx_groestl, (void*) finalhash);

	sph_skein512_init(&ctx_skein);
	sph_skein512 (&ctx_skein, (const void*) finalhash, 64);
	sph_skein512_close(&ctx_skein, (void*) finalhash);

	sph_jh512_init(&ctx_jh);
	sph_jh512 (&ctx_jh, (const void*) finalhash, 64);
	sph_jh512_close(&ctx_jh, (void*) finalhash);

	sph_keccak512_init(&ctx_keccak);
	sph_keccak512 (&ctx_keccak, (const void*) finalhash, 64);
	sph_keccak512_close(&ctx_keccak, (void*) finalhash);

	sph_luffa512_init(&ctx_luffa);
	sph_luffa512 (&ctx_luffa, (const void*) finalhash, 64);
	sph_luffa512_close (&ctx_luffa, (void*) finalhash);

	sph_cubehash512_init(&ctx_cubehash);
	sph_cubehash512 (&ctx_cubehash, (const void*) finalhash, 64);
	sph_cubehash512_close(&ctx_cubehash, (void*) finalhash);

	sph_shavite512_init(&ctx_shavite);
	sph_shavite512 (&ctx_shavite, (const void*) finalhash, 64);
	sph_shavite512_close(&ctx_shavite, (void*) finalhash);

	sph_simd512_init(&ctx_simd);
	sph_simd512 (&ctx_simd, (const void*) finalhash, 64);
	sph_simd512_close(&ctx_simd, (void*) finalhash);

	sph_echo512_init(&ctx_echo);
	sph_echo512 (&ctx_echo, (const void*) finalhash, 64);
	sph_echo512_close(&ctx_echo, (void*) finalhash);

	memcpy(output, finalhash, 32);
}

// cpu hash only used as dev test (--cputest)
extern "C" void spreadhash(void *output, const void *input)
{
	struct work work;
	struct spreadwork swork;
	memset(&work, 0, sizeof(struct work));
	memset(&swork, 0, sizeof(struct spreadwork));
	memcpy(swork.data, input, 128); // 185 but made for a simple empty
	const char hextest[371] = { 0 };
	hex2bin(swork.data, &hextest[0], 185);
	work.extradata = (void*) &swork;
	uint32_t *nonceptr = (uint32_t*) (&swork.data[84]);
	spreadx11_hash(output, &work, nonceptr[0]);
}


static bool init[MAX_GPUS] = { 0 };

// GPU Search
extern "C" int scanhash_spreadx11(int thr_id, struct work* work, uint32_t max_nonce, unsigned long *hashes_done)
{
	int dev_id = device_map[thr_id % MAX_GPUS];

	// multiple of 64 to keep things simple with signatures
	//const uint32_t throughput = opt_throughput * 1024 * 64;

	uint32_t *ptarget = work->target;

	struct spreadwork* swork = (struct spreadwork*) work->extradata;
	if (!swork) {
		work->extradata = swork = (struct spreadwork*) calloc(1, sizeof(struct spreadwork));
		applog(LOG_ERR, "GPU #%d: missing block data!", dev_id);
		sleep(1);
		//return -1;
	}

	unsigned char *blocktemplate = swork->data;
	uint32_t *pnonce = (uint32_t*) (&blocktemplate[84]);
	uint32_t nonce = (*pnonce);
	uint32_t first_nonce = nonce;

	int intensity = (device_sm[dev_id] >= 500 && !is_windows()) ? 19 : 18;
	uint32_t throughput = device_intensity(thr_id, __func__, 1U << intensity); // 19=256*256*8;
	throughput = min(throughput, max_nonce - first_nonce) & 0xFFFFFFC0UL;

	if (opt_benchmark)
		ptarget[7] = 0x00ff;

	if (!init[thr_id])
	{
		cudaSetDevice(device_map[thr_id]);
		//cudaDeviceReset();
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		// a 512-bit buffer for every nonce to hold the x11 intermediate hashes
		CUDA_SAFE_CALL(cudaMalloc(&d_hash[thr_id], (size_t)64 * throughput));

		// sha256 hashes used for signing, 32 bytes for every 64 nonces
		cudaMalloc(&d_sha256hash[thr_id], (size_t)32*(throughput>>6));
		// changing part of MinerSignature, 32 bytes for every 64 nonces
		cudaMalloc(&d_signature[thr_id], (size_t)32*(throughput>>6));
		// sha256 hashes for the whole block, 32 bytes for every 64 nonces
		cudaMalloc(&d_hashwholeblock[thr_id], (size_t)32*(throughput>>6));

		spreadx11_sha256_cpu_init(thr_id, throughput);
		spreadx11_sign_cpu_init(thr_id, throughput);
		blake_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		//spreadx11_echo512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);

		cuda_check_cpu_init(thr_id, throughput);

		// single buffer to hold the padded whole block data
		CUDA_SAFE_CALL(cudaMalloc(&d_wholeblockdata[thr_id], 200000));

		init[thr_id] = true;
	}

	struct timeval tv_start;
#if PROFILE == 1
	struct timeval tv_end;
#endif

	spreadx11_sign_cpu_setInput(work);
	spreadx11_sha256_setBlock_wholeblock(work, d_wholeblockdata[thr_id]);
	spreadx11_sha256double_setBlock_88((void *)blocktemplate);
	blake_cpu_setBlock_185((void *)blocktemplate);
	//spreadx11_echo512_cpu_setTarget(ptarget);
	cuda_check_cpu_setTarget(ptarget);

	do {
		int order = 0;
		gettimeofday(&tv_start, NULL);
		spreadx11_sha256double_cpu_hash_88(thr_id, throughput>>6, nonce, d_sha256hash[thr_id]);
		//PRINTTIME("sha256 sig");

		gettimeofday(&tv_start, NULL);
		spreadx11_sign_cpu(thr_id, throughput>>6, nonce, d_sha256hash[thr_id], d_signature[thr_id]);
		//PRINTTIME("signing   ");

		gettimeofday(&tv_start, NULL);
		spreadx11_sha256_cpu_hash_wholeblock(thr_id, throughput>>6, nonce, d_hashwholeblock[thr_id], d_signature[thr_id], d_wholeblockdata[thr_id]);
		//PRINTTIME("hashwhole ");

		gettimeofday(&tv_start, NULL);
		blake_cpu_hash_185(thr_id, throughput, nonce, d_hash[thr_id], d_signature[thr_id], d_hashwholeblock[thr_id]);
		TRACE("blake     ");

		gettimeofday(&tv_start, NULL);
		quark_bmw512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("bmw       ");

		gettimeofday(&tv_start, NULL);
		quark_groestl512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("groestl   ");

		gettimeofday(&tv_start, NULL);
		quark_skein512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("skein     ");

		gettimeofday(&tv_start, NULL);
		quark_jh512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("jh        ");

		gettimeofday(&tv_start, NULL);
		quark_keccak512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("keccak    ");

		gettimeofday(&tv_start, NULL);
		x11_luffa512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("luffa     ");

		gettimeofday(&tv_start, NULL);
		x11_cubehash512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("cubehash  ");

		gettimeofday(&tv_start, NULL);
		x11_shavite512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("shavite   ");

		gettimeofday(&tv_start, NULL);
		x11_simd512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("simd      ");


		uint32_t winnerthread = 0;
		uint32_t foundNonce;

		gettimeofday(&tv_start, NULL);
		//foundNonce = spreadx11_echo512_cpu_hash_64_final(thr_id, throughput, nonce, NULL, d_hash[thr_id], &winnerthread);
		x11_echo512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
		TRACE("echo      ");

		*hashes_done = nonce - first_nonce + throughput;

		foundNonce = cuda_check_hash(thr_id, throughput, nonce, d_hash[thr_id]);
		if (foundNonce != UINT32_MAX)
		{
			uint32_t cpuhash[8], hash[8];
			char hexbuffer[MAX_BLOCK_SIZE*2] = { 0 };

			uint32_t *resnonce = (uint32_t*) (&swork->data[84]);
			uint32_t *reshashwholeblock = (uint32_t*) (&swork->data[88]);
			uint32_t *ressignature = (uint32_t*) (&swork->data[153]);

			for (uint32_t u = 0; u < swork->txsize && u < MAX_BLOCK_SIZE; u++)
				sprintf(&hexbuffer[u*2], "%02x", swork->tx[u]);

			winnerthread = (foundNonce - nonce);

			uint32_t idx64 = winnerthread >> 6;

			if (opt_debug)
				applog(LOG_DEBUG,
					"Thread %d found a solution\n"
					"First nonce : %08x\n"
					"Found nonce : %08x\n"
					"Threadidx   : %d\n"
					"Threadidx64 : %d\n"
					"VTX (%d)     : %s\n",
					dev_id, first_nonce, foundNonce, winnerthread, idx64, swork->txsize, hexbuffer);
			else applog(LOG_INFO, "GPU #%d: found a solution, nonce %08x", dev_id, foundNonce);

			*resnonce = foundNonce;
			cudaMemcpy(reshashwholeblock, d_hashwholeblock[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
			cudaMemcpy(ressignature, d_signature[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
			cudaMemcpy(hash, d_hash[thr_id] + winnerthread * 16, 32, cudaMemcpyDeviceToHost);

			if (opt_debug) {

				memset(hexbuffer, 0, sizeof(hexbuffer));
				for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)hash)[i]);
				applog(LOG_DEBUG, "Final hash 256 : %s", hexbuffer);

				memset(hexbuffer, 0, sizeof(hexbuffer));
				for( int i = 0; i < 185; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)swork->data)[i]);
				applog(LOG_DEBUG, "Submit data    : %s", hexbuffer);

				memset(hexbuffer, 0, sizeof(hexbuffer));
				for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)reshashwholeblock)[i]);
				applog(LOG_DEBUG, "HashWholeBlock : %s", hexbuffer);

				memset(hexbuffer, 0, sizeof(hexbuffer));
				for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)ressignature)[i]);
				applog(LOG_DEBUG, "MinerSignature : %s", hexbuffer);
			}

			spreadx11_hash((void *)cpuhash, work, foundNonce);

			if(cpuhash[7] == hash[7] && fulltest(hash, ptarget) ) {
				bn_store_hash_target_ratio(cpuhash, ptarget, work);
				return 1;
			} else if(cpuhash[7] != hash[7]) {
				applog(LOG_WARNING, "GPU #%d: result for nonce %08x does not validate on CPU %08x<>%08x!", dev_id, foundNonce, cpuhash[7], hash[7]);
				applog_hash((uchar*)cpuhash);
			} else {
				applog(LOG_WARNING, "GPU #%d: result for nonce %08x does not pass the full test!", dev_id, foundNonce);
			}
		}

		nonce += throughput;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	return 0;
}

// cleanup
extern "C" void free_spreadx11(int thr_id)
{
	if (!init[thr_id])
		return;

	cudaSetDevice(device_map[thr_id]);

	cudaFree(d_hash[thr_id]);

	cuda_check_cpu_free(thr_id);
	init[thr_id] = false;

	cudaDeviceSynchronize();
}