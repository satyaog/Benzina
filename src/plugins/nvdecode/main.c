/* Includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "cuviddec.h"

#include "benzina/benzina-old.h"
#include "benzina/plugins/nvdecode.h"
#include "benzina/itu/h26x.h"
#include "benzina/itu/h265.h"
#include "kernels.h"


/* Defines */



/* Data Structures Forward Declarations and Typedefs */
typedef struct timespec             TIMESPEC;
typedef struct NVDECODE_RQ          NVDECODE_RQ;
typedef struct NVDECODE_BATCH       NVDECODE_BATCH;
typedef struct NVDECODE_READ_PARAMS NVDECODE_READ_PARAMS;
typedef struct NVDECODE_CTX         NVDECODE_CTX;



/* Data Structure & Enum Definitions */

/**
 * @brief Helper Thread Status.
 */

typedef enum NVDECODE_HLP_THRD_STATUS{
	THRD_NOT_RUNNING, /* Thread hasn't been spawned. */
	THRD_SPAWNED,     /* Thread has been spawned successfully with pthread_create(). */
	THRD_INITED,      /* Thread initialized successfully, waiting for others to
	                     do so as well. */
	THRD_RUNNING,     /* Thread running. */
	THRD_EXITING,     /* Thread is exiting. */
} NVDECODE_HLP_THRD_STATUS;


/**
 * @brief Context Status.
 */

typedef enum NVDECODE_CTX_STATUS{
	CTX_HELPERS_NOT_RUNNING, /* Context's helper threads are all not running. */
	CTX_HELPERS_RUNNING,     /* Context's helper threads are all running normally. */
	CTX_HELPERS_EXITING,     /* Context's helper threads are being asked to exit,
	                            or have begun doing so. */
	CTX_HELPERS_JOINING,     /* Context's helper threads have exited, but must
	                            still be joined. */
} NVDECODE_CTX_STATUS;


/**
 * @brief A structure containing the parameters and status of an individual
 *        request for image loading.
 */

struct NVDECODE_RQ{
	NVDECODE_BATCH* batch;             /* Batch to which this request belongs. */
	uint64_t        datasetIndex;      /* Dataset index. */
	float*          devPtr;            /* Target destination on device. */
	uint64_t        location[2];       /* Image payload location. */
	uint64_t        config_location[2];/* Video configuration offset and length. */
	float           H[3][3];           /* Homography */
	float           B   [3];           /* Bias */
	float           S   [3];           /* Scale */
	float           OOB [3];           /* Out-of-bond color */
	uint32_t        colorMatrix;       /* Color matrix selection */
	uint8_t*        data;              /* Image payload; */
	uint8_t*        hvcCData;          /* hvcC payload; */
	CUVIDPICPARAMS* picParams;         /* Picture parameters. */
	TIMESPEC        T_s_submit;        /* Time this request was submitted. */
	TIMESPEC        T_s_start;         /* Time this request began processing. */
	TIMESPEC        T_s_read;          /* Time required for reading. */
	TIMESPEC        T_s_decode;        /* Time required for decoding. */
	TIMESPEC        T_s_postproc;      /* Time required for postprocessing. */
};

/**
 * @brief A structure containing batch status data.
 */

struct NVDECODE_BATCH{
	NVDECODE_CTX*   ctx;
	uint64_t        startIndex;  /* Number of first sample submitted. */
	uint64_t        stopIndex;   /*  */
	const void*     token;
	TIMESPEC        T_s_submit;  /* Time this request was submitted. */
};

/**
 * @brief A structure containing the parameters for a disk read.
 */

struct NVDECODE_READ_PARAMS{
	int     fd;
	size_t  off;
	size_t  len;
	void*   ptr;
	ssize_t lenRead;
};

/**
 * @brief The NVDECODE context struct.
 * 
 * Terminology:
 * 
 *   - Context: This structure. Manages a pipeline of image decoding.
 *   - Job:     A unit of work comprising a compressed image read, its decoding
 *              and postprocessing.
 *   - Batch:   A group of jobs.
 *   - Lock:    The context's Big Lock, controlling access to everything.
 *              Must NOT be held more than momentarily.
 */

struct NVDECODE_CTX{
	/**
	 * All-important dataset
	 */
	
	const BENZINA_DATASET* dataset;
	const char*            datasetFile;
	size_t                 datasetLen;
	int                    datasetFd;
	
	/**
	 * Reference Count
	 */
	
	uint64_t        refCnt;
	
	/**
	 * Threaded Pipeline.
	 */
	
	pthread_mutex_t lock;
	struct{
		NVDECODE_CTX_STATUS status;
		uint64_t lifecycle;
		struct{
			uint64_t batch;
			uint64_t token;
			uint64_t sample;
		} push;
		struct{
			uint64_t batch;
			uint64_t token;
			uint64_t sample;
		} pull;
		pthread_cond_t cond;
	} master;
	struct{
		NVDECODE_HLP_THRD_STATUS status;
		int err;
		uint64_t cnt;/* # of compressed images previously read from dataset. */
		pthread_t thrd;
		pthread_cond_t cond;
	} reader;
	struct{
		NVDECODE_HLP_THRD_STATUS status;
		int err;
		uint64_t cnt;/* # of compressed images previously pushed into decoder. */
		pthread_t thrd;
		pthread_cond_t cond;
	} feeder;
	struct{
		NVDECODE_HLP_THRD_STATUS status;
		int err;
		uint64_t cnt;/* # of decompressed images previously pulled out of decoder. */
		pthread_t thrd;
		pthread_cond_t cond;
		cudaStream_t cudaStream;
	} worker;
	
	/* Tensor geometry */
	int      deviceOrdinal;
	void*    outputPtr;
	uint64_t multibuffering;
	uint64_t batchSize;
	uint64_t totalSlots;
	uint64_t outputHeight;
	uint64_t outputWidth;
	
	/* Default image transform parameters */
	struct{
		float    B  [3];/* Bias */
		float    S  [3];/* Scale */
		float    OOB[3];/* Out-of-bond color. */
		uint32_t colorMatrix;
	} defaults;
	
	/* NVDECODE state */
	CUvideodecoder           decoder;
	uint32_t                 decoderInited;
	uint32_t                 decoderRefCnt;
	CUVIDDECODECAPS          decoderCaps;
	CUVIDDECODECREATEINFO    decoderInfo;
	CUVIDPICPARAMS*          picParams;
	uint64_t                 picParamTruncLen;
	uint32_t                 mallocRefCnt;
	NVDECODE_BATCH*          batch;
	NVDECODE_RQ*             request;
};



/* Static Function Prototypes */
BENZINA_PLUGIN_STATIC const void* nvdecodeReturnAndClear          (const void**  ptr);
BENZINA_PLUGIN_STATIC int         nvdecodeTimeMonotonic           (TIMESPEC*     t);
BENZINA_PLUGIN_STATIC int         nvdecodeTimeAdd                 (TIMESPEC*     t, const TIMESPEC* a, const TIMESPEC* b);
BENZINA_PLUGIN_STATIC int         nvdecodeTimeSub                 (TIMESPEC*     t, const TIMESPEC* a, const TIMESPEC* b);
BENZINA_PLUGIN_STATIC double      nvdecodeTimeToDouble            (const TIMESPEC* t);
BENZINA_PLUGIN_STATIC void        nvdecodeDoubleToTime            (TIMESPEC*     t, double d);
BENZINA_PLUGIN_STATIC int         nvdecodeSameLifecycle           (NVDECODE_CTX* ctx, uint64_t lifecycle);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmRq     (NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetRetrBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_CTX_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdAwaitShutdown (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStart            (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStop             (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersJoin             (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAllStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAnyStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExitNow    (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExit       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdInit          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdAwaitAll      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdContinue      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdHasWork       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdWait          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdCore          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillDataRd    (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillConfigRd  (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd);
BENZINA_PLUGIN_STATIC void*       nvdecodeReaderThrdMain          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdInit          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdAwaitAll      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdContinue      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdHasWork       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdWait          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdCore          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void*       nvdecodeFeederThrdMain          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdInit          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdAwaitAll      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdContinue      (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdHasWork       (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdWait          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdCore          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void*       nvdecodeWorkerThrdMain          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC void        nvdecodeWorkerThrdCallback      (cudaStream_t  stream,
                                                                   cudaError_t   status,
                                                                   NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status);
BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut);
BENZINA_PLUGIN_STATIC int         nvdecodeSetDevice               (NVDECODE_CTX* ctx, const char* deviceId);
BENZINA_PLUGIN_STATIC int         nvdecodeAllocDataOpen           (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeAllocThreading          (NVDECODE_CTX* ctx);
BENZINA_PLUGIN_STATIC int         nvdecodeAllocCleanup            (NVDECODE_CTX* ctx, int ret);



/* Static Function Definitions */

/**
 * @brief Read pointer at the specified location, return it and clear its source.
 * @param [in]  ptrPtr  The pointer to the pointer to be read, returned and cleared.
 * @return *ptrPtr
 */

BENZINA_PLUGIN_STATIC const void* nvdecodeReturnAndClear          (const void**  ptrPtr){
	const void* ptr = *ptrPtr;
	*ptrPtr = NULL;
	return ptr;
}

/**
 * @brief Get current monotonic time using high-resolution counter.
 * 
 * Monotonic time is unsettable and always-increasing (monotonic), but it may tick
 * slightly faster than or slower than 1s/s if a clock-slewing time adjustment is
 * in progress (such as commanded by adjtime() or NTP).
 * 
 * @param [out] t  The current monotonic time.
 * @return The return code from clock_gettime(CLOCK_MONOTONIC, t).
 */

BENZINA_PLUGIN_STATIC int         nvdecodeTimeMonotonic           (TIMESPEC* t){
	return clock_gettime(CLOCK_MONOTONIC, t);
}

/**
 * @brief Add times a and b together and store sum into t.
 * 
 * The output t is normalized such that tv_nsec is always in [0, 1e9), but the
 * tv_sec field is unconstrained.
 * 
 * t,a,b may all alias each other.
 * 
 * @param [out] t = a+b
 * @param [in]  a
 * @param [in]  b
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeTimeAdd                 (TIMESPEC*       t,
                                                                   const TIMESPEC* a,
                                                                   const TIMESPEC* b){
	TIMESPEC an, bn, d;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	an.tv_sec  = a->tv_nsec/GIGA;
	bn.tv_sec  = b->tv_nsec/GIGA;
	an.tv_nsec = a->tv_nsec - an.tv_sec*GIGA;
	bn.tv_nsec = b->tv_nsec - bn.tv_sec*GIGA;
	d.tv_sec   = a->tv_sec + an.tv_sec + b->tv_sec + bn.tv_sec;
	d.tv_nsec  = an.tv_nsec            + bn.tv_nsec;
	while(d.tv_nsec < 0){
		d.tv_sec  -=    1;
		d.tv_nsec += GIGA;
	}
	while(d.tv_nsec >= GIGA){
		d.tv_sec  +=    1;
		d.tv_nsec -= GIGA;
	}
	*t = d;
	
	return 0;
}

/**
 * @brief Subtract time b from a and store difference into t.
 * 
 * The output t is normalized such that tv_nsec is always in [0, 1e9), but the
 * tv_sec field is unconstrained.
 * 
 * t,a,b may all alias each other.
 * 
 * @param [out] t = a-b
 * @param [in]  a
 * @param [in]  b
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeTimeSub                 (TIMESPEC*       t,
                                                                   const TIMESPEC* a,
                                                                   const TIMESPEC* b){
	TIMESPEC an, bn, d;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	an.tv_sec  = a->tv_nsec/GIGA;
	bn.tv_sec  = b->tv_nsec/GIGA;
	an.tv_nsec = a->tv_nsec - an.tv_sec*GIGA;
	bn.tv_nsec = b->tv_nsec - bn.tv_sec*GIGA;
	d.tv_sec   = a->tv_sec + an.tv_sec - b->tv_sec - bn.tv_sec;
	d.tv_nsec  = an.tv_nsec            - bn.tv_nsec;
	while(d.tv_nsec < 0){
		d.tv_sec  -=    1;
		d.tv_nsec += GIGA;
	}
	while(d.tv_nsec >= GIGA){
		d.tv_sec  +=    1;
		d.tv_nsec -= GIGA;
	}
	*t = d;
	
	return 0;
}

/**
 * @brief Convert time to double.
 * @param [in] t  The time or time-delta to convert.
 * @return Double-precision floating-point value, in seconds.
 */

BENZINA_PLUGIN_STATIC double      nvdecodeTimeToDouble            (const TIMESPEC* t){
	TIMESPEC d;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	d.tv_sec  = t->tv_nsec/GIGA;
	d.tv_nsec = t->tv_nsec - d.tv_sec*GIGA;
	d.tv_sec += t->tv_sec;
	
	while(d.tv_nsec < 0){
		d.tv_sec  -=    1;
		d.tv_nsec += GIGA;
	}
	
	/**
	 * The following code ensures that positive and negative times of equal magnitude
	 * render to the same-magnitude but oppositive-sign double-precision floating-point
	 * number, even after being canonicalized to d.tv_nsec in [0, 1e9). Otherwise,
	 * unpleasant surprises might occur when comparing such times.
	 */
	
	if(d.tv_sec < 0 && d.tv_nsec != 0){
		d.tv_nsec = GIGA - d.tv_nsec;
		d.tv_sec  =   -1 - d.tv_sec;
		return -d.tv_sec - 1e-9*d.tv_nsec;
	}else{
		return +d.tv_sec + 1e-9*d.tv_nsec;
	}
}

/**
 * @brief Convert double to time.
 * @param [out] t  The output time.
 * @param [in]  d  The double to convert.
 */

BENZINA_PLUGIN_STATIC void        nvdecodeDoubleToTime            (TIMESPEC* t, double d){
	double i=floor(d), f=d-i;
	const int64_t GIGA = (int64_t)1*1000*1000*1000;
	
	t->tv_nsec = GIGA*f;
	t->tv_sec  = i;
	if(t->tv_nsec >= GIGA){
		t->tv_nsec -= GIGA;
		t->tv_sec  +=    1;
	}
}

/**
 * @brief Are we still on the same lifecycle?
 * @param [in]  ctx
 * @param [in]  lifecycle
 * @return !0 if given lifecycle matches current one, 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeSameLifecycle           (NVDECODE_CTX* ctx, uint64_t lifecycle){
	return ctx->master.lifecycle == lifecycle;
}

/**
 * @brief 
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @param [in]  batchOut
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.push.batch % ctx->multibuffering];
	return 0;
}

/**
 * @brief 
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @param [in]  batchOut
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetRetrBt     (NVDECODE_CTX* ctx, NVDECODE_BATCH** batchOut){
	*batchOut = &ctx->batch[ctx->master.pull.batch % ctx->multibuffering];
	return 0;
}

/**
 * @brief Set master thread status.
 * @param [in]  ctx     The context in question.
 * @param [in]  status  The new status.
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_CTX_STATUS status){
	ctx->master.status = status;
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Wait for context to reach shutdown.
 * 
 * Called with the lock held. May release and reacquire lock.
 * 
 * @param [in]  ctx
 * @return 0 if desired status reached with no intervening helper thread lifecycle.
 *         !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdAwaitShutdown (NVDECODE_CTX* ctx){
	uint64_t lifecycle = ctx->master.lifecycle;
	do{
		if(!nvdecodeSameLifecycle(ctx, lifecycle)){
			return -1;
		}
		if(ctx->master.status == CTX_HELPERS_JOINING){
			nvdecodeHelpersJoin(ctx);
		}
		if(ctx->master.status == CTX_HELPERS_NOT_RUNNING){
			return 0;
		}
	}while(pthread_cond_wait(&ctx->master.cond, &ctx->lock) == 0);
	return -3;
}

/**
 * @brief 
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @param [in]  rqOut
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMasterThrdGetSubmRq     (NVDECODE_CTX* ctx, NVDECODE_RQ**    rqOut){
	*rqOut = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	return 0;
}

/**
 * @brief Launch helper threads.
 * 
 * Called with the lock held. Must not be called from the helper threads.
 * 
 * @param [in]  ctx
 * @return 0 if threads already running or started successfully.
 *         !0 if threads exiting, or were not running and could not be started.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStart            (NVDECODE_CTX* ctx){
	uint64_t       i;
	pthread_attr_t attr;
	sigset_t       oldset, allblockedset;
	
	switch(ctx->master.status){
		case CTX_HELPERS_NOT_RUNNING: break;
		case CTX_HELPERS_JOINING:     nvdecodeHelpersJoin(ctx); break;
		case CTX_HELPERS_EXITING:     return -1;
		case CTX_HELPERS_RUNNING:     return  0;
	}
	
	if(ctx->reader.err || ctx->feeder.err || ctx->worker.err){
		return -1;
	}
	
	memset(ctx->batch,   0, sizeof(*ctx->batch)   * ctx->multibuffering);
	memset(ctx->request, 0, sizeof(*ctx->request) * ctx->totalSlots);
	for(i=0;i<ctx->totalSlots;i++){
		ctx->request[i].picParams = &ctx->picParams[i];
		ctx->request[i].data      = NULL;
		ctx->request[i].hvcCData  = NULL;
	}
	
	if(pthread_attr_init          (&attr)                          != 0){
		return -1;
	}
	if(pthread_attr_setstacksize  (&attr,                 64*1024) != 0 ||
	   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE) != 0){
		pthread_attr_destroy(&attr);
		return -1;
	}
	sigfillset(&allblockedset);
	pthread_sigmask(SIG_SETMASK, &allblockedset, &oldset);
	ctx->reader.status = pthread_create(&ctx->reader.thrd, &attr, (void*(*)(void*))nvdecodeReaderThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->feeder.status = pthread_create(&ctx->feeder.thrd, &attr, (void*(*)(void*))nvdecodeFeederThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	ctx->worker.status = pthread_create(&ctx->worker.thrd, &attr, (void*(*)(void*))nvdecodeWorkerThrdMain, ctx) == 0 ? THRD_SPAWNED : THRD_NOT_RUNNING;
	pthread_sigmask(SIG_SETMASK, &oldset, NULL);
	ctx->reader.err    = ctx->reader.status == THRD_NOT_RUNNING ? 1 : 0;
	ctx->feeder.err    = ctx->feeder.status == THRD_NOT_RUNNING ? 1 : 0;
	ctx->worker.err    = ctx->worker.status == THRD_NOT_RUNNING ? 1 : 0;
	pthread_attr_destroy(&attr);
	
	if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
		return -1;
	}
	ctx->master.lifecycle++;
	if(nvdecodeHelpersAllStatusIs(ctx, THRD_SPAWNED)){
		nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_RUNNING);
	}else{
		nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
	}
	
	return 0;
}

/**
 * @brief Stop helper threads.
 * 
 * Called with the lock held. Must not be called from the helper threads.
 * May release and reacquire the lock.
 * 
 * @param [in]  ctx
 * @return 0 if threads not running or successfully stopped.
 *         !0 if lifecycle changes under our feet as we wait.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersStop             (NVDECODE_CTX* ctx){
	switch(ctx->master.status){
		case CTX_HELPERS_NOT_RUNNING:
		case CTX_HELPERS_JOINING:
			return nvdecodeHelpersJoin(ctx);
		default:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			return nvdecodeMasterThrdAwaitShutdown(ctx);
		case CTX_HELPERS_EXITING:
			return nvdecodeMasterThrdAwaitShutdown(ctx);
	}
}

/**
 * @brief Join helper threads.
 * 
 * Called with the lock held. Must not be called from the helper threads.
 * On successful return, all helper threads are truly no longer running and
 * have been joined, and the context is in state NOT_RUNNING.
 * 
 * Idempotent.
 * 
 * @param [in]  ctx
 * @return 0 if threads successfully joined, or not running in first place.
 *         !0 if threads were not ready to be joined.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersJoin             (NVDECODE_CTX* ctx){
	switch(ctx->master.status){
		case CTX_HELPERS_JOINING:
			pthread_join(ctx->reader.thrd, NULL);
			pthread_join(ctx->feeder.thrd, NULL);
			pthread_join(ctx->worker.thrd, NULL);
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
			return 0;
		case CTX_HELPERS_NOT_RUNNING:
			return 0;
		default:
			return -1;
	}
}

/**
 * @brief Whether all helpers have the given status.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) all helper threads have the given status.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAllStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	return ctx->reader.status == status &&
	       ctx->feeder.status == status &&
	       ctx->worker.status == status;
}

/**
 * @brief Whether any helpers have the given status.
 * 
 * @param [in]  ctx
 * @param [in]  status
 * @return Whether (!0) or not (0) any helper threads have the given status.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersAnyStatusIs      (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	return ctx->reader.status == status ||
	       ctx->feeder.status == status ||
	       ctx->worker.status == status;
}

/**
 * @brief Whether all helpers should exit *immediately*.
 * 
 * @param [in]  ctx
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExitNow    (NVDECODE_CTX* ctx){
	return ctx->reader.err ||
	       ctx->feeder.err ||
	       ctx->worker.err;
}

/**
 * @brief Whether all helpers should exit when the pipeline is empty.
 * 
 * @param [in]  ctx
 * @param [in]  now  Whether to exit *immediately* or after a pipeline flush.
 * @return 
 */

BENZINA_PLUGIN_STATIC int         nvdecodeHelpersShouldExit       (NVDECODE_CTX* ctx){
	return ctx->master.status == CTX_HELPERS_EXITING;
}

/**
 * @brief Maybe reap leftover malloc()'s from the reader.
 * @param [in]  ctx  The context in question.
 * @return 0.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMaybeReapMallocs        (NVDECODE_CTX* ctx){
	uint64_t i;
	
	if(!--ctx->mallocRefCnt){
		for(i=0;i<ctx->totalSlots;i++){
			if(ctx->request[i].data){
				free(ctx->request[i].data);
				ctx->request[i].data = NULL;
			}
			if(ctx->request[i].hvcCData){
				free(ctx->request[i].hvcCData);
				ctx->request[i].hvcCData = NULL;
			}
		}
	}
	
	return 0;
}

/**
 * @brief Possibly destroy decoder, if no longer needed.
 * 
 * The feeder and worker threads share a decoder, but because either thread
 * may fail, the other must be ready to cleanup the decoder.
 * 
 * Called with the lock held. Will release the lock and reacquire it.
 * 
 * @param [in]  ctx  The context in question.
 * @return 0.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeMaybeReapDecoder        (NVDECODE_CTX* ctx){
	CUvideodecoder decoder = ctx->decoder;
	
	if(!--ctx->decoderRefCnt && ctx->decoderInited){
		ctx->decoderInited = 0;
		
		/**
		 * We are forced to release the lock here, because deep inside
		 * cuvidDestroyDecoder(), there is a call to cuCtxSynchronize(). If
		 * we do not release the mutex, it is possible for deadlock to occur.
		 */
		
		pthread_mutex_unlock(&ctx->lock);
		cuvidDestroyDecoder (decoder);
		pthread_mutex_lock  (&ctx->lock);
	}
	
	return 0;
}

/**
 * @brief Main routine of the reader thread.
 * 
 * Does I/O as and when jobs are submitted, asynchronously from decoder thread.
 * For every job submitted, two reads are performed:
 *   - On data.bin,      for the image data payload.
 *   - On data.nvdecode, for the precomputed decode parameters.
 * 
 * @param [in]  ctx  The decoding context.
 * @return NULL.
 */

BENZINA_PLUGIN_STATIC void*       nvdecodeReaderThrdMain          (NVDECODE_CTX* ctx){
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeReaderThrdInit(ctx)){
		while(nvdecodeReaderThrdContinue(ctx)){
			nvdecodeReaderThrdCore(ctx);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize reader thread state.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose reader thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdInit          (NVDECODE_CTX* ctx){
	if(nvdecodeHelpersShouldExitNow(ctx)){
		nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ctx->mallocRefCnt++;
	
	nvdecodeReaderThrdSetStatus(ctx, THRD_INITED);
	if(nvdecodeReaderThrdAwaitAll(ctx)){
		nvdecodeReaderThrdSetStatus(ctx, THRD_RUNNING);
		return 1;
	}else{
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		nvdecodeMaybeReapMallocs(ctx);
		nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
}

/**
 * @brief Wait for full initialization of all threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return Whether (!0) or not (0) all threads reached INITED or RUNNING state.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdAwaitAll      (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersShouldExitNow(ctx)){return 0;}
		if(!nvdecodeHelpersAnyStatusIs(ctx, THRD_SPAWNED)){return 1;}
	}while(pthread_cond_wait(&ctx->reader.cond, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Determine whether the reader thread should shut down or do more work.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdContinue      (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersShouldExitNow(ctx)){
			break;
		}
		if(!nvdecodeReaderThrdHasWork(ctx)){
			if(nvdecodeHelpersShouldExit(ctx)){
				break;
			}else{
				continue;
			}
		}
		return 1;
	}while(nvdecodeReaderThrdWait(ctx));
	
	nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
	nvdecodeMaybeReapMallocs   (ctx);
	nvdecodeReaderThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Does reader thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdHasWork       (NVDECODE_CTX* ctx){
	return ctx->reader.cnt < ctx->master.push.sample;
}

/**
 * @brief Reader Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdWait          (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->reader.cond, &ctx->lock);
	return 1;
}

/**
 * @brief Perform the core operation of the reader thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdCore          (NVDECODE_CTX* ctx){
	NVDECODE_READ_PARAMS rd0 = {0}, rd1 = {0};
	NVDECODE_RQ*         rq;
	int                  readsDone;
	
	
	/* Get read parameters */
	nvdecodeReaderThrdGetCurrRq (ctx, &rq);

	if(nvdecodeReaderThrdFillDataRd  (ctx, rq, &rd0) != 0 ||
	   nvdecodeReaderThrdFillConfigRd(ctx, rq, &rd1) != 0){
		return 0;
	}
	
	
	/* Perform reads */
	pthread_mutex_unlock(&ctx->lock);
	rd0.lenRead = pread(rd0.fd, rd0.ptr, rd0.len, rd0.off);
	rd1.lenRead = pread(rd1.fd, rd1.ptr, rd1.len, rd1.off);
	pthread_mutex_lock(&ctx->lock);
	
	
	/* Handle any I/O problems */
	readsDone = (rd0.lenRead==(ssize_t)rd0.len) &&
	            (rd1.lenRead==(ssize_t)rd1.len);
	if(!readsDone){
		free(rd0.ptr);
		free(rd1.ptr);
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	/* Otherwise, report success. */
	rq->data                         = rd0.ptr;
	rq->hvcCData                     = rd1.ptr;
	rq->picParams->nBitstreamDataLen = rd0.len;
	ctx->reader.cnt++;
	pthread_cond_broadcast(&ctx->feeder.cond);
	return 0;
}

/**
 * @brief Fill the dataset read parameters structure with the current sample's
 *        details.
 * @param [in]  ctx
 * @param [in]  rq
 * @param [out] rd
 * @return 0 if successful, !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillDataRd    (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd){
	rd->ptr = NULL;
	rd->off = rq->location[0];
	rd->len = rq->location[1];
	rd->ptr = malloc(rd->len);
	if(!rd->ptr){
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	rd->fd = ctx->datasetFd;
	return 0;
}

/**
 * @brief Fill the dataset video configuration parameters structure with the
 *        current sample's details.
 * @param [in]  ctx
 * @param [in]  rq
 * @param [out] rd
 * @return 0 if successful, !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdFillConfigRd  (NVDECODE_CTX* ctx,
                                                                   const NVDECODE_RQ*    rq,
                                                                   NVDECODE_READ_PARAMS* rd){
	rd->ptr = NULL;
	rd->off = rq->config_location[0];
	rd->len = rq->config_location[1];
	rd->ptr = malloc(rd->len);
	if(!rd->ptr){
		ctx->reader.err = 1;
		nvdecodeReaderThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	rd->fd = ctx->datasetFd;
	return 0;
}

/**
 * @brief Change reader thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose reader thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->reader.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				if(ctx->master.status == CTX_HELPERS_EXITING){
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_JOINING);
				}else{
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
				}
			}else{
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			}
		break;
		case THRD_EXITING:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
		break;
		default:
		break;
	}
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Get reader thread's current processing request.
 * @param [in]  ctx    The context in question.
 * @param [out] rqOut  The pointer to the request block.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeReaderThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
	*rqOut = &ctx->request[ctx->reader.cnt % ctx->totalSlots];
	return 0;
}

/**
 * @brief Main routine of the feeder thread.
 * 
 * Feeds the data read by the reader thread into the decoders.
 * 
 * @param [in]  ctx  The decoding context.
 * @return NULL.
 */

BENZINA_PLUGIN_STATIC void*       nvdecodeFeederThrdMain          (NVDECODE_CTX* ctx){
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeFeederThrdInit(ctx)){
		while(nvdecodeFeederThrdContinue(ctx)){
			nvdecodeFeederThrdCore(ctx);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize feeder thread state.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose feeder thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdInit          (NVDECODE_CTX* ctx){
	int ret;
	
	if(nvdecodeHelpersShouldExitNow(ctx)){
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ret = cudaSetDevice(ctx->deviceOrdinal);
	if(ret != cudaSuccess){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	memset(&ctx->decoderCaps, 0, sizeof(ctx->decoderCaps));
	ctx->decoderCaps.eCodecType      = ctx->decoderInfo.CodecType;
	ctx->decoderCaps.eChromaFormat   = ctx->decoderInfo.ChromaFormat;
	ctx->decoderCaps.nBitDepthMinus8 = ctx->decoderInfo.bitDepthMinus8;
	ret = cuvidGetDecoderCaps(&ctx->decoderCaps);
	if(ret != CUDA_SUCCESS                                      ||
	   !ctx->decoderCaps.bIsSupported                           ||
	   ctx->decoderInfo.ulWidth  < ctx->decoderCaps.nMinWidth   ||
	   ctx->decoderInfo.ulWidth  > ctx->decoderCaps.nMaxWidth   ||
	   ctx->decoderInfo.ulHeight < ctx->decoderCaps.nMinHeight  ||
	   ctx->decoderInfo.ulHeight > ctx->decoderCaps.nMaxHeight  ||
	   ((ctx->decoderInfo.ulWidth*ctx->decoderInfo.ulHeight/256) > ctx->decoderCaps.nMaxMBCount)){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	ret = cuvidCreateDecoder(&ctx->decoder, &ctx->decoderInfo);
	if(ret != CUDA_SUCCESS){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	ctx->decoderInited = 1;
	ctx->decoderRefCnt++;
	
	ctx->mallocRefCnt++;
	
	nvdecodeFeederThrdSetStatus(ctx, THRD_INITED);
	if(nvdecodeFeederThrdAwaitAll(ctx)){
		nvdecodeFeederThrdSetStatus(ctx, THRD_RUNNING);
		return 1;
	}else{
		nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
		nvdecodeMaybeReapDecoder(ctx);
		nvdecodeMaybeReapMallocs(ctx);
		nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
}

/**
 * @brief Wait for full initialization of all threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return Whether (!0) or not (0) all threads reached INITED or RUNNING state.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdAwaitAll      (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersShouldExitNow(ctx)){return 0;}
		if(!nvdecodeHelpersAnyStatusIs(ctx, THRD_SPAWNED)){return 1;}
	}while(pthread_cond_wait(&ctx->feeder.cond, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Determine whether the feeder thread should shut down or do more work.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdContinue      (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersShouldExitNow(ctx)){
			break;
		}
		if(!nvdecodeFeederThrdHasWork(ctx)){
			if(nvdecodeHelpersShouldExit(ctx)){
				break;
			}else{
				continue;
			}
		}
		return 1;
	}while(nvdecodeFeederThrdWait(ctx));
	
	/**
	 * If we are the last owners of the decoder handle, destroy it.
	 * 
	 * Normally, the feeder thread will never destroy the decoder. However, if
	 * the feeder thread spawns and initializes, but the worker thread spawns
	 * and fails to initialize, we must harvest the decoder ourselves. The
	 * reverse can also happen: The worker thread could spawn and initialize,
	 * and the feeder thread could spawn but fail to initialize. In that case,
	 * the worker thread must *not* destroy the decoder, since it wasn't
	 * initialized.
	 */
	
	nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
	nvdecodeMaybeReapMallocs   (ctx);
	nvdecodeMaybeReapDecoder   (ctx);
	nvdecodeFeederThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Does feeder thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdHasWork       (NVDECODE_CTX* ctx){
	return ctx->feeder.cnt < ctx->reader.cnt;
}

/**
 * @brief Feeder Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdWait          (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->feeder.cond, &ctx->lock);
	return 1;
}

uint16_t buf_to_uint16(const uint8_t* buffer){
    uint16_t result = 0;
    for (int i = 0; i < 4; ++i)
    {
        result |= ((uint16_t)buffer[i]) << (8 - 8 * i);
    }
    return result;
}

uint32_t buf_to_uint32(const uint8_t* buffer){
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i)
    {
        result |= ((uint32_t)buffer[i]) << (24 - 8 * i);
    }
    return result;
}

uint64_t buf_to_uint64(const uint8_t* buffer){
    uint64_t result = 0;
    for (int i = 0; i < 8; ++i)
    {
        result |= ((uint64_t)buffer[i]) << (56 - 8 * i);
    }
    return result;
}

void skip_profile_tier_level(BENZ_ITU_H26XBS* bitstream, uint8_t max_sub_layers_minus1)
{
    benz_itu_h26xbs_skip_xn(bitstream, 2);  // uint8_t general_profile_space;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool    general_tier_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t general_profile_idc;
    benz_itu_h26xbs_skip_xn(bitstream, 32); // bool    general_profile_compatibility_flag[32];
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool    general_progressive_source_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool    general_interlaced_source_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool    general_non_packed_constraint_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool    general_frame_only_constraint_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 43); // bool general_max_12bit_constraint_flag;
                                            // bool general_max_10bit_constraint_flag;
                                            // bool general_max_8bit_constraint_flag;
                                            // bool general_max_422chroma_constraint_flag;
                                            // bool general_max_420chroma_constraint_flag;
                                            // bool general_max_monochrome_constraint_flag;
                                            // bool general_intra_constraint_flag;
                                            // bool general_one_picture_only_constraint_flag;
                                            // bool general_lower_bit_rate_constraint_flag;
                                            // //uint64_t general_reserved_zero_34bits;
                                            // //uint64_t general_reserved_zero_43bits;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool general_inbld_flag;
                                            // //bool general_reserved_zero_bit;
    benz_itu_h26xbs_skip_xn(bitstream, 8);  // uint8_t general_level_idc;
    benz_itu_h26xbs_bigfill(bitstream);
    // total bits skipped: 96
    int sub_layer_profile_present_flag[6] = {0};
    int sub_layer_level_present_flag[6] = {0};
    for (int i = 0; i < max_sub_layers_minus1; i++)
    {
        sub_layer_profile_present_flag[i] = benz_itu_h26xbs_read_un(bitstream, 1);
        sub_layer_level_present_flag[i] = benz_itu_h26xbs_read_un(bitstream, 1);
    }
    if (max_sub_layers_minus1 > 0)
    {
        for (int i = max_sub_layers_minus1; i < 8; i++)
        {
            benz_itu_h26xbs_skip_xn(bitstream, 2);  //uint8_t reserved_zero_2bits[6];
        }
    }
    benz_itu_h26xbs_fill64b(bitstream);
    for (int i = 0; i < max_sub_layers_minus1; i++)
    {
        if (sub_layer_profile_present_flag[i])
        {
            benz_itu_h26xbs_skip_xn(bitstream, 2);  // uint8_t sub_layer_profile_space[6];
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_layer_tier_flag[6];
            benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t sub_layer_profile_idc[6];
            benz_itu_h26xbs_skip_xn(bitstream, 32); // bool sub_layer_profile_compatibility_flag[6][32];
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_layer_progressive_source_flag[6];
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_layer_interlaced_source_flag[6];
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_layer_non_packed_constraint_flag[6];
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_layer_frame_only_constraint_flag[6];
            benz_itu_h26xbs_skip_xn(bitstream, 43); // bool sub_layer_max_12bit_constraint_flag[6];
                                                    // bool sub_layer_max_10bit_constraint_flag[6];
                                                    // bool sub_layer_max_8bit_constraint_flag[6];
                                                    // bool sub_layer_max_422chroma_constraint_flag[6];
                                                    // bool sub_layer_max_420chroma_constraint_flag[6];
                                                    // bool sub_layer_max_monochrome_constraint_flag[6];
                                                    // bool sub_layer_intra_constraint_flag[6];
                                                    // bool sub_layer_one_picture_only_constraint_flag[6];
                                                    // bool sub_layer_lower_bit_rate_constraint_flag[6];
                                                    // //uint64_t sub_layer_reserved_zero_34bits[6];
                                                    // //uint64_t sub_layer_reserved_zero_43bits[6];
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_layer_inbld_flag[6];
                                                    // //bool sub_layer_reserved_zero_bit[6];
            benz_itu_h26xbs_bigfill(bitstream);
            // total bits skipped: 88
        }
        if (sub_layer_level_present_flag[i])
        {
            benz_itu_h26xbs_skip_xn(bitstream, 8);  // uint8_t sub_layer_level_idc[6];
            // total bits skipped: 8
        }
    }

    benz_itu_h26xbs_fill64b(bitstream);
}

void set_default_scaling_list(uint8_t* dstList, uint8_t* dstDcList, uint8_t sizeId, uint8_t matrixId)
{
    // Table 7-5
    static const uint8_t DefaultScalingList0[16] = {
        16, 16, 16, 16,
        16, 16, 16, 16,
        16, 16, 16, 16,
        16, 16, 16, 16
    };

    // Table 7-6
    static const uint8_t DefaultScalingList1[64] = {
        16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 17, 16, 17, 16, 17, 18,
        17, 18, 18, 17, 18, 21, 19, 20,
        21, 20, 19, 21, 24, 22, 22, 24,
        24, 22, 22, 24, 25, 25, 27, 30,
        27, 25, 25, 29, 31, 35, 35, 31,
        29, 36, 41, 44, 41, 36, 47, 54,
        54, 47, 65, 70, 65, 88, 88, 115
    };

    static const uint8_t DefaultScalingList2[64] = {
        16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 17, 17, 17, 17, 17, 18,
        18, 18, 18, 18, 18, 20, 20, 20,
        20, 20, 20, 20, 24, 24, 24, 24,
        24, 24, 24, 24, 25, 25, 25, 25,
        25, 25, 25, 28, 28, 28, 28, 28,
        28, 33, 33, 33, 33, 33, 41, 41,
        41, 41, 54, 54, 54, 71, 71, 91
    };

    // Table 7-3-Specification of siezId
    switch (sizeId) {
    case 0: // 4x4
        memcpy(dstList, DefaultScalingList0, 16);
        break;
    case 1: // 8x8
    case 2: // 16x16
        if (matrixId <= 2)
            memcpy(dstList, DefaultScalingList1, 64);
        else
            memcpy(dstList, DefaultScalingList2, 64);
        break;
    case 3: // 32x32
        if (!matrixId)
            memcpy(dstList, DefaultScalingList1, 64);
        else
            memcpy(dstList, DefaultScalingList2, 64);
        break;
//    default:
//        ERROR("Can't get the scaling list by sizeId(%d)", sizeId);
//        return false;
    }

    if (sizeId > 1)
        dstDcList[matrixId] = 16;

//    return true;
}

void decode_scaling_list_data(CUVIDHEVCPICPARAMS* hevcPP, BENZ_ITU_H26XBS* bitstream)
{
#ifndef benz__min
#define benz__min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#endif // benz__min

    uint8_t* dstDcList = NULL;
    uint8_t* dstList = NULL;
    uint8_t* refList = NULL;

    size_t size = 64;
    uint8_t refMatrixId = 0;
    int scaling_list_pred_mode_flag = 0;
    uint8_t scaling_list_pred_matrix_id_delta = 0;
    uint8_t nextCoef;
    uint8_t coefNum;
    int16_t scaling_list_delta_coef;

    for (uint32_t sizeId = 0; sizeId < 4; sizeId++)
    {
        for (uint32_t matrixId = 0; matrixId < 6; matrixId += (sizeId == 3) ? 3 : 1)
        {
            size = 64;
            // Table 7-3
            switch (sizeId)
            {
            case 0: // 4x4
                dstList = hevcPP->ScalingList4x4[matrixId];
                size = 16;
                break;
            case 1: // 8x8
                dstList = hevcPP->ScalingList8x8[matrixId];
                break;
            case 2: // 16x16
                dstList = hevcPP->ScalingList16x16[matrixId];
                dstDcList = hevcPP->ScalingListDCCoeff16x16;
                break;
            case 3: // 32x32
                dstList = hevcPP->ScalingList32x32[matrixId];
                dstDcList = hevcPP->ScalingListDCCoeff32x32;
            }

            scaling_list_pred_mode_flag = benz_itu_h26xbs_read_un(bitstream, 1);
            if (!scaling_list_pred_mode_flag)
            {
                if (sizeId < 3)
                {
                    scaling_list_pred_matrix_id_delta = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 6]
                }
                else if (3 == sizeId)
                {
                    // as spec "7.4.5 Scaling list data semantics",
                    // matrixId should be equal to 3 when scaling_list_pred_matrix_id_delta
                    // is greater than 0.
                    scaling_list_pred_matrix_id_delta = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 2]
                }
//                else
//                {
//                    ERROR("sizeId(%u) should be in the range of[0, 3].", sizeId);
//                    return false;
//                }

                if (!scaling_list_pred_matrix_id_delta)
                {
                    set_default_scaling_list(dstList, dstDcList, sizeId, matrixId);
                }
                else
                {
                    //7-40
                    refMatrixId = matrixId - scaling_list_pred_matrix_id_delta * (sizeId == 3 ? 3 : 1);
                    // get referrence list
                    switch (sizeId)
                    {
                    case 0: // 4x4
                        refList = hevcPP->ScalingList4x4[refMatrixId];
                        break;
                    case 1: // 8x8
                        refList = hevcPP->ScalingList8x8[refMatrixId];
                        break;
                    case 2: // 16x16
                        refList = hevcPP->ScalingList16x16[refMatrixId];
                        break;
                    case 3: // 32x32
                        refList = hevcPP->ScalingList32x32[refMatrixId];
                    }

                    for (uint32_t i = 0; i < size; i++)
                    {
                        dstList[i] = refList[i];
                    }

                    if (sizeId > 1)
                    {
                        dstDcList[matrixId] = dstDcList[refMatrixId];
                    }
                }
            }
            else
            {
                nextCoef = 8;
                coefNum = benz__min(64, (1 << (4 + (sizeId << 1))));

                if (sizeId > 1)
                {
                    int32_t scaling_list_dc_coef_minus8;
                    scaling_list_dc_coef_minus8 = benz_itu_h26xbs_read_se(bitstream); // [-7, 247]
                    dstDcList[matrixId] = scaling_list_dc_coef_minus8 + 8;
                    nextCoef = dstDcList[matrixId];
                }

                for (uint32_t i = 0; i < coefNum; i++)
                {
                    scaling_list_delta_coef = benz_itu_h26xbs_read_se(bitstream); // [-128, 127]
                    nextCoef = (nextCoef + scaling_list_delta_coef + 256) % 256;
                    dstList[i] = nextCoef;
                }

                benz_itu_h26xbs_fill64b(bitstream);
            }
        }
    }

    benz_itu_h26xbs_fill64b(bitstream);

#ifndef benz__min
#undef benz__min
#endif // benz__min
}

void skip_sub_hrd_parameters(BENZ_ITU_H26XBS* bitstream, uint8_t cpb_cnt_minus1, int sub_pic_hrd_params_present_flag)
{
//    uint32_t bit_rate_value_minus1[32];
//    uint32_t cpb_size_value_minus1[32];
//    uint32_t cpb_size_du_value_minus1[32];
//    uint32_t bit_rate_du_value_minus1[32];
//    bool cbr_flag[32];
    for (uint8_t i = 0; i <= cpb_cnt_minus1; i++)
    {
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t bit_rate_value_minus1[32]; //[0, 0xFFFFFFFE]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t cpb_size_value_minus1[32]; //[0, 0xFFFFFFFE]
        benz_itu_h26xbs_bigfill(bitstream);
        if (sub_pic_hrd_params_present_flag)
        {
            benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t cpb_size_du_value_minus1[32]; //[0, 0xFFFFFFFE]
            benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t bit_rate_du_value_minus1[32]; //[0, 0xFFFFFFFE]
            benz_itu_h26xbs_bigfill(bitstream);
        }
        benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool cbr_flag[32];
    }

    benz_itu_h26xbs_fill64b(bitstream);
}

void skip_hrd_parameters(BENZ_ITU_H26XBS* bitstream, uint8_t max_sub_layers_minus1)
{
//    static const uint8_t MAXSUBLAYERS = 7;

    int nal_hrd_parameters_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int vcl_hrd_parameters_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int sub_pic_hrd_params_present_flag = 0;
    if (nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag)
    {
        sub_pic_hrd_params_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (sub_pic_hrd_params_present_flag)
        {
            benz_itu_h26xbs_skip_xn(bitstream, 8);  // uint8_t tick_divisor_minus2;
            benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t du_cpb_removal_delay_increment_length_minus1;
            benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool sub_pic_cpb_params_in_pic_timing_sei_flag;
            benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t dpb_output_delay_du_length_minus1;
        }

        benz_itu_h26xbs_skip_xn(bitstream, 4);  // uint8_t bit_rate_scale;
        benz_itu_h26xbs_skip_xn(bitstream, 4);  // uint8_t cpb_size_scale;

        if (sub_pic_hrd_params_present_flag)
        {
            benz_itu_h26xbs_skip_xn(bitstream, 4);  // uint8_t cpb_size_du_scale;
        }

        benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t initial_cpb_removal_delay_length_minus1;
        benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t au_cpb_removal_delay_length_minus1;
        benz_itu_h26xbs_skip_xn(bitstream, 5);  // uint8_t dpb_output_delay_length_minus1;
        benz_itu_h26xbs_fill64b(bitstream);
    }

//    bool fixed_pic_rate_general_flag[MAXSUBLAYERS];
//    bool fixed_pic_rate_within_cvs_flag[MAXSUBLAYERS];
//    uint16_t elemental_duration_in_tc_minus1[MAXSUBLAYERS]; //[0, 2047]
//    bool low_delay_hrd_flag[MAXSUBLAYERS];
//    uint8_t cpb_cnt_minus1[MAXSUBLAYERS]; //[0, 31]
//    SubLayerHRDParameters sublayer_hrd_params[MAXSUBLAYERS];
    for (uint8_t i = 0; i <= max_sub_layers_minus1; i++)
    {
        int fixed_pic_rate_general_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        int fixed_pic_rate_within_cvs_flag = 0;
        if (fixed_pic_rate_general_flag)
        {
            fixed_pic_rate_within_cvs_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        }

        int low_delay_hrd_flag = 0;
        if (fixed_pic_rate_within_cvs_flag)
        {
            benz_itu_h26xbs_skip_xe(bitstream);     // elemental_duration_in_tc_minus1[i]; //[0, 2047]
        }
        else
        {
            low_delay_hrd_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        }

        uint8_t cpb_cnt_minus1 = 0;
        if (!low_delay_hrd_flag)
        {
            cpb_cnt_minus1 = benz_itu_h26xbs_read_ue(bitstream); //[0, 31]
        }

        if (nal_hrd_parameters_present_flag)
        {
//            SubLayerHRDParameters sublayer_hrd_params[i];
//            sub_layer_hrd_parameters( i )
            benz_itu_h26xbs_bigfill(bitstream);
            skip_sub_hrd_parameters(bitstream, cpb_cnt_minus1, sub_pic_hrd_params_present_flag);
        }

        if (vcl_hrd_parameters_present_flag)
        {
//            SubLayerHRDParameters sublayer_hrd_params[i];
//            sub_layer_hrd_parameters( i )
            benz_itu_h26xbs_bigfill(bitstream);
            skip_sub_hrd_parameters(bitstream, cpb_cnt_minus1, sub_pic_hrd_params_present_flag);
        }
    }

    benz_itu_h26xbs_fill64b(bitstream);
}

void skip_vui_parameters(BENZ_ITU_H26XBS* bitstream, uint8_t sps_max_sub_layers_minus1)
{
    static uint8_t EXTENDED_SAR = 255;

    int aspect_ratio_info_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (aspect_ratio_info_present_flag)
    {
        uint8_t aspect_ratio_idc = benz_itu_h26xbs_read_un(bitstream, 8);
        if (aspect_ratio_idc == EXTENDED_SAR)
        {
            benz_itu_h26xbs_skip_xn(bitstream, 16); // uint16_t sar_width;
            benz_itu_h26xbs_skip_xn(bitstream, 16); // uint16_t sar_height;
        }
    }

    int overscan_info_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (overscan_info_present_flag)
    {
        benz_itu_h26xbs_skip_xn(bitstream, 1);  // int overscan_appropriate_flag;
    }

    int video_signal_type_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (video_signal_type_present_flag)
    {
        benz_itu_h26xbs_skip_xn(bitstream, 3);  // uint8_t video_format;
        benz_itu_h26xbs_skip_xn(bitstream, 1);  // int video_full_range_flag;
        int colour_description_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (colour_description_present_flag)
        {
            benz_itu_h26xbs_skip_xn(bitstream, 8);  // uint8_t colour_primaries;
            benz_itu_h26xbs_skip_xn(bitstream, 8);  // uint8_t transfer_characteristics;
            benz_itu_h26xbs_skip_xn(bitstream, 8);  // uint8_t matrix_coeffs;
        }
    }
    benz_itu_h26xbs_bigfill(bitstream);

    int chroma_loc_info_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (chroma_loc_info_present_flag)
    {
        benz_itu_h26xbs_skip_xe(bitstream);     // uint8_t chroma_sample_loc_type_top_field; //[0, 5]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint8_t chroma_sample_loc_type_bottom_field; //[0, 5]
    }

    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool neutral_chroma_indication_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool field_seq_flag;
    benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool frame_field_info_present_flag;

    int default_display_window_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (default_display_window_flag)
    {
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t def_disp_win_left_offset;   // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t def_disp_win_right_offset;  // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t def_disp_win_top_offset;    // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t def_disp_win_bottom_offset; // assumed to be within [0, 16382]
        benz_itu_h26xbs_bigfill(bitstream);
    }

    int vui_timing_info_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (vui_timing_info_present_flag)
    {
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t vui_num_units_in_tick;
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t vui_time_scale;
        benz_itu_h26xbs_bigfill(bitstream);

        int vui_poc_proportional_to_timing_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (vui_poc_proportional_to_timing_flag)
        {
            benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t vui_num_ticks_poc_diff_one_minus1; //[0, 0xFFFFFFFE]
        }

        int vui_hrd_parameters_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (vui_hrd_parameters_present_flag)
        {
//            HRDParameters hrd_params;
//            hrd_parameters( 1, sps_max_sub_layers_minus1 )
            benz_itu_h26xbs_bigfill(bitstream);
            skip_hrd_parameters(bitstream, sps_max_sub_layers_minus1);
        }
    }

    int bitstream_restriction_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (bitstream_restriction_flag)
    {
        benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool tiles_fixed_structure_flag;
        benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool motion_vectors_over_pic_boundaries_flag;
        benz_itu_h26xbs_skip_xn(bitstream, 1);  // bool restricted_ref_pic_lists_flag;
        benz_itu_h26xbs_skip_xe(bitstream);     // uint16_t min_spatial_segmentation_idc; //[0, 4095]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint8_t max_bytes_per_pic_denom; //[0, 16]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint8_t max_bits_per_min_cu_denom; //[0, 16]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint8_t log2_max_mv_length_horizontal; //[0, 16]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint8_t log2_max_mv_length_vertical; //[0, 15]
        benz_itu_h26xbs_bigfill(bitstream);
    }

    benz_itu_h26xbs_fill64b(bitstream);
}

void skip_header(BENZ_ITU_H26XBS* bitstream)
{
    // forbidden_zero_bit
    benz_itu_h26xbs_skip_xn(bitstream, 1);

    printf("NALU Header -- \n"
           "  nal_unit_type = %llu\n"
           "  nuh_layer_id = %llu\n"
           "  nuh_temporal_id_plus1 = %llu\n",
           benz_itu_h26xbs_read_un(bitstream, 6),
           benz_itu_h26xbs_read_un(bitstream, 6),
           benz_itu_h26xbs_read_un(bitstream, 3));
    benz_itu_h26xbs_fill64b(bitstream);
}

uint8_t get_pps_seq_parameter_set_id(const void* nalu, size_t nalubytelen)
{
    BENZ_ITU_H26XBS bitstream = {0};
    benz_itu_h26xbs_init(&bitstream, nalu, nalubytelen);
    skip_header(&bitstream);

    benz_itu_h26xbs_skip_xe(&bitstream); //[0, 63]
    return benz_itu_h26xbs_read_ue(&bitstream); //[0, 15]
}

uint8_t get_sps_seq_parameter_set_id(const void* nalu, size_t nalubytelen)
{
    BENZ_ITU_H26XBS bitstream = {0};
    benz_itu_h26xbs_init(&bitstream, nalu, nalubytelen);
    skip_header(&bitstream);

    benz_itu_h26xbs_skip_xn(&bitstream, 4); // sps_video_parameter_set_id
    uint8_t sps_max_sub_layers_minus1 = benz_itu_h26xbs_read_un(&bitstream, 3);
    benz_itu_h26xbs_skip_xn(&bitstream, 1); // sps_temporal_id_nesting_flag

    benz_itu_h26xbs_fill64b(&bitstream);
    skip_profile_tier_level(&bitstream, sps_max_sub_layers_minus1); // profile_tier_level( 1, sps_max_sub_layers_minus1 )

    return benz_itu_h26xbs_read_ue(&bitstream); // sps_seq_parameter_set_id [0, 15]
}

void decode_sps(CUVIDPICPARAMS* picParams, BENZ_ITU_H26XBS* bitstream){
#ifndef MAXSUBLAYERS
#define MAXSUBLAYERS 7
#endif
//    static const uint8_t MAXSUBLAYERS = 7;
//    static const uint8_t MAXSPSCOUNT = 15;

    CUVIDHEVCPICPARAMS* hevcPP = (CUVIDHEVCPICPARAMS*)&picParams->CodecSpecific;

    uint8_t sps_video_parameter_set_id = benz_itu_h26xbs_read_un(bitstream, 4);
    uint8_t sps_max_sub_layers_minus1 = benz_itu_h26xbs_read_un(bitstream, 3);
    int sps_temporal_id_nesting_flag = benz_itu_h26xbs_read_un(bitstream, 1);

//    ProfileTierLevel profile_tier_level;
//    profile_tier_level( 1, sps_max_sub_layers_minus1 )
    benz_itu_h26xbs_fill64b(bitstream);
    skip_profile_tier_level(bitstream, sps_max_sub_layers_minus1);

    uint8_t sps_seq_parameter_set_id = benz_itu_h26xbs_read_ue(bitstream); //[0, 15]

    uint8_t chroma_format_idc = benz_itu_h26xbs_read_ue(bitstream); //[0, 3]
    int separate_colour_plane_flag = 0;
    if (chroma_format_idc == 3)
    {
        separate_colour_plane_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    }

//    uint8_t chroma_array_type;
    uint16_t pic_width_in_luma_samples = benz_itu_h26xbs_read_ue(bitstream);    // assumed to be within [0, 16382]
    uint16_t pic_height_in_luma_samples = benz_itu_h26xbs_read_ue(bitstream);   // assumed to be within [0, 16382]
    benz_itu_h26xbs_fill64b(bitstream);

    int conformance_window_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (conformance_window_flag)
    {
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_left_offset;   // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_right_offset;  // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_top_offset;    // assumed to be within [0, 16382]
        benz_itu_h26xbs_skip_xe(bitstream);     // uint32_t conf_win_bottom_offset; // assumed to be within [0, 16382]
        benz_itu_h26xbs_bigfill(bitstream);
    }

//    // --
//    int32_t width;
//    int32_t height;
//    //cropped frame
//    uint32_t croppedLeft;
//    uint32_t croppedTop;
//    uint32_t croppedWidth;
//    uint32_t croppedHeight;
//    // --

    uint8_t bit_depth_luma_minus8 = benz_itu_h26xbs_read_ue(bitstream); //[0, 8]
    uint8_t bit_depth_chroma_minus8 = benz_itu_h26xbs_read_ue(bitstream); //[0, 8]
    uint8_t log2_max_pic_order_cnt_lsb_minus4 = benz_itu_h26xbs_read_ue(bitstream); //[0, 12]

    int sps_sub_layer_ordering_info_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t sps_max_dec_pic_buffering_minus1[MAXSUBLAYERS] = {0}; //[0, 15]
    uint8_t sps_max_num_reorder_pics[MAXSUBLAYERS] = {0}; //[0, 15]
    uint32_t sps_max_latency_increase_plus1[MAXSUBLAYERS] = {0}; //[0, 0xFFFFFFFE]
    for(int i = (sps_sub_layer_ordering_info_present_flag ? 0 : sps_max_sub_layers_minus1); i <= sps_max_sub_layers_minus1; i++)
    {
        sps_max_dec_pic_buffering_minus1[i] = benz_itu_h26xbs_read_ue(bitstream); //[0, 15]
        sps_max_num_reorder_pics[i] = benz_itu_h26xbs_read_ue(bitstream); //[0, 15]
        sps_max_latency_increase_plus1[i] = benz_itu_h26xbs_read_ue(bitstream); //[0, 0xFFFFFFFE]
        benz_itu_h26xbs_bigfill(bitstream);
    }

    uint8_t log2_min_luma_coding_block_size_minus3 = benz_itu_h26xbs_read_ue(bitstream); // estimation [1, 3] or [0, 8]
    uint8_t log2_diff_max_min_luma_coding_block_size = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 2] or [0, 8]
    uint8_t log2_min_transform_block_size_minus2 = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    uint8_t log2_diff_max_min_transform_block_size = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    uint8_t max_transform_hierarchy_depth_inter = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    uint8_t max_transform_hierarchy_depth_intra = benz_itu_h26xbs_read_ue(bitstream); // estimation [0, 8]
    benz_itu_h26xbs_fill64b(bitstream);

    uint8_t scaling_list_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int sps_scaling_list_data_present_flag = 0;
    if (scaling_list_enabled_flag)
    {
        sps_scaling_list_data_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
        if (sps_scaling_list_data_present_flag)
        {
//            ScalingList scaling_list;
//            scaling_list_data( )
            benz_itu_h26xbs_fill64b(bitstream);
            decode_scaling_list_data(hevcPP, bitstream);
        }
    }

    int amp_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int sample_adaptive_offset_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    int pcm_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    uint8_t pcm_sample_bit_depth_luma_minus1 = 0;
    uint8_t pcm_sample_bit_depth_chroma_minus1 = 0;
    uint8_t log2_min_pcm_luma_coding_block_size_minus3 = 0; // estimation [0, 8]
    uint8_t log2_diff_max_min_pcm_luma_coding_block_size = 0; // estimation [0, 8]
    int pcm_loop_filter_disabled_flag = 0;
    if (pcm_enabled_flag)
    {
        pcm_sample_bit_depth_luma_minus1 = benz_itu_h26xbs_read_un(bitstream, 4);
        pcm_sample_bit_depth_chroma_minus1 = benz_itu_h26xbs_read_un(bitstream, 4);
        log2_min_pcm_luma_coding_block_size_minus3 = benz_itu_h26xbs_read_ue(bitstream); // estimation [0,8]
        log2_diff_max_min_pcm_luma_coding_block_size = benz_itu_h26xbs_read_ue(bitstream); // estimation [0,8]
        pcm_loop_filter_disabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    }

    uint8_t num_short_term_ref_pic_sets = benz_itu_h26xbs_read_ue(bitstream); //[0, 64]
    for(int i = 0; i < num_short_term_ref_pic_sets; i++)
    {
//        ShortTermRefPicSet short_term_ref_pic_set[64];
//        st_ref_pic_set( i )
    }

    int long_term_ref_pics_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (long_term_ref_pics_present_flag)
    {
//        uint8_t num_long_term_ref_pics_sps; //[0,32]
//        for (i = 0; i < num_long_term_ref_pics_sps; i++)
//        {
//            uint16_t lt_ref_pic_poc_lsb_sps[32];
//            bool used_by_curr_pic_lt_sps_flag[32];
//        }
    }

    int temporal_mvp_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    int strong_intra_smoothing_enabled_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    int vui_parameters_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);
    if (vui_parameters_present_flag)
    {
//        VuiParameters vui_params;
//        vui_parameters( )
        benz_itu_h26xbs_fill64b(bitstream);
        skip_vui_parameters(bitstream, sps_max_sub_layers_minus1);
    }

    int sps_extension_present_flag = benz_itu_h26xbs_read_un(bitstream, 1);

    // ... unnecessary extension elements

    hevcPP->pic_width_in_luma_samples = pic_width_in_luma_samples;
    hevcPP->pic_height_in_luma_samples = pic_height_in_luma_samples;
    hevcPP->log2_min_luma_coding_block_size_minus3 = log2_min_luma_coding_block_size_minus3;
    hevcPP->log2_diff_max_min_luma_coding_block_size = log2_diff_max_min_luma_coding_block_size;
    hevcPP->log2_min_transform_block_size_minus2 = log2_min_transform_block_size_minus2;
    hevcPP->log2_diff_max_min_transform_block_size = log2_diff_max_min_transform_block_size;
    hevcPP->pcm_enabled_flag = pcm_enabled_flag;
    hevcPP->log2_min_pcm_luma_coding_block_size_minus3 = log2_min_pcm_luma_coding_block_size_minus3;
    hevcPP->log2_diff_max_min_pcm_luma_coding_block_size = log2_diff_max_min_pcm_luma_coding_block_size;
    hevcPP->pcm_sample_bit_depth_luma_minus1 = pcm_sample_bit_depth_luma_minus1;

    hevcPP->pcm_sample_bit_depth_chroma_minus1 = pcm_sample_bit_depth_chroma_minus1;
    hevcPP->pcm_loop_filter_disabled_flag = pcm_loop_filter_disabled_flag;
    hevcPP->strong_intra_smoothing_enabled_flag = strong_intra_smoothing_enabled_flag;
    hevcPP->max_transform_hierarchy_depth_intra = max_transform_hierarchy_depth_intra;
    hevcPP->max_transform_hierarchy_depth_inter = max_transform_hierarchy_depth_inter;
    hevcPP->amp_enabled_flag = amp_enabled_flag;
    hevcPP->separate_colour_plane_flag = separate_colour_plane_flag;
    hevcPP->log2_max_pic_order_cnt_lsb_minus4 = log2_max_pic_order_cnt_lsb_minus4;

    hevcPP->num_short_term_ref_pic_sets = num_short_term_ref_pic_sets;
    hevcPP->long_term_ref_pics_present_flag = long_term_ref_pics_present_flag;
//    hevcPP->num_long_term_ref_pics_sps = num_long_term_ref_pics_sps;
    hevcPP->sps_temporal_mvp_enabled_flag = temporal_mvp_enabled_flag;
    hevcPP->sample_adaptive_offset_enabled_flag = sample_adaptive_offset_enabled_flag;
    hevcPP->scaling_list_enable_flag = scaling_list_enabled_flag;
//    hevcPP->IrapPicFlag = 0;
//    hevcPP->IdrPicFlag = 0;

    hevcPP->bit_depth_luma_minus8 = bit_depth_luma_minus8;
    hevcPP->bit_depth_chroma_minus8 = bit_depth_chroma_minus8;

    //sps/pps extension fields
//    hevcPP->log2_max_transform_skip_block_size_minus2;
//    hevcPP->log2_sao_offset_scale_luma;
//    hevcPP->log2_sao_offset_scale_chroma;
//    hevcPP->high_precision_offsets_enabled_flag;
//    hevcPP->reserved1;

    // sps and pps extension HEVC-main 444
//    hevcPP->sps_range_extension_flag;
//    hevcPP->transform_skip_rotation_enabled_flag;
//    hevcPP->transform_skip_context_enabled_flag;
//    hevcPP->implicit_rdpcm_enabled_flag;

//    hevcPP->explicit_rdpcm_enabled_flag;
//    hevcPP->extended_precision_processing_flag;
//    hevcPP->intra_smoothing_disabled_flag;
//    hevcPP->persistent_rice_adaptation_enabled_flag;

//    hevcPP->cabac_bypass_alignment_enabled_flag;
//    hevcPP->pps_range_extension_flag;
//    hevcPP->cross_component_prediction_enabled_flag;
//    hevcPP->chroma_qp_offset_list_enabled_flag;

//    hevcPP->diff_cu_chroma_qp_offset_depth;
//    hevcPP->chroma_qp_offset_list_len_minus1;
//    hevcPP->cb_qp_offset_list;

//    hevcPP->cr_qp_offset_list;
//    hevcPP->reserved2;

//    hevcPP->reserved3;

    // 7-10
    uint32_t minCbLog2SizeY = hevcPP->log2_min_luma_coding_block_size_minus3 + 3;
    // 7-11
    uint32_t ctbLog2SizeY = minCbLog2SizeY + hevcPP->log2_diff_max_min_luma_coding_block_size;
    // 7-13
    uint32_t ctbSizeY = 1 << ctbLog2SizeY;

    picParams->PicWidthInMbs = hevcPP->pic_width_in_luma_samples / ctbSizeY;
    picParams->FrameHeightInMbs = hevcPP->pic_width_in_luma_samples / ctbSizeY;
//    picParams->CurrPicIdx;
    picParams->field_pic_flag = 0;
//    picParams->bottom_field_flag;
//    picParams->second_field;
    // Bitstream data
//    picParams->nBitstreamDataLen;
//    picParams->pBitstreamData;
    picParams->nNumSlices = 0;
    picParams->pSliceDataOffsets = 0;
    picParams->ref_pic_flag = 0;
    picParams->intra_pic_flag = 1;
    //picParams->Reserved;
}

/**
 * @brief Perform the core operation of the feeder thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdCore          (NVDECODE_CTX* ctx){
	NVDECODE_RQ*    rq;
	CUVIDPICPARAMS* pP;
	CUresult        ret;
	unsigned int    ZERO = 0;
	
	
	nvdecodeFeederThrdGetCurrRq(ctx, &rq);
	pP = rq->picParams;
	
	const uint8_t* record = &rq->hvcCData[0];

    uint32_t configurationVersion = (uint32_t)record[0];                                    // 0
    uint32_t general_profile_space = (uint32_t)(record[1] >> 6);                            // 1 : 11000000
    uint32_t general_tier_flag = (uint32_t)(record[1] >> 5 & 1);                            // 1 : 00100000
    uint32_t general_profile_idc = (uint32_t)(record[1] & 31);                              // 1 : 00011111
    uint32_t general_profile_compatibility_flags = buf_to_uint32(record + 2);               // 2 (2-5)
    uint64_t general_constraint_indicator_flags = buf_to_uint64(record + 6) >> 16;          // 6 (6-11)
    uint32_t general_level_idc = (uint32_t)record[12];                                      // 12
    // uint32_t(4) reserved = ‘1111’b;
    uint32_t min_spatial_segmentation_idc = (uint32_t)buf_to_uint16(record + 13) << 4 >> 4; // 13 (13-14) : 00001111 11111111 ...
    // uint32_t(6) reserved = ‘111111’b;
    uint32_t parallelismType = (uint32_t)(record[15] & 3);                                  // 15 : 00000011
    // uint32_t(6) reserved = ‘111111’b;
    uint32_t chromaFormat = (uint32_t)(record[16] & 3);                                     // 16 : 00000011
    // uint32_t(5) reserved = ‘11111’b;
    uint32_t bitDepthLumaMinus8 = (uint32_t)(record[17] & 7);                               // 17 : 00000111
    // uint32_t(5) reserved = ‘11111’b;
    uint32_t bitDepthChromaMinus8 = (uint32_t)(record[18] & 7);                             // 18 : 00000011
    uint32_t avgFrameRate = (uint32_t)buf_to_uint16(record + 19);                           // 19 (19-20)
    uint32_t constantFrameRate = (uint32_t)(record[21] >> 6);                               // 21 : 11000000
    uint32_t numTemporalLayers = (uint32_t)(record[21] >> 3 & 7);                           // 21 : 00111000
    uint32_t temporalIdNested = (uint32_t)(record[21] >> 2 & 1);                            // 21 : 00000100
    uint32_t lengthSizeMinusOne = (uint32_t)(record[21] & 3);                               // 21 : 00000011
    uint32_t numOfArrays = (uint32_t)record[22];                                            // 22

    // Find PPS, SPSs and get SPS id
    uint8_t pps_sps_id = 0;
    const uint8_t* sps_locations[64] = {0};
    size_t sps_lengths[64] = {0};

    record = &rq->hvcCData[0] + 23;
    for (int i=0; i < numOfArrays; i++)
    {
        uint32_t array_completeness = (uint32_t)(record[0] >> 7);                           // 0 : 10000000
        // uint32_t(1) reserved = 0;
        uint32_t NAL_unit_type = (uint32_t)(record[0] & 63);                                // 0 : 00111111
        uint32_t numNalus = (uint32_t)buf_to_uint16(record + 1);                            // 1 (1-2)

        record += 3;
        for (int j=0; j < numNalus; j++)
        {
            uint32_t nalUnitLength = (uint32_t)buf_to_uint16(record);                       // 0 (0-1)
            // uint32_t(8*nalUnitLength) nalUnit;

            if(NAL_unit_type == 32){
                //VPS_NUT
            }else if(NAL_unit_type == 33){
                //SPS_NUT
                uint8_t sps_id = get_sps_seq_parameter_set_id(record + 2, nalUnitLength);
                sps_locations[sps_id] = record + 2;
                sps_lengths[sps_id] = nalUnitLength;
            }else if(NAL_unit_type == 34){
                //PPS_NUT
                pps_sps_id = get_pps_seq_parameter_set_id(record + 2, nalUnitLength);
            }
            record += 2 + nalUnitLength;
        }
    }

    BENZ_ITU_H26XBS bitstream = {0};

    // Decode SPS
    benz_itu_h26xbs_init(&bitstream, sps_locations[pps_sps_id], sps_lengths[pps_sps_id]);
    decode_sps(pP, &bitstream);

	/**
	 * When we generated this dataset, we encoded the byte offset from
	 * the beginning of the H264 frame in the pointer field. We also
	 * must supply one slice offset of 0, since there is just one
	 * slice.
	 * 
	 * Patch up these pointers to valid values before supplying it to
	 * cuvidDecodePicture().
	 * 
	 * Also, set a CurrPicIdx value. Allegedly, it is always in the
	 * range [0, MAX_DECODE_SURFACES).
	 */
	
	pP->pBitstreamData    = rq->data+(uint64_t)pP->pBitstreamData;
	pP->pSliceDataOffsets = &ZERO;
	pP->CurrPicIdx        = ctx->feeder.cnt % ctx->decoderInfo.ulNumDecodeSurfaces;
	
	/**
	 * Drop mutex and possibly block attempting to decode image, then
	 * reacquire mutex.
	 */
	
	pthread_mutex_unlock(&ctx->lock);
	ret = cuvidDecodePicture(ctx->decoder, pP);
	pthread_mutex_lock(&ctx->lock);
	
	/* Release data. */
	free(rq->data);
	free(rq->hvcCData);
	rq->data     = NULL;
	rq->hvcCData = NULL;
	if(ret != CUDA_SUCCESS){
		ctx->feeder.err = ret;
		nvdecodeFeederThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	/* Bump counters and broadcast signal. */
	ctx->feeder.cnt++;
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Change feeder thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose feeder thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdSetStatus    (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->feeder.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				if(ctx->master.status == CTX_HELPERS_EXITING){
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_JOINING);
				}else{
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
				}
			}else{
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			}
		break;
		case THRD_EXITING:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
		break;
		default:
		break;
	}
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Get feeder thread's current processing request.
 * @param [in]  ctx    The context in question.
 * @param [out] rqOut  The pointer to the request block.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeFeederThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
	*rqOut = &ctx->request[ctx->feeder.cnt % ctx->totalSlots];
	return 0;
}

/**
 * @brief Main routine of the worker thread.
 * 
 * Accepts the data payloads
 * 
 * @param [in]  ctx  The decoding context.
 * @return NULL.
 */

BENZINA_PLUGIN_STATIC void*       nvdecodeWorkerThrdMain          (NVDECODE_CTX* ctx){
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	pthread_mutex_lock(&ctx->lock);
	if(nvdecodeWorkerThrdInit(ctx)){
		while(nvdecodeWorkerThrdContinue(ctx)){
			nvdecodeWorkerThrdCore(ctx);
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	pthread_exit(NULL);
}

/**
 * @brief Initialize worker thread state.
 * 
 * Called with the lock held and status SPAWNED.
 * 
 * @param [in]  ctx  The context whose worker thread is initializing.
 * @return Whether (!0) or not (0) initialization was successful.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdInit          (NVDECODE_CTX* ctx){
	int ret;
	
	if(nvdecodeHelpersShouldExitNow(ctx)){
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ret = cudaSetDevice(ctx->deviceOrdinal);
	if(ret != cudaSuccess){
		ctx->worker.err = ret;
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	ret = cudaStreamCreate(&ctx->worker.cudaStream);
	if(ret != cudaSuccess){
		ctx->worker.err = ret;
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
	
	ctx->decoderRefCnt++;
	
	nvdecodeWorkerThrdSetStatus(ctx, THRD_INITED);
	if(nvdecodeWorkerThrdAwaitAll(ctx)){
		nvdecodeWorkerThrdSetStatus(ctx, THRD_RUNNING);
		return 1;
	}else{
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
		nvdecodeMaybeReapDecoder(ctx);
		nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
		return 0;
	}
}

/**
 * @brief Wait for full initialization of all threads.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx
 * @return Whether (!0) or not (0) all threads reached INITED or RUNNING state.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdAwaitAll      (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersShouldExitNow(ctx)){return 0;}
		if(!nvdecodeHelpersAnyStatusIs(ctx, THRD_SPAWNED)){return 1;}
	}while(pthread_cond_wait(&ctx->worker.cond, &ctx->lock) == 0);
	return 0;
}

/**
 * @brief Determine whether the worker thread should shut down or do more work.
 * 
 * @param [in]  ctx  The
 * @return Whether (!0) or not (0) there is work to do.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdContinue      (NVDECODE_CTX* ctx){
	do{
		if(nvdecodeHelpersShouldExitNow(ctx)){
			break;
		}
		if(!nvdecodeWorkerThrdHasWork(ctx)){
			if(nvdecodeHelpersShouldExit(ctx)){
				break;
			}else{
				continue;
			}
		}
		return 1;
	}while(nvdecodeWorkerThrdWait(ctx));
	
	/**
	 * Destroy the decoder if we own the last reference to it.
	 * 
	 * Also, the worker thread is nominally responsible for the CUDA stream. We
	 * wait until work on the CUDA stream completes before exiting. We drop the
	 * lock while doing so, since the callbacks enqueued on that stream require
	 * the lock to work. We then reacquire the lock, set the status to
	 * NOT_RUNNING and exit.
	 */
	
	nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
	pthread_mutex_unlock       (&ctx->lock);
	cudaStreamSynchronize      (ctx->worker.cudaStream);
	cudaStreamDestroy          (ctx->worker.cudaStream);
	pthread_mutex_lock         (&ctx->lock);
	nvdecodeMaybeReapDecoder   (ctx);
	nvdecodeWorkerThrdSetStatus(ctx, THRD_NOT_RUNNING);
	return 0;
}

/**
 * @brief Does worker thread have work to do?
 * @param [in]  ctx  The context in question
 * @return !0 if thread has work to do; 0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdHasWork       (NVDECODE_CTX* ctx){
	return ctx->worker.cnt < ctx->feeder.cnt;
}

/**
 * @brief Worker Wait.
 * @param [in]   ctx  The context
 * @return 1
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdWait          (NVDECODE_CTX* ctx){
	pthread_cond_wait(&ctx->worker.cond, &ctx->lock);
	return 1;
}

/**
 * @brief Perform the core operation of the worker thread.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context 
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdCore          (NVDECODE_CTX* ctx){
	CUVIDPROCPARAMS    procParams;
	NVDECODE_RQ*       rq;
	unsigned long long srcPtr;
	unsigned           pitch;
	uint64_t           picIdx = 0;
	CUresult           ret;
	
	nvdecodeWorkerThrdGetCurrRq(ctx, &rq);
	memset(&procParams, 0, sizeof(procParams));
	procParams.progressive_frame = 1;
	procParams.second_field      = 0;
	procParams.top_field_first   = 0;
	procParams.unpaired_field    = 0;
	procParams.output_stream     = ctx->worker.cudaStream;
	picIdx = ctx->worker.cnt % ctx->decoderInfo.ulNumDecodeSurfaces;
	
	/**
	 * Drop the mutex and block on the decoder, then perform CUDA ops
	 * on the returned data. Then, reacquire lock.
	 */
	
	pthread_mutex_unlock(&ctx->lock);
	ret = cuvidMapVideoFrame(ctx->decoder, picIdx, &srcPtr, &pitch, &procParams);
	if(ret == CUDA_SUCCESS){
		nvdecodePostprocKernelInvoker(ctx->worker.cudaStream,
		                              rq->devPtr,
		                              ctx->outputHeight,
		                              ctx->outputWidth,
		                              rq->OOB [0], rq->OOB [1], rq->OOB [2],
		                              rq->B   [0], rq->B   [1], rq->B   [2],
		                              rq->S   [0], rq->S   [1], rq->S   [2],
		                              rq->H[0][0], rq->H[0][1], rq->H[0][2],
		                              rq->H[1][0], rq->H[1][1], rq->H[1][2],
		                              rq->H[2][0], rq->H[2][1], rq->H[2][2],
		                              rq->colorMatrix,
		                              (void*)srcPtr,
		                              pitch,
		                              ctx->decoderInfo.ulHeight,
		                              ctx->decoderInfo.ulWidth);
		cudaStreamAddCallback(ctx->worker.cudaStream,
		                      (cudaStreamCallback_t)nvdecodeWorkerThrdCallback,
		                      ctx,
		                      0);
		cuvidUnmapVideoFrame(ctx->decoder, srcPtr);
	}
	pthread_mutex_lock(&ctx->lock);
	
	
	/* Handle errors. */
	if(ret != CUDA_SUCCESS){
		ctx->worker.err = ret;
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
		return 0;
	}
	
	
	/* Exit. */
	ctx->worker.cnt++;
	return 0;
}

/**
 * @brief Post-processing Callback
 * @param [in]   stream The stream onto which this callback had been scheduled.
 * @param [in]   status The error status of this device or stream.
 * @param [in]   ctx    The context on which this callback is being executed.
 * @return 
 */

BENZINA_PLUGIN_STATIC void        nvdecodeWorkerThrdCallback      (cudaStream_t  stream,
                                                                   cudaError_t   status,
                                                                   NVDECODE_CTX* ctx){
	(void)stream;
	
	pthread_mutex_lock(&ctx->lock);
	if(status == cudaSuccess){
		ctx->master.pull.sample++;
		pthread_cond_broadcast(&ctx->master.cond);
	}else{
		ctx->worker.err = 1;
		nvdecodeWorkerThrdSetStatus(ctx, THRD_EXITING);
	}
	pthread_mutex_unlock(&ctx->lock);
}

/**
 * @brief Change worker thread's status.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx  The context whose worker thread's status is being changed.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdSetStatus     (NVDECODE_CTX* ctx, NVDECODE_HLP_THRD_STATUS status){
	ctx->worker.status = status;
	switch(status){
		case THRD_NOT_RUNNING:
			if(nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
				if(ctx->master.status == CTX_HELPERS_EXITING){
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_JOINING);
				}else{
					nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_NOT_RUNNING);
				}
			}else{
				nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
			}
		break;
		case THRD_EXITING:
			nvdecodeMasterThrdSetStatus(ctx, CTX_HELPERS_EXITING);
		break;
		default:
		break;
	}
	pthread_cond_broadcast(&ctx->master.cond);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_cond_broadcast(&ctx->feeder.cond);
	pthread_cond_broadcast(&ctx->worker.cond);
	return 0;
}

/**
 * @brief Get worker thread's current processing request.
 * @param [in]  ctx    The context in question.
 * @param [out] rqOut  The pointer to the request block.
 * @return 0
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWorkerThrdGetCurrRq     (NVDECODE_CTX* ctx, NVDECODE_RQ** rqOut){
	*rqOut = &ctx->request[ctx->worker.cnt % ctx->totalSlots];
	return 0;
}

/**
 * @brief Set the device this context will use.
 * 
 * Called with the lock held.
 * 
 * @param [in]  ctx       The context for which the device is to be set.
 * @param [in]  deviceId  A string identifying uniquely the device to be used.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeSetDevice               (NVDECODE_CTX* ctx, const char*   deviceId){
	int ret, deviceCount=0, i=-1;
	char* s;
	struct cudaDeviceProp prop;
	
	
	/* Forbid changing device ordinal while threads running. */
	if(!nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
		return BENZINA_DATALOADER_ITER_ALREADYINITED;
	}
	
	
	/**
	 * If deviceId is NULL, select current device, whatever it may be. Otherwise,
	 * parse deviceId to figure out the device.
	 */
	if(!deviceId){
		ret = cudaGetDevice(&i);
		if(ret != cudaSuccess){return ret;}
	}else{
		/* Determine maximum device ordinal. */
		ret = cudaGetDeviceCount(&deviceCount);
		if(ret != cudaSuccess){return ret;}
		
		
		/* Select a device ordinal i by one of several identification string schemes. */
		if      (strncmp(deviceId, "cuda:", strlen("cuda:")) == 0){
			if(deviceId[strlen("cuda:")] == '\0'){
				return BENZINA_DATALOADER_ITER_INVALIDARGS;
			}
			i = strtoull(deviceId+strlen("cuda:"), &s, 10);
			if(*s != '\0')      {return BENZINA_DATALOADER_ITER_INVALIDARGS;}
			if(i >= deviceCount){return BENZINA_DATALOADER_ITER_INVALIDARGS;}
		}else if(strncmp(deviceId, "pci:",  strlen("pci:"))  == 0){
			if(cudaDeviceGetByPCIBusId(&i, deviceId+strlen("pci:")) != cudaSuccess){
				return BENZINA_DATALOADER_ITER_INVALIDARGS;
			}
		}else{
			return BENZINA_DATALOADER_ITER_INVALIDARGS;
		}
	}
	
	/**
	 * Verify that the device satisfies several important requirements by
	 * inspecting its properties.
	 * 
	 * In particular, we require an NVDECODE engine, which is available only on
	 * compute-capability 3.0 and up devices, and compute-mode access from
	 * multiple host threads.
	 */
	
	if(cudaGetDeviceProperties(&prop, i) != cudaSuccess){
		return BENZINA_DATALOADER_ITER_INTERNAL;
	}
	if(prop.major        < 3                         ||
	   prop.computeMode == cudaComputeModeProhibited ||
	   prop.computeMode == cudaComputeModeExclusive){
		return BENZINA_DATALOADER_ITER_INVALIDARGS;
	}
	
	
	/* We accept the device ordinal. */
	ctx->deviceOrdinal = i;
	return 0;
}

/**
 * @brief Pull a completed batch of work from the pipeline.
 * 
 * Obviously, called with the lock held.
 * 
 * @param [in]  ctx      The iterator context in which.
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @param [in]  block    Whether the wait should be blocking or not.
 * @param [in]  timeout  A maximum amount of time to wait for the batch of data,
 *                       in seconds. If timeout <= 0, wait indefinitely.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeWaitBatchLocked         (NVDECODE_CTX*    ctx,
                                                                   const void**     token,
                                                                   int              block,
                                                                   double           timeout){
	NVDECODE_BATCH* batch;
	TIMESPEC        deadline, now;
	uint64_t        lifecycle;
	int             ret = 0;
	
	
	*token = NULL;
	if(timeout > 0){
		nvdecodeTimeMonotonic(&now);
		nvdecodeDoubleToTime (&deadline, timeout);
		nvdecodeTimeAdd      (&deadline, &now, &deadline);
	}
	
	lifecycle = ctx->master.lifecycle;
	do{
		if(!nvdecodeSameLifecycle(ctx, lifecycle)){return -2;}
		if(ctx->master.pull.batch >= ctx->master.push.batch){
			if(!block){return EAGAIN;}
			continue;
		}
		nvdecodeMasterThrdGetRetrBt(ctx, &batch);
		if(ctx->master.pull.sample >= batch->stopIndex){
			*token = nvdecodeReturnAndClear(&batch->token);
			ctx->master.pull.batch++;
			ctx->master.pull.token++;
			return 0;
		}else{
			if(ctx->reader.err || ctx->feeder.err || ctx->worker.err){
				return -1;
			}
			if(!block){return EAGAIN;}
		}
	}while((ret = (timeout > 0 ? pthread_cond_timedwait(&ctx->master.cond, &ctx->lock, &deadline) :
	                             pthread_cond_wait     (&ctx->master.cond, &ctx->lock))) == 0);
	return ret;
}


/* Plugin Interface Function Definitions */

/**
 * @brief Allocate iterator context from dataset.
 * @param [out] ctxOut   Output pointer for the context handle.
 * @param [in]  dataset  The dataset over which this iterator will iterate.
 *                       Must be non-NULL and compatible.
 * @return A pointer to the context, if successful; NULL otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeAlloc                   (void** ctxOut, const BENZINA_DATASET* dataset){
	NVDECODE_CTX* ctx = NULL;
	const char*   datasetFile = NULL;
	size_t        datasetLen;
	
	
	/**
	 * The ctxOut and dataset parameters cannot be NULL.
	 */
	
	if(!ctxOut){
		return -1;
	}
	*ctxOut = NULL;
	if(!dataset                                            ||
	   benzinaDatasetGetFile  (dataset, &datasetFile) != 0 ||
	   benzinaDatasetGetLength(dataset, &datasetLen)  != 0){
		return -1;
	}
	
	
	/**
	 * Allocate memory for context.
	 * 
	 * Also, initialize certain critical elements.
	 */
	
	*ctxOut = calloc(1, sizeof(*ctx));
	if(!*ctxOut){
		return -1;
	}else{
		ctx = (NVDECODE_CTX*)*ctxOut;
	}
	ctx->dataset           =  dataset;
	ctx->datasetFile       =  datasetFile;
	ctx->datasetLen        =  datasetLen;
	ctx->datasetFd         = -1;
	ctx->refCnt            =  1;
	ctx->deviceOrdinal     = -1;
	ctx->defaults.S[0]     = ctx->defaults.S[1] = ctx->defaults.S[2] = 1.0;
	ctx->picParams         = NULL;
	ctx->request           = NULL;
	ctx->batch             = NULL;
	
	
	/**
	 * Tail-call into context initialization procedure.
	 */
	
	return nvdecodeAllocDataOpen(ctx);
}

/**
 * @brief Initialize context's dataset handles.
 * 
 * @param [in]   ctx  The context being initialized.
 * @return Error code.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeAllocDataOpen           (NVDECODE_CTX* ctx){
	struct stat s0;
	
	ctx->datasetFd = open(ctx->datasetFile, O_RDONLY|O_CLOEXEC);
	if(ctx->datasetFd             < 0 ||
	   fstat(ctx->datasetFd, &s0) < 0){
		return nvdecodeAllocCleanup(ctx, -1);
	}

	ctx->decoderInfo.ulWidth = 512;
    ctx->decoderInfo.ulHeight = 512;
    ctx->decoderInfo.ulNumDecodeSurfaces = 4;
//    ctx->decoderInfo.CodecType = cudaVideoCodec_H264;
    ctx->decoderInfo.CodecType = cudaVideoCodec_HEVC;
    ctx->decoderInfo.ChromaFormat = cudaVideoChromaFormat_420;
    ctx->decoderInfo.bitDepthMinus8 = 0;
    ctx->decoderInfo.ulIntraDecodeOnly = 1;
    ctx->decoderInfo.display_area.left = 0;
    ctx->decoderInfo.display_area.top = 0;
    ctx->decoderInfo.display_area.right = 512;
    ctx->decoderInfo.display_area.bottom = 512;
    ctx->decoderInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
    ctx->decoderInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    ctx->decoderInfo.ulTargetWidth = 512;
    ctx->decoderInfo.ulTargetHeight = 512;
    ctx->decoderInfo.ulNumOutputSurfaces = 4;
    ctx->decoderInfo.target_rect.left = 0;
    ctx->decoderInfo.target_rect.top = 0;
    ctx->decoderInfo.target_rect.right = 0;
    ctx->decoderInfo.target_rect.bottom = 0;

	return nvdecodeAllocThreading(ctx);
}

/**
 * @brief Initialize context's threading resources.
 * 
 * This includes the condition variables and the Big Lock, but does *not*
 * include launching helper threads.
 * 
 * @param [in]   ctx  The context being initialized.
 * @return Error code.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeAllocThreading          (NVDECODE_CTX* ctx){
	pthread_condattr_t condAttr;
	
	if(pthread_condattr_init    (&condAttr)                    != 0){goto fail_attr;}
	if(pthread_condattr_setclock(&condAttr, CLOCK_MONOTONIC)   != 0){goto fail_clock;}
	if(pthread_mutex_init       (&ctx->lock,                0) != 0){goto fail_lock;}
	if(pthread_cond_init        (&ctx->master.cond, &condAttr) != 0){goto fail_master;}
	if(pthread_cond_init        (&ctx->reader.cond, &condAttr) != 0){goto fail_reader;}
	if(pthread_cond_init        (&ctx->feeder.cond, &condAttr) != 0){goto fail_feeder;}
	if(pthread_cond_init        (&ctx->worker.cond, &condAttr) != 0){goto fail_worker;}
	
	pthread_condattr_destroy(&condAttr);
	
	return nvdecodeAllocCleanup(ctx,  0);
	
	
	             pthread_cond_destroy    (&ctx->worker.cond);
	fail_worker: pthread_cond_destroy    (&ctx->feeder.cond);
	fail_feeder: pthread_cond_destroy    (&ctx->reader.cond);
	fail_reader: pthread_cond_destroy    (&ctx->master.cond);
	fail_master: pthread_mutex_destroy   (&ctx->lock);
	fail_lock:   
	fail_clock:  pthread_condattr_destroy(&condAttr);
	fail_attr:
	
	return nvdecodeAllocCleanup(ctx, -1);
}

/**
 * @brief Cleanup for context allocation.
 * 
 * @param [in]  ctx  The context being allocated.
 * @param [in]  ret  Return error code.
 * @return The value `ret`.
 */

BENZINA_PLUGIN_STATIC int         nvdecodeAllocCleanup            (NVDECODE_CTX* ctx, int ret){
	if(ret == 0){
		return ret;
	}
	
	close(ctx->datasetFd);
	ctx->datasetFd = -1;
	
	free(ctx);
	
	return ret;
}

/**
 * @brief Initialize iterator context using its current properties.
 * 
 * The current properties of the iterator will be frozen and will be
 * unchangeable afterwards.
 * 
 * @param [in]  ctx  The iterator context to initialize.
 * @return 0 if successful in initializing; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeInit                    (NVDECODE_CTX* ctx){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeHelpersStart(ctx);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Increase reference count of the iterator.
 * 
 * @param [in]  ctx  The iterator context whose reference-count is to be increased.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeRetain                  (NVDECODE_CTX* ctx){
	if(!ctx){return 0;}
	
	pthread_mutex_lock(&ctx->lock);
	ctx->refCnt++;
	pthread_mutex_unlock(&ctx->lock);
	
	return 0;
}

/**
 * @brief Decrease reference count of the iterator. Destroy iterator if its
 *        reference count drops to 0.
 * 
 * @param [in]  ctx  The iterator context whose reference-count is to be decreased.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeRelease                 (NVDECODE_CTX* ctx){
	if(!ctx){return 0;}
	
	pthread_mutex_lock(&ctx->lock);
	if(--ctx->refCnt > 0){
		pthread_mutex_unlock(&ctx->lock);
		return 0;
	}
	
	/**
	 * At this present time the mutex is held, but the reference count is 0.
	 * This makes us responsible for the destruction of the object.
	 * 
	 * Since we were the last to hold a reference to this context, we are
	 * guaranteed to succeed in tearing down the context's threads, due to
	 * there being no-one else to countermand the order. For the same reason,
	 * we are guaranteed that the current helper thread lifecycle is the last
	 * one, and a new one will not start under our feet while the lock is
	 * released.
	 */
	
	nvdecodeHelpersStop  (ctx);
	pthread_mutex_unlock (&ctx->lock);
	
	pthread_cond_destroy (&ctx->worker.cond);
	pthread_cond_destroy (&ctx->feeder.cond);
	pthread_cond_destroy (&ctx->reader.cond);
	pthread_cond_destroy (&ctx->master.cond);
	pthread_mutex_destroy(&ctx->lock);
	
	close(ctx->datasetFd);
	
	free(ctx->picParams);
	free(ctx->request);
	free(ctx->batch);
	
	memset(ctx, 0, sizeof(*ctx));
	free(ctx);
	
	return 0;
}

/**
 * @brief Ensure that this iterator context's helper threads are running.
 * 
 * This is not actually a very useful function. Technically, even if it returns
 * success, by the time it returns the threads may have shut down again already.
 * 
 * @param [in]  ctx  The iterator context whose helper threads are to be spawned.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeStartHelpers            (NVDECODE_CTX* ctx){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeHelpersStart(ctx);
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Ensure that this iterator context's helper threads are stopped.
 * 
 * @param [in]  ctx  The iterator context whose helper threads are to be stopped.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeStopHelpers             (NVDECODE_CTX* ctx){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeHelpersStop(ctx);
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Begin defining a batch of samples.
 * 
 * @param [in]  ctx       The iterator context in which.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeDefineBatch             (NVDECODE_CTX* ctx){
	NVDECODE_BATCH* batch;
	int ret = 0;
	
	pthread_mutex_lock(&ctx->lock);
	if(ctx->master.push.batch-ctx->master.pull.batch >= ctx->multibuffering){
		ret = -1;
	}else{
		nvdecodeMasterThrdGetSubmBt(ctx, &batch);
		batch->startIndex = ctx->master.push.sample;
		batch->stopIndex  = ctx->master.push.sample;
		batch->token      = NULL;
		ret =  0;
	}
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Close and push a batch of work into the pipeline.
 * 
 * @param [in]  ctx    The iterator context in which.
 * @param [in]  token  User data that will be retrieved at the corresponding
 *                     pullBatch().
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeSubmitBatch             (NVDECODE_CTX* ctx, const void* token){
	NVDECODE_BATCH* batch;
	
	pthread_mutex_lock(&ctx->lock);
	nvdecodeMasterThrdGetSubmBt(ctx, &batch);
	batch->token = token;
	ctx->master.push.batch++;
	ctx->master.push.token++;
	pthread_mutex_unlock(&ctx->lock);
	
	return 0;
}

/**
 * @brief Pull a completed batch of work from the pipeline.
 * 
 * @param [in]  ctx      The iterator context in which.
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @param [in]  block    Whether the wait should be blocking or not.
 * @param [in]  timeout  A maximum amount of time to wait for the batch of data,
 *                       in seconds. If timeout <= 0, wait indefinitely.
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeWaitBatch               (NVDECODE_CTX* ctx,
                                                                   const void**  token,
                                                                   int           block,
                                                                   double        timeout){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	ret = nvdecodeWaitBatchLocked(ctx, token, block, timeout);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Peek at a token from the pipeline.
 * 
 * @param [in]  ctx      The iterator context in question.
 * @param [in]  i        The index of the token wanted.
 * @param [in]  clear    Whether to clear (!0) the token from the buffer or not (0).
 * @param [out] token    User data that was submitted at the corresponding
 *                       pushBatch().
 * @return 0 if successful; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodePeekToken               (NVDECODE_CTX* ctx,
                                                                   uint64_t      i,
                                                                   int           clear,
                                                                   const void**  token){
	NVDECODE_BATCH* batch = NULL;
	int ret = -1;
	
	pthread_mutex_lock(&ctx->lock);
	if(i >= ctx->master.pull.token &&
	   i <  ctx->master.push.token){
		batch = &ctx->batch[i % ctx->multibuffering];
		if(clear){
			*token = nvdecodeReturnAndClear(&batch->token);
		}else{
			*token = batch->token;
		}
		ret = 0;
	}else{
		*token = NULL;
		ret = EWOULDBLOCK;
	}
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Begin defining a new sample.
 * 
 * @param [in]  ctx     
 * @param [in]  i       Index into dataset.
 * @param [in]  dstPtr  Destination Pointer.
 * @return 0 if no errors; !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeDefineSample            (NVDECODE_CTX* ctx, uint64_t i, void* dstPtr,
                                                                   uint64_t* location, uint64_t* config_location){
	NVDECODE_RQ*    rq;
	NVDECODE_BATCH* batch;
	int ret = 0;
	
	pthread_mutex_lock(&ctx->lock);
	nvdecodeMasterThrdGetSubmBt(ctx, &batch);
	nvdecodeMasterThrdGetSubmRq(ctx, &rq);
	if(batch->stopIndex-batch->startIndex >= ctx->batchSize){
		ret = -1;
	}else{
		batch->stopIndex++;
		rq->batch              = batch;
		rq->datasetIndex       = i;
		rq->devPtr             = dstPtr;
		rq->location[0]        = location[0];
		rq->location[1]        = location[1];
		rq->config_location[0] = config_location[0];
		rq->config_location[1] = config_location[1];
		ret = 0;
	}
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

/**
 * @brief Submit current sample.
 * 
 * @param [in]  ctx
 * @return 0 if submission successful and threads will soon handle it.
 *         !0 otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeSubmitSample            (NVDECODE_CTX* ctx){
	int ret;
	pthread_mutex_lock(&ctx->lock);
	switch(ctx->master.status){
		case CTX_HELPERS_JOINING:
		case CTX_HELPERS_EXITING:
			ret = nvdecodeMasterThrdAwaitShutdown(ctx);
			if(ret != 0){
				pthread_mutex_unlock(&ctx->lock);
				return -2;
			}
		break;
		default: break;
	}
    NVDECODE_RQ*    rq;
    nvdecodeMasterThrdGetSubmRq(ctx, &rq);
	ctx->master.push.sample++;
	ret = nvdecodeHelpersStart(ctx);
	pthread_cond_broadcast(&ctx->reader.cond);
	pthread_mutex_unlock(&ctx->lock);
	return ret;
}

/**
 * @brief Retrieve the number of batch pushes into the context.
 * 
 * @param [in]  ctx
 * @param [out] out
 * @return 0
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeGetNumPushes            (NVDECODE_CTX* ctx, uint64_t* out){
	pthread_mutex_lock(&ctx->lock);
	*out = ctx->master.push.batch;
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Retrieve the number of batch pulls out of the context.
 * 
 * @param [in]  ctx
 * @param [out] out
 * @return 0
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeGetNumPulls             (NVDECODE_CTX* ctx, uint64_t* out){
	pthread_mutex_lock(&ctx->lock);
	*out = ctx->master.pull.batch;
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Retrieve the multibuffering depth of the context.
 * 
 * @param [in]  ctx
 * @param [out] out
 * @return 0
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeGetMultibuffering       (NVDECODE_CTX* ctx, uint64_t* out){
	pthread_mutex_lock(&ctx->lock);
	*out = ctx->multibuffering;
	pthread_mutex_unlock(&ctx->lock);
	return 0;
}

/**
 * @brief Set Buffer and Geometry.
 * 
 * Provide to the context the target output buffer it should use.
 * 
 * @param [in]  ctx
 * @param [in]  deviceId
 * @param [in]  devicePtr
 * @param [in]  multibuffering
 * @param [in]  batchSize
 * @param [in]  outputHeight
 * @param [in]  outputWidth
 * @return Zero if successful; Non-zero otherwise.
 */

BENZINA_PLUGIN_HIDDEN int         nvdecodeSetBuffer               (NVDECODE_CTX* ctx,
                                                                   const char*   deviceId,
                                                                   void*         outputPtr,
                                                                   uint32_t      multibuffering,
                                                                   uint32_t      batchSize,
                                                                   uint32_t      outputHeight,
                                                                   uint32_t      outputWidth){
	int ret;
	
	pthread_mutex_lock(&ctx->lock);
	if(!nvdecodeHelpersAllStatusIs(ctx, THRD_NOT_RUNNING)){
		ret = BENZINA_DATALOADER_ITER_ALREADYINITED;
	}else if(!outputPtr){
		ret = BENZINA_DATALOADER_ITER_INVALIDARGS;
	}else{
		ret = nvdecodeSetDevice(ctx, deviceId);
		if(ret == 0){
			ctx->outputPtr      = outputPtr;
			ctx->multibuffering = multibuffering;
			ctx->batchSize      = batchSize;
			ctx->totalSlots     = ctx->multibuffering*ctx->batchSize;
			ctx->outputHeight   = outputHeight;
			ctx->outputWidth    = outputWidth;
			ctx->picParams      = calloc(ctx->totalSlots,     sizeof(*ctx->picParams));
			ctx->request        = calloc(ctx->totalSlots,     sizeof(*ctx->request));
			ctx->batch          = calloc(ctx->multibuffering, sizeof(*ctx->batch));
			if(ctx->picParams && ctx->request && ctx->batch){
				ret = BENZINA_DATALOADER_ITER_SUCCESS;
			}else{
				free(ctx->picParams);
				free(ctx->request);
				free(ctx->batch);
				ctx->picParams = NULL;
				ctx->request   = NULL;
				ctx->batch     = NULL;
				ret = BENZINA_DATALOADER_ITER_INTERNAL;
			}
		}
	}
	pthread_mutex_unlock(&ctx->lock);
	
	return ret;
}

BENZINA_PLUGIN_HIDDEN int         nvdecodeSetDefaultBias          (NVDECODE_CTX* ctx,
                                                                   float*        B){
	memcpy(ctx->defaults.B, B, sizeof(ctx->defaults.B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetDefaultScale         (NVDECODE_CTX* ctx,
                                                                   float*        S){
	memcpy(ctx->defaults.S, S, sizeof(ctx->defaults.S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetDefaultOOBColor      (NVDECODE_CTX* ctx,
                                                                   float*        OOB){
	memcpy(ctx->defaults.OOB, OOB, sizeof(ctx->defaults.OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSelectDefaultColorMatrix(NVDECODE_CTX* ctx,
                                                                   uint32_t      colorMatrix){
	ctx->defaults.colorMatrix = colorMatrix;
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetHomography           (NVDECODE_CTX* ctx,
                                                                   const float*  H){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	if(H){
		memcpy(rq->H, H, sizeof(rq->H));
	}else{
		rq->H[0][0] = 1.0; rq->H[0][1] = 0.0; rq->H[0][2] = 0.0;
		rq->H[1][0] = 0.0; rq->H[1][1] = 1.0; rq->H[1][2] = 0.0;
		rq->H[2][0] = 0.0; rq->H[2][1] = 0.0; rq->H[2][2] = 1.0;
	}
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetBias                 (NVDECODE_CTX* ctx,
                                                                   const float*  B){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->B, B?B:ctx->defaults.B, sizeof(rq->B));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetScale                (NVDECODE_CTX* ctx,
                                                                   const float*  S){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->S, S?S:ctx->defaults.S, sizeof(rq->S));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSetOOBColor             (NVDECODE_CTX* ctx,
                                                                   const float*  OOB){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	memcpy(rq->OOB, OOB?OOB:ctx->defaults.OOB, sizeof(rq->OOB));
	return 0;
}
BENZINA_PLUGIN_HIDDEN int         nvdecodeSelectColorMatrix       (NVDECODE_CTX* ctx,
                                                                   uint32_t      colorMatrix){
	NVDECODE_RQ* rq = &ctx->request[ctx->master.push.sample % ctx->totalSlots];
	rq->colorMatrix = colorMatrix;
	return 0;
}



/**
 * Exported Function Table.
 */

BENZINA_PLUGIN_PUBLIC BENZINA_PLUGIN_NVDECODE_VTABLE VTABLE = {
	.alloc                    = (void*)nvdecodeAlloc,
	.init                     = (void*)nvdecodeInit,
	.retain                   = (void*)nvdecodeRetain,
	.release                  = (void*)nvdecodeRelease,
	.startHelpers             = (void*)nvdecodeStartHelpers,
	.stopHelpers              = (void*)nvdecodeStopHelpers,
	.defineBatch              = (void*)nvdecodeDefineBatch,
	.submitBatch              = (void*)nvdecodeSubmitBatch,
	.waitBatch                = (void*)nvdecodeWaitBatch,
	.peekToken                = (void*)nvdecodePeekToken,
	
	.defineSample             = (void*)nvdecodeDefineSample,
	.submitSample             = (void*)nvdecodeSubmitSample,
	
	.getNumPushes             = (void*)nvdecodeGetNumPushes,
	.getNumPulls              = (void*)nvdecodeGetNumPulls,
	.getMultibuffering        = (void*)nvdecodeGetMultibuffering,
	
	.setBuffer                = (void*)nvdecodeSetBuffer,
	
	.setDefaultBias           = (void*)nvdecodeSetDefaultBias,
	.setDefaultScale          = (void*)nvdecodeSetDefaultScale,
	.setDefaultOOBColor       = (void*)nvdecodeSetDefaultOOBColor,
	.selectDefaultColorMatrix = (void*)nvdecodeSelectDefaultColorMatrix,
	
	.setHomography            = (void*)nvdecodeSetHomography,
	.setBias                  = (void*)nvdecodeSetBias,
	.setScale                 = (void*)nvdecodeSetScale,
	.setOOBColor              = (void*)nvdecodeSetOOBColor,
	.selectColorMatrix        = (void*)nvdecodeSelectColorMatrix,
};

