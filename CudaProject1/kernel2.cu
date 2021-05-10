// Inclusion of header files for running CUDA in Visual Studio Pro 2019 (v142)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Inclusion of the required CUDA libriaries and header files
#include <curand.h>
#include <cuda.h>

// Inclusion of headers from the standard library in C
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define CURAND_RNG_NON_DEFAULT 24

// Windows implementation of the Linux sys/time.h fnuctions needed in this program
#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>

#define __need_clock_t
#include <time.h>

/* Structure describing CPU time used by a process and its children.  */
struct tms
{
    clock_t tms_utime;          /* User CPU time.  */
    clock_t tms_stime;          /* System CPU time.  */

    clock_t tms_cutime;         /* User CPU time of dead children.  */
    clock_t tms_cstime;         /* System CPU time of dead children.  */
};

// CUDA 8+ requirment
struct timezone {
    int tz_minuteswest; /* minutes west of Greenwich */
    int tz_dsttime; /* type of DST correction */
};

int gettimeofday(struct timeval* t, void* timezone)
{
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    t->tv_sec = (long)timebuffer.time;
    t->tv_usec = 1000 * timebuffer.millitm;
    return 0;
}

/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t times(struct tms* __buffer) {

    __buffer->tms_utime = clock();
    __buffer->tms_stime = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}
typedef long long suseconds_t;

// CUDA error check to get error name
const char * CUDA_CHECK_VAL(cudaError_t x) {
    size_t size = 50 * sizeof('a');
    const char* msg = (char*)malloc(size);

    switch (x) {
    case 0:
        msg = "cudaSuccess";
    case 1:
        msg = "cudaErrorInvalidValue";
    case 2:
        msg = "cudaErrorMemoryAllocation";
    case 3:
        msg = "cudaErrorInitializationError";
    case 4:
        msg = "cudaErrorCudartUnloading";
    case 5:
        msg = "cudaErrorProfilerDisabled";
    case 9:
        msg = "cudaErrorInvalidConfiguration";
    case 12:
        msg = "cudaErrorInvalidPitchValue";
    case 13:
        msg = "cudaErrorInvalidSymbol";
    case 18:
        msg = "cudaErrorInvalidTexture";
    case 19:
        msg = "cudaErrorInvalidTextureBinding";
    case 20:
        msg = "cudaErrorInvalidChannelDescriptor";
    case 21:
        msg = "cudaErrorInvalidMemcpyDirection";
    case 26:
        msg = "cudaErrorInvalidFilterSetting";
    case 27:
        msg = "cudaErrorInvalidNormSetting";
    case 34:
        msg = "cudaErrorStubLibrary";
    case 35:
        msg = "cudaErrorInsufficientDriver";
    case 36:
        msg = "cudaErrorCallRequiresNewerDriver";
    case 37:
        msg = "cudaErrorInvalidSurface";
    case 43:
        msg = "cudaErrorDuplicateVariableName";
    case 44:
        msg = "cudaErrorDuplicateTextureName";
    case 45:
        msg = "cudaErrorDuplicateSurfaceName";
    case 46:
        msg = "cudaErrorDevicesUnavailable";
    case 49:
        msg = "cudaErrorIncompatibleDriverContext";
    case 52:
        msg = "cudaErrorMissingConfiguration";
    case 65:
        msg = "cudaErrorLaunchMaxDepthExceeded";
    case 66:
        msg = "cudaErrorLaunchFileScopedTex";
    case 67:
        msg = "cudaErrorLaunchFileScopedSurf";
    case 68:
        msg = "cudaErrorSyncDepthExceeded";
    case 69:
        msg = "cudaErrorLaunchPendingCountExceeded";
    case 98:
        msg = "cudaErrorInvalidDeviceFunction";
    case 100:
        msg = "cudaErrorNoDevice";
    case 101:
        msg = "cudaErrorInvalidDevice";
    case 102:
        msg = "cudaErrorDeviceNotLicensed";
    case 103:
        msg = "cudaErrorSoftwareValidityNotEstablished";
    case 127:
        msg = "cudaErrorStartupFailure";
    case 200:
        msg = "cudaErrorInvalidKernelImage";
    case 201:
        msg = "cudaErrorDeviceUninitialized";
    case 205:
        msg = "cudaErrorMapBufferObjectFailed";
    case 206:
        msg = "cudaErrorUnmapBufferObjectFailed";
    case 207:
        msg = "cudaErrorArrayIsMapped";
    case 208:
        msg = "cudaErrorAlreadyMapped";
    case 209:
        msg = "cudaErrorNoKernelImageForDevice";
    case 210:
        msg = "cudaErrorAlreadyAcquired";
    case 211:
        msg = "cudaErrorNotMapped";
    case 212:
        msg = "cudaErrorNotMappedAsArray";
    case 213:
        msg = "cudaErrorNotMappedAsPointer";
    case 214:
        msg = "cudaErrorECCUncorrectable";
    case 215:
        msg = "cudaErrorUnsupportedLimit";
    case 216:
        msg = "cudaErrorDeviceAlreadyInUse";
    case 217:
        msg = "cudaErrorPeerAccessUnsupported";
    case 218:
        msg = "cudaErrorInvalidPtx";
    case 219:
        msg = "cudaErrorInvalidGraphicsContext";
    case 220:
        msg = "cudaErrorNvlinkUncorrectable";
    case 221:
        msg = "cudaErrorJitCompilerNotFound";
    case 222:
        msg = "cudaErrorUnsupportedPtxVersion";
    case 223:
        msg = "cudaErrorJitCompilationDisabled";
    case 300:
        msg = "cudaErrorInvalidSource";
    case 301:
        msg = "cudaErrorFileNotFound";
    case 302:
        msg = "cudaErrorSharedObjectSymbolNotFound";
    case 303:
        msg = "cudaErrorSharedObjectInitFailed";
    case 304:
        msg = "cudaErrorOperatingSystem";
    case 400:
        msg = "cudaErrorInvalidResourceHandle";
    case 401:
        msg = "cudaErrorIllegalState";
    case 500:
        msg = "cudaErrorSymbolNotFound";
    case 600:
        msg = "cudaErrorNotReady";
    case 700:
        msg = "cudaErrorIllegalAddress";
    case 701:
        msg = "cudaErrorLaunchOutOfResources";
    case 702:
        msg = "cudaErrorLaunchTimeout";
    case 703:
        msg = "cudaErrorLaunchIncompatibleTexturing";
    case 704:
        msg = "cudaErrorPeerAccessAlreadyEnabled";
    case 705:
        msg = "cudaErrorPeerAccessNotEnabled";
    case 708:
        msg = "cudaErrorSetOnActiveProcess";
    case 709:
        msg = "cudaErrorContextIsDestroyed";
    case 710:
        msg = "cudaErrorAssert";
    case 711:
        msg = "cudaErrorTooManyPeers";
    case 712:
        msg = "cudaErrorHostMemoryAlreadyRegistered";
    case 713:
        msg = "cudaErrorHostMemoryNotRegistered";
    case 714:
        msg = "cudaErrorHardwareStackError";
    case 715:
        msg = "cudaErrorIllegalInstruction";
    case 716:
        msg = "cudaErrorMisalignedAddress";
    case 717:
        msg = "cudaErrorInvalidAddressSpace";
    case 718:
        msg = "cudaErrorInvalidPc";
    case 719:
        msg = "cudaErrorLaunchFailure";
    case 720:
        msg = "cudaErrorCooperativeLaunchTooLarge";
    case 800:
        msg = "cudaErrorNotPermitted";
    case 801:
        msg = "cudaErrorNotSupported";
    case 802:
        msg = "cudaErrorSystemNotReady";
    case 803:
        msg = "cudaErrorSystemDriverMismatch";
    case 804:
        msg = "cudaErrorCompatNotSupportedOnDevice";
    case 900:
        msg = "cudaErrorStreamCaptureUnsupported";
    case 901:
        msg = "cudaErrorStreamCaptureInvalidated";
    case 902:
        msg = "cudaErrorStreamCaptureMerge";
    case 903:
        msg = "cudaErrorStreamCaptureUnmatched";
    case 904:
        msg = "cudaErrorStreamCaptureUnjoined";
    case 905:
        msg = "cudaErrorStreamCaptureIsolation";
    case 906:
        msg = "cudaErrorStreamCaptureImplicit";
    case 907:
        msg = "cudaErrorCapturedEvent";
    case 908:
        msg = "cudaErrorStreamCaptureWrongThread";
    case 909:
        msg = "cudaErrorTimeout";
    case 910:
        msg = "cudaErrorGraphExecUpdateFailure";
    case 999:
        msg = "cudaErrorUnknown";
    default:
        msg = "NonValidCudaError";
    }
    return msg;
}

// CURAND error check to get error name
const char * CURAND_CHECK_VAL(curandStatus_t x) {
    size_t size = 50 * sizeof('a');
    const char* msg = (char*)malloc(size);

    switch (x) {
    case 0:
        msg = "CURAND_STATUS_SUCCESS";
    case 100:
        msg = "CURAND_STATUS_VERSION_MISMATCH";
    case 101:
        msg = "CURAND_STATUS_NOT_INITIALIZED";
    case 102:
        msg = "CURAND_STATUS_ALLOCATION_FAILED";
    case 103:
        msg = "CURAND_STATUS_TYPE_ERROR";
    case 104:
        msg = "CURAND_STATUS_OUT_OF_RANGE";
    case 105:
        msg = "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case 106:
        msg = "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case 201:
        msg = "CURAND_STATUS_LAUNCH_FAILURE";
    case 202:
        msg = "CURAND_STATUS_PREEXISTING_FAILURE";
    case 203:
        msg = "CURAND_STATUS_INITIALIZATION_FAILED";
    case 204:
        msg = "CURAND_STATUS_ARCH_MISMATCH";
    case 999:
        msg = "CURAND_STATUS_INTERNAL_ERROR";
    default:
        msg = "NON_VALID_CURAND_ERROR";
    }
    return msg;
}

// Check method for checking the error status of a CUDA call
#define CUDA_CALL(x) { if(x != cudaSuccess){ sprintf("Error: %s at %s:%d\n", CUDA_CHECK_VAL(x), __FILE__, __LINE__); return EXIT_FAILURE;}}

// Check method for checking the error status of a cuRAND call
#define CURAND_CALL(x) {if(x != CURAND_STATUS_SUCCESS){ sprintf("Error: %s at %s:%d\n", CURAND_CHECK_VAL(x), __FILE__, __LINE__); return EXIT_FAILURE;}}

// The kernel, which runs on the GPU when called
__global__ void kernel(int* a, int* b, int* c, size_t N)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) c[idx] = a[idx] * b[idx];
}

// Function for generating the same results as the GPU kernel, used for verification of results
__host__ void KernelCPUEd(int* a, int* b, int* c, size_t size)
{
    for (int i = 0; i < size; i++)
        c[i] = a[i] * b[i];
}

// Program to convert a float array to an integer array
__host__ void FtoIArray(int* dst, float* src, size_t nElem) {
    for (int i = 0; i < nElem; i++)
        dst[i] = (int)(src[i] * 1000);
}

// Function for verifying the array generated by the kernel is correct
__host__ bool inline CHECK(int* a, int* b, size_t size)
{
    double epsilon = 1.0E-8;
    for (int x = 0; x < size; x++)
    {
        if (a[x] - b[x] > epsilon)
            return true;
    }
    return false;
}

__host__ double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Entry point to the program
int main(void) {
    size_t nElem = 1 << 24;
    size_t nBytes = nElem * sizeof(int);
    size_t nBytesF = nElem * sizeof(float);

    int* h_A, * h_B, * h_C, * GpuRef;
    int* d_A, * d_B, * d_C;

    float* devNumGen, * devNumGen2, * h_AR, * h_BR;

    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CALL(cudaSetDevice(dev));

    curandGenerator_t gen, gen2;

    // Allocation of memory on the host for transferring data from host to device and vice versa
    h_A = (int*)malloc(nBytes);
    h_B = (int*)malloc(nBytes);
    h_C = (int*)malloc(nBytes);
    GpuRef = (int*)malloc(nBytes);

    // Allocation of memory on the device for storage of data needed by the kernel during runtime
    CUDA_CALL(cudaMalloc((int**)&d_A, nBytes));
    CUDA_CALL(cudaMalloc((int**)&d_B, nBytes));
    CUDA_CALL(cudaMalloc((int**)&d_C, nBytes));

    // Allocation of memory on host and device for testing the CUDA number generator
    h_AR = (float*)malloc(nBytes);
    h_BR = (float*)malloc(nBytes);
    CUDA_CALL(cudaMalloc((float**)&devNumGen, nBytesF));
    CUDA_CALL(cudaMalloc((float**)&devNumGen2, nBytesF));

    // CUDA number generator function calls and return values
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT));

    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen2, time(NULL) + 1));

    CURAND_CALL(curandGenerateUniform(gen, devNumGen, nElem));
    CURAND_CALL(curandGenerateUniform(gen2, devNumGen2, nElem));

    // Transfer random numbers generated on device to host
    CUDA_CALL(cudaMemcpy(h_AR, devNumGen, nBytesF, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_BR, devNumGen2, nBytesF, cudaMemcpyDeviceToHost));

    FtoIArray(h_A, h_AR, nElem);
    FtoIArray(h_B, h_BR, nElem);

    // Transfer of populated arrays to the device for use by the kernel
    CUDA_CALL(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // Calculate block indices
    int iLen = 1024;
    dim3 block(iLen, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // Kernel call to run the calculation n the GPU, uses 1 block and nElem amount of threads in the block
    // Max threads in a block for RTX 2060 is 4096 threads
    double iStart = cpuSecond();
    kernel <<<grid, block>>> (d_A, d_B, d_C, nElem);
    CUDA_CALL(cudaDeviceSynchronize());
    double iEnd = cpuSecond() - iStart;

    printf("Execution time of the GPU kernel <<<%d, %d>>>: %g\n", grid.x, block.x, iEnd);

    // Verification function that the kernel on the GPU is performing properly
    double iStartCPU = cpuSecond();
    KernelCPUEd(h_A, h_B, h_C, nElem);
    double iEndCPU = cpuSecond() - iStart;
    printf("Execution time of the CPU function %g\n", iEndCPU);

    // Transfer of data from Device to the host
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaMemcpy(GpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // Verification of data, compares data generated on the host to the data generated on the device
    // If the data is different, goto Exit is called and memory is freed, the the program ends
    if (CHECK(h_C, GpuRef, nElem))
    {
        printf("The arrays are not the same\n");
        goto Exit;
    }

Exit:
    // Destroy the cuRAND number generator
    CURAND_CALL(curandDestroyGenerator(gen));
    CURAND_CALL(curandDestroyGenerator(gen2));

    //Free device memory
    CUDA_CALL(cudaFree(d_A));
    CUDA_CALL(cudaFree(d_B));
    CUDA_CALL(cudaFree(d_C));
    CUDA_CALL(cudaFree(devNumGen));
    CUDA_CALL(cudaFree(devNumGen2));

    //Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(GpuRef);
    free(h_AR);
    free(h_BR);

    // Allows for the user to see the output when running in Visual Studio Pro 2019 (v142)
    char a;
    printf("Press Enter to continue");
    scanf("%c", &a);

    return 0;
}