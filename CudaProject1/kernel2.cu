// Inclusion of header files for running CUDA in Visual Studio Pro 2019 (v142)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Inclusion of the required CUDA libriaries and header files
#include <curand.h>
#include <cuda.h>

// Inclusion of headers from the standard library in C
#include <cstdio>
#include <cstdint> //uint32_t
#include <cstdlib>
#include <ctime>
#include <cmath>


// Inclusion if headers from the C++ STL
#include <any>
#include <iostream>
#include <string>
#include <thread>

using std::cout;
using std::endl;

#if _WIN32
// Windows implementation of the Linux sys/time.h fnuctions needed in this program
#include <sys/timeb.h>
#include <sys/types.h>
#include <winsock2.h>

#define __need_clock_t

/* Structure describing CPU time used by a process and its children.  */
struct tms
{
    clock_t tms_utime;          /* User CPU time.  */
    clock_t tms_stime;          /* System CPU time.  */

    clock_t tms_cutime;         /* User CPU time of dead children.  */
    clock_t tms_cstime;         /* System CPU time of dead children.  */
};

// CUDA 8+ requirment
struct timezone 
{
    int tz_minuteswest; /* minutes west of Greenwich */
    int tz_dsttime; /* type of DST correction */
};

int 
gettimeofday(struct timeval* t, std::any timezone)
{
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    t->tv_sec = static_cast<long>(timebuffer.time);
    t->tv_usec = 1000 * timebuffer.millitm;
    return 0;
}

/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t 
times(struct tms* __buffer) 
{

    __buffer->tms_utime = clock();
    __buffer->tms_stime = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}
typedef long long suseconds_t;
#else
#include <sys/time.h>
#endif

// Q_SQRT function using C
float 
quick_reverse_sqrt(const float number)
{
    const float x2 = number * 0.5F;
    const float three_halves = 1.5F;

    union {
        float f;
        uint32_t i;
    } conv;
    conv.f = number;
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f *= three_halves - (x2 * conv.f * conv.f);
    conv.f *= three_halves - (x2 * conv.f * conv.f);
    conv.f *= three_halves - (x2 * conv.f * conv.f);
    return conv.f;
}

// Error codes for any error thrown in the next function
enum QuickInverseSqrtError_t 
{
    QRSqrtSuccess = 0,
    ArrayOutOfBoundsException = 1,
    DivideByZero = 2,
    UnknownError = 3,
    NDoesNotMatchSizeOfArray = 4
};

// Run Q_rsqrt over an array
QuickInverseSqrtError_t 
calcOverArray(float* array, size_t N) 
{
    if ((sizeof(*array) / sizeof(array[0])) != N) return NDoesNotMatchSizeOfArray;

    for (int i = 0; i < static_cast<int>(N); i++) {
        if (array[i] == 0) return DivideByZero;
        array[i] = quick_reverse_sqrt(array[i]);
    }
    return QRSqrtSuccess;
}

// CUDA error check to get error name
std::string 
CUDA_CHECK_VAL(cudaError_t x) 
{
    std::string msg;

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
std::string 
CURAND_CHECK_VAL(curandStatus_t x) 
{
    std::string msg;

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
#define CUDA_CALL(x) { if(x != cudaSuccess){ cout << "Error: " << CUDA_CHECK_VAL(x) << " at " << __FILE__ << ":" << __LINE__ << endl; return EXIT_FAILURE;}}

// Check method for checking the error status of a cuRAND call
#define CURAND_CALL(x) {if(x != CURAND_STATUS_SUCCESS){ cout << "Error: " << CURAND_CHECK_VAL(x) << " at " << __FILE__ << ":" << __LINE__ << endl; return EXIT_FAILURE;}}

// The kernel, which runs on the GPU when called
__global__ void 
kernel(const int* a, const int* b, int* c, const size_t N)
{
    if (const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N) c[idx] = a[idx] * b[idx];
}

__global__ void 
make_float_larger(float* a, size_t N)
{
    if (const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N) a[idx] *= 100;
}
 
// Program to convert a float array to an integer array
__host__ void 
f_to_i_array(int* dst, const float* src, const size_t n_elem)
{
    for (int i = 0; i < n_elem; i++)
        dst[i] = static_cast<int>(src[i] * 1000);
}

// Function for verifying the array generated by the kernel is correct
__host__ bool inline 
check(const int* a, const int* b, const size_t size)
{
    for (int x = 0; x < size; x++)
    {
        if (a[x] - b[x] > 1.0E-8)
            return true;
    }
    return false;
}

__host__ double 
cpu_second()
{
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return (static_cast<double>(tp.tv_sec) + static_cast<double>(tp.tv_usec) * 1.e-6);
}

int
array_play()
{
    float* dev_a;
    curandGenerator_t test;
    const size_t n_size = 50 * sizeof(float);
    auto* a = static_cast<float*>(malloc(n_size));
    auto* b = static_cast<float*>(malloc(n_size));
    CUDA_CALL(cudaMalloc(static_cast<float**>(&dev_a), n_size))
    CURAND_CALL(curandCreateGenerator(&test, CURAND_RNG_PSEUDO_DEFAULT))
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(test, time(NULL)))
    CURAND_CALL(curandGenerateUniform(test, dev_a, 50))
    make_float_larger<<<1, 50>>>(dev_a, n_size);
    CUDA_CALL(cudaDeviceSynchronize())
    CUDA_CALL(cudaMemcpy(a, dev_a, n_size, cudaMemcpyDeviceToHost))
    memcpy(b, a, n_size);
    if (memcmp(a, b, n_size) == 0) {
        for (int i = 0; i < 50; i++)
            cout << quick_reverse_sqrt(a[i]) << " = " << 1 / std::sqrt(b[i]) << endl;
    }
    else
        printf("Error, arrays are not the same");

    CURAND_CALL(curandDestroyGenerator(test))
    CUDA_CALL(cudaFree(dev_a))
    free(a);
    free(b);
    return 0;
}

// Function for generating the same results as the GPU kernel, used for verification of results
__host__ void 
KernelCPUEd(int* a, int* b, int* c, int begin, int end)
{
    for (int i = begin; i < end; i++)
        c[i] = a[i] * b[i];
}

// Function for partitioning the array into four equal parts and executing them
void 
get_bounds_and_compute(int* a, int* b, int* c, size_t n_elem)
{
    struct data {
        struct partitionOne {
            int begin = 0;
            int end = 0;
        } partition_one;
        struct partitionTwo {
            int begin = 0;
            int end = 0;
        } partition_two;
        struct partitionThree {
            int begin = 0;
            int end = 0;
        } partition_three;
        struct partitionFour {
            int begin = 0;
            int end = 0;
        } partition_four;
    } data_struct;

    const int partition_size = static_cast<signed int>(n_elem) / 4;
    data_struct.partition_one.begin = 0;
    data_struct.partition_one.end = (1 << static_cast<int>(log2(partition_size))) - 1;

    data_struct.partition_two.begin = data_struct.partition_one.end + 1;
    data_struct.partition_two.end = data_struct.partition_two.begin + partition_size - 1;

    data_struct.partition_three.begin = data_struct.partition_two.end + 1;
    data_struct.partition_three.end = data_struct.partition_three.begin + partition_size - 1;

    data_struct.partition_four.begin = data_struct.partition_three.end + 1;
    data_struct.partition_four.end = data_struct.partition_four.begin + partition_size - 1;

    std::thread partition_one(KernelCPUEd, a, b, c, data_struct.partition_one.begin, data_struct.partition_one.end);
    std::thread partition_two(KernelCPUEd, a, b, c, data_struct.partition_two.begin, data_struct.partition_two.end);
    std::thread partition_three(KernelCPUEd, a, b, c, data_struct.partition_three.begin, data_struct.partition_three.end);
    std::thread partition_four(KernelCPUEd, a, b, c, data_struct.partition_four.begin, data_struct.partition_four.end);

    partition_one.join();
    partition_two.join();
    partition_three.join();
    partition_four.join();
}

// Entry point to the program
int 
main()
{
    size_t nElem = 1 << 28;

    size_t nBytes = nElem * sizeof(int);
    size_t nBytesF = nElem * sizeof(float);

    int* h_A, * h_B, * h_C, * GpuRef;
    int* d_A, * d_B, * d_C;

    float* devNumGen, * devNumGen2, * h_AR, * h_BR;

    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, dev))
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CALL(cudaSetDevice(dev))

    curandGenerator_t gen, gen2;

    // Allocation of memory on the host for transferring data from host to device and vice versa
    h_A = static_cast<int*>(malloc(nBytes));
    h_B = static_cast<int*>(malloc(nBytes));
    h_C = static_cast<int*>(malloc(nBytes));
    GpuRef = static_cast<int*>(malloc(nBytes));

    // Allocation of memory on the device for storage of data needed by the kernel during runtime
    CUDA_CALL(cudaMalloc(static_cast<int**>(&d_A), nBytes))
    CUDA_CALL(cudaMalloc(static_cast<int**>(&d_B), nBytes))
    CUDA_CALL(cudaMalloc(static_cast<int**>(&d_C), nBytes))

    // Allocation of memory on host and device for testing the CUDA number generator
    h_AR = static_cast<float*>(malloc(nBytes));
    h_BR = static_cast<float*>(malloc(nBytes));
    CUDA_CALL(cudaMalloc(static_cast<float**>(&devNumGen), nBytesF))
    CUDA_CALL(cudaMalloc(static_cast<float**>(&devNumGen2), nBytesF))

    // CUDA number generator function calls and return values
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT))
    CURAND_CALL(curandCreateGenerator(&gen2, CURAND_RNG_PSEUDO_DEFAULT))

    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)))
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen2, time(NULL) + 1))

    CURAND_CALL(curandGenerateUniform(gen, devNumGen, nElem))
    CURAND_CALL(curandGenerateUniform(gen2, devNumGen2, nElem))

    // Transfer random numbers generated on device to host
    CUDA_CALL(cudaMemcpy(h_AR, devNumGen, nBytesF, cudaMemcpyDeviceToHost))
    CUDA_CALL(cudaMemcpy(h_BR, devNumGen2, nBytesF, cudaMemcpyDeviceToHost))

    f_to_i_array(h_A, h_AR, nElem);
    f_to_i_array(h_B, h_BR, nElem);

    // Transfer of populated arrays to the device for use by the kernel
    CUDA_CALL(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice))
    CUDA_CALL(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice))

    // Calculate block indices
    int iLen = 1 << 8;
    dim3 block(iLen, 1);
    dim3 grid((nElem + block.x - 1) / block.x, 1);

    // Kernel call to run the calculation n the GPU, uses 1 block and nElem amount of threads in the block
    // Max threads in a block for RTX 2060 is 4096 threads
    double iStart = cpu_second();
    kernel <<<grid, block>>> (d_A, d_B, d_C, nElem);
    CUDA_CALL(cudaDeviceSynchronize())
    double iEnd = cpu_second() - iStart;

    printf("Execution time of the GPU kernel <<<%d, %d>>>: %g\n", grid.x, block.x, iEnd);

    // Verification function that the kernel on the GPU is performing properly
    double iStartCPU = cpu_second();
    get_bounds_and_compute(h_A, h_B, h_C, nElem);
    double iEndCPU = cpu_second() - iStartCPU;
    printf("Execution time of the CPU function %g\n", iEndCPU);

    // Transfer of data from Device to the host
    CUDA_CALL(cudaDeviceSynchronize())
    CUDA_CALL(cudaMemcpy(GpuRef, d_C, nBytes, cudaMemcpyDeviceToHost))

    // Verification of data, compares data generated on the host to the data generated on the device
    // If the data is different, goto Exit is called and memory is freed, the the program ends
    if (check(h_C, GpuRef, nElem))
    {
        printf("The arrays are not the same\n");
    }
	
    // Destroy the cuRAND number generator
    CURAND_CALL(curandDestroyGenerator(gen))
    CURAND_CALL(curandDestroyGenerator(gen2))

    //Free device memory
    CUDA_CALL(cudaFree(d_A))
    CUDA_CALL(cudaFree(d_B))
    CUDA_CALL(cudaFree(d_C))
    CUDA_CALL(cudaFree(devNumGen))
    CUDA_CALL(cudaFree(devNumGen2))

    //Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(GpuRef);
    free(h_AR);
    free(h_BR);

    // Allows for the user to see the output when running in Visual Studio Pro 2019 (v142)
    char end;
    printf("Press Enter to continue");
    scanf_s("%c", &end);

    return 0;
}