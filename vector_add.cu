// vector_add.cu : CUDA 맛보기 - CPU vs GPU 벡터 덧셈
// 빌드:  nvcc -O2 vector_add.cu -o vector_add
// 실행:  ./vector_add
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// GPU에서 실행되는 함수(커널). 스레드 하나가 원소 하나를 더한다.
__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // 내 스레드 번호 = 내가 맡을 원소 번호
    if (i < n) c[i] = a[i] + b[i];                   // 범위 밖 스레드는 아무것도 안 함
}

// 에러 체크를 짧게 쓰기 위한 매크로
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

int main() {
    const int N = 50 * 1000 * 1000;              // 5천만 개
    size_t bytes = N * sizeof(float);
    printf("N = %d floats (%.0f MB per array)\n", N, bytes / 1e6);

    // 1) CPU 메모리(호스트) 준비
    float *a = (float*)malloc(bytes), *b = (float*)malloc(bytes);
    float *c_cpu = (float*)malloc(bytes), *c_gpu = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) { a[i] = i * 0.5f; b[i] = i * 0.25f; }

    // 2) CPU로 더하기 (for 루프)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) c_cpu[i] = a[i] + b[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 3) GPU 메모리(디바이스) 준비 + 데이터 복사
    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc(&d_a, bytes)); CHECK(cudaMalloc(&d_b, bytes)); CHECK(cudaMalloc(&d_c, bytes));
    CHECK(cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice));

    // 4) 커널 실행 설정: 블록당 256 스레드, 필요한 블록 수 계산
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);   // 워밍업(첫 실행은 초기화 비용 포함)
    CHECK(cudaDeviceSynchronize());

    // 5) cudaEvent로 GPU 커널 시간만 정확히 재기
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    addKernel<<<blocks, threads>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0; cudaEventElapsedTime(&gpu_ms, start, stop);
    CHECK(cudaGetLastError());

    // 6) 결과를 CPU로 가져와서 정답 확인
    CHECK(cudaMemcpy(c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < N; i++) if (fabsf(c_cpu[i] - c_gpu[i]) > 1e-3f * fabsf(c_cpu[i]) + 1e-5f) wrong++;

    printf("CPU loop   : %8.2f ms\n", cpu_ms);
    printf("GPU kernel : %8.2f ms\n", gpu_ms);
    printf("Speedup    : %8.1fx\n", cpu_ms / gpu_ms);
    printf("Check      : %s (%d wrong)\n", wrong == 0 ? "PASS" : "FAIL", wrong);

    // 7) 정리
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(a); free(b); free(c_cpu); free(c_gpu);
    return 0;
}
