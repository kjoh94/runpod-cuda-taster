// beginner_error.cu : 초보자가 자주 하는 실수 2가지와 고치는 법
// 빌드:  nvcc -O2 beginner_error.cu -o beginner_error
// 실행:  ./beginner_error
#include <cstdio>
#include <cuda_runtime.h>

__global__ void fill(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 7;
}

int main() {
    const int N = 1 << 20;
    int *d, *h = new int[N]();       // h는 0으로 초기화
    cudaMalloc(&d, N * sizeof(int));

    // ---------- 실수 1: 블록당 스레드를 너무 많이 (최대 1024) ----------
    printf("[BAD ] fill<<<..., 2048>>>  (block size too big)\n");
    fill<<<(N + 2047) / 2048, 2048>>>(d, N);
    // 커널 실행 에러는 '조용히' 실패한다. 확인하지 않으면 모른다!
    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("       h[0] = %d  (expected 7)  <- 아무 에러 메시지 없이 틀린 값\n", h[0]);

    // ---------- 고치기 1: cudaGetLastError()로 확인 ----------
    fill<<<(N + 2047) / 2048, 2048>>>(d, N);
    cudaError_t err = cudaGetLastError();
    printf("[FIX ] cudaGetLastError() -> %s\n", cudaGetErrorString(err));

    printf("[GOOD] fill<<<..., 256>>>\n");
    fill<<<(N + 255) / 256, 256>>>(d, N);
    err = cudaGetLastError();
    printf("       launch -> %s\n", cudaGetErrorString(err));
    cudaMemcpy(h, d, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("       h[0] = %d  (expected 7)\n", h[0]);

    // ---------- 실수 2: 동기화 없이 시간 재기 ----------
    // 커널 실행은 '비동기' -> CPU는 GPU가 끝나기 전에 바로 다음 줄로 간다.
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    const int BIG = 1 << 26;
    int* big; cudaMalloc(&big, BIG * sizeof(int));
    clock_t c0 = clock();
    for (int r = 0; r < 50; r++) fill<<<(BIG + 255) / 256, 256>>>(big, BIG);
    clock_t c1 = clock();                                   // 동기화 없음!
    printf("[BAD ] timing without sync : %.2f ms (너무 빠름 = 거짓말)\n", 1000.0 * (c1 - c0) / CLOCKS_PER_SEC);
    cudaDeviceSynchronize();                                // GPU가 끝날 때까지 기다림
    clock_t c2 = clock();
    printf("[FIX ] timing with cudaDeviceSynchronize : %.2f ms\n", 1000.0 * (c2 - c0) / CLOCKS_PER_SEC);

    cudaFree(d); cudaFree(big); delete[] h;
    return 0;
}
