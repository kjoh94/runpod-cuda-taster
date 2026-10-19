# runpod-cuda-taster

**RunPod에서 10분 만에 CUDA 맛보기** — GPU가 없어도 클라우드 GPU 하나 빌려서 첫 CUDA 프로그램을 돌려 보는 초보자용 예제입니다.
**A 10-minute CUDA taster on RunPod** — rent one cloud GPU and run your first CUDA program, no local GPU needed.

| 파일 / File | 내용 / What it does |
|---|---|
| `vector_add.cu` | 5천만 개 float 벡터 덧셈: CPU for 루프 vs GPU 커널, `cudaEvent` 시간 측정, 정답 확인 / 50M-float vector add, CPU loop vs GPU kernel, cudaEvent timing, correctness check |
| `beginner_error.cu` | 초보자 실수 2가지와 해결법 (블록 크기 초과 → `cudaGetLastError`, 동기화 없는 시간 측정 → `cudaDeviceSynchronize`) / two classic beginner mistakes and their fixes |
| `torch_matmul.py` | 보너스: PyTorch 4096×4096 행렬곱 CPU vs GPU / bonus: PyTorch matmul CPU vs GPU |
| `run_all.sh` | 위 순서대로 전부 실행 / runs everything in order |
| `sample_output.txt` | 실제 실행 결과 / real output from one run |

## 결과 예시 / Sample results

GPU: RTX PRO 6000 Blackwell **MIG 1g.24gb 슬라이스** (GPU 한 장을 나눈 24GB 조각), CUDA 12.8, PyTorch 2.8

| 실험 / Test | CPU | GPU | 배속 / Speedup |
|---|---|---|---|
| 벡터 덧셈 50M floats / vector add | 71.9 ms | 1.61 ms | **약 45배 / ~45x** |
| 행렬곱 4096×4096 (PyTorch) / matmul | 896.5 ms | 7.44 ms | **약 120배 / ~120x** |

> 숫자는 GPU 종류, CPU, 실행할 때마다 달라집니다. Numbers vary by GPU, CPU and run.

---

## 한국어: 따라 하기

### 1. Pod 만들기
1. RunPod 가입 후 **Billing**에서 크레딧 충전 (실습은 몇백 원이면 충분)
2. **Pods → Deploy** 클릭
3. GPU 고르기: 싼 GPU 하나면 됩니다 (RTX A4000 / RTX 4000 Ada / RTX 3090 / MIG 슬라이스 등, 시간당 $0.2~0.7)
4. 템플릿: **RunPod PyTorch** (이름에 CUDA가 들어간 공식 이미지, `nvcc` 포함)
5. 디스크: Container 20GB, Volume 20GB (`/workspace`)
6. **Deploy On-Demand**

### 2. 접속하기
- Pod 카드에서 **Connect → Web Terminal** (또는 Jupyter Lab → Terminal)

### 3. GPU 확인
```bash
nvidia-smi        # GPU 이름, 메모리, 드라이버 확인
nvcc --version    # CUDA 컴파일러 버전 확인
```

### 4. 코드 받기
```bash
cd /workspace
git clone https://github.com/kjoh94/runpod-cuda-taster.git cuda_taster
cd cuda_taster
```

### 5. 컴파일하고 실행
```bash
mkdir -p ~/build
nvcc -O2 -arch=native vector_add.cu -o ~/build/vector_add
~/build/vector_add
```
> **팁:** 네트워크 볼륨(`/workspace`)에서는 실행 파일이 `Permission denied`로 안 돌 수 있어요. 소스는 `/workspace`에 두고, 빌드 결과는 `~/build` 같은 컨테이너 로컬 디스크에 만드세요.

### 6. 초보자 실수 체험
```bash
nvcc -O2 -arch=native beginner_error.cu -o ~/build/beginner_error
~/build/beginner_error
```
- 블록당 스레드 2048개(최대 1024) → 에러 메시지 없이 틀린 값. `cudaGetLastError()`로 확인하면 `invalid configuration argument`
- 커널은 비동기 → 동기화 없이 시간을 재면 0.1ms 같은 거짓 숫자. `cudaDeviceSynchronize()` 후에 재야 진짜 시간

### 7. 보너스: PyTorch
```bash
python torch_matmul.py
```

### 8. ⚠️ 꼭 Pod 끄기
- 다 썼으면 **Stop**. 켜 둔 시간만큼 요금이 나갑니다.
- **Stop**: GPU 요금은 멈추고 볼륨 데이터는 남음 (볼륨 보관 요금은 소액 발생)
- **Terminate**: Pod 삭제. 네트워크 볼륨이 아니면 데이터도 사라짐

한 번에 다 돌리려면 `bash run_all.sh`.

---

## English: step by step

1. **Create a pod** — sign up, add a little credit (Billing), **Pods → Deploy**, pick any cheap single GPU, choose the official **RunPod PyTorch** template (CUDA devel image with `nvcc`), container disk 20 GB, volume 20 GB at `/workspace`, **Deploy On-Demand**.
2. **Connect** — **Connect → Web Terminal** (or Jupyter Lab → Terminal).
3. **Check the GPU** — `nvidia-smi`, `nvcc --version`.
4. **Get the code** — `cd /workspace && git clone https://github.com/kjoh94/runpod-cuda-taster.git cuda_taster && cd cuda_taster`
5. **Build & run** — `mkdir -p ~/build && nvcc -O2 -arch=native vector_add.cu -o ~/build/vector_add && ~/build/vector_add`
   *Tip:* network volumes may refuse to execute binaries (`Permission denied`). Keep sources on `/workspace`, build into local disk such as `~/build`.
6. **Beginner mistakes** — `beginner_error.cu` shows (a) an invalid launch (2048 threads/block > 1024) failing silently until you call `cudaGetLastError()`, and (b) timing an async kernel without `cudaDeviceSynchronize()`.
7. **Bonus** — `python torch_matmul.py`.
8. **⚠️ Stop the pod** when you are done. Stop = no GPU billing, volume kept. Terminate = pod deleted.

Or just run `bash run_all.sh`.

## License
MIT
