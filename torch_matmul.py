# torch_matmul.py : 보너스 - PyTorch로 행렬곱 CPU vs GPU
# 실행: python torch_matmul.py
import time, torch

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

n = 4096
a_cpu = torch.randn(n, n)
b_cpu = torch.randn(n, n)

# CPU
t = time.perf_counter()
c_cpu = a_cpu @ b_cpu
cpu_ms = (time.perf_counter() - t) * 1000

# GPU: .to("cuda") 로 옮기기만 하면 끝
a = a_cpu.to("cuda"); b = b_cpu.to("cuda")
_ = a @ b; torch.cuda.synchronize()          # 워밍업

t = time.perf_counter()
for _ in range(10):
    c = a @ b
torch.cuda.synchronize()                     # 꼭 기다렸다가 시간 재기!
gpu_ms = (time.perf_counter() - t) * 1000 / 10

diff = (c.cpu() - c_cpu).abs().max().item()
print(f"matmul {n}x{n}  CPU: {cpu_ms:8.1f} ms")
print(f"matmul {n}x{n}  GPU: {gpu_ms:8.2f} ms")
print(f"Speedup: {cpu_ms/gpu_ms:.0f}x   (max abs diff {diff:.3f}, FP32/TF32 rounding)")
