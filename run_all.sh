#!/bin/bash
# run_all.sh : 영상에서 따라 칠 명령어 순서 그대로
# 참고: 네트워크 볼륨(/workspace)은 실행 파일을 직접 실행할 수 없는 경우가 있어
#       빌드 결과물은 컨테이너 로컬 디스크(~/build)에 만든다. 소스는 /workspace에 보관.
cd /workspace/cuda_taster
mkdir -p ~/build
set -x
nvidia-smi
nvcc --version
nvcc -O2 -arch=native vector_add.cu -o ~/build/vector_add
~/build/vector_add
nvcc -O2 -arch=native beginner_error.cu -o ~/build/beginner_error
~/build/beginner_error
python torch_matmul.py
