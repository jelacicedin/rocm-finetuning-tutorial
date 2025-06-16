#!/bin/bash
# Get the absolute path of the current directory
WORKDIR=$(pwd)


docker run -it \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 8G \
  --device=/dev/dxg \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
  -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1 \
  -v $(pwd):/workspace \
  -w /workspace \
  rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.6.0