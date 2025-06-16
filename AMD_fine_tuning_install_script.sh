#!/bin/bash

# Install `bitsandbytes` for ROCm 6.0+.
git clone --recurse https://github.com/ROCm/bitsandbytes.git
cd bitsandbytes
git checkout rocm_enabled_multi_backend
pip install -r requirements-dev.txt
cmake -DBNB_ROCM_ARCH="gfx1100" -DCOMPUTE_BACKEND=hip -S . # gfx1100 for RDNA3 - 7900 XTX 
python setup.py install

# To leverage the SFTTrainer in TRL for model fine-tuning.
pip install trl

# To leverage PEFT for efficiently adapting pre-trained language models .
pip install peft

# Install the other dependencies.
pip install transformers datasets huggingface-hub scipy