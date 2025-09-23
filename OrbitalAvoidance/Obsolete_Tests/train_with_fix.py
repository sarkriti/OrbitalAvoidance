#!/usr/bin/env python3
"""
Training script with PyTorch/JVM compatibility fixes.
"""

import os
import sys

# Set environment variables to fix PyTorch/JVM conflicts
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# Set CUDA cache directory to avoid conflicts
os.environ['CUDA_CACHE_PATH'] = '/tmp/cuda_cache'

# Import PyTorch first to initialize before JVM
import torch
torch.set_num_threads(1)

# Now import and run the regular training
if __name__ == '__main__':
    import rl_train
    rl_train.main()