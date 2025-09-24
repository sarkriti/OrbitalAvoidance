# 📦 Orbital Avoidance Library Dependencies

This document provides comprehensive information about the library dependencies required for the Orbital Avoidance RL training system.

## 🎯 Quick Setup

### Prerequisites
- **Python**: 3.10+ (tested with Python 3.10)
- **CUDA**: 12.8+ (for GPU acceleration)
- **Operating System**: Linux (tested on Ubuntu 22.04)
- **Memory**: 16GB+ RAM recommended (503GB available in test environment)
- **GPU**: NVIDIA GPU with Compute Capability 8.6+ recommended

### Installation Methods

#### Method 1: Virtual Environment (Recommended)
```bash
# Create and activate virtual environment
python3 -m venv orbital-venv
source orbital-venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

#### Method 2: Conda Environment
```bash
# Create conda environment
conda create -n orbital python=3.10
conda activate orbital

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Core Dependencies

### Machine Learning & RL Frameworks
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.8.0 | Deep learning framework with CUDA support |
| `tensorflow` | 2.15.0 | Alternative ML framework for some components |
| `stable_baselines3` | 2.7.0 | Reinforcement learning algorithms (PPO, SAC, TD3) |
| `gymnasium` | 1.2.1 | OpenAI Gym environment interface |

### Scientific Computing
| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 1.26.4 | Numerical computing (compatible with TensorFlow) |
| `pandas` | 2.3.2 | Data manipulation and analysis |
| `matplotlib` | 3.10.6 | Plotting and visualization |

### Orbital Mechanics
| Package | Version | Purpose |
|---------|---------|---------|
| `orekit_jpype` | 13.1.1.0 | High-precision orbital mechanics calculations |
| `jpype1` | 1.6.0 | Java-Python bridge for Orekit |

### Hardware Detection
| Package | Version | Purpose |
|---------|---------|---------|
| `psutil` | 7.1.0 | System resource monitoring and hardware detection |

## ⚠️ Critical Compatibility Notes

### TensorFlow & Protobuf Compatibility
**IMPORTANT**: The specific versions below are required to avoid NumPy 2.0 compatibility issues:

```
tensorflow==2.15.0
protobuf==3.20.3  # NOT 4.x - causes import failures
numpy==1.26.4     # NOT 2.0+ - breaks TensorFlow
```

### CUDA Libraries
The following NVIDIA CUDA libraries are automatically installed with PyTorch:
```
nvidia-cublas-cu12==12.8.4.1
nvidia-cudnn-cu12==9.10.2.21
nvidia-cuda-runtime-cu12==12.8.90
# ... (see requirements.txt for complete list)
```

## 🚀 Hardware Acceleration

### GPU Requirements
- **NVIDIA GPU** with CUDA Compute Capability 8.6+
- **CUDA Driver**: 12.8+
- **VRAM**: 8GB+ recommended (24GB+ for large models)

### Tested Hardware Configuration
- **GPUs**: 6x NVIDIA GPUs (4x RTX A6000 48GB, 2x RTX 3090 24GB)
- **CPU**: 64-core x86_64 with 128 logical cores
- **RAM**: 503.7GB
- **CUDA**: 12.8

### Hardware Detection Features
The system automatically detects and optimizes for:
- CPU core count and threading
- GPU availability and memory
- Optimal batch sizes based on hardware
- Memory fraction allocation

## 🛠️ Installation Troubleshooting

### Common Issues

#### 1. Protobuf Import Errors
```bash
# Error: AttributeError: module 'google.protobuf.descriptor' has no attribute '_internal_create_key'
# Solution: Use exact protobuf version
pip install protobuf==3.20.3
```

#### 2. NumPy Version Conflicts
```bash
# Error: TensorFlow incompatible with NumPy 2.0+
# Solution: Downgrade NumPy
pip install numpy==1.26.4
```

#### 3. CUDA Library Issues
```bash
# Error: CUDA runtime version mismatch
# Solution: Verify CUDA installation
nvidia-smi
nvcc --version
```

#### 4. Orekit Java Dependencies
The system requires Java runtime for orbital mechanics calculations:
```bash
# Verify Java installation
java --version
# Should be Java 11+ for Orekit 13.x
```

### Environment Variables
For GPU restriction (useful for multi-GPU systems):
```bash
# Use specific GPUs (e.g., GPUs 3 and 5)
export CUDA_VISIBLE_DEVICES=3,5
python3 OrbitalAvoidance/rl_train.py --algorithm ppo
```

## 📊 Performance Optimization

### Automatic Hardware Configuration
The system automatically configures:
- **CPU Threads**: Optimized based on core count
- **GPU Memory**: 80% allocation by default
- **Batch Sizes**: Hardware-appropriate sizing
- **Device Selection**: Best available GPU

### Memory Management
- Automatic GPU memory fraction management
- CPU memory monitoring
- Efficient batch processing based on available VRAM

## 🔄 Dependency Management

### Virtual Environment Benefits
1. **Isolation**: Prevents conflicts with system packages
2. **Reproducibility**: Exact version control
3. **Easy cleanup**: Remove environment without affecting system
4. **Testing**: Safe environment for experiments

### Version Pinning Strategy
- **Exact versions** for core ML libraries (stability)
- **Compatible ranges** for utilities (flexibility)
- **CUDA version matching** for GPU acceleration

## 🧪 Verification

### Test Hardware Detection
```bash
python3 test_hardware_detection.py
```

### Test GPU Training
```bash
python3 test_gpu_training.py
```

### Quick CUDA Test
```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

## 📝 Development Notes

### Adding New Dependencies
1. Install in virtual environment
2. Test compatibility with existing packages
3. Update `requirements.txt` with exact versions
4. Update this README with purpose and notes

### Package Update Strategy
- **Major updates**: Test thoroughly in separate environment
- **Security updates**: Apply cautiously, test compatibility
- **CUDA updates**: Ensure driver compatibility

## 🆘 Support

### Debug Information Collection
```bash
# System info
python3 -c "import sys; print(f'Python: {sys.version}')"
nvidia-smi
pip freeze > installed_packages.txt

# Hardware detection
python3 -c "from hardware_detection import HardwareDetector; print(HardwareDetector().get_full_summary())"
```

### Common Solutions
1. **Clean virtual environment** for dependency conflicts
2. **Exact version matching** for compatibility issues
3. **GPU restriction** for memory problems
4. **Java runtime check** for Orekit issues

---

*Last updated: September 23, 2025*
*Tested on Ubuntu 22.04 with CUDA 12.8*