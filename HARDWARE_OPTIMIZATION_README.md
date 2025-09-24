# Hardware Detection and Optimization System

This system provides automatic CPU and GPU detection with intelligent resource utilization for optimal performance across different hardware configurations while maintaining flexibility for various deployment environments.

## 🚀 Features

### Hardware Detection
- **CPU Detection**: Physical/logical cores, frequency, memory, architecture
- **GPU Detection**: NVIDIA GPUs with memory, utilization, compute capability
- **Cross-platform**: Works on Linux, Windows, macOS
- **Framework Support**: PyTorch, TensorFlow, stable-baselines3

### Automatic Optimization
- **Batch Size Adjustment**: Based on available GPU/CPU memory
- **Thread Configuration**: Optimal CPU thread allocation
- **Device Selection**: Intelligent GPU vs CPU selection
- **Memory Management**: GPU memory fraction control
- **Multiprocessing**: Smart process/thread usage

### Flexible Deployment
- **Fallback Mechanisms**: Graceful degradation when hardware detection fails
- **Environment Variables**: Automatic OpenMP, MKL, CUDA configuration
- **Manual Override**: Command-line options for custom configurations

## 📁 Files Overview

### Core Hardware Detection
- `hardware_detection.py` - Main hardware detection and configuration module
- `test_hardware_detection.py` - Comprehensive validation test suite

### RL Training Integration
- `OrbitalAvoidance/rl_algorithms.py` - Updated with hardware-aware algorithm configuration
- `OrbitalAvoidance/rl_train.py` - Main training script with hardware detection
- `Geo_to_Leo/satellite_rl_training.py` - Satellite training with hardware optimization

### Physics Integration
- `OrbitalAvoidance/orekit_physics.py` - Physics calculations with CPU optimization

## 🔧 Quick Start

### 1. View Hardware Configuration
```bash
python3 test_hardware_detection.py --info
```

### 2. Run Hardware Validation Tests
```bash
python3 test_hardware_detection.py
```

### 3. Train with Automatic Hardware Detection
```bash
# Main RL training with auto-detection
cd OrbitalAvoidance
python3 rl_train.py --algorithm ppo --scenario scenarios/leo_debris_avoidance.json

# Satellite training with hardware optimization
cd Geo_to_Leo
python3 satellite_rl_training.py
```

### 4. Manual Hardware Configuration
```bash
# Force CPU-only training
python3 rl_train.py --prefer-gpu false --max-threads 8

# Limit GPU memory usage
python3 rl_train.py --gpu-memory-fraction 0.5
```

## 🖥️ Hardware Configuration Examples

### High-End Workstation (Example Output)
```
💻 CPU Information:
   Brand: x86_64
   Cores: 64 physical, 128 logical
   Memory: 503.7 GB total, 493.9 GB available

🎮 GPU Information:
   GPU 0: NVIDIA RTX A6000 (48,539 MB)
   GPU 1: NVIDIA RTX A6000 (48,539 MB)
   [... additional GPUs ...]

⚙️ Optimal Configuration:
   Primary Device: cuda:0
   CPU Threads: 96
   Use GPU: Yes
   Use Multiprocessing: No
```

### Laptop/Modest Hardware
```
💻 CPU Information:
   Cores: 4 physical, 8 logical
   Memory: 16.0 GB total, 12.1 GB available

🎮 GPU Information:
   GPU 0: NVIDIA GTX 1660 Ti (6,144 MB)

⚙️ Optimal Configuration:
   Primary Device: cuda:0
   CPU Threads: 6
   Use GPU: Yes
   Use Multiprocessing: No
```

### CPU-Only Server
```
💻 CPU Information:
   Cores: 32 physical, 64 logical
   Memory: 256.0 GB total, 240.3 GB available

🎮 GPU Information: No GPUs detected

⚙️ Optimal Configuration:
   Primary Device: cpu
   CPU Threads: 48
   Use GPU: No
   Use Multiprocessing: Yes
```

## 🎛️ Configuration Options

### Command Line Arguments
```bash
# Hardware preferences
--prefer-gpu              # Prefer GPU over CPU (default: True)
--gpu-memory-fraction 0.8 # GPU memory usage fraction (default: 0.8)
--max-threads 16          # Maximum CPU threads (default: auto-detect)

# Training parameters (work with hardware detection)
--algorithm ppo           # RL algorithm (ppo, sac, td3)
--batch-size 64          # Will be adjusted based on hardware
--timesteps 100000       # Training timesteps
```

### Python Configuration
```python
from hardware_detection import get_optimal_config, configure_ml_frameworks

# Get hardware configuration
config = get_optimal_config(
    prefer_gpu=True,
    memory_fraction=0.8,
    max_threads=None  # Auto-detect
)

# Apply to ML frameworks
configure_ml_frameworks(config)

# Manual configuration
config = get_optimal_config(
    prefer_gpu=False,          # Force CPU
    memory_fraction=0.5,       # Conservative memory usage
    max_threads=8              # Limit CPU threads
)
```

## 🔍 Hardware Detection Details

### CPU Detection
- **Physical/Logical Cores**: Using `psutil.cpu_count()`
- **Memory**: Total and available RAM
- **Frequency**: Current and maximum CPU frequency
- **Architecture**: Platform detection (x86_64, arm64, etc.)
- **Usage**: Real-time CPU utilization

### GPU Detection
- **PyTorch Detection**: Primary method using `torch.cuda`
- **NVIDIA-ML**: Secondary detection with `pynvml` for detailed info
- **Memory Information**: Total, free, and used VRAM
- **Compute Capability**: CUDA compute version
- **Utilization**: Real-time GPU usage percentage

### Intelligent Configuration
- **Batch Size**: Automatically adjusted based on available memory
  - High-end GPU (≥8GB): Larger batches (128-512)
  - Mid-range GPU (4-8GB): Standard batches (64-256)
  - Low-end GPU (<4GB): Smaller batches (32-128)
  - CPU-only: Memory-based adjustment

- **Threading**: Optimal CPU thread allocation
  - Leaves 1-2 cores free for system processes
  - Considers hyperthreading efficiency
  - Adjusts for GPU workloads (fewer CPU threads when GPU active)

- **Multiprocessing**: Smart process/thread selection
  - Enabled for CPU-heavy workloads
  - Disabled when using GPU (avoids memory conflicts)
  - Considers available memory per process

## 🛠️ Integration Guide

### Adding Hardware Detection to New Components

1. **Import the module**:
```python
from hardware_detection import get_optimal_config, configure_ml_frameworks
```

2. **Get configuration**:
```python
hardware_config = get_optimal_config()
configure_ml_frameworks(hardware_config)
```

3. **Use configuration**:
```python
device = hardware_config.torch_device  # 'cuda:0' or 'cpu'
threads = hardware_config.recommended_threads
batch_size = 64 if not hardware_config.use_gpu else 128
```

### Custom Algorithm Integration
```python
class CustomAlgorithm:
    def __init__(self, config):
        # Hardware setup
        if HARDWARE_DETECTION_AVAILABLE:
            self.hardware_config = get_optimal_config()
            configure_ml_frameworks(self.hardware_config)
            self.device = self.hardware_config.torch_device
        else:
            self.device = "cpu"
            
    def train(self, data):
        # Use self.device for model placement
        model = Model().to(self.device)
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   - Check NVIDIA drivers: `nvidia-smi`
   - Verify PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
   - Install pynvml: `pip install pynvml`

2. **Memory Errors**:
   - Reduce GPU memory fraction: `--gpu-memory-fraction 0.5`
   - Force CPU mode: `--prefer-gpu false`
   - Check available memory before training

3. **Performance Issues**:
   - Monitor GPU utilization: `nvidia-smi -l 1`
   - Check CPU usage: `htop`
   - Adjust batch size manually if needed

4. **Import Errors**:
   - Install missing dependencies: `pip install psutil torch stable-baselines3`
   - Check Python environment consistency

### Debug Mode
```bash
# Enable detailed logging
export PYTHONPATH=/path/to/OrbitalAvoidance:$PYTHONPATH
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from hardware_detection import print_hardware_info
print_hardware_info()
"
```

## 🚀 Performance Benefits

### Expected Performance Improvements

1. **GPU Acceleration**: 5-20x speedup for RL training
2. **Optimal Batching**: 20-50% training efficiency improvement
3. **CPU Threading**: 2-4x speedup for CPU-intensive operations
4. **Memory Optimization**: Reduced OOM errors, larger model capacity

### Benchmark Results (Example)
```
Environment: Leo Debris Avoidance
Training Steps: 100,000

Configuration    | Time   | GPU Util | Memory
----------------|--------|----------|--------
Auto-detected   | 45 min | 85%      | 12GB
Manual (basic)  | 78 min | 60%      | 8GB
CPU-only        | 240min | N/A      | 6GB
```

## 📋 Dependencies

### Required
- `psutil` - System and process utilities
- `numpy` - Numerical computing

### Optional (for enhanced features)
- `torch` - PyTorch GPU detection and configuration
- `tensorflow` - TensorFlow optimization
- `pynvml` - NVIDIA GPU monitoring
- `stable-baselines3` - RL algorithms
- `gymnasium` - RL environments

### Installation
```bash
# Core dependencies
pip install psutil numpy

# ML frameworks (optional)
pip install torch torchvision torchaudio
pip install tensorflow
pip install stable-baselines3[extra]
pip install gymnasium[all]

# GPU monitoring (optional)
pip install pynvml
```

## 🤝 Contributing

When adding new hardware optimizations:

1. **Maintain Fallbacks**: Always provide CPU-only alternatives
2. **Test Cross-Platform**: Verify on different OS/hardware combinations
3. **Document Configuration**: Add new options to this README
4. **Validate Performance**: Benchmark improvements
5. **Handle Errors Gracefully**: Don't crash on detection failures

## 📊 Monitoring and Metrics

### Built-in Monitoring
The system logs hardware configuration and performance metrics:

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Hardware info is automatically logged during initialization
```

### External Monitoring
- **GPU**: `nvidia-smi`, `gpustat`
- **CPU**: `htop`, `top`
- **Memory**: System monitors
- **Training**: TensorBoard logs include hardware utilization

---

This hardware detection system ensures your orbital simulation and RL training runs optimally on any hardware configuration, from high-end workstations to resource-constrained environments.