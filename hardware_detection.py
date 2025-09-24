#!/usr/bin/env python3
"""
Hardware Detection and Configuration Module

This module provides centralized hardware detection and configuration capabilities
for optimal resource utilization across CPU and GPU devices while maintaining
flexibility for different deployment environments.

Features:
- Automatic CPU and GPU detection
- PyTorch and TensorFlow device configuration
- Fallback mechanisms for resource-constrained environments
- Thread and process optimization
- Memory usage monitoring and optimization
"""

import os
import sys
import platform
import psutil
import multiprocessing
import warnings
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class CPUInfo:
    """CPU hardware information."""
    cores_physical: int
    cores_logical: int
    frequency_max: float
    frequency_current: float
    architecture: str
    brand: str
    usage_percent: float
    memory_total_gb: float
    memory_available_gb: float

@dataclass
class GPUInfo:
    """GPU hardware information."""
    id: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    memory_used_mb: int
    utilization_percent: int
    compute_capability: Optional[Tuple[int, int]]
    is_available: bool
    driver_version: str

@dataclass
class HardwareConfig:
    """Hardware configuration for optimal resource utilization."""
    cpu_info: CPUInfo
    gpu_info: List[GPUInfo]
    recommended_threads: int
    recommended_processes: int
    torch_device: str
    tf_device: str
    use_gpu: bool
    use_multiprocessing: bool
    memory_fraction: float

class HardwareDetector:
    """Comprehensive hardware detection and configuration system."""
    
    def __init__(self, enable_warnings: bool = False):
        """Initialize hardware detector.
        
        Args:
            enable_warnings: Whether to show hardware-related warnings
        """
        self.enable_warnings = enable_warnings
        if not enable_warnings:
            warnings.filterwarnings('ignore', category=UserWarning)
        
        self._cpu_info = None
        self._gpu_info = None
        self._torch_available = None
        self._tf_available = None
        
    def detect_cpu(self) -> CPUInfo:
        """Detect CPU information and capabilities."""
        if self._cpu_info is not None:
            return self._cpu_info
            
        try:
            # Basic CPU info
            cores_physical = psutil.cpu_count(logical=False) or 1
            cores_logical = psutil.cpu_count(logical=True) or 1
            
            # CPU frequency
            freq_info = psutil.cpu_freq()
            frequency_max = freq_info.max if freq_info else 0.0
            frequency_current = freq_info.current if freq_info else 0.0
            
            # Architecture and brand
            architecture = platform.machine()
            brand = platform.processor() or "Unknown"
            
            # Current usage
            usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory info
            memory_info = psutil.virtual_memory()
            memory_total_gb = memory_info.total / (1024**3)
            memory_available_gb = memory_info.available / (1024**3)
            
            self._cpu_info = CPUInfo(
                cores_physical=cores_physical,
                cores_logical=cores_logical,
                frequency_max=frequency_max,
                frequency_current=frequency_current,
                architecture=architecture,
                brand=brand,
                usage_percent=usage_percent,
                memory_total_gb=memory_total_gb,
                memory_available_gb=memory_available_gb
            )
            
            logger.info(f"CPU detected: {cores_physical}/{cores_logical} cores, {memory_total_gb:.1f}GB RAM")
            
        except Exception as e:
            logger.warning(f"CPU detection failed: {e}")
            # Fallback to minimal info
            self._cpu_info = CPUInfo(
                cores_physical=1,
                cores_logical=1,
                frequency_max=0.0,
                frequency_current=0.0,
                architecture="unknown",
                brand="unknown",
                usage_percent=0.0,
                memory_total_gb=4.0,  # Conservative estimate
                memory_available_gb=2.0
            )
            
        return self._cpu_info
    
    def detect_pytorch_devices(self) -> Tuple[bool, List[GPUInfo]]:
        """Detect PyTorch-compatible devices."""
        gpu_info = []
        torch_available = False
        
        try:
            import torch
            torch_available = True
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                
                for i in range(device_count):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        
                        # Memory info
                        memory_total = props.total_memory // (1024 * 1024)  # MB
                        memory_reserved = torch.cuda.memory_reserved(i) // (1024 * 1024)
                        memory_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)
                        memory_free = memory_total - memory_reserved
                        
                        # Compute capability
                        compute_capability = (props.major, props.minor)
                        
                        gpu_info.append(GPUInfo(
                            id=i,
                            name=props.name,
                            memory_total_mb=memory_total,
                            memory_free_mb=memory_free,
                            memory_used_mb=memory_allocated,
                            utilization_percent=0,  # PyTorch doesn't provide this easily
                            compute_capability=compute_capability,
                            is_available=True,
                            driver_version=torch.version.cuda or "unknown"
                        ))
                        
                        logger.info(f"PyTorch GPU {i}: {props.name} ({memory_total}MB)")
                        
                    except Exception as e:
                        logger.warning(f"Failed to get info for GPU {i}: {e}")
                        
        except ImportError:
            logger.info("PyTorch not available")
        except Exception as e:
            logger.warning(f"PyTorch GPU detection failed: {e}")
            
        return torch_available, gpu_info
    
    def detect_tensorflow_devices(self) -> Tuple[bool, List[str]]:
        """Detect TensorFlow-compatible devices."""
        tf_available = False
        tf_devices = []
        
        try:
            import tensorflow as tf
            tf_available = True
            
            # Suppress TF logging
            tf.get_logger().setLevel('ERROR')
            
            # Get physical devices
            physical_devices = tf.config.list_physical_devices()
            gpu_devices = tf.config.list_physical_devices('GPU')
            
            tf_devices = [device.name for device in physical_devices]
            
            if gpu_devices:
                logger.info(f"TensorFlow GPUs: {len(gpu_devices)} devices")
                
                # Configure GPU memory growth to avoid allocation issues
                for gpu in gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except:
                        pass
                        
        except ImportError:
            logger.info("TensorFlow not available")
        except Exception as e:
            logger.warning(f"TensorFlow device detection failed: {e}")
            
        return tf_available, tf_devices
    
    def detect_nvidia_gpu(self) -> List[GPUInfo]:
        """Detect NVIDIA GPUs using nvidia-ml-py if available."""
        gpu_info = []
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Basic info
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Memory info
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_total = memory_info.total // (1024 * 1024)  # MB
                    memory_free = memory_info.free // (1024 * 1024)
                    memory_used = memory_info.used // (1024 * 1024)
                    
                    # Utilization
                    try:
                        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilization = util_info.gpu
                    except:
                        utilization = 0
                    
                    # Driver version
                    try:
                        driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                    except:
                        driver_version = "unknown"
                    
                    gpu_info.append(GPUInfo(
                        id=i,
                        name=name,
                        memory_total_mb=memory_total,
                        memory_free_mb=memory_free,
                        memory_used_mb=memory_used,
                        utilization_percent=utilization,
                        compute_capability=None,  # Would need CUDA for this
                        is_available=True,
                        driver_version=driver_version
                    ))
                    
                    logger.info(f"NVIDIA GPU {i}: {name} ({memory_total}MB, {utilization}% util)")
                    
                except Exception as e:
                    logger.warning(f"Failed to get NVIDIA GPU {i} info: {e}")
                    
        except ImportError:
            logger.debug("pynvml not available for NVIDIA GPU detection")
        except Exception as e:
            logger.warning(f"NVIDIA GPU detection failed: {e}")
            
        return gpu_info
    
    def detect_gpu(self) -> List[GPUInfo]:
        """Detect all available GPUs."""
        if self._gpu_info is not None:
            return self._gpu_info
            
        all_gpu_info = []
        
        # Try PyTorch detection first (most comprehensive for ML)
        torch_available, torch_gpus = self.detect_pytorch_devices()
        if torch_gpus:
            all_gpu_info.extend(torch_gpus)
        
        # Try NVIDIA-ML detection for additional info
        nvidia_gpus = self.detect_nvidia_gpu()
        
        # Merge information (prefer PyTorch info but add NVIDIA details)
        for nvidia_gpu in nvidia_gpus:
            # Find corresponding PyTorch GPU
            torch_gpu = next(
                (gpu for gpu in all_gpu_info if gpu.id == nvidia_gpu.id), 
                None
            )
            
            if torch_gpu:
                # Update with NVIDIA-specific info
                torch_gpu.utilization_percent = nvidia_gpu.utilization_percent
                torch_gpu.driver_version = nvidia_gpu.driver_version
            else:
                # Add NVIDIA GPU if not found in PyTorch
                all_gpu_info.append(nvidia_gpu)
        
        self._gpu_info = all_gpu_info
        return self._gpu_info
    
    def get_optimal_config(self, 
                          prefer_gpu: bool = True,
                          memory_fraction: float = 0.8,
                          max_threads: Optional[int] = None) -> HardwareConfig:
        """Get optimal hardware configuration for ML workloads.
        
        Args:
            prefer_gpu: Whether to prefer GPU over CPU when available
            memory_fraction: Fraction of GPU memory to use (0.1-1.0)
            max_threads: Maximum number of CPU threads to use
            
        Returns:
            HardwareConfig: Optimized configuration
        """
        cpu_info = self.detect_cpu()
        gpu_info = self.detect_gpu()
        
        # Determine GPU usage
        use_gpu = prefer_gpu and len(gpu_info) > 0 and any(gpu.is_available for gpu in gpu_info)
        
        # CPU thread optimization
        if max_threads is None:
            # Use 75% of logical cores, but leave at least 1 core free
            recommended_threads = max(1, int(cpu_info.cores_logical * 0.75))
        else:
            recommended_threads = min(max_threads, cpu_info.cores_logical)
        
        # Process optimization (for multiprocessing)
        if cpu_info.cores_physical >= 4:
            recommended_processes = max(2, cpu_info.cores_physical // 2)
        else:
            recommended_processes = 1
        
        # Device selection for frameworks
        if use_gpu and gpu_info:
            # Select best GPU (most free memory)
            best_gpu = max(gpu_info, key=lambda g: g.memory_free_mb if g.is_available else 0)
            torch_device = f"cuda:{best_gpu.id}" if best_gpu.is_available else "cpu"
            tf_device = f"/GPU:{best_gpu.id}" if best_gpu.is_available else "/CPU:0"
        else:
            torch_device = "cpu"
            tf_device = "/CPU:0"
        
        # Multiprocessing decision
        use_multiprocessing = (
            cpu_info.cores_physical >= 2 and 
            cpu_info.memory_available_gb >= 4.0 and
            not use_gpu  # Avoid multiprocessing with GPU to prevent memory issues
        )
        
        config = HardwareConfig(
            cpu_info=cpu_info,
            gpu_info=gpu_info,
            recommended_threads=recommended_threads,
            recommended_processes=recommended_processes,
            torch_device=torch_device,
            tf_device=tf_device,
            use_gpu=use_gpu,
            use_multiprocessing=use_multiprocessing,
            memory_fraction=memory_fraction
        )
        
        return config
    
    def apply_pytorch_config(self, config: HardwareConfig) -> None:
        """Apply PyTorch-specific optimizations."""
        try:
            import torch
            
            # Set number of threads
            torch.set_num_threads(config.recommended_threads)
            
            if config.use_gpu and torch.cuda.is_available():
                # Set default device
                torch.cuda.set_device(config.torch_device)
                
                # Configure memory management
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(config.memory_fraction)
                
                logger.info(f"PyTorch configured for {config.torch_device}")
            else:
                logger.info(f"PyTorch configured for CPU with {config.recommended_threads} threads")
                
        except ImportError:
            logger.debug("PyTorch not available for configuration")
        except Exception as e:
            logger.warning(f"PyTorch configuration failed: {e}")
    
    def apply_tensorflow_config(self, config: HardwareConfig) -> None:
        """Apply TensorFlow-specific optimizations."""
        try:
            import tensorflow as tf
            
            # CPU configuration
            tf.config.threading.set_inter_op_parallelism_threads(config.recommended_threads)
            tf.config.threading.set_intra_op_parallelism_threads(config.recommended_threads)
            
            if config.use_gpu:
                # GPU configuration
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    logger.info(f"TensorFlow configured for GPU with memory growth")
            else:
                # Disable GPU if not using it
                tf.config.set_visible_devices([], 'GPU')
                logger.info(f"TensorFlow configured for CPU with {config.recommended_threads} threads")
                
        except ImportError:
            logger.debug("TensorFlow not available for configuration")
        except Exception as e:
            logger.warning(f"TensorFlow configuration failed: {e}")
    
    def apply_environment_config(self, config: HardwareConfig) -> None:
        """Apply environment-wide optimizations."""
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(config.recommended_threads)
        os.environ['MKL_NUM_THREADS'] = str(config.recommended_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(config.recommended_threads)
        
        # CUDA-specific environment variables
        if config.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_info[0].id) if config.gpu_info else '0'
        
        logger.info(f"Environment configured: {config.recommended_threads} threads, GPU={config.use_gpu}")
    
    def print_hardware_summary(self, config: HardwareConfig) -> None:
        """Print a comprehensive hardware summary."""
        print("\n" + "="*60)
        print("🖥️  HARDWARE CONFIGURATION SUMMARY")
        print("="*60)
        
        # CPU Information
        cpu = config.cpu_info
        print(f"\n💻 CPU Information:")
        print(f"   Brand: {cpu.brand}")
        print(f"   Architecture: {cpu.architecture}")
        print(f"   Cores: {cpu.cores_physical} physical, {cpu.cores_logical} logical")
        print(f"   Frequency: {cpu.frequency_max:.0f} MHz (max), {cpu.frequency_current:.0f} MHz (current)")
        print(f"   Memory: {cpu.memory_total_gb:.1f} GB total, {cpu.memory_available_gb:.1f} GB available")
        print(f"   Current Usage: {cpu.usage_percent:.1f}%")
        
        # GPU Information
        if config.gpu_info:
            print(f"\n🎮 GPU Information:")
            for i, gpu in enumerate(config.gpu_info):
                status = "✅ Available" if gpu.is_available else "❌ Unavailable"
                print(f"   GPU {gpu.id}: {gpu.name} ({status})")
                print(f"           Memory: {gpu.memory_total_mb} MB total, {gpu.memory_free_mb} MB free")
                print(f"           Utilization: {gpu.utilization_percent}%")
                if gpu.compute_capability:
                    print(f"           Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
                print(f"           Driver: {gpu.driver_version}")
        else:
            print(f"\n🎮 GPU Information: No GPUs detected")
        
        # Configuration
        print(f"\n⚙️  Optimal Configuration:")
        print(f"   Primary Device: {config.torch_device}")
        print(f"   CPU Threads: {config.recommended_threads}")
        print(f"   CPU Processes: {config.recommended_processes}")
        print(f"   Use GPU: {'Yes' if config.use_gpu else 'No'}")
        print(f"   Use Multiprocessing: {'Yes' if config.use_multiprocessing else 'No'}")
        print(f"   GPU Memory Fraction: {config.memory_fraction:.1%}")
        
        print("="*60)

# Global hardware detector instance
_global_detector = None

def get_hardware_detector(enable_warnings: bool = False) -> HardwareDetector:
    """Get global hardware detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = HardwareDetector(enable_warnings=enable_warnings)
    return _global_detector

def get_optimal_config(**kwargs) -> HardwareConfig:
    """Get optimal hardware configuration using global detector."""
    detector = get_hardware_detector()
    return detector.get_optimal_config(**kwargs)

def configure_ml_frameworks(config: Optional[HardwareConfig] = None) -> HardwareConfig:
    """Configure ML frameworks with optimal settings."""
    if config is None:
        config = get_optimal_config()
    
    detector = get_hardware_detector()
    detector.apply_environment_config(config)
    detector.apply_pytorch_config(config)
    detector.apply_tensorflow_config(config)
    
    return config

def print_hardware_info() -> HardwareConfig:
    """Print comprehensive hardware information and return config."""
    detector = get_hardware_detector()
    config = detector.get_optimal_config()
    detector.print_hardware_summary(config)
    return config

if __name__ == "__main__":
    # Demo script
    print("🔍 Hardware Detection Demo")
    config = print_hardware_info()
    
    print("\n🔧 Applying ML framework configurations...")
    configure_ml_frameworks(config)
    print("✅ Configuration complete!")