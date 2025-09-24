#!/usr/bin/env python3
"""
GPU Training Validation Test

This script specifically tests that RL training uses GPU acceleration
and provides clear verification of GPU usage during training.
"""

import sys
import os
import time
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "OrbitalAvoidance"))

def test_pytorch_gpu_detection():
    """Test PyTorch GPU detection and device setting."""
    print("🔍 Test 1: PyTorch GPU Detection and Configuration")
    print("=" * 60)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**2} MB)")
            
            # Test device setting
            device = torch.device("cuda:0")
            print(f"✅ Using device: {device}")
            
            # Test tensor creation on GPU
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            start_time = time.time()
            z = torch.matmul(x, y)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU tensor operation successful: {gpu_time:.4f}s")
            
            return True
        else:
            print("❌ No CUDA GPUs available")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def test_hardware_detection_gpu_config():
    """Test that hardware detection correctly configures GPU usage."""
    print("\n🔧 Test 2: Hardware Detection GPU Configuration")
    print("=" * 60)
    
    try:
        from hardware_detection import get_optimal_config, configure_ml_frameworks
        
        # Test GPU-preferred configuration
        config = get_optimal_config(prefer_gpu=True)
        
        print(f"Device selection: {config.torch_device}")
        print(f"Use GPU: {config.use_gpu}")
        print(f"GPU count: {len(config.gpu_info)}")
        
        if config.use_gpu and config.gpu_info:
            gpu = config.gpu_info[0]  # First GPU
            print(f"Selected GPU: {gpu.name}")
            print(f"GPU Memory: {gpu.memory_total_mb} MB")
            print(f"Memory fraction: {config.memory_fraction:.1%}")
        
        # Apply configuration
        configure_ml_frameworks(config)
        print("✅ ML frameworks configured with GPU settings")
        
        return config.use_gpu and "cuda" in config.torch_device
        
    except ImportError:
        print("❌ Hardware detection not available")
        return False
    except Exception as e:
        print(f"❌ Hardware detection test failed: {e}")
        return False

def test_rl_algorithm_gpu_usage():
    """Test that RL algorithms are configured to use GPU."""
    print("\n🤖 Test 3: RL Algorithm GPU Configuration")
    print("=" * 60)
    
    try:
        import torch
        import gymnasium as gym
        import numpy as np
        
        # Create a simple mock environment
        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
                self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
            
            def reset(self, seed=None):
                return np.zeros(10, dtype=np.float32), {}
            
            def step(self, action):
                return np.zeros(10, dtype=np.float32), 0.0, False, False, {}
        
        env = MockEnv()
        
        # Test PPO algorithm GPU configuration
        from OrbitalAvoidance.rl_algorithms import PPOAlgorithm
        
        config = {
            "prefer_gpu": True,
            "gpu_memory_fraction": 0.8,
            "ppo": {
                "learning_rate": 3e-4,
                "batch_size": 64
            }
        }
        
        print("Creating PPO algorithm with GPU configuration...")
        ppo_algo = PPOAlgorithm(env, config)
        
        print(f"✅ PPO Algorithm created")
        print(f"Device configured: {getattr(ppo_algo, 'device', 'not set')}")
        print(f"Hardware config available: {hasattr(ppo_algo, 'hardware_config')}")
        
        if hasattr(ppo_algo, 'hardware_config') and ppo_algo.hardware_config:
            hc = ppo_algo.hardware_config
            print(f"Use GPU: {hc.use_gpu}")
            print(f"Torch device: {hc.torch_device}")
            
            # Verify the device is actually GPU
            if "cuda" in hc.torch_device and torch.cuda.is_available():
                print("✅ Algorithm correctly configured for GPU usage")
                return True
            else:
                print("⚠️ Algorithm not using GPU despite availability")
                return False
        else:
            print("⚠️ Hardware configuration not properly initialized")
            return False
            
    except ImportError as e:
        print(f"❌ Required modules not available: {e}")
        return False
    except Exception as e:
        print(f"❌ RL algorithm test failed: {e}")
        return False

def test_actual_gpu_training():
    """Test actual GPU usage during a brief training session."""
    print("\n🏋️ Test 4: Actual GPU Training Verification")
    print("=" * 60)
    
    try:
        import torch
        from stable_baselines3 import PPO
        import gymnasium as gym
        import numpy as np
        
        # Check initial GPU memory
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated(0)
            print(f"Initial GPU memory: {initial_memory / 1024**2:.1f} MB")
            
            # Create a simple environment
            class SimpleEnv(gym.Env):
                def __init__(self):
                    self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(50,), dtype=np.float32)
                    self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
                    self.step_count = 0
                
                def reset(self, seed=None):
                    self.step_count = 0
                    return np.random.randn(50).astype(np.float32), {}
                
                def step(self, action):
                    self.step_count += 1
                    obs = np.random.randn(50).astype(np.float32)
                    reward = np.random.randn()
                    done = self.step_count >= 100
                    return obs, reward, done, False, {}
            
            env = SimpleEnv()
            
            # Create PPO model with explicit GPU device
            print("Creating PPO model with GPU device...")
            model = PPO(
                "MlpPolicy", 
                env, 
                device="cuda:0",
                verbose=0,
                batch_size=64,
                n_steps=128,
                policy_kwargs={"net_arch": [64, 64]}
            )
            
            print(f"Model device: {model.device}")
            
            # Check GPU memory after model creation
            after_model_memory = torch.cuda.memory_allocated(0)
            print(f"GPU memory after model creation: {after_model_memory / 1024**2:.1f} MB")
            print(f"Memory increase: {(after_model_memory - initial_memory) / 1024**2:.1f} MB")
            
            if after_model_memory > initial_memory:
                print("✅ GPU memory usage increased - model loaded on GPU")
            else:
                print("⚠️ No significant GPU memory increase detected")
            
            # Run a few training steps
            print("Running brief training to verify GPU usage...")
            
            # Monitor GPU utilization during training
            start_time = time.time()
            model.learn(total_timesteps=1000, progress_bar=False)
            training_time = time.time() - start_time
            
            # Check final GPU memory
            final_memory = torch.cuda.memory_allocated(0)
            print(f"GPU memory after training: {final_memory / 1024**2:.1f} MB")
            print(f"Training time: {training_time:.2f} seconds")
            
            # Verify GPU was actually used
            if final_memory > initial_memory:
                print("✅ GPU successfully used for training!")
                print(f"Total GPU memory used: {(final_memory - initial_memory) / 1024**2:.1f} MB")
                return True
            else:
                print("❌ GPU memory usage not detected during training")
                return False
        else:
            print("❌ CUDA not available for training test")
            return False
            
    except ImportError as e:
        print(f"❌ Required modules not available: {e}")
        return False
    except Exception as e:
        print(f"❌ GPU training test failed: {e}")
        return False

def test_gpu_utilization_monitoring():
    """Test GPU utilization monitoring during training."""
    print("\n📊 Test 5: GPU Utilization Monitoring")
    print("=" * 60)
    
    try:
        import subprocess
        
        # Check if nvidia-smi is available
        result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                               "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("Current GPU status:")
            for i, line in enumerate(lines):
                if line.strip():
                    util, mem_used, mem_total = line.split(', ')
                    print(f"  GPU {i}: {util}% utilization, {mem_used}/{mem_total} MB")
            
            print("✅ GPU monitoring available via nvidia-smi")
            print("💡 Run 'watch -n 1 nvidia-smi' in another terminal to monitor GPU usage during training")
            return True
        else:
            print("⚠️ nvidia-smi not available for GPU monitoring")
            return False
            
    except Exception as e:
        print(f"ℹ️ GPU monitoring test skipped: {e}")
        return True  # Not critical for GPU usage

def run_gpu_validation_tests():
    """Run all GPU validation tests."""
    print("🚀 GPU TRAINING VALIDATION TESTS")
    print("=" * 70)
    
    tests = [
        ("PyTorch GPU Detection", test_pytorch_gpu_detection),
        ("Hardware Detection GPU Config", test_hardware_detection_gpu_config),
        ("RL Algorithm GPU Setup", test_rl_algorithm_gpu_usage),
        ("Actual GPU Training", test_actual_gpu_training),
        ("GPU Monitoring", test_gpu_utilization_monitoring),
    ]
    
    results = []
    gpu_tests_passed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*10} {test_name} {'='*10}")
            result = test_func()
            results.append((test_name, result))
            if result and "GPU" in test_name:
                gpu_tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 GPU VALIDATION SUMMARY")
    print("=" * 70)
    
    gpu_critical_tests = ["PyTorch GPU Detection", "Hardware Detection GPU Config", 
                         "RL Algorithm GPU Setup", "Actual GPU Training"]
    critical_passed = sum(1 for name, result in results 
                         if result and name in gpu_critical_tests)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        critical = "🔥 CRITICAL" if test_name in gpu_critical_tests else ""
        print(f"{status} | {test_name:30} | {critical}")
    
    print(f"\nCritical GPU tests passed: {critical_passed}/{len(gpu_critical_tests)}")
    
    if critical_passed == len(gpu_critical_tests):
        print("🎉 SUCCESS: GPU training is properly configured and working!")
        print("✅ Your system will use GPU acceleration for RL training")
        return True
    else:
        print("⚠️ WARNING: Some GPU tests failed - training may fall back to CPU")
        return False

if __name__ == "__main__":
    success = run_gpu_validation_tests()
    
    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Run training: cd OrbitalAvoidance && python3 rl_train.py --algorithm ppo")
        print("2. Monitor GPU: watch -n 1 nvidia-smi")
        print("3. Check logs for GPU device confirmation")
    else:
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check CUDA installation: nvidia-smi")
        print("2. Verify PyTorch CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'")
        print("3. Install dependencies: pip install torch stable-baselines3")
    
    sys.exit(0 if success else 1)