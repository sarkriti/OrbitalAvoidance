#!/usr/bin/env python3
"""
Simple GPU Training Test - avoiding NumPy compatibility issues
Tests GPU usage for RL training without problematic dependencies
"""

import os
import sys
import time
import torch
import warnings

# Suppress warnings to clean up output
warnings.filterwarnings('ignore')

def test_pytorch_gpu_training():
    """Test actual GPU training with PyTorch"""
    print("🔥 Testing PyTorch GPU Training")
    print("=" * 60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"📊 Found {gpu_count} GPUs")
    
    # Select primary GPU (cuda:1 as configured)
    device = torch.device("cuda:1" if gpu_count > 1 else "cuda:0")
    print(f"🎯 Using device: {device}")
    
    # Create a simple neural network
    class SimpleNet(torch.nn.Module):
        def __init__(self, input_size=128, hidden_size=256, output_size=64):
            super().__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Move model to GPU
    model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print(f"🧠 Model created on device: {next(model.parameters()).device}")
    
    # Create training data on GPU
    batch_size = 512  # Large batch to utilize GPU
    input_size = 128
    output_size = 64
    
    # Generate random training data
    x_train = torch.randn(batch_size, input_size, device=device)
    y_train = torch.randn(batch_size, output_size, device=device)
    
    print(f"📊 Training data shape: {x_train.shape}, device: {x_train.device}")
    
    # Training loop with GPU utilization
    print("🏋️ Starting GPU training...")
    start_time = time.time()
    
    for epoch in range(50):  # 50 epochs to see GPU usage
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:2d}: Loss = {loss.item():.6f}")
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # Test memory allocation
    try:
        # Allocate a large tensor to test GPU memory
        large_tensor = torch.zeros(1000, 1000, device=device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**2   # MB
        
        print(f"💾 GPU Memory - Allocated: {memory_allocated:.1f}MB, Reserved: {memory_reserved:.1f}MB")
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"⚠️ Memory test failed: {e}")
    
    return True

def test_hardware_detection_gpu():
    """Test our hardware detection system"""
    print("\n🔧 Testing Hardware Detection System")
    print("=" * 60)
    
    try:
        # Import our hardware detection
        sys.path.append('/home/brysonbrown/Orbital/OrbitalAvoidance')
        from hardware_detection import HardwareDetector
        
        detector = HardwareDetector()
        config = detector.get_hardware_config()
        
        print(f"🎯 Selected device: {config.device}")
        print(f"📊 GPU count: {config.gpu_count}")
        print(f"🔥 Use GPU: {config.use_gpu}")
        
        if config.use_gpu and config.gpu_info:
            gpu = config.gpu_info[0]
            print(f"🚀 Primary GPU: {gpu['name']} ({gpu['memory_mb']}MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False

def check_gpu_utilization():
    """Check current GPU utilization"""
    print("\n📊 Current GPU Status")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 5:
                    idx, name, util, mem_used, mem_total = parts
                    print(f"  GPU {idx}: {name.strip()} - {util}% utilization, {mem_used}/{mem_total}MB")
        else:
            print("❌ nvidia-smi not available")
            
    except Exception as e:
        print(f"❌ GPU monitoring failed: {e}")

def main():
    print("🚀 SIMPLE GPU TRAINING VALIDATION")
    print("=" * 70)
    
    tests = [
        ("PyTorch GPU Training", test_pytorch_gpu_training),
        ("Hardware Detection", test_hardware_detection_gpu),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} | {test_name}")
        except Exception as e:
            print(f"❌ FAIL | {test_name} - Exception: {e}")
            results.append((test_name, False))
    
    # Show current GPU status
    check_gpu_utilization()
    
    # Summary
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} | {test_name}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - GPU training is ready!")
        print("\n💡 Your system will use GPU acceleration for RL training")
        print("💡 Run 'watch -n 1 nvidia-smi' to monitor GPU usage during training")
    else:
        print("⚠️ Some tests failed - check GPU configuration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)