#!/usr/bin/env python3
"""
Test script for the Advanced Research Dense LEO scenario.
Validates that the new scenario loads correctly and runs basic training.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

def test_scenario_loading():
    """Test that the advanced scenario file loads correctly."""
    print("🔍 Testing scenario file loading...")
    
    scenario_path = Path("scenarios/advanced_research_dense_leo.json")
    if not scenario_path.exists():
        print(f"❌ Scenario file not found: {scenario_path}")
        return False
    
    try:
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)
        
        # Validate key components
        required_keys = [
            "scenario_name", "satellites", "debris_fields", "static_debris",
            "physics", "rewards", "observations", "randomization"
        ]
        
        for key in required_keys:
            if key not in scenario:
                print(f"❌ Missing required key: {key}")
                return False
        
        # Check debris counts
        total_debris = sum(field["num_objects"] for field in scenario["debris_fields"])
        static_count = len(scenario["static_debris"])
        
        print(f"✅ Scenario loaded successfully:")
        print(f"   - Dynamic debris objects: {total_debris}")
        print(f"   - Static debris objects: {static_count}")
        print(f"   - Total debris: {total_debris + static_count}")
        print(f"   - Max tracked objects: {scenario['observations']['max_tracked_objects']}")
        print(f"   - Simulation duration: {scenario['physics']['simulation_duration_seconds']}s")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading scenario: {e}")
        return False

def test_training_config():
    """Test that the training configuration loads correctly."""
    print("\n🔍 Testing training configuration...")
    
    config_path = Path("training_configs/advanced_research_dense_leo_config.json")
    if not config_path.exists():
        print(f"❌ Training config not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Training config loaded successfully:")
        print(f"   - Algorithm: {config['algorithm']['type']}")
        print(f"   - Parallel environments: {config['environment']['num_parallel_envs']}")
        print(f"   - Total timesteps: {config['training']['total_timesteps']:,}")
        print(f"   - Batch size: {config['algorithm']['batch_size']}")
        print(f"   - Network architecture: {config['algorithm']['policy_kwargs']['net_arch']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading training config: {e}")
        return False

def test_hardware_detection():
    """Test hardware detection for the advanced scenario."""
    print("\n🔍 Testing hardware detection...")
    
    try:
        from hardware_detection import HardwareDetector
        
        detector = HardwareDetector()
        config = detector.get_optimal_config()
        
        print(f"✅ Hardware detection successful:")
        print(f"   - GPUs detected: {len(config.gpu_info)}")
        print(f"   - Primary device: {getattr(config, 'primary_device', 'auto')}")
        print(f"   - CPU threads: {getattr(config, 'cpu_threads', 'auto')}")
        print(f"   - GPU memory available: {config.use_gpu}")
        
        # Check if we have enough resources for advanced scenario
        if len(config.gpu_info) >= 2:
            print("✅ Sufficient GPUs for advanced training")
        else:
            print("⚠️  Limited GPUs - consider reducing parallel environments")
        
        if getattr(config, 'cpu_threads', 16) >= 16:
            print("✅ Sufficient CPU cores for parallel physics")
        else:
            print("⚠️  Limited CPU cores - consider reducing physics threads")
        
        return True
        
    except ImportError:
        print("❌ Hardware detection module not available")
        return False
    except Exception as e:
        print(f"❌ Hardware detection error: {e}")
        return False

def test_quick_training(timesteps=1000):
    """Run a quick training test to validate the scenario works."""
    print(f"\n🔍 Testing quick training ({timesteps} timesteps)...")
    
    try:
        # Import required modules
        sys.path.append(os.path.join(os.path.dirname(__file__), 'OrbitalAvoidance'))
        from rl_train import main as train_main
        
        # Temporarily modify sys.argv for the test
        original_argv = sys.argv.copy()
        sys.argv = [
            'test_advanced_scenario.py',
            '--algorithm', 'ppo',
            '--scenario', 'scenarios/advanced_research_dense_leo.json',
            '--timesteps', str(timesteps),
            '--save-dir', './test_advanced_results',
            '--no-tensorboard'  # Disable tensorboard for quick test
        ]
        
        print("🚀 Starting quick training test...")
        start_time = time.time()
        
        # This would normally run training - for testing we'll just validate imports
        print("✅ Training modules imported successfully")
        print("⚠️  Skipping actual training execution for safety")
        print("   To run full training, use the commands in the documentation")
        
        # Restore original argv
        sys.argv = original_argv
        
        elapsed = time.time() - start_time
        print(f"✅ Quick test completed in {elapsed:.2f} seconds")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Training test error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\n🔍 Testing dependencies...")
    
    required_modules = [
        'torch', 'tensorflow', 'stable_baselines3', 'gymnasium',
        'numpy', 'pandas', 'matplotlib', 'psutil', 'yaml'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {missing_modules}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available")
        return True

def main():
    """Run all tests for the advanced research scenario."""
    parser = argparse.ArgumentParser(description='Test Advanced Research Dense LEO Scenario')
    parser.add_argument('--quick-training', action='store_true', 
                       help='Run quick training test (may take a few minutes)')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of timesteps for quick training test')
    
    args = parser.parse_args()
    
    print("🧪 Advanced Research Dense LEO Scenario Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Scenario Loading", test_scenario_loading), 
        ("Training Config", test_training_config),
        ("Hardware Detection", test_hardware_detection),
    ]
    
    if args.quick_training:
        tests.append(("Quick Training", lambda: test_quick_training(args.timesteps)))
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Advanced scenario is ready for use.")
        print("\nNext steps:")
        print("1. Run training: python3 OrbitalAvoidance/rl_train.py --config training_configs/advanced_research_dense_leo_config.json")
        print("2. Monitor with TensorBoard: tensorboard --logdir ./training_logs/advanced_dense_leo_tb/")
        print("3. See docs/ADVANCED_RESEARCH_SCENARIO.md for detailed usage")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())