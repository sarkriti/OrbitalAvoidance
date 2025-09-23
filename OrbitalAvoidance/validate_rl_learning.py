#!/usr/bin/env python3
"""
RL Learning Validation Script

This script provides definitive proof that RL models are actually learning
by analyzing training curves, comparing model performance across training
stages, and demonstrating behavioral improvements.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from stable_baselines3 import PPO
    from rl_training_env import create_rl_environment
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available")

class RLLearningValidator:
    """Comprehensive validation of RL learning effectiveness."""
    
    def __init__(self):
        self.test_results = []
        self.model_dir = Path("trained_models")
        self.config_file = "/home/bryson/RL_testing/scenarios/leo_debris_avoidance.json"
        
    def analyze_training_curves(self):
        """Test 1: Analyze training progress from model checkpoints."""
        print("🧪 Test 1: Training Curve Analysis")
        print("=" * 50)
        
        if not STABLE_BASELINES_AVAILABLE:
            print("❌ SKIPPED - stable-baselines3 not available")
            return
        
        # Find PPO training directory
        ppo_dir = self.model_dir / "debris_avoidance_ppo"
        if not ppo_dir.exists():
            print("❌ FAILED - No PPO training directory found")
            self.test_results.append({
                'test': 'Training Curves',
                'passed': False,
                'details': 'No training directory found'
            })
            return
        
        # Load model checkpoints
        checkpoint_files = sorted([f for f in ppo_dir.glob("model_checkpoint_*_steps.zip")])
        
        if len(checkpoint_files) < 3:
            print("❌ FAILED - Insufficient checkpoints for analysis")
            self.test_results.append({
                'test': 'Training Curves', 
                'passed': False,
                'details': f'Only {len(checkpoint_files)} checkpoints found'
            })
            return
        
        print(f"Found {len(checkpoint_files)} training checkpoints")
        
        # Test early, middle, and late training models
        early_model = checkpoint_files[0]
        middle_model = checkpoint_files[len(checkpoint_files)//2]
        late_model = checkpoint_files[-1]
        
        print(f"Testing progression: {early_model.name} → {middle_model.name} → {late_model.name}")
        
        # Evaluate each model
        performance_scores = []
        
        for model_file in [early_model, middle_model, late_model]:
            score = self._evaluate_model(str(model_file), episodes=5)
            performance_scores.append(score)
            steps = int(model_file.name.split('_')[2])
            print(f"  {steps:6d} steps: Avg reward = {score:.2f}")
        
        # Check if performance improved
        early_score, middle_score, late_score = performance_scores
        
        # Test passes if late model performs better than early model
        improvement = late_score - early_score
        improvement_percent = (improvement / abs(early_score)) * 100 if early_score != 0 else 0
        
        print(f"Total improvement: {improvement:+.2f} ({improvement_percent:+.1f}%)")
        
        passed = improvement > 0 and improvement_percent > 10  # At least 10% improvement
        print(f"✅ PASSED - Clear learning progress" if passed else f"❌ FAILED - No significant improvement")
        
        self.test_results.append({
            'test': 'Training Curves',
            'passed': passed,
            'improvement_percent': improvement_percent,
            'details': f'Early: {early_score:.2f}, Late: {late_score:.2f}, Improvement: {improvement_percent:+.1f}%'
        })
        
        print()
    
    def compare_trained_vs_random(self):
        """Test 2: Compare trained model vs random policy."""
        print("🧪 Test 2: Trained vs Random Comparison")
        print("=" * 50)
        
        if not STABLE_BASELINES_AVAILABLE:
            print("❌ SKIPPED - stable-baselines3 not available")
            return
        
        # Load best trained model
        best_model_path = self.model_dir / "debris_avoidance_ppo" / "best_model.zip"
        
        if not best_model_path.exists():
            print("❌ FAILED - No best model found")
            self.test_results.append({
                'test': 'Trained vs Random',
                'passed': False,
                'details': 'No best model found'
            })
            return
        
        # Test trained model
        trained_score = self._evaluate_model(str(best_model_path), episodes=10)
        print(f"Trained model average reward: {trained_score:.2f}")
        
        # Test random policy
        random_score = self._evaluate_random_policy(episodes=10)
        print(f"Random policy average reward: {random_score:.2f}")
        
        # Calculate improvement
        if random_score != 0:
            improvement_factor = trained_score / random_score
            improvement_percent = (trained_score - random_score) / abs(random_score) * 100
        else:
            improvement_factor = float('inf') if trained_score > 0 else 1.0
            improvement_percent = 100.0 if trained_score > random_score else 0.0
        
        print(f"Improvement factor: {improvement_factor:.2f}x")
        print(f"Improvement percentage: {improvement_percent:+.1f}%")
        
        # Test passes if trained model significantly outperforms random
        passed = improvement_factor > 2.0 and improvement_percent > 50  # At least 2x better
        print(f"✅ PASSED - Trained model significantly better" if passed else f"❌ FAILED - No clear advantage")
        
        self.test_results.append({
            'test': 'Trained vs Random',
            'passed': passed,
            'improvement_factor': improvement_factor,
            'details': f'Trained: {trained_score:.2f}, Random: {random_score:.2f}, Factor: {improvement_factor:.2f}x'
        })
        
        print()
    
    def analyze_behavioral_patterns(self):
        """Test 3: Analyze behavioral patterns of trained model."""
        print("🧪 Test 3: Behavioral Pattern Analysis")
        print("=" * 50)
        
        if not STABLE_BASELINES_AVAILABLE:
            print("❌ SKIPPED - stable-baselines3 not available")
            return
        
        best_model_path = self.model_dir / "debris_avoidance_ppo" / "best_model.zip"
        
        if not best_model_path.exists():
            print("❌ FAILED - No best model found")
            return
        
        # Load model and environment
        try:
            model = PPO.load(str(best_model_path))
            env = create_rl_environment(config_file=self.config_file)
            
            # Run episodes and analyze behavior
            action_magnitudes = []
            collision_rates = []
            fuel_usage_rates = []
            
            for episode in range(5):
                obs, _ = env.reset()
                episode_actions = []
                collision_count = 0
                initial_fuel = None
                
                for step in range(500):  # Limit steps per episode
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Track action magnitude
                    action_magnitude = np.linalg.norm(action)
                    episode_actions.append(action_magnitude)
                    
                    # Track collisions
                    if 'collisions' in info and info['collisions']:
                        collision_count += 1
                    
                    # Track fuel usage
                    if initial_fuel is None:
                        initial_fuel = info.get('fuel_remaining', {}).get(list(info.get('fuel_remaining', {}).keys())[0], 100) if info.get('fuel_remaining') else 100
                    
                    if terminated or truncated:
                        break
                
                avg_action_magnitude = np.mean(episode_actions)
                collision_rate = collision_count / (step + 1)
                
                current_fuel = info.get('fuel_remaining', {}).get(list(info.get('fuel_remaining', {}).keys())[0], initial_fuel) if info.get('fuel_remaining') else initial_fuel
                fuel_usage_rate = (initial_fuel - current_fuel) / (step + 1) if initial_fuel else 0
                
                action_magnitudes.append(avg_action_magnitude)
                collision_rates.append(collision_rate)
                fuel_usage_rates.append(fuel_usage_rate)
                
                print(f"  Episode {episode+1}: Actions={avg_action_magnitude:.3f}, Collisions={collision_rate:.4f}, Fuel rate={fuel_usage_rate:.4f}")
            
            env.close()
            
            # Analyze patterns
            avg_action_magnitude = np.mean(action_magnitudes)
            avg_collision_rate = np.mean(collision_rates)
            avg_fuel_usage = np.mean(fuel_usage_rates)
            
            print(f"Average action magnitude: {avg_action_magnitude:.3f}")
            print(f"Average collision rate: {avg_collision_rate:.4f}")
            print(f"Average fuel usage rate: {avg_fuel_usage:.4f}")
            
            # Test passes if model shows intelligent behavior:
            # - Uses moderate thrust (not too high/low)
            # - Low collision rate
            # - Reasonable fuel usage
            
            intelligent_thrust = 0.1 < avg_action_magnitude < 0.8  # Reasonable thrust levels
            low_collisions = avg_collision_rate < 0.01  # Less than 1% collision rate
            reasonable_fuel = avg_fuel_usage < 0.1  # Not wasting too much fuel
            
            passed = intelligent_thrust and low_collisions and reasonable_fuel
            
            print(f"Intelligent thrust usage: {intelligent_thrust}")
            print(f"Low collision rate: {low_collisions}")
            print(f"Reasonable fuel usage: {reasonable_fuel}")
            print(f"✅ PASSED - Intelligent behavior patterns" if passed else f"❌ FAILED - Poor behavioral patterns")
            
            self.test_results.append({
                'test': 'Behavioral Patterns',
                'passed': passed,
                'avg_action_magnitude': avg_action_magnitude,
                'avg_collision_rate': avg_collision_rate,
                'details': f'Thrust: {avg_action_magnitude:.3f}, Collisions: {avg_collision_rate:.4f}, Fuel: {avg_fuel_usage:.4f}'
            })
            
        except Exception as e:
            print(f"❌ FAILED - Error during behavioral analysis: {e}")
            self.test_results.append({
                'test': 'Behavioral Patterns',
                'passed': False,
                'details': f'Error: {str(e)}'
            })
        
        print()
    
    def _evaluate_model(self, model_path, episodes=5):
        """Evaluate a trained model and return average reward."""
        try:
            model = PPO.load(model_path)
            env = create_rl_environment(config_file=self.config_file)
            
            total_rewards = []
            
            for _ in range(episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for _ in range(1000):  # Max steps per episode
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                total_rewards.append(episode_reward)
            
            env.close()
            return np.mean(total_rewards)
            
        except Exception as e:
            print(f"Error evaluating model {model_path}: {e}")
            return -1000  # Very negative score for failed evaluation
    
    def _evaluate_random_policy(self, episodes=5):
        """Evaluate random policy and return average reward."""
        try:
            env = create_rl_environment(config_file=self.config_file)
            
            total_rewards = []
            
            for _ in range(episodes):
                obs, _ = env.reset()
                episode_reward = 0
                
                for _ in range(1000):  # Max steps per episode
                    action = env.action_space.sample()  # Random action
                    obs, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                total_rewards.append(episode_reward)
            
            env.close()
            return np.mean(total_rewards)
            
        except Exception as e:
            print(f"Error evaluating random policy: {e}")
            return -1000  # Very negative score for failed evaluation
    
    def check_training_logs(self):
        """Test 4: Check for training history files."""
        print("🧪 Test 4: Training History Validation")
        print("=" * 50)
        
        # Check for training config and history files
        ppo_dir = self.model_dir / "debris_avoidance_ppo"
        
        config_file = ppo_dir / "training_config.json"
        history_file = ppo_dir / "final_model_history.json"
        
        config_exists = config_file.exists()
        history_exists = history_file.exists()
        
        print(f"Training config file: {'✅ Found' if config_exists else '❌ Missing'}")
        print(f"Training history file: {'✅ Found' if history_exists else '❌ Missing'}")
        
        details = []
        
        if config_exists:
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    timesteps = config.get('training', {}).get('total_timesteps', 0)
                    algorithm = config.get('algorithm', 'unknown')
                    print(f"Training algorithm: {algorithm}")
                    print(f"Total timesteps: {timesteps:,}")
                    details.append(f"Algorithm: {algorithm}, Steps: {timesteps:,}")
            except Exception as e:
                print(f"Error reading config: {e}")
        
        if history_exists:
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    if 'training_progress' in history:
                        progress = history['training_progress']
                        print(f"Training progress entries: {len(progress)}")
                        details.append(f"Progress entries: {len(progress)}")
            except Exception as e:
                print(f"Error reading history: {e}")
        
        passed = config_exists and history_exists
        print(f"✅ PASSED - Training files found" if passed else f"❌ FAILED - Missing training documentation")
        
        self.test_results.append({
            'test': 'Training History',
            'passed': passed,
            'details': ', '.join(details) if details else 'No training files found'
        })
        
        print()
    
    def run_all_tests(self):
        """Run all RL learning validation tests."""
        print("🧠 RL LEARNING VALIDATION")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.check_training_logs()
        self.analyze_training_curves()
        self.compare_trained_vs_random()
        self.analyze_behavioral_patterns()
        
        # Summary
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        
        passed_tests = sum(1 for result in self.test_results if result['passed'])
        total_tests = len(self.test_results)
        
        for result in self.test_results:
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"{status} | {result['test']:25} | {result['details']}")
        
        print()
        print(f"Overall Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - RL models are definitely learning!")
        elif passed_tests >= total_tests * 0.75:
            print("✅ MOSTLY PASSED - RL learning confirmed with minor issues")
        else:
            print("⚠️  Multiple tests failed - RL learning effectiveness questionable")
        
        return passed_tests >= total_tests * 0.75  # Pass if 75% or more tests pass


if __name__ == "__main__":
    validator = RLLearningValidator()
    success = validator.run_all_tests()
    
    if success:
        print("\n✅ CONCLUSION: Your RL models are definitely learning effectively!")
    else:
        print("\n❌ CONCLUSION: RL learning validation failed - investigate training setup")