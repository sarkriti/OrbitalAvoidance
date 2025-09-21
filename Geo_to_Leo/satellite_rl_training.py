#!/usr/bin/env python3
"""
Satellite RL Training Script

Train reinforcement learning agents to control satellites using realistic physics.
Demonstrates various RL algorithms for orbital transfer missions.
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import time

# Import our satellite environment
try:
    from satellite_env import SatelliteOrbitControlEnv as SatelliteEnv
    print("✅ Using main satellite_env.py")
except ImportError:
    print("⚠️ Using test environment (satellite_env.py not found)")
    from test_physics_env import TestHighFidelitySatelliteEnv
    
    # Wrapper to add render_mode support
    class SatelliteEnv(TestHighFidelitySatelliteEnv):
        def __init__(self, render_mode=None):
            super().__init__()
            self.render_mode = render_mode

def create_training_environment():
    """Create and configure the training environment"""
    
    def _init():
        env = SatelliteEnv()
        return Monitor(env, filename=None)
    
    return _init

def train_ppo_agent(total_timesteps=50000, save_path="models/"):
    """Train a PPO agent for satellite control"""
    
    print("🚀 TRAINING PPO AGENT FOR SATELLITE CONTROL")
    print("=" * 50)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("logs/", exist_ok=True)
    
    # Create vectorized environment
    env = make_vec_env(create_training_environment(), n_envs=4)
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/ppo_satellite/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    print(f"✅ PPO Model created")
    print(f"   - Policy: MlpPolicy")
    print(f"   - Learning rate: {model.learning_rate}")
    print(f"   - Training steps: {total_timesteps}")
    
    # Training callback for evaluation
    eval_env = SatelliteEnv()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # Train the model
    print(f"\n🏋️ Starting training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Save the model
    model_path = os.path.join(save_path, "ppo_satellite_final")
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    env.close()
    eval_env.close()
    
    return model, model_path

def train_sac_agent(total_timesteps=50000, save_path="models/"):
    """Train a SAC agent for satellite control"""
    
    print("\n🚀 TRAINING SAC AGENT FOR SATELLITE CONTROL")
    print("=" * 50)
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    
    # Create environment
    env = make_vec_env(create_training_environment(), n_envs=1)
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/sac_satellite/",
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
    )
    
    print(f"✅ SAC Model created")
    print(f"   - Policy: MlpPolicy")
    print(f"   - Buffer size: {model.buffer_size}")
    print(f"   - Training steps: {total_timesteps}")
    
    # Train the model
    print(f"\n🏋️ Starting SAC training...")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"✅ SAC training completed in {training_time:.1f} seconds")
    
    # Save the model
    model_path = os.path.join(save_path, "sac_satellite_final")
    model.save(model_path)
    print(f"✅ SAC model saved to {model_path}")
    
    env.close()
    
    return model, model_path

def evaluate_trained_agent(model_path, num_episodes=5):
    """Evaluate a trained agent"""
    
    print(f"\n🧪 EVALUATING TRAINED AGENT")
    print("=" * 30)
    
    # Load the model
    if "ppo" in model_path:
        model = PPO.load(model_path)
        algorithm = "PPO"
    elif "sac" in model_path:
        model = SAC.load(model_path)
        algorithm = "SAC"
    else:
        model = PPO.load(model_path)
        algorithm = "Unknown"
    
    print(f"✅ Loaded {algorithm} model from {model_path}")
    
    # Create evaluation environment
    env = SatelliteEnv(render_mode='human')
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\n🛰️ Episode {episode + 1}/{num_episodes}")
        
        while True:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            # Render every 10 steps
            if episode_length % 10 == 0:
                env.render()
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"   Episode reward: {episode_reward:.2f}")
        print(f"   Episode length: {episode_length} steps")
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n📊 EVALUATION RESULTS:")
    print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"   Mean episode length: {mean_length:.1f} steps")
    print(f"   Success rate: {sum(1 for r in episode_rewards if r > -50) / len(episode_rewards) * 100:.1f}%")
    
    env.close()
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'rewards': episode_rewards
    }

def plot_training_results(log_path="logs/"):
    """Plot training results if available"""
    
    try:
        # Try to load and plot results
        results = load_results(log_path)
        
        if len(results) > 0:
            x, y = ts2xy(results, 'timesteps')
            
            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.xlabel('Timesteps')
            plt.ylabel('Episode Reward')
            plt.title('Training Progress')
            plt.grid(True)
            plt.savefig('training_progress.png')
            plt.show()
            
            print("✅ Training results plotted")
        else:
            print("⚠️ No training results found to plot")
            
    except Exception as e:
        print(f"⚠️ Could not plot results: {e}")

def demonstrate_random_baseline():
    """Demonstrate random baseline performance"""
    
    print("\n🎲 RANDOM BASELINE DEMONSTRATION")
    print("=" * 35)
    
    env = SatelliteEnv(render_mode='human')
    
    # Run random episodes
    episode_rewards = []
    
    for episode in range(3):
        obs, _ = env.reset()
        episode_reward = 0
        
        print(f"\n🛰️ Random Episode {episode + 1}/3")
        
        for step in range(50):
            # Random action
            action = env.action_space.sample() * 0.1  # Small random actions
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if step % 10 == 0:
                env.render()
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        print(f"   Random episode reward: {episode_reward:.2f}")
    
    mean_random_reward = np.mean(episode_rewards)
    print(f"\n📊 Random baseline mean reward: {mean_random_reward:.2f}")
    
    env.close()
    
    return mean_random_reward

def main():
    """Main training and evaluation pipeline"""
    
    print("🛰️ SATELLITE RL TRAINING PIPELINE")
    print("=" * 40)
    
    # Check environment
    try:
        env = SatelliteEnv()
        obs, _ = env.reset()
        print(f"✅ Environment check passed")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        return
    
    # Demonstrate random baseline
    random_baseline = demonstrate_random_baseline()
    
    # Training options
    print(f"\n🎯 TRAINING OPTIONS:")
    print("1. Train PPO agent (fast, stable)")
    print("2. Train SAC agent (sample efficient)")
    print("3. Train both algorithms")
    print("4. Load and evaluate existing model")
    print("5. Quick demo only")
    
    try:
        choice = input("\nEnter choice (1-5, or press Enter for quick demo): ").strip()
    except KeyboardInterrupt:
        choice = "5"
    
    if choice == "" or choice == "5":
        print("\n✅ Quick demo completed!")
        print(f"Random baseline performance: {random_baseline:.2f}")
        print("To train agents, run again and select options 1-3")
        return
    
    elif choice == "1":
        # Train PPO
        model, model_path = train_ppo_agent(total_timesteps=25000)
        evaluate_trained_agent(model_path)
        
    elif choice == "2":
        # Train SAC
        model, model_path = train_sac_agent(total_timesteps=25000)
        evaluate_trained_agent(model_path)
        
    elif choice == "3":
        # Train both
        ppo_model, ppo_path = train_ppo_agent(total_timesteps=25000)
        sac_model, sac_path = train_sac_agent(total_timesteps=25000)
        
        print(f"\n🏆 COMPARING ALGORITHMS:")
        print("-" * 25)
        
        ppo_results = evaluate_trained_agent(ppo_path, num_episodes=3)
        sac_results = evaluate_trained_agent(sac_path, num_episodes=3)
        
        print(f"\nPPO vs SAC Comparison:")
        print(f"PPO mean reward: {ppo_results['mean_reward']:.2f}")
        print(f"SAC mean reward: {sac_results['mean_reward']:.2f}")
        
        better_algorithm = "PPO" if ppo_results['mean_reward'] > sac_results['mean_reward'] else "SAC"
        print(f"🏆 {better_algorithm} performed better!")
        
    elif choice == "4":
        # Load existing model
        model_files = []
        if os.path.exists("models/"):
            model_files = [f for f in os.listdir("models/") if f.endswith('.zip')]
        
        if model_files:
            print(f"\nAvailable models:")
            for i, f in enumerate(model_files):
                print(f"{i+1}. {f}")
            
            try:
                model_choice = int(input("Select model: ")) - 1
                model_path = os.path.join("models/", model_files[model_choice])
                evaluate_trained_agent(model_path.replace('.zip', ''))
            except (ValueError, IndexError):
                print("Invalid selection")
        else:
            print("No trained models found. Train a model first.")
    
    # Plot results if available
    plot_training_results()
    
    print(f"\n🎉 TRAINING PIPELINE COMPLETED!")
    print("Models saved in 'models/' directory")
    print("Logs saved in 'logs/' directory")

if __name__ == "__main__":
    main()