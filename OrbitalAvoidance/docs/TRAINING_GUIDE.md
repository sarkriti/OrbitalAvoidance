# RL Training Guide for Orbital Debris Avoidance

## 🚀 Quick Start Training

### Basic Training Commands

```bash
# Train PPO agent with best model saving (recommended)
python3 rl_train.py --algorithm ppo --scenario scenarios/leo_debris_avoidance.json --reward balanced --timesteps 100000

# Train SAC agent with all checkpoint saving
python3 rl_train.py --algorithm sac --scenario scenarios/leo_debris_avoidance.json --reward aggressive --timesteps 500000 --save-all-checkpoints

# Use custom configuration file
python3 rl_train.py --config training_configs/custom_training.json
```

### Testing Trained Models

```bash
# Test a trained model
python rl_train.py --mode test --model-path trained_models/debris_avoidance_ppo/model_checkpoint_100000_steps.zip --scenario scenarios/leo_debris_avoidance.json --test-episodes 20
```

## 🎯 Modifying Reward Functions

### Built-in Reward Types
- `balanced` - Balances safety, fuel efficiency, mission success
- `conservative` - Emphasizes safety over efficiency  
- `aggressive` - Prioritizes mission completion over fuel

### Custom Reward Configuration

Create a custom reward function by modifying the reward weights in your scenario configuration:

```json
{
  "rewards": {
    "reward_type": "multi_objective",
    "collision_penalty": -1000.0,
    "fuel_cost_weight": 0.1,
    "safety_distance_km": 1.0,
    "close_approach_penalty_weight": 10.0,
    "mission_success_reward": 100.0
  }
}
```

### Reward Function Parameters

**Collision Rewards:**
- `collision_penalty`: Penalty for collisions (-1000.0)
- `safety_distance_km`: Minimum safe distance (1.0 km)
- `close_approach_penalty_weight`: Weight for close approach penalty (10.0)

**Fuel Rewards:**
- `fuel_cost_weight`: Weight for fuel efficiency (0.1)
- Higher values = more fuel conservative

**Mission Rewards:**
- `mission_success_reward`: Bonus for completing mission (100.0)

### Creating Custom Reward Functions

To create completely custom reward logic, modify the `_calculate_*_reward` methods in `rl_training_env.py`:

```python
def _calculate_collision_reward(self) -> float:
    """Your custom collision reward logic"""
    # Custom implementation here
    pass

def _calculate_fuel_reward(self, action: np.ndarray) -> float:
    """Your custom fuel reward logic"""
    # Custom implementation here  
    pass
```

## ⚙️ Algorithm-Specific Parameters

### PPO (Proximal Policy Optimization)
```json
{
  "ppo": {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "policy_kwargs": {
      "net_arch": {"pi": [256, 256], "vf": [256, 256]}
    }
  }
}
```

### SAC (Soft Actor-Critic)
```json
{
  "sac": {
    "learning_rate": 0.0003,
    "buffer_size": 1000000,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99
  }
}
```

### TD3 (Twin Delayed Deep Deterministic)
```json
{
  "td3": {
    "learning_rate": 0.001,
    "buffer_size": 1000000,
    "batch_size": 100,
    "tau": 0.005,
    "policy_delay": 2
  }
}
```

## 📊 Training Configuration

### Complete Training Configuration Example
```json
{
  "algorithm": "ppo",
  "scenario_file": "scenarios/leo_debris_avoidance.json",
  "reward_function": "balanced",
  "training": {
    "total_timesteps": 500000,
    "checkpoint_freq": 50000,
    "eval_freq": 25000,
    "reward_threshold": null
  },
  "environment": {
    "max_debris_tracked": 10,
    "render_mode": null
  },
  "output": {
    "log_dir": "./training_logs",
    "save_dir": "./trained_models", 
    "experiment_name": "my_custom_training"
  }
}
```

### Environment Parameters
- `max_debris_tracked`: Maximum debris objects in observation space (5-20)
- `render_mode`: Visualization during training (null for faster training)

### Training Parameters
- `total_timesteps`: Total training steps (100k-1M+)
- `checkpoint_freq`: How often to save models (10k-50k) - ignored if save_best_only=True
- `eval_freq`: How often to evaluate performance (5k-25k)
- `reward_threshold`: Stop training when reward reached (optional)
- `save_best_only`: Only save the best model based on evaluation performance (default: True)

### Model Saving Options

**Best Model Only (Recommended):**
```json
{
  "training": {
    "save_best_only": true,
    "eval_freq": 5000
  }
}
```
- Only saves the model with the highest evaluation performance
- Saves disk space and prevents model clutter
- Automatically tracks best performance during training
- Model saved as `best_model.zip` in the save directory

**All Checkpoint Models:**
```json
{
  "training": {
    "save_best_only": false,
    "checkpoint_freq": 10000
  }
}
```
- Saves models at regular intervals regardless of performance
- Useful for analyzing training progression
- Can consume significant disk space

**Command Line Usage:**
```bash
# Save only best model (default)
python3 rl_train.py --algorithm ppo --scenario scenarios/leo_debris_avoidance.json

# Save all checkpoint models
python3 rl_train.py --algorithm ppo --scenario scenarios/leo_debris_avoidance.json --save-all-checkpoints
```

## 🛠️ Advanced Customization

### Physics Configuration

Modify physics parameters in your scenario file:

```json
{
  "physics": {
    "propagator_type": "orekit_numerical",
    "time_step_seconds": 60.0,
    "simulation_duration_seconds": 86400.0,
    "include_drag": true,
    "include_solar_radiation": true,
    "include_j2_perturbation": true,
    "collision_threshold_km": 0.01
  }
}
```

### Observation Space Configuration

```json
{
  "observations": {
    "include_relative_positions": true,
    "include_relative_velocities": true,
    "include_fuel_levels": true,
    "include_time_to_collision": true,
    "max_tracked_objects": 10,
    "observation_radius_km": 100.0
  }
}
```

### Spacecraft Configuration

```json
{
  "satellites": [
    {
      "name": "main_satellite",
      "mass_kg": 1000.0,
      "max_thrust_n": 10.0,
      "fuel_capacity_kg": 100.0,
      "collision_radius_m": 5.0,
      "orbit": {
        "altitude_km": 400.0,
        "inclination_deg": 51.6,
        "eccentricity": 0.001,
        "longitude_of_ascending_node_deg": 0.0,
        "argument_of_perigee_deg": 0.0,
        "true_anomaly_deg": 0.0
      }
    }
  ]
}
```

## 📈 Monitoring Training

### Training Logs
- Logs saved to: `training_logs/[experiment_name]/`
- TensorBoard logs: `tensorboard --logdir training_logs/`
- Model checkpoints: `trained_models/[experiment_name]/`

### Key Metrics to Monitor
- **Episode Reward**: Average reward per episode
- **Success Rate**: Percentage of collision-free episodes  
- **Fuel Efficiency**: Average fuel consumption
- **Training Loss**: Algorithm convergence

## 🔍 Debugging Training Issues

### Common Problems & Solutions

**Training Not Converging:**
- Reduce learning rate (0.0001 instead of 0.0003)
- Increase training timesteps
- Adjust reward function weights
- Try different algorithm (SAC for continuous control)

**Reward Too Sparse:**
- Add intermediate rewards
- Reduce collision penalty magnitude
- Increase fuel efficiency rewards

**Model Too Conservative:**
- Reduce collision penalty
- Increase mission success reward
- Use "aggressive" reward function

**Model Too Aggressive:**
- Increase collision penalty
- Add safety distance rewards
- Use "conservative" reward function

### Training Tips
1. Start with shorter training runs (50k-100k steps) to test configurations
2. Use balanced reward function initially
3. Monitor TensorBoard for training progress
4. Save checkpoints frequently for experimentation
5. Test different algorithms if one doesn't work well
6. Validate training with: `python validation/research_grade_validation.py`

## 🎮 Model Usage After Training

### Loading and Using Trained Models

```python
from stable_baselines3 import PPO
import gymnasium as gym
from rl_training_env import create_rl_environment

# Load trained model
model = PPO.load("trained_models/debris_avoidance_ppo/model_checkpoint_100000_steps.zip")

# Create environment
env = create_rl_environment("scenarios/leo_debris_avoidance.json")

# Use model for inference
obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Integration with Web Simulator

Trained models automatically integrate with the enhanced web simulator:
```bash
python enhanced_web_simulator.py --config scenarios/leo_debris_avoidance.json
```

The simulator will use the latest trained model for real-time RL control.