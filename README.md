# Satellite Control with Reinforcement Learning

This project combines orbital mechanics simulation with reinforcement learning to create an intelligent satellite control system.

## Setup and Installation

### 1. Activate the Python Environment
```bash
source /home/bryson/RL_testing/orbit/orbit_env/bin/activate
```

### 2. Verify Required Packages
The environment already has these key packages:
- `gymnasium` - RL environment framework
- `stable_baselines3` - State-of-the-art RL algorithms
- `poliastro` - Orbital mechanics and space mission design
- `astropy` - Astronomy and astrophysics library

## Running the Environment

### Quick Demo
```bash
cd /home/bryson/RL_testing
python quick_demo.py
```

### Full Satellite Control Environment
```bash
python satellite_env.py
```

Choose from:
1. **Test environment** - See how the simulation works
2. **Train RL agent** - Train a PPO agent to control satellite orbits
3. **Both** - Run test then training

## How It Works

### Satellite Environment (`satellite_env.py`)

**Observation Space (12 dimensions):**
- Current satellite position (x, y, z) in km
- Current satellite velocity (vx, vy, vz) in km/s
- Target orbit position (x, y, z) in km  
- Target orbit velocity (vx, vy, vz) in km/s

**Action Space (3 dimensions):**
- Thrust commands (thrust_x, thrust_y, thrust_z) normalized between -1 and 1
- Scaled to maximum 0.1 m/s thrust per action

**Reward System:**
- Negative reward for distance from target orbit
- Bonus reward (+100) for successfully reaching target
- Small positive reward (+1) for maintaining stable orbit
- Large penalty (-100) for orbit decay or invalid maneuvers

**Mission Goal:**
- Start in a random Low Earth Orbit (300-500 km altitude)
- Learn to reach Geostationary Earth Orbit (35,786 km altitude)
- Maintain the target orbit within tolerance

### RL Algorithm
- **PPO (Proximal Policy Optimization)** - Proven effective for continuous control
- **Neural Network Policy** - Maps observations to thrust actions
- **Training Time** - ~100,000 steps (adjustable)

## Advanced Usage

### Custom Target Orbits
Modify the target orbit in `satellite_env.py`:

```python
# Example: Target a specific altitude and inclination
self.target_orbit = Orbit.circular(
    Earth,
    alt=800 * u.km,      # 800 km altitude
    inc=98.7 * u.deg     # Sun-synchronous orbit
)
```

### Hyperparameter Tuning
Adjust training parameters in the `train_satellite_controller()` function:

```python
model = PPO(
    "MlpPolicy", 
    env, 
    learning_rate=3e-4,    # Learning rate
    n_steps=2048,          # Steps per update
    batch_size=64,         # Batch size
    n_epochs=10,           # Training epochs
    clip_range=0.2,        # PPO clip range
)
```

### Monitoring Training
Training logs are saved to `./satellite_tensorboard/`. View with:

```bash
tensorboard --logdir=./satellite_tensorboard/
```

### Loading Trained Models
```python
from stable_baselines3 import PPO
from satellite_env import SatelliteOrbitControlEnv

# Load trained model
model = PPO.load("satellite_control_ppo")

# Test on environment
env = SatelliteOrbitControlEnv()
obs, _ = env.reset()

for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

## Integration Options

### 1. With Ground Control Systems
Create interfaces to real satellite control protocols:
```python
def send_to_ground_station(thrust_commands):
    # Convert RL actions to actual satellite commands
    # Interface with existing ground control software
    pass
```

### 2. Multi-Satellite Coordination
Extend to control multiple satellites:
```python
class MultiSatelliteEnv(gym.Env):
    def __init__(self, num_satellites=3):
        self.satellites = [SatelliteOrbitControlEnv() for _ in range(num_satellites)]
        # Implement coordination logic
```

### 3. Real-Time Operation
For real satellite control (simulation mode):
```python
class RealTimeControlEnv(SatelliteOrbitControlEnv):
    def __init__(self):
        super().__init__()
        self.real_time_mode = True
        self.dt = 60.0  # 1-minute real-time updates
```

## Safety Considerations

**For Real Satellite Control:**
1. **Always test in simulation first**
2. **Implement thrust magnitude limits**
3. **Add collision avoidance**
4. **Include emergency stop mechanisms**
5. **Validate all maneuvers before execution**

**Current Safety Features:**
- Maximum thrust limits (0.1 m/s per action)
- Orbit validity checking
- Automatic episode termination on failures
- Reward penalties for dangerous maneuvers

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure virtual environment is activated
2. **Orbit Decay**: Reduce maximum thrust or add altitude constraints
3. **Training Not Converging**: Adjust reward function or learning parameters
4. **Memory Issues**: Reduce batch size or number of training steps

### Debug Mode:
Add verbose output to track satellite behavior:
```python
env = SatelliteOrbitControlEnv(render_mode='human')
# Will print detailed orbital information each step
```

## Next Steps

1. **Experiment with different target orbits**
2. **Add more realistic orbital perturbations (J2, atmospheric drag)**
3. **Implement fuel consumption constraints**
4. **Add multi-objective optimization (time, fuel, accuracy)**
5. **Test different RL algorithms (SAC, TD3, A2C)**
6. **Create visualization tools for orbit trajectories**

## Research Extensions

- **Formation Flying**: Multiple satellites maintaining relative positions
- **Constellation Management**: Coordinating large satellite networks
- **Debris Avoidance**: Real-time collision avoidance with space debris
- **Mission Planning**: Long-term autonomous mission execution
- **Fault Recovery**: Handling satellite system failures