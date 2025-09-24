# 🚀 Advanced Research Dense LEO Training Scenario

This document describes the new **Advanced Research Dense LEO** scenario designed to align with current academic research standards while leveraging your high-performance hardware capabilities.

## 📊 Scenario Overview

### Key Research Improvements
| Aspect | Previous | New Advanced | Research Alignment |
|--------|----------|--------------|-------------------|
| **Total Debris** | 21 objects | **164 objects** | ✅ Academic standard (50-200) |
| **Static Debris** | 1 large object | **4 large objects** | ✅ Multiple trackable debris |
| **Dynamic Fields** | 1 field | **3 specialized fields** | ✅ Realistic debris distribution |
| **Observation Tracking** | 15 objects | **20 objects** | ✅ Enhanced situational awareness |
| **Simulation Duration** | 60 minutes | **120 minutes** | ✅ Multiple orbital periods |
| **Collision Penalty** | -1000 | **-100** (multi-objective) | ✅ Balanced reward structure |
| **Randomization** | Fixed seed | **Full randomization** | ✅ Generalization capability |

## 🎯 Debris Environment Design

### 1. Primary LEO Cloud (80 objects)
- **Altitude Range**: 390-420 km
- **Distribution**: Clustered (3 clusters, 50km radius each)
- **Dynamic Behavior**: Random perturbations every 5 minutes
- **Purpose**: Simulates main debris population from past collisions

### 2. Secondary LEO Cloud (50 objects)
- **Altitude Range**: 410-430 km  
- **Distribution**: Uniform spread
- **Dynamic Behavior**: Slower perturbations every 10 minutes
- **Purpose**: Overlapping orbital region for complex encounters

### 3. Small Debris Swarm (30 objects)
- **Altitude Range**: 400-415 km
- **Distribution**: Concentrated swarm
- **Dynamic Behavior**: Frequent perturbations every 3 minutes
- **Purpose**: High-density challenge area

### 4. Large Static Debris (4 objects)
- **Varied Altitudes**: 395, 405, 415, 422 km
- **Varied Inclinations**: 49.8°, 51.6°, 53.2°, 55.1°
- **Purpose**: Predictable large threats requiring precise avoidance

## 🧠 Enhanced Observation System

### Tracked Information (20 objects max)
- **Relative positions and velocities** (normalized by safety radius)
- **Time-to-collision estimates** (6-hour horizon)
- **Collision probabilities** (probabilistic screening)
- **Miss distance predictions**
- **Uncertainty ellipsoids** for each tracked object
- **Delta-V remaining** and fuel efficiency metrics
- **Orbital elements** for trajectory prediction

### Advanced Features
- **State history**: 10-step trajectory memory
- **Safety radius normalization**: Relative distances scaled by collision radius
- **Time features**: Mission elapsed time, orbital phase

## 🎯 Multi-Objective Reward System

### Reward Components
```json
Safety (60%):           Collision avoidance, proximity penalties
Fuel Efficiency (20%):  Delta-V optimization, maneuver efficiency  
Mission Completion (20%): Orbital stability, target maintenance
```

### Penalty Structure
- **Collision**: -100 (severe but not episode-ending for learning)
- **Close Approach** (<2km): -15 per object per timestep
- **Warning Zone** (<5km): -10 per object
- **Caution Zone** (<10km): -2 per object
- **Fuel Cost**: 0.05 per m/s delta-V

### Reward Bonuses
- **Fuel Efficiency**: +5 for staying under 70% delta-V budget
- **Mission Success**: +200 for completing full episode safely
- **Orbital Stability**: +3 for maintaining target orbit

## 🔄 Randomization System

### Episode-Level Randomization
- **Satellite Initial State**: ±10km altitude, ±2° inclination, random RAAN/anomaly
- **Debris Positions**: ±50km scatter from nominal positions
- **Debris Velocities**: ±0.1 m/s random perturbations
- **Environmental Conditions**: Solar activity, atmospheric density

### Benefits
- **Generalization**: Agent learns robust policies
- **Overfitting Prevention**: No memorization of fixed scenarios  
- **Real-World Applicability**: Handles uncertain conditions

## ⚙️ Hardware Optimization

### Your System Utilization
- **Parallel Environments**: 16 simultaneous training environments
- **CPU Cores**: 32 cores for physics computation
- **GPU Memory**: 80% allocation on GPUs 3 and 5
- **Memory Budget**: 128GB for large state spaces
- **Target Performance**: 50 FPS training throughput

### Physics Acceleration
- **Parallel Propagation**: 8 threads for orbital mechanics
- **Vectorized Operations**: Batch debris updates
- **GPU Physics**: CUDA-accelerated collision detection
- **Adaptive Integration**: High-precision (1e-10) with adaptive timesteps

## 🚀 Training Configuration

### Curriculum Learning (5M timesteps total)
1. **Easy Stage** (1M steps): 50% debris density, 60-min episodes
2. **Medium Stage** (2M steps): 75% debris density, 90-min episodes  
3. **Full Stage** (2M steps): 100% debris density, 120-min episodes

### Network Architecture
- **Policy Network**: [512, 512, 256] neurons
- **Value Network**: [512, 512, 256] neurons
- **Batch Size**: 512 (optimized for your GPUs)
- **Learning Rate**: 3e-4 with adaptive scheduling

## 📈 Research Metrics

### Tracked Performance Indicators
- **Collision Rate**: Percentage of episodes with collisions
- **Fuel Efficiency**: Average delta-V consumption per episode
- **Conjunction Events**: Number of close approaches (<10km)
- **Maneuver Count**: Average maneuvers per episode
- **Mission Success Rate**: Percentage of completed episodes
- **Average Miss Distance**: Safety margin effectiveness

## 🛠️ Usage Instructions

### Quick Start
```bash
# Activate your virtual environment
source ~/tf-fix-venv/bin/activate

# Run advanced training with GPU restriction
export CUDA_VISIBLE_DEVICES=3,5
python3 OrbitalAvoidance/rl_train.py \
    --config training_configs/advanced_research_dense_leo_config.json \
    --timesteps 5000000 \
    --save-dir ./advanced_dense_results
```

### Alternative: Direct Scenario Use
```bash
# Use scenario directly with custom parameters
python3 OrbitalAvoidance/rl_train.py \
    --algorithm ppo \
    --scenario scenarios/advanced_research_dense_leo.json \
    --timesteps 1000000 \
    --save-dir ./test_advanced_scenario
```

### Monitoring Training
```bash
# TensorBoard visualization
tensorboard --logdir ./training_logs/advanced_dense_leo_tb/

# Real-time logs
tail -f ./training_logs/advanced_dense_leo_eval/evaluations.npz
```

## 🔬 Research Comparison

### Alignment with Academic Standards
- **✅ Debris Count**: 164 objects (within 50-200 range)
- **✅ Multi-Field Design**: 3 specialized debris populations
- **✅ Dynamic Environment**: Moving debris with perturbations
- **✅ Extended Duration**: 2-hour episodes (multiple orbits)
- **✅ Enhanced Observations**: 20 tracked objects, 100km radius
- **✅ Multi-Objective Rewards**: Safety, fuel, mission weights
- **✅ Randomization**: Full episode-level randomization
- **✅ Research Metrics**: Comprehensive performance tracking

### Computational Requirements
- **Minimum**: 16GB RAM, 8-core CPU, 1 GPU
- **Recommended**: 64GB RAM, 16+ cores, 2+ GPUs (your setup)
- **Training Time**: ~12-24 hours for full 5M timesteps on your hardware

## 🆚 Scenario Comparison

| Use Case | Scenario | Debris Count | Duration | Hardware |
|----------|----------|--------------|----------|----------|
| **Quick Testing** | `leo_debris_avoidance.json` | 21 | 60 min | Any |
| **Standard Training** | `research_grade_leo.json` | 51 | 24 hours | Medium |
| **Research/Publication** | `advanced_research_dense_leo.json` | 164 | 120 min | High-end |

This advanced scenario positions your research at the cutting edge of RL-based space debris avoidance while maximizing utilization of your powerful hardware setup.

---

*Created: September 23, 2025*  
*Hardware: 6x NVIDIA GPU, 64-core CPU, 503GB RAM*  
*Research Standards: Current academic literature (2024-2025)*