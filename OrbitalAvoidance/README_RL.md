# Research-Grade Orbital Debris Avoidance RL

This package provides a Gymnasium environment with high-fidelity orbital physics suitable for research on autonomous debris avoidance.

Highlights:
- Cowell propagation with perturbations (J2, drag, optional SRP)
- RTN impulsive actions (delta-v) with fuel costs
- Observations include relative kinematics to K-nearest debris
- Reward balances collision risk, safe separation, and fuel use
- Deterministic scenario generation that guarantees repeated close approaches

## Install

Create/activate your Python env (Python 3.10+ recommended), then install deps:

```bash
pip install -U pip
pip install numpy matplotlib gymnasium stable-baselines3 astropy poliastro
```

## Quick smoke test

```bash
python - <<'PY'
from OrbitalAvoidance.research_grade_env import make_env
import numpy as np

env = make_env(seed=0)
obs, info = env.reset()
print('obs shape:', obs.shape, 'info:', info)
for _ in range(20):
    a = env.action_space.sample()
    obs, r, d, tr, info = env.step(a)
    if _ % 5 == 0:
        print('step', _, 'min_d[km]=', info['min_distance_km'], 'reward=', r)
    if d:
        print('terminated early due to collision')
        break
print('done')
PY
```

## Train PPO

```bash
python OrbitalAvoidance/train_ppo.py --steps 300000 --seed 7 --logdir runs/ppo
```

Artifacts will be written under `runs/ppo` with best checkpoints.

## Live Visualizer

Run the live visualizer (requires `matplotlib`):

```bash
# If running from anywhere, ensure PYTHONPATH points to the repo root
export PYTHONPATH=/home/bryson/RL_testing:$PYTHONPATH
python3 /home/bryson/RL_testing/OrbitalAvoidance/live_visualizer.py --mode zero --seed 7 --interval-ms 150
```

Controls:
- `p` = pause/resume stepping
- `z` = zero actions (no maneuvers)
- `r` = random actions (demonstration)
- `q` = quit

The left panel shows a 3D view (Earth, satellite trail, debris). The right panels show min distance (log scale), reward, and cumulative delta‑v.

Headless recording (no GUI, saves a file):

```bash
# Save MP4 (requires ffmpeg available on PATH)
export PYTHONPATH=/home/bryson/RL_testing:$PYTHONPATH
python3 /home/bryson/RL_testing/OrbitalAvoidance/live_visualizer.py --mode random --save /home/bryson/RL_testing/visualization.mp4 --frames 1200 --interval-ms 100

# Or save GIF (requires pillow)
python3 /home/bryson/RL_testing/OrbitalAvoidance/live_visualizer.py --mode random --save /home/bryson/RL_testing/visualization.gif --frames 600 --interval-ms 150
```

WSL note: If you see no window, enable an X server (VcXsrv/X410) and export your display, or use the `--save` option to record without a GUI.

Browser-based (WebAgg) server (no X server required):

```bash
export PYTHONPATH=/home/bryson/RL_testing:$PYTHONPATH
python3 /home/bryson/RL_testing/OrbitalAvoidance/live_visualizer.py --web --bind 127.0.0.1 --port 8080
# Then open http://127.0.0.1:8080 in your browser
```

Use `--bind 0.0.0.0` to expose on your LAN (security implications). Controls work via the browser focus.

## Configuration

See `EnvConfig` and `ForceConfig` in `research_grade_env.py` / `physics_models.py`.
- Increase `k_nearest` for more debris in observations
- Increase `n_debris` for denser fields
- Toggle `ForceConfig` flags for SRP, J2, drag
- Tune reward weights (`w_*`) to emphasize safety vs fuel

## Notes

- The environment currently uses simplified SRP and exponential atmosphere models; J2 is included. Extend with Moon/Sun third-body via astropy ephemerides for higher orbits.
- All units are SI unless noted. Internal state vectors use km and km/s for compatibility with poliastro.
