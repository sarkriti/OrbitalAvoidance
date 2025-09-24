import os
import sys
import numpy as np

def test_env_smoke():
    # Ensure src is on PYTHONPATH for tests run from repo root
    repo_root = os.path.dirname(os.path.dirname(__file__))
    src_path = os.path.join(repo_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from orbital_avoidance.envs.rl_training_env import create_rl_environment
    env = create_rl_environment(config_file='scenarios/leo_debris_avoidance.json')
    obs, info = env.reset()
    assert obs is not None
    for _ in range(3):
        action = np.zeros(env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        if terminated:
            break
