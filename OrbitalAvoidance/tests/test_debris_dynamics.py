import numpy as np
import pytest

try:
    from orbital_avoidance.envs.rl_training_env import create_rl_environment
    OREKIT_AVAILABLE = True
except Exception:
    # If imports fail (e.g., Orekit/Java not available on CI), mark unavailable and skip tests.
    create_rl_environment = None
    OREKIT_AVAILABLE = False


@pytest.mark.skipif(not OREKIT_AVAILABLE, reason="Orekit or environment imports unavailable on this runner")
def test_debris_are_dynamic_and_move(tmp_path):
    # Create environment with the provided scenario
    env = create_rl_environment(config_file='scenarios/leo_debris_avoidance.json', max_debris_tracked=10)
    obs, info = env.reset()

    # Ensure debris list non-empty
    base = env._base_env
    debris_objects = base.debris_objects
    # Support both dict and list storage shapes
    if isinstance(debris_objects, dict):
        items = list(debris_objects.items())
    else:
        items = [(getattr(d, 'name', f'Debris_{i}'), d) for i, d in enumerate(debris_objects)]

    assert len(items) > 0, "No debris objects found in scenario"

    # Record previous positions (support both dict and object shapes)
    prev_positions = {}
    for name, d in items:
        if isinstance(d, dict):
            prev_positions[name] = np.array(d.get('position', d.get('state', [0.0, 0.0, 0.0])))
        else:
            # object-like
            prev_positions[name] = np.array(getattr(d, 'position', getattr(d, 'state', [0.0, 0.0, 0.0])))

    action = [0.0, 0.0, 0.0]
    obs, reward, terminated, truncated, info = env.step(action)

    moved_any = False
    for name, d in items:
        if isinstance(d, dict):
            pos = np.array(d.get('position', d.get('state', [0.0, 0.0, 0.0])))
            dynamic_flag = d.get('dynamic', None)
        else:
            pos = np.array(getattr(d, 'position', getattr(d, 'state', [0.0, 0.0, 0.0])))
            dynamic_flag = getattr(d, 'dynamic', None)

        delta = pos - prev_positions[name]
        if not np.allclose(delta, 0.0):
            moved_any = True
        # If the object has an explicit 'dynamic' flag in its dict/object, assert it's True
        if dynamic_flag is not None:
            assert dynamic_flag is True, f"Debris {name} has dynamic flag False"

    assert moved_any, "No debris moved after one environment step"
