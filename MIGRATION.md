# Migration Guide: New `orbital_avoidance` Package Structure

The repository has been reorganized to provide a clean, maintainable Python package under `src/orbital_avoidance`.

## Summary of Changes

Old top-level modules (in `OrbitalAvoidance/`) have been migrated into a namespaced package:

```
src/orbital_avoidance/
  physics/orekit_physics.py
  envs/rl_training_env.py
  envs/configurable_env.py
  algo/rl_algorithms.py
  config/config_system.py
  __init__.py (exports public API)
```

Legacy files at the old paths now contain thin wrappers that:
- Emit a `DeprecationWarning`
- Re-export all symbols from the new module

## New Import Style

Before:
```python
from rl_training_env import RLOrbitalEnvironment
from rl_algorithms import RLAlgorithmFactory
from orekit_physics import OrekitOrbitPropagator
from config_system import SimulationConfig
```

After:
```python
from orbital_avoidance import (
    RLOrbitalEnvironment, RLAlgorithmFactory,
    OrekitOrbitPropagator, SimulationConfig
)
```
Or explicitly:
```python
from orbital_avoidance.envs import rl_training_env
from orbital_avoidance.physics import orekit_physics
```

## Backward Compatibility
All legacy imports still work for now, e.g.:
```python
from rl_training_env import RLOrbitalEnvironment  # Works but deprecated
```
You will see a `DeprecationWarning`. Update your imports to the new package style to silence the warning.

## Path Setup
When running scripts directly (e.g. `python OrbitalAvoidance/rl_train.py`), ensure the `src` directory is on `PYTHONPATH`. The training script now auto-adds it:
```python
REPO_ROOT/src  # auto-inserted by rl_train.py
```
If you write custom scripts, add:
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'src'))
```
Or install the package locally:
```bash
pip install -e .
```
(Add a `pyproject.toml` later if packaging.)

## Public API (exported via `orbital_avoidance.__init__`)
- Physics: `OrekitOrbitPropagator`, `OrbitElements`, `SpacecraftProperties`, `create_propagator`
- Environments: `RLOrbitalEnvironment`, `ConfigurableOrbitalEnvironment`, `create_rl_environment`
- Algorithms: `RLAlgorithmFactory`, `TrainingManager`, `RewardFunctionManager`, `create_algorithm`, `get_reward_config`
- Configuration: `SimulationConfig`, `SpacecraftConfig`, `OrbitConfig`, `DebrisFieldConfig`, `PhysicsConfig`, `RewardConfig`, `ObservationConfig`, `ConfigurationLoader`, `DebrisGenerator`

## Recommendations
1. Update all custom scripts to the new import style.
2. Treat legacy modules as temporary—removal planned after stabilization.
3. Add unit tests targeting the namespaced package paths.
4. Consider adding a packaging file (`pyproject.toml`) for editable installs.

## Next Steps (Planned)
- Archive or prune legacy debug/test directories.
- Add lightweight unit tests for propagation & environment reset/step.
- Provide example notebooks using new API.

## Need Help?
Open an issue or search the codebase for `DeprecationWarning` to find remaining legacy import points.

Happy navigating! 🚀
