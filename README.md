# jaxFlows

jaxFlows is a hydrologic modeling framework built on top of JAX. It provides typed
parameter structures and JAX-compatible model components for representing a wide
range of hydrologic processes.

## Features
- Parameter classes such as `SnowParams`, `CanopyParams`, `SoilParams`,
  `GroundwaterParams` and more defined as `NamedTuple` objects.
- Utilities to build per-cell parameter sets and stack them for multi-cell
  simulations.
- Land use lookup tables with helpers to encode and decode land use types.
- Core model routines for snow, soil, groundwater, glacier, permafrost,
  wetland and open water processes, all compatible with JAX's JIT and
  automatic differentiation.

## Usage
Example of running a simple single cell simulation:
```python
import jax.numpy as jnp
from jaxFlows import SnowParams, HydroParams, single_cell_model

snow = SnowParams(day_fraction=jnp.array(0.35))
params = HydroParams(snow=snow, landuse=None)

# precip, pet and temp are 1‑D arrays of shape (time,)
q = single_cell_model(precip, pet, temp, params, landuse_code=0)
```

## Installation
jaxFlows requires Python 3.8+ and `jax`. Install JAX and then install
jaxFlows from source:
```bash
pip install jax
pip install -e .
```

## License
MIT
