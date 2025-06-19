import jax
import jax.numpy as jnp
from typing import Sequence, Tuple, Dict, Optional

from .core import HydroState, _ensure_scalar_state, _require_jax
from .parameters import HydroParams
from .branches import get_branch_functions, _encode_branch_index
from .utils import encode_landuse_types, _validate_inputs

def single_cell_model(
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,  # ← Should be HydroParams, not LandUseParams
    landuse_code: int
) -> jnp.ndarray:
    """Run one cell's simulation over time."""
    _require_jax()
    
    branch_idx = _encode_branch_index(jnp.array(landuse_code))
    
    def step_fun(state: HydroState, inputs: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
        operand = (state, inputs, params)  # ← Pass full params, not params.landuse
        new_state, runoff = jax.lax.switch(branch_idx, get_branch_functions(), operand)
        return new_state, runoff
    
    # Initialize all state variables as JAX scalars
    init_state = HydroState(
        s_snow=jnp.array(0.0),
        s_snow_liq=jnp.array(0.0),
        s_skin=jnp.array(0.0),
        s_canopy=jnp.array(0.0),
        s_soil=jnp.array(0.0),
        s_surface=jnp.array(0.0),
        s_ground=jnp.array(0.0),
        s_glacier=jnp.array(0.0),
        s_active_layer=jnp.array(0.0)
    )
    
    _, runoff_ts = jax.lax.scan(step_fun, init_state, (precip, pet, temp))
    return runoff_ts

def hydrologic_model(
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    temp: jnp.ndarray,
    params: HydroParams,
    landuse_types: Sequence[str] = None
) -> jnp.ndarray:
    """Run batched hydrologic model across all cells."""
    _require_jax()
    _validate_inputs(precip, pet, temp)
    T, n_cells = precip.shape
    
    if landuse_types is not None:
        if len(landuse_types) != n_cells:
            raise ValueError(f"Landuse types length mismatch: {len(landuse_types)} vs {n_cells}")
        
        codes = encode_landuse_types(landuse_types)
        
        def per_cell_fun(p, e, t, p_params, code):
            return single_cell_model(p, e, t, p_params, code)
        
        return jax.vmap(per_cell_fun, in_axes=(1, 1, 1, 0, 0), out_axes=1)(
            precip, pet, temp, params, codes)
    else:
        return jax.vmap(
            lambda p, e, t, p_params: single_cell_model(p, e, t, p_params, -1),
            in_axes=(1, 1, 1, 0), out_axes=1
        )(precip, pet, temp, params)