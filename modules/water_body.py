import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import WaterBodyParams

@jax.jit
def water_body_module(
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    params: WaterBodyParams,
    s_surface: jnp.ndarray,
    inflow: jnp.ndarray = 0.0
) -> Dict[str, jnp.ndarray]:
    """Water body storage and evaporation."""
    
    # Total available water
    available = precip + s_surface + inflow
    
    # Enhanced evaporation from water body
    et_water = jnp.minimum(pet * params.et_factor, available)
    remaining = jnp.maximum(0.0, available - et_water)
    
    # Fixed outflow rate
    outflow_rate = 0.1
    r_water = remaining * outflow_rate
    s_surface_new = jnp.maximum(0.0, remaining - r_water)
    
    return {
        "s_surface_new": s_surface_new,
        "et_water": et_water,
        "r_water": r_water,
        "outflow_rate": jnp.array(outflow_rate),
    }