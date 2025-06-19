import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import WetlandParams

@jax.jit
def wetland_module(
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    params: WetlandParams,
    s_surface: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Wetland water balance with enhanced evaporation."""
    
    # Wetland receives its fraction of precipitation
    inflow_wet = precip * params.f_wet
    available = s_surface + inflow_wet
    
    # Enhanced evaporation
    e_wet = jnp.minimum(pet * params.f_wet, available)
    remaining = available - e_wet
    
    # Delayed outflows
    r_wet_surface = remaining / (params.lag_sw + 1.0)
    r_wet_ground = remaining * 0.05 / (params.lag_gw + 1.0)  # Minimal groundwater
    
    total_outflow = r_wet_surface + r_wet_ground
    s_surface_new = jnp.maximum(0.0, remaining - total_outflow)
    
    return {
        "s_surface_new": s_surface_new,
        "e_wet": e_wet,
        "r_wet_surface": r_wet_surface,
        "r_wet_ground": r_wet_ground,
    }