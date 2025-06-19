import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import GroundwaterParams

@jax.jit
def groundwater_process(
    precip: jnp.ndarray,
    snow: Dict[str, jnp.ndarray],
    soil: Dict[str, jnp.ndarray],
    pet: jnp.ndarray,
    params: GroundwaterParams,
    s_surface: jnp.ndarray,
    s_ground: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Groundwater routing with proper water accounting."""
    
    # Water inputs (avoid double-counting)
    surface_inputs = soil["r_surface"]  # Surface runoff from soil
    groundwater_inputs = soil["r_gr"]   # Groundwater recharge from soil
    
    # Surface water processes
    f_sw = jnp.maximum(params.f_lake, params.f_wetland)
    e_sw_potential = pet * f_sw
    e_sw_actual = jnp.minimum(e_sw_potential, s_surface + surface_inputs)
    
    remaining_surface = s_surface + surface_inputs - e_sw_actual
    r_sw = remaining_surface / (params.lag_sw + 1.0)
    s_surface_new = jnp.maximum(0.0, remaining_surface - r_sw)
    
    # Groundwater processes
    remaining_ground = s_ground + groundwater_inputs
    r_ground = remaining_ground / (params.lag_gw + 1.0)
    s_ground_new = jnp.maximum(0.0, remaining_ground - r_ground)
    
    return {
        "f_sw": f_sw,
        "p_rain": precip - snow["p_snow"],  # For compatibility
        "r_surface": surface_inputs,
        "r_sw": r_sw,
        "e_sw": e_sw_actual,
        "delta_s_surface": s_surface_new - s_surface,
        "s_surface_new": s_surface_new,
        "r_ground": r_ground,
        "delta_s_ground": s_ground_new - s_ground,
        "s_ground_new": s_ground_new,
    }