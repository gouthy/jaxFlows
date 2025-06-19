import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import PermafrostParams

@jax.jit
def permafrost_module(
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    pet: jnp.ndarray,
    params: PermafrostParams,
    s_active: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Calibratable permafrost module with temperature-dependent processes."""
    
    # Snow/rain partitioning using calibratable thresholds
    frac_snow = jnp.clip(
        (params.t_snow_max - temp) / (params.t_snow_max - params.t_snow_min), 
        0.0, 1.0
    )
    p_snow, p_rain = precip * frac_snow, precip * (1 - frac_snow)
    
    # Snow melt with calibratable degree-day factor
    melt_potential = jnp.maximum(0.0, params.ddf_permafrost * (temp - params.t_melt))
    snow_melt = jnp.minimum(melt_potential, p_snow)
    remaining_snow = p_snow - snow_melt
    
    # Temperature-dependent sublimation
    sublimation_factor = jnp.where(
        temp < params.sublimation_temp_threshold,
        params.sublimation_factor_cold,
        params.sublimation_factor_warm
    )
    snow_sublimation = jnp.minimum(remaining_snow, pet * sublimation_factor)
    delayed_melt = remaining_snow - snow_sublimation
    
    # Total liquid water
    liquid_water = p_rain + snow_melt + delayed_melt
    
    # Active layer dynamics
    thaw_depth = jnp.minimum(
        params.max_active_depth,
        params.thaw_rate * jnp.maximum(temp - params.t_melt, 0.0)
    )
    effective_thaw_depth = jnp.maximum(thaw_depth, 0.1)  # Minimum thaw depth
    capacity = params.theta_sat * effective_thaw_depth
    
    # Water balance
    available_capacity = jnp.maximum(0.0, capacity - s_active)
    infiltration = jnp.minimum(liquid_water, available_capacity)
    surface_runoff = liquid_water - infiltration
    
    # Temperature-dependent evaporation and drainage
    available_for_et = s_active + infiltration
    cold_et_factor = jnp.where(
        temp < params.sublimation_temp_threshold,
        params.cold_et_factor,
        1.0
    )
    e_active = jnp.minimum(pet * cold_et_factor, available_for_et)
    
    remaining_water = available_for_et - e_active
    drainage_rate = jnp.where(
        temp < params.sublimation_temp_threshold,
        params.drainage_rate_cold,
        params.drainage_rate_warm
    )
    subsurface_flow = remaining_water * drainage_rate
    
    # Updated storage
    s_active_new = jnp.maximum(0.0, remaining_water - subsurface_flow)
    
    return {
        "s_active_new": s_active_new,
        "r_permafrost": surface_runoff + subsurface_flow,
        "e_active": e_active + snow_sublimation,
        # Diagnostics
        "p_snow": p_snow,
        "p_rain": p_rain,
        "snow_melt": snow_melt,
        "snow_sublimation": snow_sublimation,
        "delayed_melt": delayed_melt,
        "infiltration": infiltration,
        "surface_runoff": surface_runoff,
        "subsurface_flow": subsurface_flow,
        "thaw_depth": effective_thaw_depth,
        "frac_snow": frac_snow,
    }