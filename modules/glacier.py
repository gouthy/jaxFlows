import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import GlacierParams

def _calculate_phase_partition(temp: jnp.ndarray, params: GlacierParams, precip: jnp.ndarray):
    """Calculate snow/rain partitioning - shared logic"""
    frac_snow = jnp.clip(
        (params.t_snow_max - temp) / (params.t_snow_max - params.t_snow_min), 
        0.0, 1.0
    )
    return precip * frac_snow, precip * (1 - frac_snow), frac_snow

@jax.jit
def glacier_module(
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    pet: jnp.ndarray,
    params: GlacierParams,
    s_glacier: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Glacier mass balance with proper water accounting."""
    
    # Snow/rain partitioning
    p_snow, p_rain, frac_snow = _calculate_phase_partition(temp, params, precip)
    
    # Glacier melt potential
    melt_potential = jnp.maximum(0.0, params.ddf * (temp - params.t_melt))
    
    # Available glacier ice for melting
    glacier_ice_melt = jnp.minimum(melt_potential, s_glacier)
    
    # Remaining melt potential can melt new snow
    remaining_melt_potential = melt_potential - glacier_ice_melt
    new_snow_melt = jnp.minimum(remaining_melt_potential, p_snow)
    
    # Total actual melt and refreeze
    actual_melt = glacier_ice_melt + new_snow_melt
    refreeze = params.refreeze_frac * actual_melt
    net_melt = actual_melt - refreeze
    
    # Snow accumulation and sublimation
    snow_accumulation = p_snow - new_snow_melt
    available_for_sublimation = snow_accumulation + jnp.minimum(s_glacier * 0.01, 10.0)
    sublimation = jnp.minimum(params.sublimation_frac * pet, available_for_sublimation)
    
    # Update glacier storage
    glacier_gain = snow_accumulation + refreeze
    glacier_loss = glacier_ice_melt + sublimation
    s_glacier_new = jnp.maximum(0.0, s_glacier + glacier_gain - glacier_loss)
    
    # Total outputs
    total_runoff = p_rain + net_melt
    total_et = sublimation
    
    return {
        "s_glacier_new": s_glacier_new,
        "r_glacier": total_runoff,
        "e_glacier": total_et,
        "p_snow": p_snow,
        "p_rain": p_rain,
        "snow_accumulation": snow_accumulation,
        "actual_melt": actual_melt,
        "net_melt": net_melt,
        "refreeze": refreeze,
        "sublimation": sublimation,
        "frac_snow": frac_snow,
        "melt_potential": melt_potential
    }