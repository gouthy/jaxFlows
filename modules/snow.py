import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import SnowParams

@jax.jit
def snow_module(
    precip: jnp.ndarray,
    t_surf: jnp.ndarray,
    params: SnowParams,
    s_snow: jnp.ndarray,
    s_snlq: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
    """Snow accumulation, melt, retention, and refreeze."""
    # fraction of precip that falls as snow
    frac_snow = (params.t_snow_max - t_surf) / (params.t_snow_max - params.t_snow_min)
    frac_snow = jnp.clip(frac_snow, 0.0, 1.0)
    p_snow = precip * frac_snow
    
    # degree-day melt
    ddf = params.day_fraction * 8.3 + 0.7
    melt_pot = jnp.maximum(0.0, ddf * (t_surf - params.t_melt))
    
    # raw melt limited by available snow
    r_raw = jnp.minimum(melt_pot, s_snow + p_snow)
    
    # liquid retention
    max_liq = params.f_snlq_max * (s_snow + p_snow)
    possible = max_liq - s_snlq
    f_snlq = jnp.clip(r_raw, 0.0, possible)
    r_snow = r_raw - f_snlq
    
    # refreeze
    refreeze = jnp.where(t_surf < params.t_melt, s_snlq + f_snlq, 0.0)
    ds_snlq = f_snlq - refreeze
    s_snlq_n = jnp.clip(s_snlq + ds_snlq, 0.0, jnp.inf)
    
    # update snow storage
    ds_snow = p_snow - r_raw + refreeze
    s_snow_n = jnp.clip(s_snow + ds_snow, 0.0, jnp.inf)
    
    return {
        "p_snow": jnp.maximum(0.0, p_snow),
        "r_snow": jnp.maximum(0.0, r_snow),
        "f_snlq": jnp.maximum(0.0, f_snlq),
        "refreeze": jnp.maximum(0.0, refreeze),
        "s_snow_new": jnp.maximum(0.0, s_snow_n),
        "s_snow_liq_new": jnp.maximum(0.0, s_snlq_n),
    }