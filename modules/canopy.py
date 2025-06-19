import jax
import jax.numpy as jnp
from ..parameters import LandUseParams

@jax.jit
def canopy_module(
    precip: jnp.ndarray,
    p_snow: jnp.ndarray,
    r_snow: jnp.ndarray,
    pet: jnp.ndarray,
    lu: LandUseParams,
    s_skin: jnp.ndarray,
    s_canopy: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    """Canopy interception and evaporation partitioning."""
    p_rain = precip - p_snow
    r_intercept = p_rain * lu.impervious_fraction
    p_rain_perv = p_rain * (1 - lu.impervious_fraction)
    total_perv = p_rain_perv + r_snow
    
    # fractional areas
    f_skin = lu.canopy.f_bare * (1 - lu.impervious_fraction)
    f_can = lu.canopy.f_veg * (1 - lu.impervious_fraction)
    f_perv_total = f_skin + f_can
    tiny = 1e-6
    
    # Process skin and canopy cells
    def process_cell(s_current, f_area, capacity_factor):
        alloc = jnp.where(f_perv_total > tiny,
                         total_perv * (f_area / f_perv_total), 0.0)
        avail = s_current + alloc
        cap = jnp.maximum(capacity_factor * f_area, tiny)
        fw = jnp.minimum(1.0, avail / cap)
        pet_eff = pet * lu.crop_coefficient * f_area * fw
        e_actual = jnp.minimum(avail, pet_eff)
        remaining = avail - e_actual
        runoff = jnp.maximum(remaining - cap, 0.0)
        s_new = jnp.clip(remaining - runoff, 0.0, cap)
        return s_new, e_actual, runoff
    
    # Skin cell
    s_skin_new, e_skin, r_skin = process_cell(s_skin, f_skin, lu.canopy.capacity)
    
    # Canopy cell  
    s_can_new, e_can, r_can = process_cell(s_canopy, f_can, lu.canopy.lai)
    
    return {
        "s_skin_new": s_skin_new,
        "e_skin": e_skin,
        "r_skin": r_skin,
        "s_can_new": s_can_new,
        "e_can": e_can,
        "r_can": r_can,
        "r_intercept": r_intercept,
        "r_transpiration": r_skin + r_can,
    }