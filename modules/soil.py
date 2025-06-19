import jax
import jax.numpy as jnp
from typing import Dict
from ..parameters import LandUseParams

@jax.jit
def soil_module(
    precip: jnp.ndarray,
    t_surface: jnp.ndarray,
    pet: jnp.ndarray,
    snow: Dict[str, jnp.ndarray],
    canopy: Dict[str, jnp.ndarray],
    lu: LandUseParams,
    s_soil: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Infiltration, percolation, evaporation from soil."""
    params = lu.soil
    f_bare = lu.canopy.f_bare * (1 - lu.impervious_fraction)
    f_veg = lu.canopy.f_veg * (1 - lu.impervious_fraction)
    
    r_tr = canopy["r_transpiration"]
    e_can = canopy["e_can"]
    e_skin = canopy["e_skin"]
    
    # Sub-grid storage calculation
    s_sg = jnp.where(
        s_soil > params.s_sg_min,
        params.s_sg_max - (params.s_sg_max - params.s_sg_min) * 
        (1 - (s_soil - params.s_sg_min) / (params.s_max - params.s_sg_min)) ** (1 / (1 + params.exponent_b)),
        s_soil
    )
    
    D_sg = params.s_sg_max - params.s_sg_min
    c1 = jnp.where(D_sg > 0, jnp.minimum(1.0, ((params.s_sg_max - s_sg) / D_sg) ** (1 + params.exponent_b)), 1.0)
    c2 = jnp.where(D_sg > 0, jnp.maximum(0.0, ((params.s_sg_max - s_sg - r_tr) / D_sg) ** (1 + params.exponent_b)), 0.0)
    
    # Infiltration logic
    is_frozen = t_surface < 273.15
    no_through = (r_tr <= 0) | (s_sg + r_tr <= params.s_sg_min)
    over_sub = (s_sg + r_tr > params.s_sg_max)
    
    ideal_infil = jnp.where(
        is_frozen, 0.0,
        jnp.where(
            no_through, r_tr,
            jnp.where(
                over_sub,
                jnp.maximum(0.0, params.s_max - s_soil),
                jnp.minimum(r_tr, jnp.maximum(0.0, s_soil - params.s_max) + 
                           (D_sg / (1 + params.exponent_b)) * (c1 - c2))
            )
        )
    )
    
    avail_cap = jnp.maximum(0.0, params.s_max - s_soil)
    infiltration = jnp.clip(ideal_infil, 0.0, jnp.minimum(r_tr, avail_cap))
    r_surface = r_tr - infiltration
    
    # Evaporation calculations
    pet_available_t = jnp.maximum(0.0, pet * lu.crop_coefficient - e_can)
    pet_available_bs = jnp.maximum(0.0, pet * lu.crop_coefficient - e_skin)
    
    theta_t = jnp.clip((s_soil - params.s_wilt) / (params.f_so_crit * params.s_max - params.s_wilt), 0.0, 1.0)
    theta_bs = jnp.clip((s_soil - params.f_so_bs_low * params.s_max) / ((1 - params.f_so_bs_low) * params.s_max), 0.0, 1.0)
    
    e_t_cell = pet_available_t * theta_t * f_veg
    e_bs_cell = pet_available_bs * theta_bs * f_bare
    
    # Groundwater release
    r_gr_low = jnp.where(params.s_max > 0, params.r_gr_min * params.dt * (s_soil / params.s_max), 0.0)
    
    frac2 = jnp.where(
        (params.s_max - params.s_gr_max) > 0,
        jnp.maximum(0.0, s_soil - params.s_gr_max) / (params.s_max - params.s_gr_max),
        0.0
    )
    r_gr_high = jnp.where(
        s_soil > params.s_gr_max,
        (params.r_gr_max - params.r_gr_min) * params.dt * (frac2 ** params.r_gr_exp),
        0.0
    )
    
    cond_low = (s_soil <= params.s_gr_min) | (t_surface < 273.15)
    cond_mid = (s_soil <= params.s_gr_max)
    r_gr = jnp.where(cond_low, 0.0, jnp.where(cond_mid, r_gr_low, r_gr_low + r_gr_high))
    
    # Update soil storage
    d_s_soil = infiltration - r_gr - e_t_cell - e_bs_cell
    s_soil_new = jnp.clip(s_soil + d_s_soil, 0.0, params.s_max)
    
    return {
        "infiltration": infiltration,
        "r_surface": r_surface,
        "r_gr": r_gr,
        "e_t": e_t_cell,
        "e_bs": e_bs_cell,
        "delta_s_soil": d_s_soil,
        "s_soil_new": s_soil_new,
    }