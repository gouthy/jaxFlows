import jax.numpy as jnp
from typing import Tuple, Dict
from .core import HydroState, _ensure_scalar_state
from .parameters import HydroParams
from .state import *

def step_standard_branch(
    operand: Tuple[HydroState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], HydroParams]
) -> Tuple[HydroState, jnp.ndarray]:
    """Standard hydrologic processes for most land use types"""
    state, inputs, params = operand
    precip, pet, temp = inputs
    
    # Sequential state updates
    state_snow, out_snow = update_snow_state(state, precip, temp, params)
    state_canopy, out_canopy = update_canopy_state(state_snow, precip, out_snow, pet, params)
    state_soil, out_soil = update_soil_state(state_canopy, precip, temp, pet, out_snow, out_canopy, params)
    state_gw, out_gw = update_groundwater_state(state_soil, precip, pet, out_snow, out_soil, params)
    
    final_state = _ensure_scalar_state(state_gw)
    runoff = jnp.squeeze(jnp.asarray(
        out_canopy["r_intercept"] + out_soil["r_surface"] + out_gw["r_sw"] + out_gw["r_ground"]
    ))
    
    return final_state, runoff

def step_water_body_branch(
    operand: Tuple[HydroState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], HydroParams]
) -> Tuple[HydroState, jnp.ndarray]:
    """Water body processes"""
    state, inputs, params = operand
    precip, pet, temp = inputs
    
    new_state, out = update_water_body_state(state, precip, pet, params.landuse.water_body)
    final_state = _ensure_scalar_state(new_state)
    runoff = jnp.squeeze(jnp.asarray(out["r_water"]))
    
    return final_state, runoff

def step_wetland_branch(
    operand: Tuple[HydroState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], HydroParams]
) -> Tuple[HydroState, jnp.ndarray]:
    """Wetland processes"""
    state, inputs, params = operand
    precip, pet, temp = inputs
    
    new_state, out = update_wetland_state(state, precip, pet, params.landuse.wetland)
    final_state = _ensure_scalar_state(new_state)
    runoff = jnp.squeeze(jnp.asarray(out["r_wet_surface"] + out["r_wet_ground"]))
    
    return final_state, runoff

def step_glacier_branch(
    operand: Tuple[HydroState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], HydroParams]
) -> Tuple[HydroState, jnp.ndarray]:
    """Glacier processes"""
    state, inputs, params = operand
    precip, pet, temp = inputs
    
    new_state, out = update_glacier_state(state, precip, temp, pet, params.landuse.glacier)
    final_state = _ensure_scalar_state(new_state)
    runoff = jnp.squeeze(jnp.asarray(out["r_glacier"]))
    
    return final_state, runoff

def step_permafrost_branch(
    operand: Tuple[HydroState, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], HydroParams]
) -> Tuple[HydroState, jnp.ndarray]:
    """Permafrost processes"""
    state, inputs, params = operand
    precip, pet, temp = inputs
    
    new_state, out = update_permafrost_state(state, precip, temp, pet, params.landuse.permafrost)
    final_state = _ensure_scalar_state(new_state)
    runoff = jnp.squeeze(jnp.asarray(out["r_permafrost"]))
    
    return final_state, runoff

# Branch function registry
def get_branch_functions():
    """Return list of branch functions for JAX switch"""
    return [
        step_water_body_branch,    # 0
        step_wetland_branch,       # 1
        step_glacier_branch,       # 2
        step_permafrost_branch,    # 3
        step_standard_branch       # 4 (default)
    ]

def _encode_branch_index(landuse_code: jnp.ndarray) -> jnp.ndarray:
    """Map landuse code to branch index"""
    from .utils import get_landuse_encoding
    
    encoding = get_landuse_encoding()
    water_codes = [encoding["WATR"], encoding["LAKE"]]
    
    is_water = jnp.isin(landuse_code, jnp.array(water_codes))
    is_wet = landuse_code == encoding["WET"]
    is_glac = landuse_code == encoding["GLAC"]
    is_perm = landuse_code == encoding["PERM"]
    
    return jnp.where(is_water, 0,
           jnp.where(is_wet, 1,
           jnp.where(is_glac, 2,
           jnp.where(is_perm, 3, 4))))