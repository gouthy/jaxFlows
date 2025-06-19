from typing import Tuple, Dict, Any
import jax.numpy as jnp
from .core import HydroState
from .parameters import HydroParams, GlacierParams, PermafrostParams, WetlandParams, WaterBodyParams
from .modules import *

def update_snow_state(
    state: HydroState,
    precip: jnp.ndarray, 
    temp: jnp.ndarray,
    params: HydroParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update snow state and return new state + outputs"""
    out = snow_module(precip, temp, params.snow, state.s_snow, state.s_snow_liq)
    new_state = state._replace(s_snow=out["s_snow_new"], s_snow_liq=out["s_snow_liq_new"])
    return new_state, out

def update_canopy_state(
    state: HydroState,
    precip: jnp.ndarray,
    snow: Dict[str, float], 
    pet: jnp.ndarray,
    params: HydroParams
) -> Tuple[HydroState, Dict[str, Any]]:
    """Update canopy state"""
    out = canopy_module(precip, snow["p_snow"], snow["r_snow"], pet, params.landuse, state.s_skin, state.s_canopy)
    new_state = state._replace(s_skin=out["s_skin_new"], s_canopy=out["s_can_new"])
    return new_state, out

def update_soil_state(
    state: HydroState,
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    pet: jnp.ndarray,
    snow: Dict[str, float],
    canopy: Dict[str, Any], 
    params: HydroParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update soil state"""
    out = soil_module(precip, temp, pet, snow, canopy, params.landuse, state.s_soil)
    new_state = state._replace(s_soil=out["s_soil_new"])
    return new_state, out

def update_groundwater_state(
    state: HydroState,
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    snow: Dict[str, float],
    soil: Dict[str, float],
    params: HydroParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update groundwater state"""
    out = groundwater_process(precip, snow, soil, pet, params.landuse.groundwater, state.s_surface, state.s_ground)
    new_state = state._replace(s_surface=out["s_surface_new"], s_ground=out["s_ground_new"])
    return new_state, out

# Specialized state updates for different land use types
def update_glacier_state(
    state: HydroState,
    precip: jnp.ndarray,
    temp: jnp.ndarray, 
    pet: jnp.ndarray,
    params: GlacierParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update glacier state"""
    out = glacier_module(precip, temp, pet, params, state.s_glacier)
    new_state = state._replace(s_glacier=out["s_glacier_new"])
    return new_state, out

def update_permafrost_state(
    state: HydroState,
    precip: jnp.ndarray,
    temp: jnp.ndarray,
    pet: jnp.ndarray, 
    params: PermafrostParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update permafrost state"""
    out = permafrost_module(precip, temp, pet, params, state.s_active_layer)
    new_state = state._replace(s_active_layer=out["s_active_new"])
    return new_state, out

def update_wetland_state(
    state: HydroState,
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    params: WetlandParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update wetland state"""
    out = wetland_module(precip, pet, params, state.s_surface)
    new_state = state._replace(s_surface=out["s_surface_new"])
    return new_state, out

def update_water_body_state(
    state: HydroState,
    precip: jnp.ndarray,
    pet: jnp.ndarray,
    params: WaterBodyParams
) -> Tuple[HydroState, Dict[str, float]]:
    """Update water body state"""
    out = water_body_module(precip, pet, params, state.s_surface)
    new_state = state._replace(s_surface=out["s_surface_new"])
    return new_state, out