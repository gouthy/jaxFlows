from typing import NamedTuple
import jax.numpy as jnp

class SnowParams(NamedTuple):
    day_fraction: jnp.ndarray
    t_snow_min: jnp.ndarray = jnp.array(272.05)
    t_snow_max: jnp.ndarray = jnp.array(276.45)
    t_melt: jnp.ndarray = jnp.array(273.15)
    f_snlq_max: jnp.ndarray = jnp.array(0.06)

class CanopyParams(NamedTuple):
    f_bare: jnp.ndarray
    f_veg: jnp.ndarray
    lai: jnp.ndarray
    capacity: jnp.ndarray

class SoilParams(NamedTuple):
    s_max: jnp.ndarray
    s_wilt: jnp.ndarray
    s_gr_min: jnp.ndarray
    s_gr_max: jnp.ndarray
    s_sg_min: jnp.ndarray
    s_sg_max: jnp.ndarray
    exponent_b: jnp.ndarray
    r_gr_min: jnp.ndarray
    r_gr_max: jnp.ndarray
    dt: jnp.ndarray
    f_so_crit: jnp.ndarray = jnp.array(0.75)
    f_so_bs_low: jnp.ndarray = jnp.array(0.05)
    r_gr_exp: jnp.ndarray = jnp.array(1.5)

class GlacierParams(NamedTuple):
    ddf: jnp.ndarray
    refreeze_frac: jnp.ndarray
    sublimation_frac: jnp.ndarray
    lag: jnp.ndarray
    t_snow_min: jnp.ndarray = jnp.array(271.15)
    t_snow_max: jnp.ndarray = jnp.array(274.15)
    t_melt: jnp.ndarray = jnp.array(273.15)

class PermafrostParams(NamedTuple):
    max_active_depth: jnp.ndarray
    thaw_rate: jnp.ndarray
    theta_sat: jnp.ndarray
    sublimation_factor_cold: jnp.ndarray = jnp.array(0.8)
    sublimation_factor_warm: jnp.ndarray = jnp.array(0.1)
    sublimation_temp_threshold: jnp.ndarray = jnp.array(273.15)
    cold_et_factor: jnp.ndarray = jnp.array(0.1)
    drainage_rate_cold: jnp.ndarray = jnp.array(0.01)
    drainage_rate_warm: jnp.ndarray = jnp.array(0.05)
    ddf_permafrost: jnp.ndarray = jnp.array(1.5)
    t_snow_min: jnp.ndarray = jnp.array(272.05)
    t_snow_max: jnp.ndarray = jnp.array(276.45)
    t_melt: jnp.ndarray = jnp.array(273.15)

class WetlandParams(NamedTuple):
    f_wet: jnp.ndarray
    lag_sw: jnp.ndarray
    lag_gw: jnp.ndarray

class WaterBodyParams(NamedTuple):
    is_water_body: bool
    et_factor: jnp.ndarray

class GroundwaterParams(NamedTuple):
    f_lake: jnp.ndarray = jnp.array(0.0)
    f_wetland: jnp.ndarray = jnp.array(0.0)
    lag_sw: jnp.ndarray = jnp.array(0.0)
    lag_gw: jnp.ndarray = jnp.array(0.0)

class LandUseParams(NamedTuple):
    canopy: CanopyParams
    soil: SoilParams
    groundwater: GroundwaterParams
    impervious_fraction: jnp.ndarray
    glacier: GlacierParams
    permafrost: PermafrostParams
    wetland: WetlandParams
    water_body: WaterBodyParams
    crop_coefficient: jnp.ndarray = jnp.array(1.0)

class HydroParams(NamedTuple):
    snow: SnowParams
    landuse: LandUseParams

# Default inactive parameters factory
def create_inactive_params():
    """Factory for creating inactive parameter sets"""
    return {
        'glacier': GlacierParams(
            ddf=jnp.array(0.0), 
            refreeze_frac=jnp.array(0.0), 
            sublimation_frac=jnp.array(0.0), 
            lag=jnp.array(1.0)
        ),
        'permafrost': PermafrostParams(
            max_active_depth=jnp.array(0.0), 
            thaw_rate=jnp.array(0.0), 
            theta_sat=jnp.array(0.0)
        ),
        'wetland': WetlandParams(
            f_wet=jnp.array(0.0), 
            lag_sw=jnp.array(1.0), 
            lag_gw=jnp.array(1.0)
        ),
        'water_body': WaterBodyParams(
            is_water_body=False,
            et_factor=jnp.array(1.0)
        )
    }