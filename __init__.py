"""
Hydrologic modeling package with JAX support.
"""

from .core import HydroState, _require_jax
from .parameters import (
    SnowParams, CanopyParams, SoilParams, GroundwaterParams,
    GlacierParams, PermafrostParams, WetlandParams, WaterBodyParams,
    LandUseParams, HydroParams
)
from .landuse import default_landuse_lookup
from .utils import encode_landuse_types, decode_landuse_types, validate_cell_params
from .builder import build_cell_params, build_cell_params_from_landuse
from .model import hydrologic_model, single_cell_model

# Public API - matches original __all__
__all__ = [
    "SnowParams", "CanopyParams", "SoilParams", "GroundwaterParams",
    "GlacierParams", "PermafrostParams", "WetlandParams", "WaterBodyParams", 
    "LandUseParams", "HydroParams", "HydroState",
    "default_landuse_lookup", "encode_landuse_types", "decode_landuse_types",
    "build_cell_params", "build_cell_params_from_landuse",
    "hydrologic_model", "single_cell_model", "validate_cell_params"
]

# Version info
__version__ = "1.0.0"
__author__ = "Goutam Konapala"