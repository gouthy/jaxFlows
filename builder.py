from typing import Sequence, Tuple, Dict, Any, Optional, List
import jax.numpy as jnp
from jax.tree_util import tree_map
from .core import _require_jax
from .parameters import HydroParams, SnowParams
from .landuse import default_landuse_lookup

def build_cell_params(
    per_cell: Sequence[HydroParams],
    landuse_types: Sequence[str] = None
) -> Tuple[HydroParams, Optional[Sequence[str]]]:
    """Build stacked parameters for multiple cells"""
    _require_jax()
    
    if not per_cell:
        raise ValueError("per_cell sequence cannot be empty")
    if not isinstance(per_cell, (list, tuple)):
        raise ValueError("per_cell must be a list or tuple of HydroParams")
    
    # Validate inputs
    for i, p in enumerate(per_cell):
        if not isinstance(p, HydroParams):
            raise ValueError(f"Element {i} is not a HydroParams: {type(p)}")
    
    # Validate landuse types if provided
    lu_seq = None
    if landuse_types is not None:
        if len(landuse_types) != len(per_cell):
            raise ValueError(f"Number of landuse types ({len(landuse_types)}) must match number of cells ({len(per_cell)})")
        
        valid = set(default_landuse_lookup().keys())
        for i, lu in enumerate(landuse_types):
            if not isinstance(lu, str):
                raise ValueError(f"Landuse type {i} must be string, got {type(lu)}")
            if lu not in valid:
                raise ValueError(f"Unknown landuse '{lu}' at index {i}. Valid: {sorted(valid)}")
        lu_seq = list(landuse_types)
    
    # Stack parameters
    try:
        stacked = tree_map(lambda *xs: jnp.stack(xs), *per_cell)
        n = len(per_cell)
        
        # Validate stacking worked correctly
        if hasattr(stacked.snow, "day_fraction"):
            if stacked.snow.day_fraction.shape != (n,):
                raise ValueError(f"Stacking failed: expected shape ({n},), got {stacked.snow.day_fraction.shape}")
        
        return stacked, lu_seq
    except Exception as e:
        raise ValueError(f"Failed to stack cell parameters: {e}") from e

def build_cell_params_from_landuse(
    landuse_types: Sequence[str],
    snow_params: SnowParams = None,
    custom_overrides: Dict[str, Dict[str, Any]] = None
) -> Tuple[HydroParams, Sequence[str]]:
    """Build cell parameters from landuse types"""
    _require_jax()
    
    if not landuse_types:
        raise ValueError("landuse_types sequence cannot be empty")
    
    lookup = default_landuse_lookup()
    if snow_params is None:
        snow_params = SnowParams(day_fraction=jnp.array(0.35))
    
    per_cell: List[HydroParams] = []
    for i, lu in enumerate(landuse_types):
        if not isinstance(lu, str):
            raise ValueError(f"Landuse type at index {i} must be string, got {type(lu)}")
        if lu not in lookup:
            valid = sorted(lookup.keys())
            raise ValueError(f"Unknown landuse '{lu}' at index {i}. Valid: {valid}")
        
        lu_params = lookup[lu]
        
        # Apply custom overrides if provided
        if custom_overrides and lu in custom_overrides:
            overrides = custom_overrides[lu]
            # Apply parameter overrides recursively
            lu_params = _apply_overrides(lu_params, overrides)
        
        hydro_p = HydroParams(snow=snow_params, landuse=lu_params)
        per_cell.append(hydro_p)
    
    return build_cell_params(per_cell, landuse_types)

def _apply_overrides(params, overrides: Dict[str, Any]):
    """Apply parameter overrides recursively"""
    if hasattr(params, '_replace'):  # NamedTuple
        updates = {}
        for key, value in overrides.items():
            if hasattr(params, key):
                current_val = getattr(params, key)
                if isinstance(value, dict) and hasattr(current_val, '_replace'):
                    updates[key] = _apply_overrides(current_val, value)
                else:
                    updates[key] = value
        return params._replace(**updates)
    else:
        return params