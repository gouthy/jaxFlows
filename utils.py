import jax.numpy as jnp
from typing import Dict, List, Sequence
from .landuse import default_landuse_lookup

def create_landuse_encoding() -> Dict[str, int]:
    """Create encoding for landuse types"""
    types = list(default_landuse_lookup().keys())
    return {t: i for i, t in enumerate(types)}

def get_landuse_encoding() -> Dict[str, int]:
    """Get landuse encoding (cached)"""
    if not hasattr(get_landuse_encoding, '_cache'):
        get_landuse_encoding._cache = create_landuse_encoding()
    return get_landuse_encoding._cache

def encode_landuse_types(landuse_types: Sequence[str]) -> jnp.ndarray:
    """Encode landuse types to integers"""
    encoding = get_landuse_encoding()
    encoded = [encoding[l] for l in landuse_types]
    return jnp.array(encoded, dtype=jnp.int32)

def decode_landuse_types(codes: jnp.ndarray) -> List[str]:
    """Decode integer codes to landuse types"""
    encoding = get_landuse_encoding()
    decoding = {v: k for k, v in encoding.items()}
    return [decoding[int(c)] for c in codes]

def _validate_inputs(precip: jnp.ndarray, pet: jnp.ndarray, temp: jnp.ndarray) -> None:
    """Validate meteorological inputs"""
    # Shape validation
    if not (precip.shape == pet.shape == temp.shape):
        raise ValueError(f"Input shapes must match. Got precip: {precip.shape}, pet: {pet.shape}, temp: {temp.shape}")
    
    # Value validation
    if jnp.any(precip < 0):
        raise ValueError("Precipitation cannot be negative")
    if jnp.any(pet < 0):
        raise ValueError("Potential evapotranspiration cannot be negative")
    if jnp.any(temp < 173.15):  # -100°C
        raise ValueError("Temperature values seem unreasonably low (< -100°C)")
    if jnp.any(temp > 373.15):  # 100°C
        raise ValueError("Temperature values seem unreasonably high (> 100°C)")
    
    # NaN/Inf validation
    for name, arr in [("precip", precip), ("pet", pet), ("temp", temp)]:
        if jnp.any(jnp.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
        if jnp.any(jnp.isinf(arr)):
            raise ValueError(f"{name} contains infinite values")

def validate_cell_params(params, landuse_types: Sequence[str] = None) -> None:
    """Validate parameter consistency"""
    def get_first_dim(x):
        return getattr(x, "shape", (1,))[0] if hasattr(x, "shape") else 1
    
    n_cells = get_first_dim(params.snow.day_fraction)
    
    def check_shapes(obj, path=""):
        if hasattr(obj, "_fields"):  # NamedTuple
            for field in obj._fields:
                val = getattr(obj, field)
                check_shapes(val, f"{path}.{field}")
        elif hasattr(obj, "shape"):  # JAX array
            if obj.shape[0] != n_cells:
                raise ValueError(f"Parameter {path} has shape {obj.shape}, expected first dimension {n_cells}")
        elif isinstance(obj, (int, float, bool)):  # Scalar
            pass
        else:
            raise ValueError(f"Unexpected parameter type at {path}: {type(obj)}")
    
    check_shapes(params, "params")
    
    if landuse_types is not None:
        if len(landuse_types) != n_cells:
            raise ValueError(f"Number of landuse types ({len(landuse_types)}) doesn't match parameter arrays ({n_cells})")
    
    # Value range validation
    if jnp.any(params.snow.day_fraction < 0) or jnp.any(params.snow.day_fraction > 1):
        raise ValueError("Snow day_fraction must be between 0 and 1")
    if jnp.any(params.landuse.impervious_fraction < 0) or jnp.any(params.landuse.impervious_fraction > 1):
        raise ValueError("Impervious fraction must be between 0 and 1")