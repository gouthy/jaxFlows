from __future__ import annotations
from typing import NamedTuple, Sequence, Tuple, Dict, Any, Optional, List

try:
    import jax
    import jax.numpy as jnp
    from jax.tree_util import tree_map
except Exception:
    jax = None
    jnp = None
    tree_map = None

def _require_jax():
    if jnp is None:
        raise ImportError("JAX is required. Please install jax.")

class HydroState(NamedTuple):
    s_snow: jnp.ndarray
    s_snow_liq: jnp.ndarray
    s_skin: jnp.ndarray
    s_canopy: jnp.ndarray
    s_soil: jnp.ndarray
    s_surface: jnp.ndarray
    s_ground: jnp.ndarray
    s_glacier: jnp.ndarray
    s_active_layer: jnp.ndarray

def _ensure_scalar_state(state: HydroState) -> HydroState:
    """Ensure all HydroState fields are JAX scalars with shape ()"""
    return HydroState(
        s_snow=jnp.squeeze(jnp.asarray(state.s_snow)),
        s_snow_liq=jnp.squeeze(jnp.asarray(state.s_snow_liq)),
        s_skin=jnp.squeeze(jnp.asarray(state.s_skin)),
        s_canopy=jnp.squeeze(jnp.asarray(state.s_canopy)),
        s_soil=jnp.squeeze(jnp.asarray(state.s_soil)),
        s_surface=jnp.squeeze(jnp.asarray(state.s_surface)),
        s_ground=jnp.squeeze(jnp.asarray(state.s_ground)),
        s_glacier=jnp.squeeze(jnp.asarray(state.s_glacier)),
        s_active_layer=jnp.squeeze(jnp.asarray(state.s_active_layer))
    )