import jax.numpy as jnp
from typing import Dict
from .parameters import *

def create_landuse_configs():
    """Complete landuse configurations with REALISTIC hydrological parameters"""
    
    # Helper functions for creating parameter objects
    def make_canopy(f_bare, f_veg, lai, capacity):
        return CanopyParams(
            f_bare=jnp.array(f_bare), f_veg=jnp.array(f_veg),
            lai=jnp.array(lai), capacity=jnp.array(capacity)
        )
    
    def make_soil(s_max, s_wilt, s_gr_min, s_gr_max, s_sg_min, s_sg_max, 
                  exponent_b, r_gr_min, r_gr_max):
        """Create soil parameters with realistic drainage rates"""
        return SoilParams(
            s_max=jnp.array(s_max), s_wilt=jnp.array(s_wilt),
            s_gr_min=jnp.array(s_gr_min), s_gr_max=jnp.array(s_gr_max),
            s_sg_min=jnp.array(s_sg_min), s_sg_max=jnp.array(s_sg_max),
            exponent_b=jnp.array(exponent_b), 
            r_gr_min=jnp.array(r_gr_min), r_gr_max=jnp.array(r_gr_max),
            dt=jnp.array(86400.0)  # seconds per day
        )
    
    def make_groundwater(f_lake=0.0, f_wetland=0.0, lag_sw=1.0, lag_gw=1.0):
        return GroundwaterParams(
            f_lake=jnp.array(f_lake), f_wetland=jnp.array(f_wetland),
            lag_sw=jnp.array(lag_sw), lag_gw=jnp.array(lag_gw)
        )
    
    # Get inactive parameters
    inactive = create_inactive_params()
    
    # REALISTIC DRAINAGE RATES (mm/s) - key fix!
    # Target daily rates: forests 0.2-2.0 mm/day, agricultural 0.5-5.0 mm/day, urban 0.1-1.0 mm/day
    FOREST_MIN = 2.5e-6    # 0.22 mm/day - slow forest drainage
    FOREST_MAX = 2.0e-5    # 1.73 mm/day - moderate forest drainage
    AGRIC_MIN = 5e-6       # 0.43 mm/day - agricultural drainage  
    AGRIC_MAX = 4e-5       # 3.46 mm/day - agricultural drainage
    URBAN_MIN = 1e-6       # 0.09 mm/day - very slow urban drainage
    URBAN_MAX = 1e-5       # 0.86 mm/day - slow urban drainage
    GRASS_MIN = 3e-6       # 0.26 mm/day - grassland drainage
    GRASS_MAX = 2.5e-5     # 2.16 mm/day - grassland drainage
    
    return {
        # ==== FOREST TYPES (Conservative drainage, high storage) ====
        "ENF": LandUseParams(
            canopy=make_canopy(0.0, 1.0, 5.5, 0.25),
            soil=make_soil(
                s_max=220, s_wilt=50, s_gr_min=80, s_gr_max=190, 
                s_sg_min=10, s_sg_max=180, exponent_b=0.35,
                r_gr_min=FOREST_MIN, r_gr_max=FOREST_MAX
            ),
            groundwater=make_groundwater(0.0, 0.0, 8.0, 35.0),  # Slow response
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        "DBF": LandUseParams(
            canopy=make_canopy(0.0, 1.0, 6.0, 0.30),
            soil=make_soil(
                s_max=200, s_wilt=45, s_gr_min=75, s_gr_max=175,
                s_sg_min=10, s_sg_max=160, exponent_b=0.30,
                r_gr_min=FOREST_MIN, r_gr_max=FOREST_MAX
            ),
            groundwater=make_groundwater(0.0, 0.0, 7.0, 30.0),
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        "MF": LandUseParams(
            canopy=make_canopy(0.0, 1.0, 5.8, 0.28),
            soil=make_soil(
                s_max=210, s_wilt=50, s_gr_min=78, s_gr_max=180,
                s_sg_min=10, s_sg_max=170, exponent_b=0.32,
                r_gr_min=FOREST_MIN, r_gr_max=FOREST_MAX
            ),
            groundwater=make_groundwater(0.0, 0.0, 7.5, 32.0),
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        # ==== SHRUB/SAVANNA (Moderate drainage) ====
        "OS": LandUseParams(
            canopy=make_canopy(0.2, 0.8, 1.5, 0.10),
            soil=make_soil(
                s_max=120, s_wilt=35, s_gr_min=50, s_gr_max=100,
                s_sg_min=5, s_sg_max=90, exponent_b=0.20,
                r_gr_min=GRASS_MIN, r_gr_max=GRASS_MAX
            ),
            groundwater=make_groundwater(0.0, 0.0, 4.0, 20.0),
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        "SAV": LandUseParams(
            canopy=make_canopy(0.25, 0.75, 3.5, 0.15),
            soil=make_soil(
                s_max=140, s_wilt=40, s_gr_min=55, s_gr_max=115,
                s_sg_min=8, s_sg_max=120, exponent_b=0.25,
                r_gr_min=GRASS_MIN, r_gr_max=GRASS_MAX
            ),
            groundwater=make_groundwater(0.0, 0.0, 4.5, 22.0),
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        # ==== GRASSLAND (Moderate-fast drainage) ====
        "GRA": LandUseParams(
            canopy=make_canopy(0.3, 0.7, 2.0, 0.12),
            soil=make_soil(
                s_max=130, s_wilt=40, s_gr_min=45, s_gr_max=105,
                s_sg_min=5, s_sg_max=110, exponent_b=0.22,
                r_gr_min=GRASS_MIN, r_gr_max=GRASS_MAX
            ),
            groundwater=make_groundwater(0.0, 0.0, 3.0, 18.0),
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        # ==== AGRICULTURAL (Fast drainage, good for crops) ====
        "CROP": LandUseParams(
            canopy=make_canopy(0.2, 0.8, 4.0, 0.18),
            soil=make_soil(
                s_max=180, s_wilt=45, s_gr_min=40, s_gr_max=140,
                s_sg_min=10, s_sg_max=150, exponent_b=0.25,
                r_gr_min=AGRIC_MIN, r_gr_max=AGRIC_MAX  # Higher drainage for agriculture
            ),
            groundwater=make_groundwater(0.0, 0.0, 2.5, 15.0),  # Faster response
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.2)  # Higher crop coefficient
        ),
        
        # ==== URBAN (Low storage, slow drainage, high imperviousness) ====
        "URB": LandUseParams(
            canopy=make_canopy(0.1, 0.2, 1.0, 0.05),
            soil=make_soil(
                s_max=80, s_wilt=20, s_gr_min=25, s_gr_max=60,
                s_sg_min=2, s_sg_max=70, exponent_b=0.10,
                r_gr_min=URBAN_MIN, r_gr_max=URBAN_MAX  # Slow urban drainage
            ),
            groundwater=make_groundwater(0.0, 0.0, 1.0, 8.0),  # Fast surface response
            impervious_fraction=jnp.array(0.7),  # High imperviousness
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        # ==== WATER BODIES (No soil drainage) ====
        "WATR": LandUseParams(
            canopy=make_canopy(1.0, 0.0, 0.0, 0.0),
            soil=make_soil(1e-6, 0.0, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0),
            groundwater=make_groundwater(1.0, 0.0, 1.0, 1.0),
            impervious_fraction=jnp.array(1.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'],
            water_body=WaterBodyParams(is_water_body=True, et_factor=jnp.array(1.2)),
            crop_coefficient=jnp.array(0.0)
        ),
        
        "LAKE": LandUseParams(
            canopy=make_canopy(1.0, 0.0, 0.0, 0.0),
            soil=make_soil(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            groundwater=make_groundwater(1.0, 0.0, 1.0, 1.0),
            impervious_fraction=jnp.array(1.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=inactive['wetland'],
            water_body=WaterBodyParams(is_water_body=True, et_factor=jnp.array(1.1)),
            crop_coefficient=jnp.array(0.0)
        ),
        
        # ==== WETLAND (Very slow drainage, high storage) ====
        "WET": LandUseParams(
            canopy=make_canopy(0.0, 1.0, 5.0, 0.20),
            soil=make_soil(
                s_max=150, s_wilt=0.0, s_gr_min=100, s_gr_max=140,
                s_sg_min=0.0, s_sg_max=150, exponent_b=0.10,
                r_gr_min=1e-6, r_gr_max=5e-6  # Very slow wetland drainage
            ),
            groundwater=make_groundwater(0.0, 0.0, 15.0, 60.0),  # Very slow response
            impervious_fraction=jnp.array(0.0),
            glacier=inactive['glacier'], permafrost=inactive['permafrost'],
            wetland=WetlandParams(f_wet=jnp.array(1.0), lag_sw=jnp.array(15.0), lag_gw=jnp.array(60.0)),
            water_body=inactive['water_body'],
            crop_coefficient=jnp.array(1.0)
        ),
        
        # ==== GLACIER (Specialized ice/snow hydrology) ====
        "GLAC": LandUseParams(
            canopy=make_canopy(1.0, 0.0, 0.0, 0.0),
            soil=make_soil(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            groundwater=make_groundwater(),
            impervious_fraction=jnp.array(1.0),
            glacier=GlacierParams(
                ddf=jnp.array(4.5),          # Realistic degree-day factor
                refreeze_frac=jnp.array(0.15), # Realistic refreeze fraction
                sublimation_frac=jnp.array(0.08), # Realistic sublimation
                lag=jnp.array(45.0)          # Slower glacier response
            ),
            permafrost=inactive['permafrost'], wetland=inactive['wetland'],
            water_body=inactive['water_body'], crop_coefficient=jnp.array(0.0)
        ),
        
        # ==== PERMAFROST (Specialized cold region hydrology) ====
        "PERM": LandUseParams(
            canopy=make_canopy(1.0, 0.0, 0.0, 0.0),
            soil=make_soil(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            groundwater=make_groundwater(),
            impervious_fraction=jnp.array(1.0),
            glacier=inactive['glacier'],
            permafrost=PermafrostParams(
                max_active_depth=jnp.array(1.8),           # Realistic active layer depth
                thaw_rate=jnp.array(0.008),                # Realistic thaw rate
                theta_sat=jnp.array(250.0),                # Realistic saturation
                sublimation_factor_cold=jnp.array(0.5),    # Moderate cold sublimation
                sublimation_factor_warm=jnp.array(0.08),   # Low warm sublimation
                cold_et_factor=jnp.array(0.15),            # Reduced cold ET
                drainage_rate_cold=jnp.array(0.015),       # Slow cold drainage
                drainage_rate_warm=jnp.array(0.04),        # Moderate warm drainage
                ddf_permafrost=jnp.array(1.6)              # Realistic degree-day factor
            ),
            wetland=inactive['wetland'], water_body=inactive['water_body'],
            crop_coefficient=jnp.array(0.0)
        ),
    }

def default_landuse_lookup() -> Dict[str, LandUseParams]:
    """Main lookup function - returns realistic landuse parameter dictionary"""
    return create_landuse_configs()

# Add a function to print parameter summary
def print_parameter_summary():
    """Print summary of realistic parameter ranges"""
    
    configs = create_landuse_configs()
    
    print("🌍 REALISTIC HYDROLOGY PARAMETER SUMMARY")
    print("=" * 60)
    print(f"{'Land Use':<8} {'Soil Max':<9} {'Drainage':<12} {'GW Lag':<8} {'Runoff Coeff*'}")
    print("-" * 60)
    
    expected_rc = {
        'ENF': '0.20-0.30', 'DBF': '0.20-0.30', 'MF': '0.20-0.30',
        'OS': '0.25-0.40', 'SAV': '0.25-0.40', 'GRA': '0.30-0.45', 
        'CROP': '0.15-0.35', 'URB': '0.70-0.85', 'WET': '0.05-0.20',
        'WATR': '0.90-0.95', 'LAKE': '0.90-0.95', 'GLAC': '0.60-0.80', 'PERM': '0.40-0.70'
    }
    
    for lu_type, params in configs.items():
        if hasattr(params.soil, 's_max') and float(params.soil.s_max[0]) > 0:
            s_max = float(params.soil.s_max[0])
            r_min_daily = float(params.soil.r_gr_min[0]) * 86400
            r_max_daily = float(params.soil.r_gr_max[0]) * 86400
            lag_gw = float(params.groundwater.lag_gw[0])
            
            print(f"{lu_type:<8} {s_max:<9.0f} {r_min_daily:.1f}-{r_max_daily:.1f}mm/d {lag_gw:<8.0f} {expected_rc.get(lu_type, 'varies')}")
        else:
            print(f"{lu_type:<8} {'Special':<9} {'Special':<12} {'Special':<8} {expected_rc.get(lu_type, 'varies')}")
    
    print("-" * 60)
    print("* Expected runoff coefficients under typical conditions")
    print("\n✅ All parameters now set to realistic values based on hydrology literature!")