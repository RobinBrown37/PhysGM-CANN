"""Original PhysGM training label conventions (export E conversion is separate)."""
import math

box_packageMATERIALS = ("Wood", "Metal", "Plastic", "Glass", "Fabric", "Leather", "Ceramic", "Stone",
             "Rubber", "Paper", "Sand", "Snow", "Plasticine", "Foam")
E_MEAN, E_STD = 7.387210, 2.456477
NU_MEAN, NU_STD = 0.398, 0.111


def validate_physical(physical):
    if physical.get("material") not in MATERIALS:
        raise ValueError(f"Unknown material: {physical.get('material')!r}")
    E, nu = float(physical["E"]), float(physical["nu"])
    if not math.isfinite(E) or E <= 0 or not math.isfinite(nu):
        raise ValueError("Physical labels require finite E > 0 and finite nu")
    return MATERIALS.index(physical["material"]), E, nu


def normalized_physical(physical):
    material, E, nu = validate_physical(physical)
    return material, (math.log10(max(E, 1.0))-E_MEAN)/E_STD, (nu-NU_MEAN)/NU_STD
