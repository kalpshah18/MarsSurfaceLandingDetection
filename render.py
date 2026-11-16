"""Physical image formation for the synthetic dataset (shared by the generator and sweeps).

Image model
-----------
* Lambertian shading with unit albedo: I = max(0, n . l), where n is the DTM surface
  normal and l the unit vector toward the sun.
* Cast shadows from a height-field ray march toward the sun: a pixel is in shadow if
  any terrain point along the ray rises above the ray. Shadowed pixels get I = 0
  (no atmospheric skylight is modelled).
* No other darkening. In particular there is no depth-based ("painted") shadow.

Sun direction convention (image coordinates, x to the right, y down):
the azimuth is measured from the +x axis toward +y, so azimuth 45 deg puts the sun
toward the lower-right corner of the image. Surfaces facing the lower right are
lit; shadows fall toward the upper left of the obstacle that casts them.
"""
import cv2
import numpy as np


def sun_vector(azimuth_deg, elevation_deg):
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)


def lambertian(z, azimuth_deg, elevation_deg, gsd=1.0):
    """Unit-albedo Lambertian intensity in [0, 1] (no shadows)."""
    dz_dx = np.gradient(z, axis=1) / gsd
    dz_dy = np.gradient(z, axis=0) / gsd
    lx, ly, lz = sun_vector(azimuth_deg, elevation_deg)
    norm = np.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1.0)
    return np.clip((-dz_dx * lx - dz_dy * ly + lz) / norm, 0.0, 1.0)


def cast_shadow(z, azimuth_deg, elevation_deg, gsd=1.0, step_px=0.5):
    """Boolean cast-shadow mask by ray marching toward the sun.

    Heights along the ray are sampled bilinearly every `step_px` pixels, so smooth
    bowls give smooth shadow edges. Points outside the tile cannot occlude.
    """
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    tan_el = np.tan(el)
    zf = np.ascontiguousarray(z, dtype=np.float32)
    H, W = zf.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    relief = float(zf.max() - zf.min())
    n_steps = int(np.ceil(relief / (tan_el * gsd * step_px))) + 1
    dx, dy = np.cos(az) * step_px, np.sin(az) * step_px
    shadow = np.zeros((H, W), dtype=bool)
    for k in range(1, n_steps + 1):
        zs = cv2.remap(zf, xs + k * dx, ys + k * dy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=-1e9)
        shadow |= zs > zf + k * step_px * gsd * tan_el
    return shadow


def render(z, azimuth_deg=45.0, elevation_deg=30.0, gsd=1.0):
    """Return (uint8 image, boolean cast-shadow mask)."""
    intensity = lambertian(z, azimuth_deg, elevation_deg, gsd)
    shadow = cast_shadow(z, azimuth_deg, elevation_deg, gsd)
    intensity[shadow] = 0.0
    return (intensity * 255).astype(np.uint8), shadow
