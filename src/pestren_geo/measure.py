"""Area, length and volume. The only place in this library a number is made.

WHY MEASUREMENT IS ONE MODULE AND NOT A METHOD ON EVERYTHING.

Concentrating it means there is one place to audit, one place the equal-area
check lives, and one place that knows a mask's area is a pixel count rather than
the area of a polygon traced around it. Scattered across modules, each of those
decisions gets made again slightly differently.

THE THREE RULES THIS MODULE ENFORCES.

1. Never in degrees. `GridSpec` already refuses geographic CRSs; `polygon_area`
   re-checks because polygons arrive from outside.

2. A mask's area is `count * pixel_area`, never the area of its contour. A
   rasterised shape's traced outline carries a half-pixel boundary bias that
   grows with perimeter, so a long thin ditch is biased far more than a square
   field of the same area. Count pixels to measure; trace contours to export
   geometry; never cross the two.

3. Volume needs a datum and will not be guessed at. Water volume from imagery is
   not possible — lidar's near-infrared return maps the surface, not the bed —
   so `depression_volume` works from terrain and says so.
"""

from __future__ import annotations

import numpy as np

from .grid import SQUARE_METRES_PER_ACRE, GridSpec, _is_geographic
from .provenance import Measurement, Source


def mask_area(
    mask: np.ndarray,
    grid: GridSpec,
    *,
    unit: str = "acre",
    sources: tuple[Source, ...] = (),
) -> Measurement:
    """Area of the True pixels of `mask`, by counting them.

    `mask` must be boolean and shaped like the grid. Counting is exact with
    respect to the mask — the uncertainty that matters is in the mask itself,
    not in the arithmetic, which is why no uncertainty is invented here.
    """
    _check_shape(mask, grid, "mask")
    if mask.dtype != np.bool_:
        raise TypeError(
            f"mask must be boolean, got {mask.dtype}. An integer mask hides the "
            "question of what counts as inside; make the decision explicit first."
        )

    m2 = float(np.count_nonzero(mask)) * grid.pixel_area
    return Measurement(
        value=_convert_area(m2, unit),
        unit=unit,
        method="pixel_count",
        gsd_m=grid.gsd,
        support_scale_m=grid.support_scale,
        crs=grid.crs,
        sources=sources,
    )


def polygon_area(
    geometry,
    crs: str,
    *,
    unit: str = "acre",
    sources: tuple[Source, ...] = (),
) -> Measurement:
    """Area of a shapely polygon already in a projected CRS.

    This does not reproject. Reprojection is the caller's decision because the
    right target depends on what the number is for — an equal-area CRS for area,
    a local UTM zone for distance — and silently picking one hides that choice.
    """
    if _is_geographic(crs):
        raise ValueError(
            f"{crs} is geographic; its 'area' is in square degrees, which is not "
            "an area. Reproject to an equal-area CRS (EPSG:5070 for CONUS) first."
        )
    return Measurement(
        value=_convert_area(float(geometry.area), unit),
        unit=unit,
        method="shapely_polygon_area",
        crs=crs,
        sources=sources,
    )


def depression_volume(
    filled: np.ndarray,
    original: np.ndarray,
    grid: GridSpec,
    *,
    unit: str = "m3",
    sources: tuple[Source, ...] = (),
) -> Measurement:
    """Storage capacity of the depressions between a filled and raw surface.

    `filled` comes from a depression-filling pass (see `terrain.fill_depressions`);
    the difference is depth, and depth integrated over area is the volume the
    terrain could hold before it spills.

    This is capacity to the spill point, which is a property of the ground. It is
    NOT the volume of water currently present — that needs a water-surface
    elevation, which imagery cannot supply because near-infrared does not
    penetrate water. Use `terrain.stage_area_volume` with an observed stage.
    """
    _check_shape(filled, grid, "filled")
    _check_shape(original, grid, "original")

    if grid.is_oversampled:
        return Measurement.absent(
            unit, "depression_volume",
            because=(
                f"grid is {grid.gsd:g} m but the source is {grid.support_scale:g} m. "
                "Depression volume on an upsampled surface measures the "
                "interpolator, not the ground."
            ),
            gsd_m=grid.gsd, support_scale_m=grid.support_scale, crs=grid.crs,
        )

    depth = np.asarray(filled, float) - np.asarray(original, float)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)
    m3 = float(depth.sum()) * grid.pixel_area

    return Measurement(
        value=_convert_volume(m3, unit),
        unit=unit,
        method="depression_fill_integration",
        gsd_m=grid.gsd,
        support_scale_m=grid.support_scale,
        crs=grid.crs,
        uncertainty=_volume_uncertainty(depth, grid, unit),
        sources=sources,
    )


def _volume_uncertainty(depth: np.ndarray, grid: GridSpec, unit: str) -> float | None:
    """Vertical error dominates volume; horizontal error barely registers.

    A wetted area times the DEM's vertical RMSE is the honest first-order bar.
    Without a stated RMSE there is nothing to report, and inventing one would be
    worse than leaving it out.
    """
    rmse = getattr(grid, "vertical_rmse_m", None)
    if not rmse:
        return None
    wet_m2 = float(np.count_nonzero(depth > 0)) * grid.pixel_area
    return _convert_volume(wet_m2 * rmse, unit)


# ----------------------------------------------------------------- conversions

_AREA = {
    "m2": 1.0,
    "ha": 1e-4,
    "acre": 1.0 / SQUARE_METRES_PER_ACRE,
    "km2": 1e-6,
}
_VOLUME = {
    "m3": 1.0,
    "acre_ft": 1.0 / 1233.4818375475,
    "litre": 1000.0,
}


def _convert_area(m2: float, unit: str) -> float:
    try:
        return m2 * _AREA[unit]
    except KeyError:
        raise ValueError(f"unknown area unit {unit!r}; have {sorted(_AREA)}") from None


def _convert_volume(m3: float, unit: str) -> float:
    try:
        return m3 * _VOLUME[unit]
    except KeyError:
        raise ValueError(f"unknown volume unit {unit!r}; have {sorted(_VOLUME)}") from None


def _check_shape(arr: np.ndarray, grid: GridSpec, name: str) -> None:
    if arr.shape != grid.shape:
        raise ValueError(
            f"{name} is {arr.shape} but the grid is {grid.shape}. Two arrays on "
            "different grids is the single most common cause of a wrong answer "
            "here, so it is an error rather than a broadcast."
        )
