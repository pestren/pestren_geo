"""The grid every measurement in this library is made on.

WHY THIS TYPE EXISTS AT ALL.

Almost every wrong number in raster geospatial work comes from two arrays that
were never on the same grid, or from an area computed in degrees. Both are silent:
nothing raises, the result looks plausible, and it is wrong by an amount nobody
can reconstruct later.

So this library has exactly one rule, and `GridSpec` is how it is enforced: every
raster in a computation lands on one grid, in a metric CRS, with a snapped origin.
Functions take arrays plus the grid they belong to. There is no code path that
accepts a bare array and guesses.

WHY THE ORIGIN IS SNAPPED.

An origin derived from an arbitrary bounding box means the same ground requested
twice lands on two different pixel lattices, and the two runs return different
areas. The difference is small, irreproducible, and will be found by a user rather
than a test. Snapping the origin to a multiple of the ground sample distance makes
the lattice a property of the CRS and the GSD, not of the request.

The guarantee this buys is stronger than reproducibility: every snapped origin
lies on one global lattice, so any two grids at the same GSD differ by a WHOLE
number of pixels. Two overlapping reads can therefore be compared by integer
offset, with no resampling of either — and resampling is where sub-pixel error
enters a comparison that is supposed to be measuring change.

WHY support_scale IS CARRIED, NOT INFERRED.

A slope computed on a 1 m grid and a slope computed on a 10 m grid are different
physical quantities, not different estimates of one quantity. Worse, services will
happily resample a 10 m survey to a 1 m request and return it without complaint,
which produces a confident and entirely fictional slope map. `support_scale` is
the resolution of the SOURCE, which may be coarser than `gsd`; it travels with
every measurement so a reader can tell what was actually observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterator

#: Area computations default here. CONUS Albers is equal-area, so a pixel is the
#: same number of square metres everywhere in the grid — which is what makes
#: `count * pixel_area` a legitimate area rather than an approximation.
DEFAULT_AREA_CRS = "EPSG:5070"

SQUARE_METRES_PER_ACRE = 4046.8564224


@dataclass(frozen=True)
class GridSpec:
    """A north-up raster grid in a projected CRS.

    `transform` is the affine in rasterio's convention
    ``(a, b, c, d, e, f)`` → ``x = a*col + b*row + c``, ``y = d*col + e*row + f``.
    Only north-up, axis-aligned grids are representable, which is deliberate:
    rotated grids are a source of confusion that buys nothing for parcel work.
    """

    crs: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    gsd: float
    support_scale: float | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"degenerate grid: {self.width}x{self.height}")
        if self.gsd <= 0:
            raise ValueError(f"gsd must be positive, got {self.gsd}")
        if _is_geographic(self.crs):
            raise ValueError(
                f"{self.crs} is a geographic CRS. Grids must be metric — a degree "
                "is not a metre and its ground length changes with latitude. Use a "
                f"projected CRS such as {DEFAULT_AREA_CRS} or a local UTM zone."
            )

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[float, float, float, float],
        gsd: float,
        crs: str = DEFAULT_AREA_CRS,
        *,
        support_scale: float | None = None,
        tile: int | None = None,
        overlap: int = 0,
    ) -> "GridSpec":
        """Build a grid covering `bounds` (minx, miny, maxx, maxy) in `crs`.

        The origin is snapped outward to a multiple of `gsd`, so the lattice is
        reproducible. When `tile` is given the extent is padded up to a whole
        number of tile strides, so tiling covers the grid exactly with no partial
        tile at the edge to special-case later.
        """
        minx, miny, maxx, maxy = bounds
        if not (maxx > minx and maxy > miny):
            raise ValueError(f"bounds are not increasing: {bounds}")

        x0 = math.floor(minx / gsd) * gsd
        y1 = math.ceil(maxy / gsd) * gsd
        width = math.ceil((maxx - x0) / gsd)
        height = math.ceil((y1 - miny) / gsd)

        if tile:
            stride = tile - overlap
            if stride <= 0:
                raise ValueError(f"overlap {overlap} must be smaller than tile {tile}")
            width = _round_up_to_stride(width, tile, stride)
            height = _round_up_to_stride(height, tile, stride)

        return cls(
            crs=crs,
            transform=(gsd, 0.0, x0, 0.0, -gsd, y1),
            width=width,
            height=height,
            gsd=gsd,
            support_scale=support_scale,
        )

    # ---------------------------------------------------------------- geometry

    @property
    def pixel_area(self) -> float:
        """Square metres per pixel — constant across the grid, by construction."""
        return self.gsd * self.gsd

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        a, _, c, _, e, f = self.transform
        return (c, f + e * self.height, c + a * self.width, f)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def xy(self, row: float, col: float) -> tuple[float, float]:
        """Projected coordinate of a pixel's centre."""
        a, b, c, d, e, f = self.transform
        return (a * (col + 0.5) + b * (row + 0.5) + c,
                d * (col + 0.5) + e * (row + 0.5) + f)

    # ----------------------------------------------------------------- tiling

    def tiles(self, size: int = 512, overlap: int = 64) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(row_off, col_off, height, width)`` windows covering the grid.

        Overlap exists so a model's edge effects can be cropped before stitching.
        512 with 64 of overlap is the default because it matches the chip size
        most aerial-pretrained encoders expect, and because at 0.6 m it is a
        307 m footprint — about the width of a quarter-quarter section.
        """
        stride = size - overlap
        if stride <= 0:
            raise ValueError(f"overlap {overlap} must be smaller than tile {size}")
        for row in range(0, max(self.height - overlap, 1), stride):
            for col in range(0, max(self.width - overlap, 1), stride):
                yield (row, col, min(size, self.height - row), min(size, self.width - col))

    # ------------------------------------------------------------- provenance

    def at_support(self, support_scale: float) -> "GridSpec":
        """Same grid, re-labelled with the resolution actually observed.

        Call this after reading a source that turned out to be coarser than the
        grid. It does not resample anything — it records the truth so downstream
        measurements can refuse to over-claim.
        """
        return replace(self, support_scale=support_scale)

    @property
    def is_oversampled(self) -> bool:
        """True when the grid is finer than the data behind it.

        Measurements that depend on local shape — slope, curvature, micro-relief —
        are not meaningful here, and `pestren_geo.terrain` refuses them rather
        than returning a smooth confident fiction.
        """
        return self.support_scale is not None and self.support_scale > self.gsd * 1.5

    def describe(self) -> dict:
        """The provenance block to attach to anything measured on this grid."""
        return {
            "crs": self.crs,
            "gsd_m": self.gsd,
            "support_scale_m": self.support_scale,
            "oversampled": self.is_oversampled,
            "shape": list(self.shape),
            "pixel_area_m2": self.pixel_area,
        }


def _round_up_to_stride(n: int, tile: int, stride: int) -> int:
    if n <= tile:
        return tile
    return tile + math.ceil((n - tile) / stride) * stride


def _is_geographic(crs: str) -> bool:
    """Catch the common geographic CRS codes without requiring pyproj.

    Deliberately conservative: it is better to let an exotic projected CRS through
    than to depend on pyproj for a guard that exists to catch EPSG:4326 typed by
    reflex. `measure` re-checks with pyproj when it is available.
    """
    known_geographic = {"EPSG:4326", "EPSG:4269", "EPSG:4267", "OGC:CRS84", "CRS84"}
    return crs.strip().upper().replace("_", ":") in known_geographic
