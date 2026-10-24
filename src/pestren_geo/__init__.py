"""Pestren Geo — raster measurement with its provenance attached.

A small library for turning aerial and satellite rasters into measurements that
survive being argued with. It knows about grids, terrain and change. It knows
nothing about counties, deeds or owners — that boundary is deliberate and is
described in docs/ARCHITECTURE.md.
"""

from .grid import GridSpec, DEFAULT_AREA_CRS, SQUARE_METRES_PER_ACRE
from .provenance import Measurement, Delta, Source
from .measure import mask_area, polygon_area, depression_volume
from .align import coregister, warp_onto, Alignment

__version__ = "0.1.0"
__all__ = [
    "GridSpec", "DEFAULT_AREA_CRS", "SQUARE_METRES_PER_ACRE",
    "Measurement", "Delta", "Source",
    "mask_area", "polygon_area", "depression_volume",
    "coregister", "warp_onto", "Alignment",
]
