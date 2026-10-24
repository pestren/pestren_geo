"""The invariants are the product. These tests are where they live."""

import numpy as np
import pytest

from pestren_geo import GridSpec, mask_area, depression_volume
from pestren_geo.grid import DEFAULT_AREA_CRS

BOUNDS = (-2_000_000.0, 2_000_000.0, -1_999_600.0, 2_000_400.0)  # 400 m square


def grid(gsd=0.6, **kw):
    return GridSpec.from_bounds(BOUNDS, gsd, **kw)


class TestGeographicCrsIsRefused:
    """Area in degrees is the most common silent error in this field."""

    @pytest.mark.parametrize("crs", ["EPSG:4326", "epsg:4326", "OGC:CRS84", "EPSG:4269"])
    def test_constructor_refuses(self, crs):
        with pytest.raises(ValueError, match="geographic"):
            GridSpec.from_bounds((-121.9, 38.6, -121.8, 38.7), 0.6, crs)

    def test_projected_is_accepted(self):
        assert grid().crs == DEFAULT_AREA_CRS


class TestOriginSnapping:
    """The same ground twice must produce the same lattice, or areas drift."""

    def test_origin_lands_on_a_multiple_of_gsd(self):
        g = grid(gsd=0.6)
        assert g.transform[2] % 0.6 == pytest.approx(0, abs=1e-9)

    def test_identical_bounds_give_an_identical_lattice(self):
        assert GridSpec.from_bounds(BOUNDS, 0.6) == GridSpec.from_bounds(BOUNDS, 0.6)

    @pytest.mark.parametrize("shift", [0.3, 1.7, -4.2, 100.0])
    def test_two_grids_always_differ_by_a_whole_number_of_pixels(self, shift):
        """The guarantee that matters: overlapping grids align by integer offset.

        Snapping does not force two different extents to share an origin — it
        puts every origin on one global lattice, so any two of them differ by a
        whole pixel. That is what lets two overlapping reads be compared without
        resampling either of them, which is where sub-pixel error creeps in.
        """
        a = GridSpec.from_bounds(BOUNDS, 0.6)
        b = GridSpec.from_bounds((BOUNDS[0] + shift, BOUNDS[1], BOUNDS[2], BOUNDS[3]), 0.6)
        offset_px = (a.transform[2] - b.transform[2]) / 0.6
        assert offset_px == pytest.approx(round(offset_px), abs=1e-9)


class TestPixelAreaIsConstant:
    def test_area_of_a_full_mask_matches_the_extent(self):
        g = grid(gsd=0.6)
        full = np.ones(g.shape, bool)
        m = mask_area(full, g, unit="m2")
        assert m.value == pytest.approx(g.width * g.height * 0.36)

    def test_acre_conversion(self):
        g = grid(gsd=1.0)
        mask = np.zeros(g.shape, bool)
        mask[:100, :100] = True          # 10,000 m²
        assert mask_area(mask, g).value == pytest.approx(10_000 / 4046.8564224)

    def test_integer_mask_is_refused(self):
        g = grid()
        with pytest.raises(TypeError, match="boolean"):
            mask_area(np.ones(g.shape, np.uint8), g)

    def test_wrong_shape_is_an_error_not_a_broadcast(self):
        g = grid()
        with pytest.raises(ValueError, match="different grids"):
            mask_area(np.ones((4, 4), bool), g)


class TestSupportScaleGovernsWhatMayBeMeasured:
    """A 10 m survey resampled to 1 m is not a 1 m observation."""

    def test_oversampled_grid_is_flagged(self):
        assert grid(gsd=1.0, support_scale=10.0).is_oversampled
        assert not grid(gsd=1.0, support_scale=1.0).is_oversampled

    def test_volume_is_refused_on_an_oversampled_grid(self):
        g = grid(gsd=1.0, support_scale=10.0)
        flat = np.zeros(g.shape)
        m = depression_volume(flat + 1.0, flat, g)
        assert not m.is_present
        assert "upsampled" in m.absent_because

    def test_volume_is_produced_when_support_matches(self):
        g = grid(gsd=1.0, support_scale=1.0)
        original = np.zeros(g.shape)
        filled = np.zeros(g.shape)
        filled[:10, :10] = 2.0                      # 100 cells, 2 m deep
        m = depression_volume(filled, original, g, unit="m3")
        assert m.value == pytest.approx(200.0)
        assert m.support_scale_m == 1.0


class TestTiling:
    def test_tiles_cover_the_grid(self):
        g = grid(gsd=0.6, tile=512, overlap=64)
        tiles = list(g.tiles(512, 64))
        assert tiles
        far_row = max(r + h for r, _, h, _ in tiles)
        far_col = max(c + w for _, c, _, w in tiles)
        assert far_row >= g.height and far_col >= g.width

    def test_overlap_must_be_smaller_than_the_tile(self):
        with pytest.raises(ValueError, match="overlap"):
            list(grid().tiles(256, 256))


class TestProvenanceTravels:
    def test_measurement_carries_the_grid_it_was_made_on(self):
        g = grid(gsd=0.6, support_scale=0.6)
        m = mask_area(np.ones(g.shape, bool), g)
        assert (m.gsd_m, m.support_scale_m, m.crs) == (0.6, 0.6, DEFAULT_AREA_CRS)
        assert m.method == "pixel_count"
        assert "computed_at" in m.to_dict()
