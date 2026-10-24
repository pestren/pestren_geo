# Roadmap

Built in the open, in the order that each piece becomes load-bearing.

## Shipped — 0.1

- **`grid`** — `GridSpec`. Refuses geographic CRSs, snaps origins so every grid at
  a given GSD shares one global lattice, carries `support_scale`, tiles.
- **`provenance`** — `Measurement`, `Delta`, `Source`. Every number travels with
  its method, grid, source resolution and time.
- **`measure`** — `mask_area`, `polygon_area`, `depression_volume`. Pixel counting
  rather than contour area, and refusal on an oversampled grid.
- **`align`** — `coregister`, `warp_onto`. Learned matching behind a backend
  choice, with a residual gate that refuses rather than warping on a bad fit.

## Next — 0.2, terrain

Slope and aspect by Horn's method on true per-axis ground distance; profile and
plan curvature; detrending; Planchon–Darboux depression filling; and the
stage–area–volume curve that turns a filled depression into a storage capacity.

The constraint that shapes this module: a slope on a 1 m grid and a slope on a
10 m grid are different physical quantities. Every function refuses where
`support_scale` makes the answer meaningless, rather than returning a smooth
confident fiction.

## Then — 0.3, change

Differencing two aligned observations; morphological cleanup with a kernel bounded
by the narrowest real feature; connected components; polygonisation. The output is
a set of changed regions with areas and both acquisition dates — a dated delta,
never an assertion that either observation was wrong.

## Later

- **`read`** — windowed COG reads that land directly on a `GridSpec`, with the
  resampling rules enforced: bilinear for continuous, nearest for categorical,
  and never cubic on anything feeding a depression fill.
- **`segment`** — a frozen-trunk, fixed-class mask head. Deliberately last. The
  measurement discipline has to be solid before anything learned is layered on it,
  and a classifier is only worth shipping once it beats a random forest on
  spatially blocked error.

## Not planned

Anything that reads a land record, knows what a county is, or asserts where a
boundary lies. That is a different problem with different failure modes, and
mixing the two is how a measurement becomes a legal claim by accident.
