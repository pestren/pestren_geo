# pestren-geo

Raster measurement with its provenance attached.

A small Python library for turning aerial and satellite imagery into measurements
that survive being argued with. It knows about grids, terrain, alignment and
change. It knows nothing about counties, deeds or owners.

```python
from pestren_geo import GridSpec, mask_area

grid = GridSpec.from_bounds(bounds_in_metres, gsd=0.6, crs="EPSG:5070")
acres = mask_area(farmable, grid)

print(acres)          # 38.42 acre @0.6m (pixel_count)
print(acres.to_dict())
# {'value': 38.42, 'unit': 'acre', 'method': 'pixel_count', 'gsd_m': 0.6,
#  'support_scale_m': 0.6, 'crs': 'EPSG:5070', 'computed_at': '...'}
```

## Why it exists

Most wrong numbers in raster geospatial work are silent. Nothing raises, the
result looks plausible, and it is wrong by an amount nobody can reconstruct
afterwards. Four causes account for most of it:

- **Area computed in degrees.** A degree is not a metre and its ground length
  changes with latitude.
- **Two arrays that were never on the same grid.** They broadcast fine.
- **A resolution that was asked for rather than observed.** Elevation services
  will resample a 10 m survey to a 1 m request and return it without complaint,
  producing a smooth, confident, entirely fictional slope map.
- **A number with no record of how it was made**, compared later against another
  number made differently.

This library makes each of those an error rather than a possibility.

## The four rules, enforced in code

**1. Grids are metric and snapped.** `GridSpec` refuses geographic CRSs outright.
Origins snap to a multiple of the ground sample distance, so the same ground
requested twice lands on the same lattice and returns the same area.

**2. Support scale travels with every measurement.** The resolution of the
*source*, which may be coarser than the grid. A slope on a 1 m grid and a slope
on a 10 m grid are different physical quantities, not two estimates of one.

**3. The library refuses rather than caveats.** Where the support scale makes a
measurement meaningless, you get an absence with a reason, not a number with a
footnote. A caveat nobody reads is worse than a gap.

```python
volume = depression_volume(filled, original, grid)   # grid is 1 m, source is 10 m
volume.is_present        # False
volume.absent_because    # 'grid is 1 m but the source is 10 m. Depression volume
                         #  on an upsampled surface measures the interpolator…'
```

**4. Alignment is proven before change is claimed.** 2 px at 0.6 m is 1.2 m on
the ground — 40% of a 3 m drainage ditch. `coregister` returns a refusal above
its residual gate, and `warp_onto` will not accept one.

## Install

```bash
pip install pestren-geo[opencv,io]
```

Extras are the backends. `opencv` brings OpenCV 5 for classical raster ops and
the torch-free feature-matching path; `kornia` brings the reference PyTorch
matcher; `io` brings rasterio, shapely and pyproj for reading and reprojection.
The core depends only on numpy.

## Using it from an agent

The library is the measuring half. If you are building an agent that answers
questions about land, the shortest path is in [`docs/AGENTS.md`](docs/AGENTS.md):
measure everything up front into a scene graph, keep it in your agent's state,
and give the model a table to read rather than an image to search.

```python
scene = build_scene(parcel_id)        # once per parcel, cached on its dates
farmable = mask_area(cropped, grid)   # arithmetic — no model in the path
select("the trees on the top left", scene["blobs"])   # reads the table
```

## Status

Early, and built in the open. `grid`, `measure`, `provenance` and `align` are
implemented and tested. `terrain` (slope, curvature, depression filling,
stage–area–volume) and `change` (differencing, cleanup, polygonisation) are
next — see `docs/ROADMAP.md`.

## Feature matching backends

Learned matching is the right tool for farmland, which is low-texture with
strongly repeating rows and large seasonal appearance change — the regime
classical detectors are weakest in. Two implementations, and the choice is a
deployment decision rather than a quality one:

| backend | needs | notes |
|---|---|---|
| `opencv` | OpenCV 5, ~120 MB | DISK + LightGlue as ONNX, no deep-learning framework. Verified at 0.2449 px median residual on a synthetic control. |
| `kornia` | torch | The reference implementation. Better maintained, free of the ONNX export quirks. |

If you use the `opencv` backend, three non-obvious things each return *zero
matches* silently, so they are worth knowing:

1. The matcher must be the `.trt.onnx` export. The `_cpu` and `_fused` builds
   embed a `com.microsoft` operator OpenCV's ONNX parser rejects.
2. `score_threshold` must be passed explicitly as `0.1`. Omitting it, or passing
   a *lower* value, returns nothing.
3. Images must be grayscale `uint8`. Three-channel BGR returns nothing.

Weights are not bundled; `docs/BACKENDS.md` has the download commands.

## What this library will never do

It will not tell you where a boundary is. It reports that two observations
differ, by how much, and where — establishing a boundary is the licensed
practice of land surveying, and a measurement from imagery is not a survey.
That constraint is also the more useful product: a difference localised to one
edge tells a surveyor exactly where to look.

## Documentation

| | |
|---|---|
| [**Architecture**](docs/ARCHITECTURE.md) | Why the library has this shape, and the one dependency rule that keeps it publishable |
| [**Research**](docs/RESEARCH.md) | The three papers behind the design — what each measured, which decision it drove, and what we deliberately left |
| [**Classes**](docs/CLASSES.md) | What the segmentation model labels, why only six, and which requested classes belong to a registry rather than to imagery |
| [**Agents**](docs/AGENTS.md) | Wiring this into LangGraph or any tool-calling agent, with the scene-graph pattern |
| [**OpenCV 5**](docs/OPENCV5.md) | An honest audit — what is genuinely new, what is a speedup, and six API quirks that silently return nothing |
| [**Backends**](docs/BACKENDS.md) | Feature-matching backends and their weights |
| [**Roadmap**](docs/ROADMAP.md) | What is next, and what is deliberately out of scope |

### The one number to take away

If you read nothing else: the same models score **4.67** when asked to locate
something in an image, and **~90** when asked to reason about positions given in
text. Measure positions before the question is asked, and hand over a table.
[Where that comes from →](docs/RESEARCH.md)

## Licence

Apache-2.0.

Imagery and elevation sources carry their own terms — JRC Global Surface Water
requires attribution, and NAIP is public domain. Attribute what you use.
