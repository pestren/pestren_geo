# Architecture

## Why this library has the shape it does

A parcel of land has two kinds of truth, and they change independently.

**Physical truth** is what is there. It changes by process — erosion, accretion,
construction, road widening, impoundment — continuously, and it leaves no entry
in any registry.

**Legal truth** is what is owned. It changes by instrument — deed, plat, easement,
condemnation, quiet title — discretely and with a date, and it leaves no trace in
any image.

Neither is derivable from the other. You cannot infer a quiet-title action from
imagery, and you cannot infer twenty years of bank erosion from a deed. A system
that observes one channel is blind to half the state space, and that is a
*measurement* gap rather than a quality gap: no model closes it, only a second
instrument does.

**This library is the physical channel, and nothing else.** It takes geometry and
rasters in and returns measurements. It does not know what a county is.

## The dependency rule

Nothing in `pestren_geo` may import a records layer, hold credentials, or reach a
registry. The direction is one-way: a records application may depend on this
library, and this library may never depend on one.

That is not a style preference. The boundary is what makes publishing possible —
an engine that measures ground is a reasonable thing to open-source, and a graph
of county record sources is not. If the line is ever crossed the library cannot
stay public, so it is enforced by dependency direction rather than by intention.

```
   ┌──────────────────────────────────┐
   │  records application   (private) │   deeds, parcels, authorities,
   │                                  │   reconciliation, reporting
   └───────────────┬──────────────────┘
                   │ depends on
                   ▼
   ┌──────────────────────────────────┐
   │  pestren_geo           (public)  │   grids, measurement, provenance,
   │                                  │   alignment, terrain, change
   └──────────────────────────────────┘
                   │ depends on
                   ▼
            numpy · rasterio · OpenCV · (torch, optional)
```

## The three invariants

Almost every wrong number in raster work is silent. Nothing raises, the output
looks plausible, and it is wrong by an amount nobody can reconstruct afterwards.
Four causes account for most of it, and each is made loud here.

**Grids are metric and snapped.** `GridSpec` refuses geographic CRSs, because an
area in square degrees is not an area. Origins snap to a multiple of the ground
sample distance, which does something stronger than making a request
reproducible: every grid at a given GSD lies on *one global lattice*, so any two
overlapping reads differ by a whole number of pixels. They can therefore be
compared by integer offset without resampling either — and resampling is exactly
where sub-pixel error enters a comparison that exists to measure change.

**Support scale travels with the number.** This is the resolution of the *source*,
which is frequently coarser than the resolution requested. Elevation services will
resample a 10 m survey to a 1 m request and return it without complaint; the
result is a smooth, confident, entirely fictional slope map. A slope measured on a
1 m grid and a slope measured on a 10 m grid are different physical quantities,
not two estimates of one, and publishing either as a bare float invites a
comparison that is not valid.

**Refusal beats caveat.** Where support scale makes a measurement meaningless, the
library returns an absence carrying a reason rather than a number carrying a
footnote. A caveat nobody reads is worse than a gap, because an absence is a
usable answer and a wrong number that looks right is not.

## Deployment shape

The library is pure computation with no server in it, so how it is deployed is the
caller's decision. One topology worth describing, because it follows from the
routing rather than from infrastructure preference:

```
   client
     │
     ▼
   web tier  (small, always on)
     │        arithmetic only — area, slope, curvature, ponding, volume
     │        no model is ever loaded here
     │
     │  async job, for the minority of questions that need a mask
     ▼
   inference tier  (GPU, bursty)
              torch, learned matching, segmentation
              results persisted; the web tier reads them, never computes them
```

Most questions about a parcel — how many acres, how steep, where water sits, how
much the pond holds — are arithmetic on a grid and never reach a model at all.
Sizing an always-on machine for the rare case means paying continuously for
capacity that is almost never used.

The interface matters more than the hardware: **batch results are persisted and
the web tier reads them.** Honour that and the inference tier can be a GPU
workstation, a cloud instance, or a laptop, and can change later without the
web tier noticing.

## What this library will never do

It will not tell you where a boundary is. It reports that two observations differ,
by how much, and where. Establishing a boundary is the licensed practice of land
surveying in every US state, and a measurement derived from imagery is not a
survey.

That constraint also happens to produce the more useful result. A difference
localised to one edge of a parcel tells a surveyor precisely where to look, which
is worth more than an assertion they would have to re-verify anyway.
