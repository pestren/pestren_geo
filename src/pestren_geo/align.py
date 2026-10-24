"""Aligning two observations of the same ground, taken at different times.

WHY THIS IS THE FIRST THING, NOT A PREPROCESSING DETAIL.

Every question of the form "what changed here" is really two questions, and the
second one is usually skipped: are the two rasters actually describing the same
ground, pixel for pixel? Aerial flights differ in year, sun angle, and
orthorectification solution. Residual misregistration of 2 px at 0.6 m is 1.2 m
on the ground. Against a 3 m drainage ditch that is 40% of the feature, so an
unaligned difference lights up every field edge in the scene and buries the real
change in false positives.

So alignment is not cleanup before the interesting part. It is the step that
decides whether the interesting part means anything, and this module's most
important behaviour is refusing to proceed when it has not succeeded.

ON BACKENDS.

Learned feature matching is the right tool here — classical detectors struggle on
farmland, which is low-texture with strongly repeating rows and large seasonal
appearance change. Two implementations exist and the choice is a deployment
decision rather than a quality one:

  "opencv" — OpenCV 5's Features module running DISK and LightGlue as ONNX.
             No deep-learning framework, around 120 MB installed. Verified at
             0.2449 px median residual on a synthetic control.
  "kornia" — the reference PyTorch implementation. Better maintained and free of
             the ONNX export quirks, at the cost of carrying torch.

Neither is the architecture. If a third appears that is better, it becomes a
third name in this dict and nothing else in the library changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Above this, two rasters are not aligned well enough to difference and the
#: honest result is a refusal. At 0.6 m ground sample distance, 0.5 px is 30 cm —
#: comfortably finer than the narrowest feature we claim to resolve.
DEFAULT_RESIDUAL_GATE_PX = 0.5


@dataclass(frozen=True)
class Alignment:
    """The transform taking image B onto image A's grid, and how well it fits."""

    homography: np.ndarray | None
    median_residual_px: float | None
    p95_residual_px: float | None
    matches: int
    inliers: int
    backend: str
    refused_because: str | None = None

    @property
    def ok(self) -> bool:
        return self.homography is not None and self.refused_because is None

    @property
    def shift_px(self) -> tuple[float, float] | None:
        if self.homography is None:
            return None
        return (float(self.homography[0, 2]), float(self.homography[1, 2]))

    def to_dict(self) -> dict:
        d = {
            "ok": self.ok,
            "backend": self.backend,
            "median_residual_px": self.median_residual_px,
            "p95_residual_px": self.p95_residual_px,
            "matches": self.matches,
            "inliers": self.inliers,
            "shift_px": self.shift_px,
        }
        if self.refused_because:
            d["refused_because"] = self.refused_because
        return {k: v for k, v in d.items() if v is not None}


def coregister(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    backend: str = "opencv",
    gate_px: float = DEFAULT_RESIDUAL_GATE_PX,
    max_keypoints: int = 2048,
    **backend_kw,
) -> Alignment:
    """Find the transform putting `image_b` onto `image_a`'s grid.

    Both images are single-channel uint8. For multi-year aerial imagery prefer
    the near-infrared band: vegetation reflectance in NIR is far more stable
    across seasons than visible colour, which is what defeats matching on RGB.

    Returns an `Alignment` whose `ok` is False when the fit does not clear
    `gate_px`. Callers must honour that — warping with a bad fit and reporting
    the difference anyway produces confident, plausible, wrong change.
    """
    if image_a.ndim != 2 or image_b.ndim != 2:
        raise ValueError(
            f"expected single-channel images, got {image_a.shape} and {image_b.shape}. "
            "Pick one band deliberately rather than averaging to grey."
        )

    try:
        match_fn = _BACKENDS[backend]
    except KeyError:
        raise ValueError(f"unknown backend {backend!r}; have {sorted(_BACKENDS)}") from None

    pts_a, pts_b = match_fn(image_a, image_b, max_keypoints, **backend_kw)

    if len(pts_a) < 4:
        return Alignment(None, None, None, len(pts_a), 0, backend,
                         refused_because=f"only {len(pts_a)} matches; 4 is the minimum")

    H, residuals, inliers = _fit(pts_a, pts_b)
    if H is None:
        return Alignment(None, None, None, len(pts_a), 0, backend,
                         refused_because="no transform converged on these matches")

    median = float(np.median(residuals))
    refused = None
    if median > gate_px:
        refused = (
            f"median residual {median:.2f} px exceeds the {gate_px} px gate. "
            "Differencing at this alignment would report misregistration as change."
        )

    return Alignment(
        homography=H,
        median_residual_px=median,
        p95_residual_px=float(np.percentile(residuals, 95)),
        matches=len(pts_a),
        inliers=int(inliers),
        backend=backend,
        refused_because=refused,
    )


def warp_onto(image_b: np.ndarray, alignment: Alignment, like: np.ndarray) -> np.ndarray:
    """Resample `image_b` onto the grid of `like`, using a passing alignment.

    Bilinear, never cubic. Cubic overshoots at breaklines, which invents local
    minima that a depression fill then dutifully finds and reports as ponds.
    """
    import cv2

    if not alignment.ok:
        raise ValueError(
            f"refusing to warp with a failed alignment: {alignment.refused_because}"
        )
    h, w = like.shape[:2]
    return cv2.warpPerspective(
        image_b, np.linalg.inv(alignment.homography), (w, h), flags=cv2.INTER_LINEAR
    )


# ------------------------------------------------------------------- backends

def _match_opencv(a, b, max_keypoints, *, disk_model="models/disk.onnx",
                  glue_model="models/disk_lightglue.trt.onnx", score_threshold=0.1):
    """OpenCV 5 Features: DISK + LightGlue as ONNX, no DL framework.

    Three non-obvious requirements, each of which silently returns zero matches:
    the matcher must be the `.trt.onnx` export (the `_cpu` and `_fused` builds
    embed a `com.microsoft` operator the DNN parser rejects); `score_threshold`
    must be passed explicitly as 0.1, and lower values return *fewer* matches,
    not more; and the images must be grayscale uint8, as BGR returns nothing.
    """
    import cv2

    disk = cv2.DISK.create(disk_model, max_keypoints)
    glue = cv2.LightGlueMatcher.create(glue_model, score_threshold)

    kp_a, desc_a = disk.detectAndCompute(a, None)
    kp_b, desc_b = disk.detectAndCompute(b, None)
    pts_a = np.float32([k.pt for k in kp_a])
    pts_b = np.float32([k.pt for k in kp_b])

    glue.setPairInfo(pts_a, pts_b, (a.shape[1], a.shape[0]), (b.shape[1], b.shape[0]))
    matches = glue.match(desc_a, desc_b)
    if not matches:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return (pts_a[[m.queryIdx for m in matches]],
            pts_b[[m.trainIdx for m in matches]])


def _match_kornia(a, b, max_keypoints, *, device="cpu"):
    """Kornia's LightGlueMatcher — the reference implementation, needs torch."""
    import kornia.feature as KF
    import torch

    to_t = lambda x: torch.from_numpy(x).float()[None, None] / 255.0
    ta, tb = to_t(a).to(device), to_t(b).to(device)

    disk = KF.DISK.from_pretrained("depth").to(device).eval()
    matcher = KF.LightGlueMatcher("disk").to(device).eval()

    with torch.inference_mode():
        fa, fb = disk(ta.repeat(1, 3, 1, 1), max_keypoints), disk(tb.repeat(1, 3, 1, 1), max_keypoints)
        ka, da = fa[0].keypoints, fa[0].descriptors
        kb, db = fb[0].keypoints, fb[0].descriptors
        _, idx = matcher(da, db, KF.laf_from_center_scale_ori(ka[None]),
                         KF.laf_from_center_scale_ori(kb[None]))
    idx = idx.cpu().numpy()
    return ka.cpu().numpy()[idx[:, 0]], kb.cpu().numpy()[idx[:, 1]]


_BACKENDS = {"opencv": _match_opencv, "kornia": _match_kornia}


def _fit(pts_a: np.ndarray, pts_b: np.ndarray):
    """Robust homography plus per-inlier reprojection error.

    MAGSAC rather than plain RANSAC: on farmland the repeating crop rows produce
    many self-consistent wrong correspondences, and MAGSAC is markedly better at
    not locking onto them.
    """
    import cv2

    src = pts_a.reshape(-1, 1, 2)
    dst = pts_b.reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
    if H is None:
        return None, None, 0
    keep = mask.ravel().astype(bool)
    residuals = np.linalg.norm(cv2.perspectiveTransform(src, H) - dst, axis=2).ravel()[keep]
    return H, residuals, keep.sum()
