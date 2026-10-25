# OpenCV 5, honestly audited

OpenCV 5 landed at CVPR 2026 with a rewritten DNN engine, a new Features module,
a point-cloud module, and a long list of removals. This page says which of it is
actually useful for overhead raster work, which is a speedup dressed as a
feature, and which is irrelevant — plus the six API quirks that cost a day each
if nobody warns you.

Everything below was run against `opencv-python-headless==5.0.0.93`.

---

## The audit

Twelve headline additions. **Four carry weight, and one carries most of it.**

| What is new in 5 | Verdict | What it changes for raster work |
|---|---|---|
| **Rewritten DNN engine**<br/>ONNX coverage ~22% → 64–80%, dynamic shapes | **Core** | OpenCV 4 could not import a transformer graph at all — LayerNorm, GELU, Einsum and attention subgraphs all sat outside its 22%. This is a binary capability change, not a speedup. |
| **FP16 types** (`CV_16F`) | **Core** | Halves activation memory. Relevant whenever inference must fit a small worker. |
| **Features module**<br/>ALIKED, DISK, LightGlueMatcher | **Core** | Learned feature matching **without a deep-learning framework.** The one genuinely new capability for this domain — see below. |
| **ptcloud module** | Partial | Voxel/FPS sampling, octree, USAC plane segmentation. **Reads .ply and .obj, not .LAZ** — so for lidar you still need laspy to read and hand it numpy arrays. The processing is OpenCV's; the IO is not. |
| **USAC default** for robust estimation | Supporting | The homography fit behind co-registration. Markedly better than plain RANSAC where repeating patterns produce self-consistent wrong matches — which describes crop rows exactly. |
| **TRUCO `findContours`** | Throughput | Mask → polygon, several times faster. OpenCV 4 did this. Call it cost, not capability. |
| **Warping 10–300% faster** | Throughput | Grid alignment. Same honesty applies. |
| **ONNX Runtime backend** via CMake | Escape hatch | If `cv2.dnn` chokes on a graph, swap the backend rather than adding a second dependency. |
| LLM/VLM inside DNN | *Unused* | The headline feature of the release, and irrelevant here. An 8B model doing autoregressive decode on a CPU worker is an edge story, not a raster one. |
| calib3d split → geometry/calib/stereo | *Unused* | No cameras to calibrate, no stereo pairs. Only the ptcloud quarter matters. |
| RISC-V / Qualcomm optimisations | *Unused* | Not a deployment target for most server work. |
| C++17 minimum, C API removed,<br/>OpenVX and classic ML removed | *Migration cost* | We use none of what was removed — worth checking rather than assuming. |

---

## The one thing with no substitute

Three of the four "core" rows have alternatives. One does not.

| Need | Without OpenCV 5 | |
|---|---|---|
| Morphology, contours, warping | scipy, ~88 MB | replaceable |
| Transformer inference | `onnxruntime` — smaller, fuller ONNX coverage, genuinely the better standalone choice | replaceable |
| Point-cloud geometry | Open3D or PDAL, and laspy either way | replaceable |
| **Learned feature matching** | LightGlue's reference implementation and kornia are **both PyTorch**. There is no CPU path to learned matching that avoids a deep-learning framework — roughly 2 GB installed. | **no substitute** |

So the precise claim is: **OpenCV 5 is the only way to run learned feature
matching without carrying torch.** That is narrow, falsifiable, and survives a
sceptical question.

What it is *not*: "OpenCV 5 unlocked inference." If torch is available, kornia
is the better matcher and `onnxruntime` the better runtime. OpenCV earns its
place here through the classical ops you were carrying it for anyway.

---

## Why learned matching matters for farmland

Classical detectors — SIFT, ORB — look for corners and distinctive texture.
Farmland has neither. It is low-contrast, strongly repeating (crop rows produce
many self-consistent *wrong* matches), and changes appearance dramatically
between seasons and flight years.

DISK is documented as strong on wide-baseline and low-texture scenes, which is
that regime exactly.

**Verified on a synthetic control:**

| | |
|---|---|
| Median reprojection residual | **0.2449 px** |
| True shift | +12 / −7 px |
| Recovered | +11.97 / −6.98 px |
| Matches / inliers | 1469 / 1469 |
| Timing, CPU | ~2.8 s per image extract, ~3.3 s match at 2048 keypoints |
| `cv2` installed size | **119 MB** |

> **Caveat worth stating:** this is a synthetic control, not real imagery. On a
> synthetic test with crisp geometric shapes — ORB's best case — ORB was
> *competitive*. Run your own comparison on real imagery before claiming DISK
> wins on accuracy. The claim that always holds is the deployment one: learned
> matching at 119 MB rather than ~2 GB.

---

## The six quirks, every one of which silently returns zero matches

These produce **no error and no matches**, which is a miserable thing to debug
from scratch.

**1 · OpenCV ships no weights.** Both factories take a *required* `modelPath`,
and `opencv_zoo` does not carry these models.

```bash
B=https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0
curl -sSL -o models/disk.onnx               $B/disk.onnx                #  4.2 MB
curl -sSL -o models/disk_lightglue.trt.onnx $B/disk_lightglue.trt.onnx  # 43.9 MB
```

**2 · The matcher must be the `.trt.onnx` export.** The `_cpu` and plain `_fused`
builds embed `com.microsoft::MultiHeadAttention`, an ONNX Runtime custom operator
OpenCV's parser rejects outright. The TensorRT-targeted export uses standard
operators and **runs fine on CPU despite the name.** This one is genuinely
counterintuitive — the `_cpu` build is the one you would reach for.

**3 · Never an end2end or pipeline model.** OpenCV wants extractor and matcher as
two separate graphs; a fused pipeline cannot fit the API however it is called.

**4 · `match()` takes no keypoints.** Call `setPairInfo()` first.

**5 · Keypoints must be an Nx2 float32 array, not a `cv2.KeyPoint` list — and
image sizes are `(width, height)`,** the reverse of numpy's `.shape`. Both wrong
forms raise an unhelpful overload error.

**6 · `score_threshold` must be passed explicitly as `0.1`.** The worst of the
six. Omitting it returns **zero matches** — and so does passing a *lower* value,
which is the opposite of how a confidence floor normally behaves. The same image
pair gives 462 matches with it and 0 without.

### Putting it together

```python
import cv2, numpy as np

disk = cv2.DISK.create("models/disk.onnx", 2048)
glue = cv2.LightGlueMatcher.create("models/disk_lightglue.trt.onnx", 0.1)  # ← quirk 6

# grayscale uint8 only — BGR returns nothing
kp_a, desc_a = disk.detectAndCompute(img_a, None)
kp_b, desc_b = disk.detectAndCompute(img_b, None)

pts_a = np.float32([k.pt for k in kp_a])          # ← quirk 5
pts_b = np.float32([k.pt for k in kp_b])
wh = lambda im: (im.shape[1], im.shape[0])        # (width, height), not .shape

glue.setPairInfo(pts_a, pts_b, wh(img_a), wh(img_b))   # ← quirk 4
matches = glue.match(desc_a, desc_b)

src = pts_a[[m.queryIdx for m in matches]].reshape(-1, 1, 2)
dst = pts_b[[m.trainIdx for m in matches]].reshape(-1, 1, 2)
H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0)
```

`pestren_geo.align.coregister(a, b, backend="opencv")` wraps all of this, adds
the residual gate, and refuses rather than returning a bad fit.

**One benign warning to ignore:** `setPreferableTarget: Targets are not supported
by the new graph engine`. Separately — `CV_16F` is available in the Python
bindings but **`CV_16BF` is not**, so the bfloat16 half of the release is C++
only for now.

---

## What OpenCV is genuinely for here

Strip away the release notes and the durable reason to carry OpenCV in a raster
pipeline is the classical operations, which are best-in-class and tiny:

| Operation | Used for |
|---|---|
| `morphologyEx` | Cleaning a mask before totalling. **Kernel never wider than the narrowest real feature** — a 5×5 open at 0.6 m erases every 3 m ditch in the scene, and the acreage total gets *better looking* as it does. |
| `connectedComponentsWithStats` | Separating touching blobs. Three treelines become three objects. |
| `findContours` | Mask → exportable vector geometry. |
| `warpPerspective` / `remap` | Putting one raster on another's grid. **`INTER_LINEAR`, never cubic** — cubic overshoots at breaklines and invents pits that a depression fill then dutifully finds. |
| `Scharr` / `Sobel` | Gradients, with better rotational symmetry than a naive 3×3. |

None of that is new in 5. All of it is why the dependency is there.

---

## Should you upgrade?

**If you need learned matching without torch:** yes, it is the only route.

**If torch is already in your stack:** the upgrade buys you throughput and FP16.
Use kornia for matching and `onnxruntime` for inference, and treat OpenCV as the
classical-ops library it has always been.

**Migration cost is low** if you are on modern OpenCV 4 — C++17 minimum, the C
API and OpenVX gone, `Features2D` renamed to `Features`, `calib3d` split four
ways. Check whether you import anything removed before assuming it is free.
