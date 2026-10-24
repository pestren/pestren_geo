# Feature matching backends

Co-registration needs learned feature matching. Farmland is the regime classical
detectors are weakest in — low texture, strongly repeating crop rows, and large
seasonal appearance change between flight years.

Two implementations sit behind one interface. The choice is a deployment
decision, not a quality one.

```python
coregister(a, b, backend="kornia")   # reference implementation, needs torch
coregister(a, b, backend="opencv")   # ONNX, no deep-learning framework
```

## kornia — the default where torch is available

```bash
pip install "pestren-geo[kornia]"
```

The reference LightGlue and DISK implementations. Actively maintained, no export
quirks, and the obvious choice when a GPU or torch is already present.

## opencv — for a torch-free deployment

```bash
pip install "pestren-geo[opencv]"

mkdir -p models
B=https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0
curl -sSL -o models/disk.onnx               $B/disk.onnx                #  4.2 MB
curl -sSL -o models/disk_lightglue.trt.onnx $B/disk_lightglue.trt.onnx  # 43.9 MB
```

OpenCV 5's Features module running DISK and LightGlue as ONNX graphs. Around
120 MB installed against roughly 2 GB for torch, which is the entire reason this
path exists.

Verified on `opencv-python-headless==5.0.0.93`: **0.2449 px median residual** on
a synthetic control, recovering a +12/−7 px shift to 0.03 px.

### Six things that silently return zero matches

Each of these produces no error and no matches, which is a difficult failure to
debug from scratch.

1. **OpenCV ships no weights.** `DISK.create()` and `LightGlueMatcher.create()`
   both take a *required* `modelPath`. `opencv_zoo` does not carry these models.
2. **The matcher must be the `.trt.onnx` export.** The `_cpu` and plain `_fused`
   builds embed `com.microsoft::MultiHeadAttention`, an ONNX Runtime custom
   operator OpenCV's parser rejects. The TensorRT-targeted export uses standard
   operators and runs on CPU despite its name.
3. **Never an end2end or pipeline model.** OpenCV needs extractor and matcher as
   two separate graphs.
4. **`match()` takes no keypoints.** Call `setPairInfo()` first, with Nx2 float32
   arrays rather than a list of `cv2.KeyPoint`.
5. **Image sizes are `(width, height)`**, the reverse of numpy's `.shape`.
6. **`score_threshold` must be passed explicitly as `0.1`.** Omitting it returns
   nothing — and so does passing a *lower* value, which is the opposite of how a
   confidence floor normally behaves.

A benign warning, `setPreferableTarget: Targets are not supported by the new
graph engine`, can be ignored. Separately, `CV_16F` is available in the Python
bindings but `CV_16BF` is not.

## Adding a third

Implement `(image_a, image_b, max_keypoints, **kw) -> (pts_a, pts_b)` returning
two Nx2 float32 arrays of corresponding points, and register it in `_BACKENDS`.
Nothing else in the library changes.
