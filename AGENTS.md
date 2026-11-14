# AGENTS.md

xblick-boxer: Meta's Boxer as a Durchblick vision node (Python 3.12, uv project). OWLv2 finds 2D boxes for text
prompts (`--labels`), BoxerNet lifts them to oriented 3D boxes from DINOv3 features, depth points and gravity (from
the camera's pose on `<camera>/tf`). Human documentation: README.md (entry), docs/topics.md, docs/performance.md,
docs/design.md. Style of all documentation: the core's docs/project/writing-docs.md. Transport-wide
traps (Zenoh sessions and routers, CUDA rings, hanging tests): AGENTS.md of the core repository
(github.com/xblick/xblick).

## Commands

```bash
tools/setup.sh                            # submodule, .venv from uv.lock, then xblick-boxer --download-only
.venv/bin/python -m pytest tests/test_weights.py -q   # CPU, 10 s: downloads from a local HTTP server
.venv/bin/xblick-boxer --download-only    # checkpoints (1.2 GB) into third_party/boxer/ckpts if missing, SHA-256
.venv/bin/python -m pytest tests -q       # always name tests/: plain pytest collects third_party/boxer and fails
.venv/bin/ruff check src tests            # lint: 120 columns, third_party excluded (pyproject [tool.ruff])
.venv/bin/xblick-boxer --camera cam/azurekinect --depth cam/azurekinect/depth_to_color \
  --labels "cracker box,sugar box" [--features]      # needs zenohd on the PC (7447)
xblick rerun --spawn                       # viewer
```

- `tests/test_node.py`: conversions (quaternions, vision_msgs, depth), no GPU.
- `tests/test_weights.py`: the download against a local HTTP server, no GPU.
- `--no-download`: a start with a missing checkpoint ends with one line (exit 1). Excludes `--download-only`.
- `tests/test_boxer.py` (GPU, checkpoints): GPU semi-dense points against Boxer's own on a synthetic depth image. Two
  tests need the simulation frame `tests/data/sim_ycb*.png`, which this branch does not contain: they skip.
- Durchblick (`xblick`): github.com/xblick/xblick at the commit in `pyproject.toml`. Newer Durchblick needed: raise
  the pin and the rev in `pyproject.toml`, then `uv lock`.
- Without a camera: a simulation that publishes colour, depth and the camera's pose.

## Rules

- Fail fast, never process a frame without its parts. Skip and count a frame whose pose (`--pose-wait-ms`) or depth
  (`--max-offset-ms`) does not come. `--local cuda` with a same-host camera without a CUDA ring is an error. A missing
  submodule stops the node with what to do (`tools/setup.sh`); never clone it at run time.
- Boxer's code stays unchanged. It is no package: `boxer.py` puts `third_party/boxer` on `sys.path`. Make speed-ups
  around it (inputs on the GPU, bfloat16 weights). Boxes must stay as they were: compare against the current boxes
  on the test frame after any change.
- Nothing between frame and model on the CPU: resize (bilinear as OpenCV's, without rounding to uint8) and
  semi-dense points (`sdp_from_depth`) on the GPU.
- A frame without boxes publishes empty arrays, so viewers clear their boxes.
- Weights: `weights.py` is the one list (`CHECKPOINTS`: SHA-256, size; `REVISION`: the commit of `facebook/boxer` that main was on 2026-11-14;
  `boxer.py` imports it). `weights.ensure` runs at the start before the models load; `--download-only` runs it alone.
  All there: no network, no file written (`test_weights.py` checks it). `REQUIRED`: the submodule first, the
  checkpoints go into it. Downloads: `<name>.part` in `ckpts/`, renamed after the SHA-256 check, under the lock
  `ckpts/.download.lock`. Never let a partial or unchecked file get its final name. No fallbacks.
- Licence: Boxer's code (submodule at a fixed commit) and checkpoints are CC-BY-NC 4.0. Copy nothing of them into
  this repository.
- README.md and docs/ are the documentation of record. Speed numbers with GPU, date and load. Keep options and
  topics in sync with the code.

## Traps

- **Hidden CPU work costs most.** Symptoms measured: BoxerNet's input check `img.max()` on the CPU 17 ms,
  `img_to_tensor` 12 ms, autocast casting float32 weights every frame (~300 cast kernels). Fix: inputs as GPU
  tensors, Linear/Conv2d weights stored in bfloat16.
- **Launch-bound.** A frame is bound by kernel launches on the CPU. `torch.compile`, CUDA graphs and
  `--dino-beside-owl` gained nothing. Do not retry without a new idea.
- **Two threads, one CUDA stream each** (`Boxer.detect`, `Boxer.lift`): OWLv2 of the next frame while BoxerNet lifts
  this one. Same time per frame, more frames per second on a busy GPU.
- **Shared GPUs distort timings.** Compare variants interleaved in one process. Report the fastest frames too.
- **Checkpoints directory.** Symptom: untracked content in the submodule. Cause: Boxer's `.gitignore` ignores
  `ckpts/` only as a directory, a symlink named `ckpts` is not ignored. Fix: `third_party/boxer/ckpts` must be a real
  directory.
- **`--download-only` and required options.** `weights.parse_args` lifts `required` of `--camera` and `--labels`
  when the token `--download-only` is spelled out. `download_only` is in `leave_out` of `declare_options`;
  `no_download` is a read-only parameter.
- **Wrong SHA-256 after a download.** The `.part` file stays, the error names it. Remove it, start again. Ctrl-C
  during a download: exit 130, SIGTERM: 143 (`weights._stoppable`), the `.part` file removed.
- **OWLv2 confuses similar objects** (cracker box named "sugar box"). A prompt problem, no lifting problem.
- **`--labels`.** Words, each also comma-separated.
- **Depth** must be in the colour camera's geometry (`depth_to_color` on the Azure Kinect), 16UC1 mm or
  32FC1 m.
- **From the core.** Every program uses the PC's Zenoh router (`zenohd` on 7447). Without an MPS daemon several nodes
  on one GPU take turns with their kernels, silently. Keep `.venv` and checkpoints on local disk, never on NFS.

## Layout

| Path | Content |
| --- | --- |
| `src/xblick_boxer/node.py` | the node: options, inputs, pairing, two-stage pipeline, publishing |
| `src/xblick_boxer/weights.py` | `CHECKPOINTS`, `TERMS`, `needed`, `ensure`, `--download-only`/`--no-download`; from `Weight` on the same in every node |
| `src/xblick_boxer/boxer.py` | Boxer on one image: OWLv2, BoxerNet, `sdp_from_depth`, Boxer's imports |
| `tests/` | node and GPU tests |
| `third_party/boxer` (submodule), its `ckpts/` | Boxer's code at a fixed commit, the checkpoints the node downloads, `.download.lock` |
| `tools/setup.sh` | submodule, venv, then `xblick-boxer --download-only` |
| `docs/` | topics, performance, design |

## Defaults

| What | Value |
| --- | --- |
| Inputs | `<camera>/color` + `camera_info`, `<camera>/depth` (16UC1 mm or 32FC1 m), `<camera>/tf` |
| Outputs | `<camera>/boxer/detections_2d`, `<camera>/boxer/boxes_3d` (`vision_msgs`), status `boxer` on `<camera>/diagnostics` |
| `--features` | image stream `<camera>/boxer/dino`, `16FC384`, 60 x 60 patches, read by xblick-nvblox |
| `--node` | `boxer` |
| `--thresh2d` / `--thresh3d` | 0.2 / 0.5 |
| `--max-offset-ms` / `--pose-wait-ms` | half the depth's frame period / 200 |
| `--max-hz` / `--local` / `--device` | 30 / `auto` / `cuda` |
| Environment | `UV` (setup), `CUDA_MPS_PIPE_DIRECTORY` (MPS expected) |
