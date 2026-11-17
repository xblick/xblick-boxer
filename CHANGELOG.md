# Changelog

## Branch ros2 (not released)

The node with the core's option `--ros` (ROS 2 on `rmw_zenoh`).

### Changed

- The core comes from its unreleased branch `ros2` (`xblick @ git+https://github.com/xblick/xblick@ros2`), not from
  PyPI. `uv sync` builds its C++ part from source.

### Added

- `--ros`, `--ros-domain`, `--ros-node` and `--ros-params-writable` from the core. With `--ros` the 2D and 3D boxes go
  to ROS 2 as well, and the parameters are a ROS node. README: section "ROS 2".
- `tests/test_ros.py`: a stand-in model (no checkpoint, a CUDA device); with `--ros` a ROS subscriber gets each output
  with a ROS type, with ROS's attachment and tokens; without it nothing of ROS is on the wire.

Compatible with earlier clients: nothing on our keys changed.

## 0.2.6 (2026-11-15)

A fixed camera's pose.

### Changed

- The node looks the camera's pose up with the core's `poses.TransformBuffer`, on `<camera>/tf_static` and
  `<camera>/tf`, for the frame id of the colour image. A fixed camera with `xblick cam --pose` (its pose on
  `tf_static`, valid at every stamp) now works as a moving one does. A frame without a pose is skipped and counted, as
  before. The status `boxer` names the reason of the last missing pose.
- The pose on `tf` must be that of the image's frame, or of a frame that `tf_static` links to it. Before, the node took
  any pose in `world` on `tf`, whatever its child frame.
- `--max-offset-ms` takes its default, 20 ms, from the core (`history.PAIR_MAX_OFFSET_NS`).
- The node needs `xblick>=0.5.11,<0.6`.

Compatible with earlier clients: nothing on the wire changed.

## 0.2.5 (2026-11-15)

A fixed pairing offset and a fixed GPU.

### Changed

- Changed default: `--max-offset-ms` is a fixed 20 ms, no longer half the depth stream's measured frame period. It pairs
  the colour and depth of one capture of the Azure Kinect (up to 17 ms apart) and of the rc_visard (equal stamps). A
  colour frame without a depth frame that close is skipped and counted, as before.
- `--device cuda` is GPU 0, `cuda:N` GPU N. Another value, or a GPU that cannot be used, ends the node with one line.
- The start line says where the model computes (`model on cuda:0`).
- README: how to cite Boxer.

Compatible with earlier clients: nothing on the wire changed.

## 0.2.4 (2026-11-14)

- The weights moved out of the checkout into the node's cache directory, `~/.cache/xblick/boxer/`. `XBLICK_HOME` and
  `XDG_CACHE_HOME` move it, and `--weights-dir DIR` names another directory. Weights at the former place, the
  checkout's `third_party/boxer/ckpts/`, are not used. The node prints the command that moves them. In the checkout it is
  `mkdir -p ~/.cache/xblick && mv third_party/boxer/ckpts ~/.cache/xblick/boxer`.
- `--weights-dir DIR` is a new start option and a read-only parameter.
- The message about a missing file names the full download command.
- The node needs `xblick>=0.5.5,<0.6`.
- Boxer's own code reads OWLv2 and DINOv3 from module globals that name its `ckpts/`. The node sets them to its
  cache directory after importing Boxer (`boxer.use_checkpoints`), without a change of Boxer's files. OWLv2's
  text-embedding caches now go there too. A test on a GPU confirmed that Boxer reads all three checkpoints from the
  cache directory and never creates `ckpts/`.
- `Boxer` takes the weights' directory (`weights=`, default `weights.STORE.root`).

## 0.2.3 (2026-11-14)

- The download of the weights is the core's `xblick.weights` (xblick 0.5.3). The node's own copy of that code is
  gone. `weights.py` keeps the node's files, its terms notice and its `STORE`. The options, the messages, and the
  files and their places stay as they were.
- The node needs `xblick>=0.5.3,<0.6`.
- `tests/test_weights.py` keeps the node's own checks. The core's tests cover the download itself.

## 0.2.2 (2026-11-14)

- The node downloads missing checkpoints at its start, before the models load. It prints Boxer's terms, the files
  and their size first. With all checkpoints there it uses no network and writes no file.
- `--download-only` downloads and checks the checkpoints, then exits 0. It needs no camera, Zenoh router or GPU, and
  `--camera` and `--labels` may be left out. `tools/setup.sh` calls it after the submodule and `uv sync`, and no
  longer needs curl.
- `--no-download` ends a start at once when a checkpoint is missing and names it. It is a read-only parameter.
- `weights.py` is the one list of the checkpoints (names, SHA-256 and the terms moved from `tools/setup.sh`, the
  terms unchanged). `tools/setup.sh` now prints them only when a checkpoint is downloaded, after it has fetched
  Boxer's source.
- Without Boxer's source (the submodule) the node downloads nothing and names `tools/setup.sh`.
- A download goes to `<name>.part` under the lock `ckpts/.download.lock` and gets its final name after the SHA-256
  check.
- The download takes the checkpoints from a fixed commit of `facebook/boxer` (bfba912, where `main` was on
  2026-11-14), no longer from `main`. Hugging Face states the same SHA-256 values for the files there. A download is
  also checked against the exact size.

## 0.2.1 (2026-11-14)

`tools/setup.sh` prints Boxer's terms (CC BY-NC 4.0) before it downloads Boxer's code and checkpoints.

## 0.2.0 (2026-11-13)

Every name in code, on the wire and on disk is `xblick` now. Only the prose name of the project stays Durchblick.
The node needs the core `xblick` 0.5 and does not work with programs of Durchblick 0.4 (`blick`) or older: run every
program of a setup at 0.5.

| What | 0.1.0 | 0.2.0 |
| --- | --- | --- |
| Repository | github.com/xblick/durchblick-boxer | github.com/xblick/xblick-boxer |
| Distribution | `durchblick-boxer` | `xblick-boxer` |
| Python package | `durchblick_boxer` | `xblick_boxer` |
| Commands | `durchblick-boxer` | `xblick-boxer` |
| Program name in log lines, the logger and reports | `durchblick-boxer` | `xblick-boxer` |
| Core dependency | `blick>=0.4,<0.5`, git main of github.com/xblick/durchblick | `xblick>=0.5.0,<0.6`, github.com/xblick/xblick at 45908e6 |
| Core imports and command | `import blick`, `blick rerun`, `blick play`, `blick param` | `import xblick`, `xblick rerun`, `xblick play`, `xblick param` |
| Links to the core's docs | github.com/xblick/durchblick/blob/main/docs/... | github.com/xblick/xblick/blob/main/docs/... |
| Other nodes named in the docs | `durchblick-fast-foundationstereo`, `durchblick-mapping`, `durchblick-nvblox` | `xblick-fast-foundationstereo`, `xblick-mapping`, `xblick-nvblox` |

## 0.1.0

The node as `durchblick-boxer` (Python package `durchblick_boxer`) on the core `blick` 0.4.
