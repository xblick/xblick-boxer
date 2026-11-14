# xblick-boxer

[![xblick core](https://img.shields.io/pypi/v/xblick?label=xblick%20core&color=blue)](https://github.com/xblick/xblick)

xblick-boxer finds the objects that a camera sees and publishes an oriented 3D box for each of them. It runs
Meta's [Boxer](https://huggingface.co/facebook/boxer) on the camera's colour, depth and pose streams and keeps up
with a 30 Hz camera.

It is a node of [Durchblick](https://github.com/xblick/xblick). You name the objects with text
prompts, so the node needs no fixed class list. Use it when you need the place, size and orientation of objects.

## What it publishes

| Topic | Message type | Content |
| --- | --- | --- |
| `<camera>/boxer/detections_2d` | `vision_msgs/Detection2DArray` | The 2D boxes of the detector OWLv2, in pixels of the colour image. |
| `<camera>/boxer/boxes_3d` | `vision_msgs/Detection3DArray` | Boxer's 3D boxes: centre, orientation and size in metres, in the camera frame. |
| `<camera>/boxer/dino` | image stream, `16FC384` | With `--features`: the DINOv3 features of every frame, 60 x 60 patches. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | The status `boxer`, once a second. |

The node reads `<camera>/color` with its `camera_info`, `<camera>/depth` and the camera's pose on `<camera>/tf`.
Every result has the stamp and the camera frame of its colour frame.

Details: [docs/topics.md](docs/topics.md).

## Requirements

- An NVIDIA GPU with CUDA. We measured on an RTX PRO 6000. Alone on that GPU the node held 1.7 GiB of VRAM
  (four labels, `--features`, 1280 x 720 at 30 frames per second, 2026-11-02).
- Durchblick 0.5 or newer, below 0.6 (Python package `xblick`). The setup script installs it from source.
- [uv](https://docs.astral.sh/uv/), curl and git.
- A depth stream in the geometry of the colour camera, and the camera's pose in a world whose z axis points up.
- Boxer's code and checkpoints are under CC BY-NC 4.0, and so is this repository. You may not use them commercially.

## Install

Clone this repository and run the setup script.

```bash
tools/setup.sh
```

The script does three things. It fetches Boxer's source as a git submodule. It downloads Boxer's checkpoints
(1.2 GB) from Hugging Face and checks their SHA-256. It creates `.venv` exactly as `uv.lock` fixes it, with
PyTorch for CUDA 13. A second call does only what changed.

Before it downloads Boxer, the script prints Boxer's terms (CC BY-NC 4.0: no commercial use, credit to its
authors).

## Run

Start the node on a camera. `cam/azurekinect`, the Azure Kinect node's default, is an example name. The node
connects to the Zenoh router of the PC (`zenohd` on port 7447), which [Sessions](https://github.com/xblick/xblick/blob/main/docs/concepts/sessions.md) explains.

```bash
.venv/bin/xblick-boxer --camera cam/azurekinect --depth cam/azurekinect/depth_to_color \
  --labels "cracker box,sugar box,tomato soup can,mustard bottle"
```

Look at the result in Durchblick's viewer. Without options, it shows every stream and every node result that it
finds.

```bash
xblick rerun --spawn
```

The options that most users change:

| Option | Default | Meaning |
| --- | --- | --- |
| `--labels` | required | The text prompts. Separate them with spaces or commas. Live: it changes while the node runs. |
| `--depth` | `<camera>/depth` | The depth stream in the geometry of the colour camera. |
| `--no-depth` | off | Work without depth. Boxer then estimates distances from the image alone. |
| `--thresh2d` | 0.2 | The score threshold of OWLv2. Live: it changes while the node runs. |
| `--thresh3d` | 0.5 | The confidence threshold of BoxerNet. Live: it changes while the node runs. |
| `--features` | off | Publish the DINOv3 features while somebody subscribes. |
| `--max-hz` | 30 | The highest number of colour frames per second. `0` or `none` takes every frame. |
| `--local` | `auto` | How the node reads a camera on the same host: `auto`, `shm` or `cuda`. |
| `--connect HOST` | the router of the PC | Connect to a router on another host. |

All options: [docs/topics.md](docs/topics.md#options).

Every option is also a parameter of the node `<camera>/<node>`, for example `cam/azurekinect/boxer`. Durchblick's
terminal interface (`xblick`, tab Parameters) and `xblick param` show them. `labels`, `thresh2d`, `thresh3d` and
`pose_wait_ms` change while the node runs: new labels are OWLv2's prompts from the next frame on. The other options
apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them.

## Why it is fast

A frame took 140 ms in the median at first and takes 28 ms now. The fastest frames went from 85 ms to 17.4 ms.

| Change | Effect |
| --- | --- |
| OWLv2 computes in bfloat16, and BoxerNet's inputs stay on the GPU. | fastest frame 85 ms to 34 ms |
| The node converts the image to float on the GPU. | 34 ms to 21 ms |
| BoxerNet stores its weights in bfloat16. | 19.5 ms to 17.4 ms in a direct comparison |
| The two stages work in two threads with one CUDA stream each. | 21-26 to 27-29 frames per second on a busy GPU |

The measurements, the method and what did not help: [docs/performance.md](docs/performance.md).

## Limits

- Version 0.2.1. We tested the node on a simulation. We have not tested it on a robot.
- OWLv2 confuses similar objects. It names the cracker box "sugar box" in most frames.
- The node does not use Boxer's tracking or its multi-view fusion. A box has no id over frames.
- The node skips a frame whose pose or depth does not arrive.

## Documentation

- [docs/topics.md](docs/topics.md): inputs, messages and all options.
- [docs/performance.md](docs/performance.md): measurements, what we changed, what did not help.
- [docs/design.md](docs/design.md): how the node works inside, the decisions, the tests and their results.
- Durchblick: [Streams and topics](https://github.com/xblick/xblick/blob/main/docs/concepts/streams-and-topics.md), [Transports](https://github.com/xblick/xblick/blob/main/docs/concepts/transports.md),
  [Pairing streams](https://github.com/xblick/xblick/blob/main/docs/guides/pairing-streams.md), [Making a node fast](https://github.com/xblick/xblick/blob/main/docs/guides/making-a-node-fast.md).

## Licence and credits

Copyright 2026 David Schneider. This repository is under the Creative Commons Attribution-NonCommercial 4.0
International licence (CC BY-NC 4.0), see [LICENSE](LICENSE). You may not use it commercially.

The node runs Meta's [Boxer](https://github.com/facebookresearch/boxer). Boxer, BoxerNet and the checkpoints are
Meta's work under CC BY-NC 4.0. Meta's `LICENSE` and `NOTICE` are in `third_party/boxer`. Boxer is a git submodule at
a fixed commit, and the setup script downloads the checkpoints. This repository contains no copy of them.

The function `sdp_from_depth` in `src/xblick_boxer/boxer.py` is adapted from `BaseLoader.sdp_from_depth` in
Boxer's `loaders/base_loader.py`. It computes the same points with PyTorch on the device of the depth image.
[NOTICE](NOTICE) lists the changes.
