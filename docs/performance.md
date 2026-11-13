# Performance

This page states how fast durchblick-boxer is, how we measured it, which changes made it fast, and which changes
did not help. The [README](../README.md) has the short version.

## Result

Boxer needs about 17 to 21 ms per frame on an RTX PRO 6000. The node keeps up with a 30 Hz camera.

| Version | Fastest frame | Median | Date |
| --- | --- | --- | --- |
| The first version | 85 ms | 140 ms | 2026-10-28 |
| OWLv2 in bfloat16, BoxerNet's inputs on the GPU | 34 ms | 65 ms | 2026-10-28 |
| The image converted to float on the GPU | 21 ms | 29 ms | 2026-10-29 |
| BoxerNet's Linear and Conv2d weights in bfloat16 | 17.4 ms | 28 ms | 2026-10-29 |

Setup: a simulated frame of a table with YCB objects (640 x 480, with depth) on an RTX PRO 6000. The frame is not
part of this repository. The time runs from the image to the boxes.

## How we measured

Other jobs used the GPUs during all measurements. The medians therefore contain the load of these jobs, and the
fastest frames say more about the code.

For the last row, the training jobs of another user kept every GPU of the machine at 46 to 100 % load. We
therefore measured the unchanged code and the changed code alternately in one process. The unchanged code needed
19.5 ms for its fastest frame and 32 ms in the median, in the same measurement.

A GPU executes work asynchronously. A Python timer around one line measures only the time to queue the work. For
the breakdown below, we made Python wait for the GPU after each step in turn (a *synchronisation*).

## Where the time goes

A *kernel* is one function that the GPU executes, for example one convolution. The CPU needs a few microseconds to
start each kernel. This is a *kernel launch*.

| Step | Time (fastest frames) |
| --- | --- |
| Resize and upload | 0.6 ms |
| OWLv2 | 6.2 ms |
| Depth points | 0.7 ms |
| BoxerNet | 13.8 ms |

Inside BoxerNet, DINOv3 takes 5.3 ms, its self-attention 1.0 ms, its cross-attention 1.4 ms and its head 1.5 ms.
Small steps between these parts take the rest. Some of them wait for checks inside Boxer's code that copy a value
from the GPU to the CPU. We do not change Boxer's code.

The GPU computes for 12 ms per frame. A frame launches 1600 kernels.

| Part | Kernel launches per frame |
| --- | --- |
| DINOv3 | 700 |
| OWLv2 | 200 |
| BoxerNet's cross-attention | 190 |
| BoxerNet's depth patches | 130 |

The rest of the frame time is the launching itself and the share of the GPU that the other jobs take. The CPU
cannot launch the kernels faster, and the GPU waits between them. The frame is bound by kernel launches on the
CPU. This explains why several changes below did not help.

We measured this breakdown on 2026-10-29 on a shared GPU.

## What we changed

### 1. OWLv2 in bfloat16, BoxerNet's inputs on the GPU

*bfloat16* is a 16-bit number format that modern GPUs compute much faster than 32-bit floats. OWLv2 now computes in
it.

BoxerNet checks its input with `img.max()`. The first version handed it an image in host memory, and this check
took 17 ms on the CPU. The node now hands BoxerNet its inputs as GPU tensors. The fastest frame went from 85 ms to
34 ms.

### 2. The image converted to float on the GPU

Boxer's function `img_to_tensor` converted the image to float on the CPU, which took 12 ms. The node now converts
it on the GPU. The fastest frame went from 34 ms to 21 ms.

### 3. BoxerNet's weights in bfloat16

PyTorch's *autocast* converts values to bfloat16 on the fly, inside a block of code. BoxerNet works under autocast,
and its weights were 32-bit floats. Autocast therefore converted the weights in every frame, which launched about
300 extra kernels. The node now stores the weights of the Linear and Conv2d layers in bfloat16, once at start.

The fastest frame went from 19.5 ms to 17.4 ms, and the median from 32 ms to 28 ms. The boxes on the test frame are
bit-identical, because autocast computes these layers in bfloat16 in both cases.

### 4. Two threads, one CUDA stream each

A *CUDA stream* is a queue of GPU work. The GPU can overlap work from two CUDA streams.

The node has two stages, `Boxer.detect` and `Boxer.lift`. Each stage has its own thread and its own CUDA stream.
OWLv2 works on the next frame while BoxerNet lifts the boxes of this frame.

The time per frame stays the same. The frame rate on a busy GPU rises from 21-26 to 27-29 frames per second on the
test frame. On the live simulation, the node now takes every camera frame.

### 5. No CPU work between the frame and the models

The node resizes the image on the GPU. It uses the same bilinear method as OpenCV, without OpenCV's rounding to
uint8. The node also computes Boxer's semi-dense points from the depth image on the GPU, in `sdp_from_depth`.
`tests/test_boxer.py` compares these points with the points of Boxer's own function.

With `--local cuda`, the node reads the frames from the camera's CUDA ring, and they never leave the GPU.

We made this change on 2026-10-30 and have not timed it on the test frame. The README of durchblick-mapping has a
measurement with the mapping nodes on one GPU, in its section "Measured". It states about 50 frames per second at
720p, for the host ring and the CUDA ring alike.

## The cost of `--features`

Setup: 2026-10-29, an idle GPU, a reader of the stream in another process, 60 frames, the two variants measured
alternately.

| Case | With features | Without |
| --- | --- | --- |
| A frame with 2D boxes, features over the CUDA ring | 19.5 ms median | 20.5 ms |
| A frame with 2D boxes, features over the host ring | 18.0 ms | 17.7 ms |
| A frame without 2D boxes | 10.9 ms | 5.7 ms |

On a frame with 2D boxes, BoxerNet computes the features anyway. Publishing them over the CUDA ring costs nothing
that we can measure. Over the host ring it costs 0.3 ms, which is the copy to host memory.

On a frame without 2D boxes, BoxerNet does not work. DINOv3 then computes the features for the stream alone.

Without a subscriber, the node does none of this work.

## What did not help

| Change | Result |
| --- | --- |
| `torch.compile` of DINOv3, modes `default` and `reduce-overhead` | 18.7-23 ms, within the spread of the unchanged code (18.7-22.5 ms) |
| OWLv2's detector with CUDA graphs | within the same spread |
| `--dino-beside-owl` | slower on a busy GPU, no faster on an idle one |

*`torch.compile`* translates a PyTorch model into optimised code. A *CUDA graph* records a fixed sequence of
kernels and starts it with one call. The mode `reduce-overhead` uses CUDA graphs.

**`--dino-beside-owl`.** OWLv2 and DINOv3 both need only the image. With this option, the detection stage computes
the DINOv3 features on a CUDA stream of their own while OWLv2 works. BoxerNet then takes them through its
`dino.forward`. The boxes are the same as without the option.

| Measurement | With the option | Without |
| --- | --- | --- |
| Busy GPUs, 2026-10-29, fastest frame | 24 ms | 18 ms |
| Idle GPU, later that day, 80 frames measured alternately, median | 18.8 ms | 18.2 ms |
| Idle GPU, two-thread pipeline, two rounds | 75 and 67 frames/s | 68 and 71 frames/s |

The frame is bound by kernel launches on the CPU. The work of the GPU does not limit it, so an overlap of that work gains
nothing. The option is off by default.

## Status and limits

The node keeps up with a 30 Hz camera at about 17 ms per frame. The following ideas would shorten the time from
the frame to the boxes. They would not raise the rate.

- **TensorRT for OWLv2 and DINOv3.** *TensorRT* is NVIDIA's compiler for networks, which merges layers and selects
  the fastest implementation of each for your GPU. The kernels of the two networks are about 8 of the 12 ms of GPU
  work. TensorRT could save perhaps 3 to 4 ms. It needs a `torch-tensorrt` that matches the torch build. We would
  have to compare the boxes with the current ones again.
- **Another 2D detector.** Boxer's paper names DETIC with 35 ms and SAM3. This is a choice of quality. It does not
  make the node faster.
- We have not timed the GPU resize and the GPU depth points on the test frame.
- We have not recorded the VRAM that the node needs.

[Making a node fast](https://github.com/xblick/durchblick/blob/main/docs/guides/making-a-node-fast.md) compares these findings with Durchblick's other nodes.
