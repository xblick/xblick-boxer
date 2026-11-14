# Topics and options

This page lists what xblick-boxer reads, what it publishes, and every option of the node.

## Inputs

| Topic | Type | Use |
| --- | --- | --- |
| `<camera>/color` | image stream with `camera_info` | The frames in which OWLv2 looks for the prompts. |
| `<camera>/depth` | image stream, `16UC1` in mm or `32FC1` in m | The depth, in the geometry of the colour camera. |
| `<camera>/tf` | poses | The pose of the camera in the world at the stamp of the frame. |

**Depth.** The depth image must have the geometry of the colour camera. On an Azure Kinect, the stream
`depth_to_color` has it. `--depth` names the stream.

The node pairs each colour frame with the depth frame whose stamp is nearest. The two stamps may differ by at most
`--max-offset-ms`. The default is half the frame period of the depth stream, which the node measures. The node
reads the depth frame together with the colour frame.
[Pairing streams](https://github.com/xblick/xblick/blob/main/docs/guides/pairing-streams.md) describes the mechanism.

With `--no-depth` the node needs no depth stream. Boxer then estimates distances from the image alone.

**Pose.** BoxerNet needs the direction of gravity. The node takes it from the pose of the camera in the world and
assumes that the world's z axis points up. The node waits up to `--pose-wait-ms` for the pose at the stamp of the
frame. If no pose has exactly that stamp, the node interpolates between the two poses around it.

**Skipped frames.** The node skips a frame whose pose or depth does not arrive, and counts it. It never processes a
frame without one of its parts.

## Outputs

Every result has the stamp of its colour frame and the frame id of the camera. `--node` changes the `boxer` part
of the topic names, so several nodes can work on one camera.

A frame without boxes publishes empty arrays. A viewer then removes its boxes.

| Topic | Type | Content |
| --- | --- | --- |
| `<camera>/boxer/detections_2d` | `vision_msgs/Detection2DArray` | OWLv2's boxes in pixels of the colour image. `class_id` is the prompt, and `score` is OWLv2's score. |
| `<camera>/boxer/boxes_3d` | `vision_msgs/Detection3DArray` | Boxer's boxes. The centre and the orientation are in the camera frame with OpenCV axes. The orientation gives the axes of the box, and the size is in metres along these axes. `score` is the mean of OWLv2's and BoxerNet's scores. |
| `<camera>/boxer/dino` | image stream, `16FC384` | BoxerNet's DINOv3 features, see below. |
| `<camera>/diagnostics` | `diagnostic_msgs/DiagnosticArray` | The status `boxer`, once a second. |

### The features

With `--features`, the node publishes BoxerNet's DINOv3 features of every frame on `<camera>/boxer/dino`. It does
so only while somebody subscribes.

- BoxerNet scales the image to 960 x 960 and computes one feature vector per patch of 16 x 16 pixels.
- The stream is an image of 60 x 60 pixels with 384 channels of float16, encoding `16FC384`.
- Its `camera_info` describes the patch grid.
- A reader on the same host gets the features on the GPU through the `cuda` transport. Other readers receive
  2.8 MB per frame.

xblick-nvblox fuses these features into its map.

### The status

The status `boxer` contains the frames per second, the boxes per frame and the milliseconds per frame. It also
counts the skipped frames by cause: no pose, no depth, no `camera_info`.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--camera` | required | The base name of the camera, for example `cam/azurekinect`. |
| `--labels` | required | The text prompts, separated by spaces or commas, for example `box can "tomato soup can"`. Live: it changes while the node runs. |
| `--color` | `<camera>/color` | The colour stream. |
| `--depth` | `<camera>/depth` | The depth stream in the geometry of the colour camera. |
| `--no-depth` | off | Work without depth. You cannot combine it with `--depth`. |
| `--node` | `boxer` | The part of the topic names after the camera. |
| `--thresh2d` | 0.2 | The score threshold of OWLv2. Live: it changes while the node runs. |
| `--thresh3d` | 0.5 | The confidence threshold of BoxerNet. Live: it changes while the node runs. |
| `--max-offset-ms` | half the depth's frame period | The largest difference between the stamps of a colour frame and its depth frame. `0` demands equal stamps. |
| `--pose-wait-ms` | 200 | How long the node waits for the pose of a frame. Live: it changes while the node runs. |
| `--device` | `cuda` | The GPU of the node. |
| `--no-download` | off | End at once when a checkpoint is missing, and name it. The node downloads nothing. |
| `--download-only` | off | Download the missing checkpoints, check their SHA-256 and end. The node needs no camera, Zenoh router or GPU for it, and `--camera` and `--labels` may be left out. |
| `--dino-beside-owl` | off | Compute the DINOv3 features beside OWLv2 on a CUDA stream of their own. It did not help in our measurements. |
| `--features` | off | Publish the DINOv3 features. |
| `--max-hz` | 30 | The highest number of colour frames per second. `0` or `none` takes every frame. |
| `--local` | `auto` | The transport for a camera on the same host: `auto`, `shm` or `cuda`. |
| `--connect`, `--listen`, `--config` | the router of the PC | The Zenoh session, as for every Durchblick program. |

**`--max-hz`.** The node takes at most this many colour frames per second, and always the newest one. It does not
decode the frames that it leaves out. It decodes a depth frame only for a colour frame that it takes. The report in
the log states the rate in effect and the number of frames that the node left out.

**`--local`.** With `auto`, the node reads a camera on the same host from its CUDA ring if that ring is on the
node's GPU. Otherwise it reads from the host ring in shared memory. `shm` and `cuda` force one of the two. With
`cuda`, the frames never leave the GPU, and the node stops if the camera offers no CUDA ring. The start log and
every report state which transport each stream uses.

**MPS.** If you set `CUDA_MPS_PIPE_DIRECTORY`, the node expects an MPS daemon and stops at start when none answers.
[Sharing a GPU](https://github.com/xblick/xblick/blob/main/docs/guides/sharing-a-gpu.md) explains MPS.

## Parameters

Every option is also a parameter of the node `<camera>/<node>`, for example `cam/azurekinect/boxer`. The live ones
change while the node runs: `labels`, `thresh2d`, `thresh3d`, `pose_wait_ms`; the tables mark them. The others are
read-only and apply at the next start, where `--param NAME:=VALUE` or `--params-file FILE` set them. New labels are
OWLv2's prompts from the next frame on. Durchblick's terminal interface (`xblick`, tab Parameters) and `xblick
param` show them. `--download-only` is a mode of the command line and has no parameter.
