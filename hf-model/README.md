---
license: apache-2.0
pipeline_tag: keypoint-detection
library_name: onnx
tags:
  - piano
  - keyboard
  - music
  - keypoints
  - onnx
  - onnxruntime-web
  - webgpu
  - computer-vision
  - browser
---

# pianoCV

KeyNet finds the keypoints of a piano or electronic keyboard in a camera frame: the keybed
corners, the white-key gaps and the black keys' edges. A small geometry core fits a real
keyboard layout to those points and draws every key in the camera's own perspective. It is the
model behind https://github.com/matheusfillipe/pianoCV.

| file | what it finds |
|---|---|
| `keynet.onnx` | keybed corners, white-key gaps and black-key edges as heatmaps, and whether a keyboard is in view |

| | |
|---|---|
| input | `[1, 3, H, W]` float32 RGB normalised on ImageNet statistics, mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. Search mode feeds the whole frame squashed to 256x256, track mode feeds a 768x160 crop around the last fit. |
| outputs | `heatmaps` `[1, 12, H/2, W/2]`, one channel per keypoint kind, and `presence`, how likely a keyboard is in view |
| channels | 0-3 keybed corners (back-low, back-high, front-high, front-low), 4 white-key gaps at the front edge, 5-6 black keys' front corners on the keybed (low and high side), 7-8 the same corners on the black keys' tops, 9 white-key gaps at the back edge, 10-11 the black keys' back top corners |
| precision | weights in float16, inputs and outputs in float32 |

It is a plain [ONNX](https://onnx.ai) file, so you can run it from any language with
[ONNX Runtime](https://onnxruntime.ai).

## Limitations

- Trained mostly on synthetic renders plus hand-checked real frames, so unusual boards and
  camera angles are where it is weakest.
- It finds keys only. It does not detect hands, read notes, or identify the instrument.

## Licence

Apache 2.0.
