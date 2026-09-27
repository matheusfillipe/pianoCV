---
license: apache-2.0
pipeline_tag: image-segmentation
library_name: onnx
tags:
  - piano
  - keyboard
  - music
  - segmentation
  - onnx
  - onnxruntime-web
  - webgpu
  - computer-vision
  - browser
---

# pianoCV

A piano segmentation model: given a camera frame of a piano or electronic keyboard, it marks
every pixel that belongs to a white key, a black key, or the gap between two white keys, so
each key comes out as its own region with its own outline.

It ships with two smaller companions, a detector that finds roughly where the keyboard is and
a matcher that reads the key edges along it, and the three together find every key live in a
web browser. They are the models behind
https://github.com/matheusfillipe/pianoCV, which fits a real keyboard's layout to what they see
and draws every key in the camera's own perspective.

They are deliberately small and fast. Each one runs through onnxruntime-web, and the key
segmenter runs on the GPU through WebGPU in about 30 ms per frame.

| file | what it finds | size |
|---|---|---|
| `keybed_seg2.onnx` | roughly where the keyboard is in the frame | 6.6 MB |
| `keyseg.onnx` | every pixel that is a white key, a black key, or the gap between two white keys | 6.6 MB |
| `keymatch.onnx` | where the white-key gaps and the black-key edges are along the keyboard | 0.3 MB |

All three take RGB input normalised on ImageNet statistics, mean `[0.485, 0.456, 0.406]` and
std `[0.229, 0.224, 0.225]`, as planar float32.

## keybed_seg2.onnx

| | |
|---|---|
| input | `image`, `[1, 3, 288, 288]`, the whole frame squashed to a square |
| output | `mask`, `[1, 1, 144, 144]`, how likely each patch is to be keyboard, 0 to 1 |

A MobileNetV3-Small backbone pretrained on ImageNet with a small U-Net decoder. It gives a
rough region, which the pipeline only uses to decide where to look.

## keyseg.onnx

| | |
|---|---|
| input | `crop`, `[1, 3, 224, 1024]`, a crop around the keyboard, rotated so the keys run left to right with the player's edge at the bottom |
| output | `classes`, `[1, 4, 224, 1024]`, per-pixel probabilities for background, white key, black key and white-key gap |

The crop only rotates and scales the frame, so keys keep the shape the camera gives them. The
same MobileNetV3-Small encoder with a decoder that climbs back to full resolution, since the
gap between two white keys is a pixel or two wide. A black key's label covers its whole
visible outline, raised top and sides included. On held out synthetic renders it overlaps the
true key pixels with an IoU of 0.86 for black keys and 0.65 for white keys.

## keymatch.onnx

| | |
|---|---|
| input | `strip`, `[1, 3, 64, 768]`, the keybed rectified flat, far edge at the top |
| output | `heatmaps`, `[1, 3, 768]`, probability along the strip of a white-key gap, a black key's left edge and a black key's right edge |

A small 1D network over a rectified strip. The pipeline reads it on two strips of different
depth, and how far each gap moves between them gives the slant of every key line, which is
how it straightens a skewed outline. On held out synthetic renders it finds edges with a
precision of 0.85 and places them within 0.75 px on average.

## How they were built

All three are trained on synthetic renders. A three.js generator builds a procedural keyboard
with real key proportions and varies its size, case, lighting, background and camera, and
saves every key's exact outline with each frame. The keybed detector is also fine tuned on
hand labelled frames from real recordings.

## Limitations, honestly

- **Mostly synthetic training data.** The models transfer to real cameras, but the far end of
  a keyboard at a steep angle is still where they are weakest: small, distant black keys can
  merge. The pipeline covers that by keeping a key's fitted shape when the segmenter cannot
  separate it.
- **Very low camera angles**, below about 20 degrees above the keys, mostly fail.
- **They only find the keys.** They do not detect hands, read notes, or identify the
  instrument.

## Licence

Apache 2.0.
