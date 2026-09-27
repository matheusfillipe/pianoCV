# pianoCV

Point a webcam at a piano and pianoCV finds every key, live, right in your browser.

**[Try it here](https://matheusfillipe.github.io/pianoCV/)**. Nothing gets uploaded: the
camera never leaves your machine.

Plug in a MIDI keyboard and the keys you play light up where they really are in the picture.
The idea is to draw visuals onto a real piano, like a video filter.

## How it works

A piano segmentation model looks at the camera and marks which pixels are white keys, black
keys, and the gaps between them. Two smaller models help it: one finds roughly where the
keyboard is, and one reads the edges of the keys. Then a bit of geometry fits a real keyboard
layout on top, so every key is found, even the blurry ones far away, and each one gets a
note.

![From a camera frame to every key: find the keyboard, segment every key, draw them all](.github/readme/pipeline.png)

The models run in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/),
on your GPU through [WebGPU](https://developer.mozilla.org/docs/Web/API/WebGPU_API) when it can.
They were trained with [PyTorch](https://pytorch.org) on thousands of fake pianos rendered in
[three.js](https://threejs.org), and they live on
[Hugging Face](https://huggingface.co/mattf/pianoCV). Your hands are found with
[MediaPipe](https://ai.google.dev/edge/mediapipe) so the glow stays behind them, and the
notes come in through [Web MIDI](https://developer.mozilla.org/docs/Web/API/Web_MIDI_API).

## The models

The trained weights are on Hugging Face at
[mattf/pianoCV](https://huggingface.co/mattf/pianoCV), with a
[model card](https://huggingface.co/mattf/pianoCV) that explains what each one takes and
returns:

- [`keyseg.onnx`](https://huggingface.co/mattf/pianoCV/resolve/main/keyseg.onnx), the piano
  segmentation model that marks every key
- [`keybed_seg2.onnx`](https://huggingface.co/mattf/pianoCV/resolve/main/keybed_seg2.onnx),
  which finds roughly where the keyboard is
- [`keymatch.onnx`](https://huggingface.co/mattf/pianoCV/resolve/main/keymatch.onnx), which
  reads the edges of the keys

They are plain [ONNX](https://onnx.ai) files, so you can use them from any language with
[ONNX Runtime](https://onnxruntime.ai).

## Run it yourself

Install [Bun](https://bun.sh/docs/installation) and
[uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```
git clone https://github.com/matheusfillipe/pianoCV.git
cd pianoCV
make install
make model
make dev
```

`make install` sets up the browser app and the Python tools, `make model` downloads the three
models from Hugging Face, and `make dev` starts the app. Open http://localhost:5273/, allow the
camera, and point it at a keyboard. `make help` shows everything else, including how to render
training pianos and train the models yourself.

## Licence

Apache 2.0.
